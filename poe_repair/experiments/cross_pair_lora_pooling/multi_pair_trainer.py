"""Multi-pair train_epoch with per-step embedding lookup.

The base ``lora.trainer.train_epoch`` binds one ``(seq_a, seq_b, seq_e)``
tuple for the whole epoch. For cross-pair pooling, each cached step
may come from a different pair, so the embeddings must vary per step.

Batch K >= 1 is supported by sampling pair-first then K steps from
that pair's subset. Every entry in the batch therefore shares one
embedding tuple, so the existing 3K-wide forward in ``_train_one_step``
works unchanged. Per-step sampling probabilities remain uniform over
the 40-cell pool (pair-uniform * step-uniform-within-pair = global
uniform when pair populations are equal).
"""

from __future__ import annotations

import gc
import logging
import math
import time
from typing import Any

import torch
from torch.optim import AdamW

from poe_repair.experiments.one_pair_one_seed import trainer as lora_trainer
from poe_repair.experiments.one_pair_one_seed.config import RunConfig


log = logging.getLogger(__name__)


def train_epoch_multi_pair(
    *,
    unet: torch.nn.Module,
    scheduler,
    optimizer: AdamW,
    dataset_by_pair: dict[str, list[lora_trainer.CachedStep]],
    embeddings_by_pair: dict[str, dict[str, torch.Tensor]],
    cfg: RunConfig,
    state: lora_trainer.TrainerState,
    device: torch.device,
    train_dtype: torch.dtype,
    rng: torch.Generator,
    grad_scaler: "torch.cuda.amp.GradScaler | None" = None,
    logger_callback=None,
) -> bool:
    """One epoch over the pooled multi-pair dataset.

    Per step:
      1. Sample one pair uniformly from ``dataset_by_pair``.
      2. Sample K entries uniformly from that pair's subset.
      3. Fetch that pair's encoded prompts from ``embeddings_by_pair``.
      4. Forward / backward via ``_train_one_step`` (3K-wide forward
         shares one embedding tuple — that's the constraint).

    Kill criteria and bucket-loss bookkeeping match
    ``lora.trainer.train_epoch`` line-for-line.
    """
    K = int(cfg.schedule.train_batch_size)
    if K < 1:
        raise RuntimeError(f"train_batch_size must be >=1; got {K}")

    pairs = sorted(dataset_by_pair)
    if not pairs:
        raise RuntimeError("dataset_by_pair is empty")
    for p in pairs:
        if not dataset_by_pair[p]:
            raise RuntimeError(f"dataset_by_pair[{p!r}] is empty")
        if p not in embeddings_by_pair:
            raise RuntimeError(
                f"no embeddings for pair={p!r}; "
                f"available={list(embeddings_by_pair)}"
            )

    unet.train()
    if state.bucket_loss_running is None:
        state.bucket_loss_running = {"early": 0.0, "commit": 0.0, "late": 0.0}
        state.bucket_count_running = {"early": 0, "commit": 0, "late": 0}

    n_steps = int(cfg.schedule.epoch_size)
    grad_clip = float(cfg.optim.grad_clip)

    step_t0 = time.time()
    for _ in range(n_steps):
        pair_idx = int(torch.randint(0, len(pairs), (1,), generator=rng).item())
        pair = pairs[pair_idx]
        subset = dataset_by_pair[pair]
        idxs = torch.randint(0, len(subset), (K,), generator=rng).tolist()
        entries = [subset[int(i)] for i in idxs]
        emb = embeddings_by_pair[pair]

        loss, info = lora_trainer._train_one_step(
            unet=unet, scheduler=scheduler, step_entries=entries,
            seq_a=emb["seq_a"], pool_a=emb["pool_a"],
            seq_b=emb["seq_b"], pool_b=emb["pool_b"],
            seq_e=emb["seq_e"], pool_e=emb["pool_e"],
            cfg=cfg, device=device, train_dtype=train_dtype,
        )

        optimizer.zero_grad(set_to_none=True)
        if grad_scaler is not None and grad_scaler.is_enabled():
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            lora_params = [p for n, p in unet.named_parameters() if "lora_" in n and p.requires_grad]
            grad_norm_pre = lora_trainer._global_grad_norm(lora_params)
            torch.nn.utils.clip_grad_norm_(lora_params, grad_clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            lora_params = [p for n, p in unet.named_parameters() if "lora_" in n and p.requires_grad]
            grad_norm_pre = lora_trainer._global_grad_norm(lora_params)
            torch.nn.utils.clip_grad_norm_(lora_params, grad_clip)
            optimizer.step()

        loss_val = float(loss.detach().item())
        # Attribute loss to whichever bucket dominates the batch.
        bucket_counts = {"early": 0, "commit": 0, "late": 0}
        for e in entries:
            bucket_counts[lora_trainer._bucket(
                int(e.step_index),
                commit_window=cfg.probe.commit_window,
            )] += 1
        bucket = max(bucket_counts, key=bucket_counts.get)
        state.bucket_loss_running[bucket] = (
            0.95 * state.bucket_loss_running[bucket] + 0.05 * loss_val
            if state.bucket_count_running[bucket] > 0 else loss_val
        )
        state.bucket_count_running[bucket] += 1
        state.optimizer_step += 1

        if logger_callback is not None:
            logger_callback({
                "train/loss": loss_val,
                "train/loss_bucket/early": state.bucket_loss_running["early"],
                "train/loss_bucket/commit": state.bucket_loss_running["commit"],
                "train/loss_bucket/late": state.bucket_loss_running["late"],
                "train/grad_norm": grad_norm_pre,
                "train/lr": float(optimizer.param_groups[0]["lr"]),
                "train/batch_size": int(info["batch_size"]),
                "train/epoch": state.epoch,
                "train/optimizer_step": state.optimizer_step,
                "train/delta_target_norm": info["delta_target_norm"],
                "train/delta_hat_norm": info["delta_hat_norm"],
                "train/source_pair": pair,
                "train/source_seeds": [int(e.source_seed) for e in entries],
            })

        # Kill criteria (mirror lora.trainer.train_epoch).
        if (
            state.optimizer_step >= int(cfg.kill.after_steps)
            and state.bucket_loss_running["commit"] > float(cfg.kill.loss_threshold)
            and state.bucket_count_running["commit"] >= 20
        ):
            state.aborted_reason = (
                f"commit-bucket loss {state.bucket_loss_running['commit']:.4f} > "
                f"{cfg.kill.loss_threshold} after {state.optimizer_step} steps"
            )
            log.warning("kill: %s", state.aborted_reason)
            return False
        if (
            state.optimizer_step == 200
            and state.bucket_count_running["commit"] >= 20
        ):
            state.initial_commit_loss = state.bucket_loss_running["commit"]
        if (
            state.initial_commit_loss is not None
            and state.optimizer_step >= int(cfg.kill.commit_bucket_halve_after_steps)
            and state.bucket_loss_running["commit"]
            > 0.5 * state.initial_commit_loss
        ):
            state.aborted_reason = (
                f"commit-bucket loss {state.bucket_loss_running['commit']:.4f} did not "
                f"halve from initial {state.initial_commit_loss:.4f} by step "
                f"{state.optimizer_step}"
            )
            log.warning("kill: %s", state.aborted_reason)
            return False

    state.epoch += 1

    if logger_callback is not None:
        elapsed = max(1e-6, time.time() - step_t0)
        logger_callback({
            "train/throughput_steps_per_sec": float(n_steps / elapsed),
            "train/peak_vram_gb": (
                float(torch.cuda.max_memory_allocated() / 1024**3)
                if torch.cuda.is_available() else 0.0
            ),
        })
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        gc.collect()

    return True
