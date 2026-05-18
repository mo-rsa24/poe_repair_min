"""Group A trainer — dataset loader, sampling schedule, training step, kill criteria.

The trainer does not invoke SDXL during training for A1 / A2 — the cache
already contains the raw ε tensors needed to compute ``r_t`` in closed
form. For A3 (frozen-feature MLP), the corrector's own forward runs
SDXL internally; the trainer just passes ``(z_t, t, e_J_seq, e_J_pool)``.
"""

from __future__ import annotations

import gc
import logging
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from poe_repair.experiments.group_a_failure.config import RunConfig
from poe_repair.training_cache import (
    CellPath,
    delta_t_from_raw,
    load_step_raw,
)
from poe_repair.students import (
    build_frozen_feature_mlp,
    build_latent_cnn,
    build_latent_unet,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class CachedStep:
    step_index: int
    timestep: int
    x_t: torch.Tensor          # (1, 4, H, W), float32 CPU
    delta_t: torch.Tensor      # (1, 4, H, W), float32 CPU — the guided residual r_t


def load_cached_steps(
    cell: CellPath,
    *,
    guidance_scale: float,
) -> list[CachedStep]:
    """Load all cached steps and pre-compute r_t = ε̃_J - ε̃_PoE per step."""
    entries: list[CachedStep] = []
    for path in cell.step_files():
        raw = load_step_raw(path)
        delta = delta_t_from_raw(
            raw["eps_a_raw"], raw["eps_b_raw"],
            raw["eps_j_raw"], raw["eps_uncond"],
            guidance_scale,
        )
        entries.append(
            CachedStep(
                step_index=raw["step_index"],
                timestep=raw["timestep"],
                x_t=raw["x_t"],
                delta_t=delta,
            )
        )
    if not entries:
        raise FileNotFoundError(f"no cached steps under {cell.residuals_dir}")
    entries.sort(key=lambda e: e.step_index)
    return entries


# ---------------------------------------------------------------------------
# t sampler
# ---------------------------------------------------------------------------


@dataclass
class TSampler:
    weights: torch.Tensor      # (N,), sums to 1
    indices: torch.Tensor      # (N,), index of dataset[i]
    rng: torch.Generator

    @classmethod
    def build(
        cls,
        dataset: list[CachedStep],
        *,
        mode: str,
        commit_window: tuple[int, int],
        floor: float,
        seed: int,
    ) -> "TSampler":
        n = len(dataset)
        indices = torch.arange(n)
        if mode == "uniform":
            weights = torch.ones(n)
        elif mode == "commit_window":
            lo, hi = int(commit_window[0]), int(commit_window[1])
            weights = torch.tensor(
                [1.0 if (lo <= e.step_index < hi) else 0.0 for e in dataset]
            )
            # If commit-only would zero everything (shouldn't happen), fall back.
            if float(weights.sum().item()) <= 0.0:
                weights = torch.ones(n)
        elif mode == "sigma_weighted":
            weights = torch.tensor(
                [float(e.delta_t.float().norm().item()) for e in dataset]
            )
            if float(weights.sum().item()) <= 0.0:
                weights = torch.ones(n)
        else:
            raise ValueError(f"unknown t_sampler mode: {mode!r}")
        weights = weights.float()
        weights = weights / weights.sum()
        if floor > 0.0:
            # Mix in a uniform floor so tail timesteps still see updates.
            uniform = torch.full_like(weights, 1.0 / n)
            weights = (1.0 - floor) * weights + floor * uniform
            weights = weights / weights.sum()
        rng = torch.Generator(device="cpu").manual_seed(int(seed))
        return cls(weights=weights, indices=indices, rng=rng)

    def sample_indices(self, k: int) -> list[int]:
        idxs = torch.multinomial(self.weights, k, replacement=True, generator=self.rng)
        return [int(i) for i in idxs.tolist()]


# ---------------------------------------------------------------------------
# Corrector factory
# ---------------------------------------------------------------------------


def build_corrector(
    cfg: RunConfig,
    *,
    unet: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """Construct the corrector specified by ``cfg.technique.name``.

    For A1/A2 the corrector is independent of SDXL. For A3 it holds a
    reference to the frozen UNet and installs a forward hook on its
    mid-block.
    """
    tc = cfg.technique
    name = tc.name
    if name == "latent_cnn":
        net = build_latent_cnn(
            body_channels=int(tc.body_channels),
            n_blocks=int(tc.n_blocks),
            cond_dim=int(tc.cond_dim),
        )
    elif name == "latent_unet":
        net = build_latent_unet(
            base_channels=int(tc.base_channels),
            blocks_per_level=int(tc.blocks_per_level),
            cond_dim=int(tc.cond_dim),
        )
    elif name == "frozen_feature_mlp":
        net = build_frozen_feature_mlp(
            unet,
            feature_channels=int(tc.feature_channels),
            head_channels=int(tc.head_channels),
            n_blocks=int(tc.head_blocks),
            cond_dim=int(tc.cond_dim),
            height=int(cfg.sampler.height),
            width=int(cfg.sampler.width),
            unet_dtype=dtype,
        )
    else:
        raise ValueError(f"unknown technique: {name!r}")
    # Train the head in fp32. The body of A1/A2 already casts to fp32
    # internally; A3's head also operates on fp32 features.
    net.to(device=device, dtype=torch.float32)
    return net


def attach_summary(net: nn.Module, cfg: RunConfig) -> dict:
    return {
        "technique": cfg.technique.name,
        "num_parameters": int(sum(p.numel() for p in net.parameters())),
        "num_trainable": int(sum(p.numel() for p in net.parameters() if p.requires_grad)),
        "config": {
            "body_channels": cfg.technique.body_channels,
            "n_blocks": cfg.technique.n_blocks,
            "base_channels": cfg.technique.base_channels,
            "blocks_per_level": cfg.technique.blocks_per_level,
            "feature_channels": cfg.technique.feature_channels,
            "head_channels": cfg.technique.head_channels,
            "head_blocks": cfg.technique.head_blocks,
            "cond_dim": cfg.technique.cond_dim,
        },
    }


# ---------------------------------------------------------------------------
# Trainer state
# ---------------------------------------------------------------------------


@dataclass
class TrainerState:
    optimizer_step: int = 0
    epoch: int = 0
    bucket_loss_running: dict | None = None
    bucket_count_running: dict | None = None
    r_collapse_streak: int = 0
    aborted_reason: str | None = None


def make_optimizer(net: nn.Module, cfg: RunConfig) -> AdamW:
    params = [p for p in net.parameters() if p.requires_grad]
    return AdamW(
        params,
        lr=float(cfg.optim.lr),
        weight_decay=float(cfg.optim.weight_decay),
        betas=tuple(cfg.optim.betas),
    )


def _bucket(step_index: int, *, commit_window: tuple[int, int]) -> str:
    lo, hi = int(commit_window[0]), int(commit_window[1])
    if step_index < lo:
        return "early"
    if step_index < hi:
        return "commit"
    return "late"


def _global_grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        total += float(p.grad.detach().float().norm().item()) ** 2
    return math.sqrt(total)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def _train_one_step(
    *,
    net: nn.Module,
    dataset: list[CachedStep],
    sampler: TSampler,
    seq_j: torch.Tensor, pool_j: torch.Tensor,
    cfg: RunConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    K = int(cfg.schedule.train_batch_size)
    idxs = sampler.sample_indices(K)
    entries = [dataset[i] for i in idxs]

    # Batch the cached tensors. Targets and z_t live in fp32 throughout.
    z_t = torch.cat([e.x_t for e in entries], dim=0).to(device=device, dtype=torch.float32)
    r_t = torch.cat([e.delta_t for e in entries], dim=0).to(device=device, dtype=torch.float32)
    t = torch.tensor([int(e.timestep) for e in entries], device=device, dtype=torch.long)
    # Broadcast joint conditioning across the batch.
    seq_b = seq_j.to(device=device).expand(K, -1, -1).contiguous()
    pool_b = pool_j.to(device=device).expand(K, -1).contiguous()

    r_hat = net(z_t, t, seq_b, pool_b)
    loss = F.mse_loss(r_hat.float(), r_t.float())

    r_hat_norms = r_hat.float().flatten(1).norm(dim=1)
    r_t_norms = r_t.float().flatten(1).norm(dim=1)
    info = {
        "step_indices": [int(e.step_index) for e in entries],
        "batch_size": K,
        "r_hat_norm_mean": float(r_hat_norms.mean().item()),
        "r_t_norm_mean": float(r_t_norms.mean().item()),
    }
    return loss, info


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_epoch(
    *,
    net: nn.Module,
    optimizer: AdamW,
    dataset: list[CachedStep],
    sampler: TSampler,
    seq_j: torch.Tensor, pool_j: torch.Tensor,
    cfg: RunConfig,
    state: TrainerState,
    device: torch.device,
    logger_callback=None,
) -> bool:
    """One epoch = ``cfg.schedule.epoch_size`` optimizer steps. Returns
    True if training should continue, False if a kill criterion fires.
    """
    net.train()
    if state.bucket_loss_running is None:
        state.bucket_loss_running = {"early": 0.0, "commit": 0.0, "late": 0.0}
        state.bucket_count_running = {"early": 0, "commit": 0, "late": 0}

    n_steps = int(cfg.schedule.epoch_size)
    grad_clip = float(cfg.optim.grad_clip)
    t0 = time.time()

    for _ in range(n_steps):
        loss, info = _train_one_step(
            net=net, dataset=dataset, sampler=sampler,
            seq_j=seq_j, pool_j=pool_j,
            cfg=cfg, device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = [p for p in net.parameters() if p.requires_grad]
        grad_norm_pre = _global_grad_norm(params)
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()

        loss_val = float(loss.detach().item())
        bucket_counts = {"early": 0, "commit": 0, "late": 0}
        for s_idx in info["step_indices"]:
            bucket_counts[_bucket(int(s_idx), commit_window=cfg.probe.commit_window)] += 1
        bucket = max(bucket_counts, key=bucket_counts.get)
        if state.bucket_count_running[bucket] > 0:
            state.bucket_loss_running[bucket] = (
                0.95 * state.bucket_loss_running[bucket] + 0.05 * loss_val
            )
        else:
            state.bucket_loss_running[bucket] = loss_val
        state.bucket_count_running[bucket] += 1

        state.optimizer_step += 1

        if logger_callback is not None:
            logger_callback(
                {
                    "train/loss": loss_val,
                    "train/loss_bucket/early": state.bucket_loss_running["early"],
                    "train/loss_bucket/commit": state.bucket_loss_running["commit"],
                    "train/loss_bucket/late": state.bucket_loss_running["late"],
                    "train/grad_norm": grad_norm_pre,
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                    "train/batch_size": int(info["batch_size"]),
                    "train/epoch": state.epoch,
                    "train/optimizer_step": state.optimizer_step,
                    "train/r_hat_norm_mean": info["r_hat_norm_mean"],
                    "train/r_t_norm_mean": info["r_t_norm_mean"],
                }
            )

    state.epoch += 1

    if logger_callback is not None:
        elapsed = max(1e-6, time.time() - t0)
        logger_callback(
            {
                "train/throughput_steps_per_sec": float(n_steps / elapsed),
                "train/peak_vram_gb": (
                    float(torch.cuda.max_memory_allocated() / 1024**3)
                    if torch.cuda.is_available() else 0.0
                ),
            }
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        gc.collect()

    return True
