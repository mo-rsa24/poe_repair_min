"""LoRA trainer: LoRA injection + training loop on the cached (x_t, t, Δ_t) set.

Single-cell training. The cache holds (50 steps) raw eps tensors and x_t;
we compute Δ_t in closed form per step. LoRA is attached to SDXL UNet
cross-attention projections (``attn2``) at rank 8. The loss is MSE on the
guided residual:

    Δ̂ = guided(ε_LoRA(joint)) − guided_PoE(ε_a, ε_b, ε_uncond)
    loss = ‖Δ̂ − Δ_t‖²

Algebraically this reduces to ``MSE(eps_lora_raw, eps_j_raw_cached)``
(see consolidated plan §1.3 equivalence note), but we keep the explicit
residual form so wiring bugs in either branch surface clearly.
"""

from __future__ import annotations

import gc
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from poe_repair.experiments.lora.config import RunConfig
from poe_repair.training_cache import (
    CellPath,
    delta_t_from_raw,
    load_step_raw,
)
from poe_repair.runtime import guided_eps, poe_eps


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class CachedStep:
    step_index: int
    timestep: int
    x_t: torch.Tensor          # (1, 4, H, W), float32 CPU
    eps_a_raw: torch.Tensor
    eps_b_raw: torch.Tensor
    eps_j_raw: torch.Tensor
    eps_uncond: torch.Tensor
    delta_t: torch.Tensor      # guided Δ_t target
    source_seed: int = -1      # -1 = legacy single-seed (unset)


def load_cached_steps(
    cell: CellPath,
    *,
    guidance_scale: float,
) -> list[CachedStep]:
    """Load all step files for a cell. ~250 MB for 50 SDXL steps."""
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
                eps_a_raw=raw["eps_a_raw"],
                eps_b_raw=raw["eps_b_raw"],
                eps_j_raw=raw["eps_j_raw"],
                eps_uncond=raw["eps_uncond"],
                delta_t=delta,
                source_seed=int(cell.seed),
            )
        )
    if not entries:
        raise FileNotFoundError(
            f"no cached steps under {cell.residuals_dir}"
        )
    return entries


def load_cached_steps_pooled(
    cells: list[CellPath],
    *,
    guidance_scale: float,
) -> list[CachedStep]:
    """Concatenate cached steps across multiple cells. Order: cells[0]'s
    steps first, then cells[1], etc. Each entry tagged with ``source_seed``.
    Uniform random sampling over the returned list gives per-seed mixing
    for free (no special scheduler needed)."""
    out: list[CachedStep] = []
    for c in cells:
        out.extend(load_cached_steps(c, guidance_scale=guidance_scale))
    if not out:
        raise FileNotFoundError(
            f"no cached steps across {len(cells)} cells"
        )
    return out


# ---------------------------------------------------------------------------
# Bucket helpers
# ---------------------------------------------------------------------------


def _bucket(step_index: int, *, commit_window: tuple[int, int]) -> str:
    lo, hi = int(commit_window[0]), int(commit_window[1])
    if step_index < lo:
        return "early"
    if step_index < hi:
        return "commit"
    return "late"


# ---------------------------------------------------------------------------
# LoRA injection
# ---------------------------------------------------------------------------


def attach_lora(unet: torch.nn.Module, cfg: RunConfig) -> dict[str, Any]:
    """Attach a LoRA adapter to the UNet's cross-attention projections.

    Returns a dict with ``{"adapter_name", "matched_modules",
    "trainable_params"}`` for sanity-printing at run start.
    """
    from peft import LoraConfig

    lora_cfg = LoraConfig(
        r=int(cfg.lora.rank),
        lora_alpha=int(cfg.lora.alpha),
        lora_dropout=float(cfg.lora.dropout),
        bias="none",
        target_modules=list(cfg.lora.target_modules),
        init_lora_weights=(
            True if cfg.lora.init == "gaussian"
            else cfg.lora.init  # passthrough for "loftq" / etc.
        ),
    )

    # Diffusers UNet PeftAdapterMixin: ``add_adapter`` attaches the LoRA
    # and registers it under ``adapter_name``.
    if not hasattr(unet, "add_adapter"):
        raise RuntimeError(
            "UNet does not expose add_adapter — diffusers >=0.21 is required."
        )
    unet.add_adapter(lora_cfg, adapter_name=cfg.lora.adapter_name)

    # A module is "matched" if it has injected lora_A / lora_B sub-modules.
    matched: list[str] = []
    for name, module in unet.named_modules():
        sub_names = {n for n, _ in module.named_children()}
        if "lora_A" in sub_names and "lora_B" in sub_names:
            matched.append(name)

    # Enable grad on adapter params only. Cast LoRA params to fp32 — fp16
    # LoRA training underflows immediately (loss NaN at step 1). Frozen
    # base UNet stays in its original dtype; peft's LoraLayer.forward casts
    # the LoRA delta back to the base layer's input dtype before adding,
    # so the mixed-precision path is safe.
    n_trainable = 0
    n_total = 0
    with torch.no_grad():
        for name, p in unet.named_parameters():
            n_total += p.numel()
            if "lora_" in name:
                p.data = p.data.to(torch.float32)
                p.requires_grad_(True)
                n_trainable += p.numel()
            else:
                p.requires_grad_(False)

    if n_trainable == 0:
        raise RuntimeError(
            "LoRA attach produced 0 trainable parameters. "
            "Check target_modules regex — SDXL cross-attn lives at attn2."
        )

    return {
        "adapter_name": cfg.lora.adapter_name,
        "matched_modules": matched,
        "n_matched": len(matched),
        "trainable_params": int(n_trainable),
        "total_params": int(n_total),
    }


def lora_state_dict(unet: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Extract LoRA-only state for checkpointing (~20 MB at rank 8)."""
    out: dict[str, torch.Tensor] = {}
    for name, p in unet.named_parameters():
        if "lora_" in name:
            out[name] = p.detach().to("cpu", torch.float32).clone()
    return out


def load_lora_state(unet: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Inverse of ``lora_state_dict``. Tensors are placed on the matching
    module's device/dtype."""
    own_params = dict(unet.named_parameters())
    with torch.no_grad():
        for name, t in state.items():
            if name not in own_params:
                continue
            p = own_params[name]
            p.copy_(t.to(device=p.device, dtype=p.dtype))


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def _build_added_cond_kwargs_3branch(
    pool_a: torch.Tensor, pool_b: torch.Tensor, pool_e: torch.Tensor,
    *,
    height: int, width: int,
    device: torch.device, dtype: torch.dtype,
) -> tuple[torch.Tensor, dict]:
    """Concat 3-branch (A, B, ∅) embeddings + the time-id batch."""
    from poe_repair.methods._sampling import add_time_ids
    pool_3 = torch.cat([pool_a, pool_b, pool_e], dim=0).to(device=device, dtype=dtype)
    cond = {
        "text_embeds": pool_3,
        "time_ids": add_time_ids(
            height=height, width=width, batch_size=3, device=device, dtype=dtype,
        ),
    }
    return pool_3, cond


def _train_one_step(
    *,
    unet: torch.nn.Module,
    scheduler,
    step_entries: list[CachedStep],
    seq_a: torch.Tensor, pool_a: torch.Tensor,
    seq_b: torch.Tensor, pool_b: torch.Tensor,
    seq_e: torch.Tensor, pool_e: torch.Tensor,
    cfg: RunConfig,
    device: torch.device,
    train_dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One forward+backward step over a batch of K cached entries.

    Per cached ``(x_t, t)`` in the batch:
      1. Run LoRA-enabled UNet on (A, B, ∅) → ε_a_lora, ε_b_lora, ε_∅_lora.
      2. Compose ε̃_PoE_lora = w·(ε̃_a + ε̃_b − ε_∅).
      3. Target ε̃_J_cached (guided from cached eps_j_raw + cached eps_∅).
      4. Loss_k = ‖ε̃_PoE_lora − ε̃_J_cached‖² for cached step k.
      5. Total loss = mean over batch.

    All K samples run through a single 3K-wide UNet forward; each sample's
    A/B/∅ branches sit at contiguous indices and use their own timestep.
    Wall-clock per optimizer step grows sub-linearly with K because the
    UNet underutilises the GPU at batch=1.
    """
    assert len(step_entries) >= 1
    K = len(step_entries)
    gs = float(cfg.sampler.guidance_scale)
    H, W = cfg.sampler.height, cfg.sampler.width

    # Build the batched 3K input. Per-sample tensors are concatenated in
    # blocks of 3 (A, B, ∅), so the output can be reshaped (K, 3, ...).
    with torch.no_grad():
        x_t_list, eps_j_target_list, eps_poe_frozen_list = [], [], []
        timestep_per_sample = []
        delta_target_norms = []
        for e in step_entries:
            x_t   = e.x_t.to(device=device, dtype=train_dtype)
            ej    = e.eps_j_raw.to(device=device, dtype=train_dtype)
            eu    = e.eps_uncond.to(device=device, dtype=train_dtype)
            ea    = e.eps_a_raw.to(device=device, dtype=train_dtype)
            eb    = e.eps_b_raw.to(device=device, dtype=train_dtype)
            ej_g  = guided_eps(ej, eu, gs)
            ea_g  = guided_eps(ea, eu, gs)
            eb_g  = guided_eps(eb, eu, gs)
            x_t_list.append(x_t.repeat(3, 1, 1, 1))           # (3, 4, H, W)
            eps_j_target_list.append(ej_g)                    # (1, 4, H, W)
            eps_poe_frozen_list.append(poe_eps(ea_g, eb_g, eu))
            timestep_per_sample.append(int(e.timestep))
            delta_target_norms.append(float(e.delta_t.float().norm().item()))

        latent_input_3K = torch.cat(x_t_list, dim=0)          # (3K, 4, H, W)
        eps_j_target = torch.cat(eps_j_target_list, dim=0)    # (K, 4, H, W)
        eps_poe_frozen = torch.cat(eps_poe_frozen_list, dim=0)  # (K, 4, H, W)

        # Per-sample timesteps, expanded to 3K for the 3-branch repeat.
        timestep_b = torch.tensor(
            timestep_per_sample, device=device, dtype=torch.long,
        ).repeat_interleave(3)                                 # (3K,)

        # DDIM scale_model_input is a no-op for our scheduler; still call it
        # so any future swap to a non-DDIM scheduler stays correct.
        latent_input_3K = scheduler.scale_model_input(latent_input_3K, timestep_b[0])

        pe_3 = torch.cat([seq_a, seq_b, seq_e], dim=0).to(device=device, dtype=train_dtype)
        encoder_3K = pe_3.repeat(K, 1, 1)                     # (3K, 77, 2048)
        from poe_repair.methods._sampling import add_time_ids
        pool_3 = torch.cat([pool_a, pool_b, pool_e], dim=0).to(device=device, dtype=train_dtype)
        cond_3K = {
            "text_embeds": pool_3.repeat(K, 1),               # (3K, 1280)
            "time_ids": add_time_ids(
                height=H, width=W, batch_size=3 * K,
                device=device, dtype=train_dtype,
            ),
        }

    # LoRA-enabled 3K-wide forward. Cast UNet output to fp32 immediately —
    # fp16 cancellation otherwise crushes gradients.
    noise = unet(
        latent_input_3K, timestep_b, encoder_hidden_states=encoder_3K,
        added_cond_kwargs=cond_3K, timestep_cond=None,
    ).sample.float()                                          # (3K, 4, H, W)

    # Reshape into (K, 3, 4, H, W) → split A / B / ∅ along the 3-axis.
    noise = noise.view(K, 3, *noise.shape[1:])
    eps_a_raw_l = noise[:, 0]
    eps_b_raw_l = noise[:, 1]
    eps_uncond_l = noise[:, 2]
    eps_a_l = guided_eps(eps_a_raw_l, eps_uncond_l, gs)
    eps_b_l = guided_eps(eps_b_raw_l, eps_uncond_l, gs)
    eps_poe_lora = poe_eps(eps_a_l, eps_b_l, eps_uncond_l)
    loss = F.mse_loss(eps_poe_lora, eps_j_target.float())

    info = {
        "step_indices": [int(e.step_index) for e in step_entries],
        "timesteps": timestep_per_sample,
        "batch_size": K,
        "delta_target_norm": float(sum(delta_target_norms) / K),
        "delta_hat_norm": float(
            (eps_poe_lora - eps_poe_frozen.float()).norm().item() / K
        ),
    }
    return loss, info


# ---------------------------------------------------------------------------
# Top-level training driver
# ---------------------------------------------------------------------------


@dataclass
class TrainerState:
    optimizer_step: int = 0
    epoch: int = 0
    bucket_loss_running: dict[str, float] | None = None
    bucket_count_running: dict[str, int] | None = None
    initial_commit_loss: float | None = None
    aborted_reason: str | None = None


def make_optimizer(unet: torch.nn.Module, cfg: RunConfig) -> AdamW:
    lora_params = [p for n, p in unet.named_parameters() if "lora_" in n and p.requires_grad]
    return AdamW(
        lora_params,
        lr=float(cfg.optim.lr),
        weight_decay=float(cfg.optim.weight_decay),
        betas=tuple(cfg.optim.betas),
    )


def make_grad_scaler(
    enabled: bool = True,
    *,
    init_scale: float = 2.0 ** 16,
):
    """fp16 LoRA training underflows in the backward path through 70
    cross-attn blocks of fp16 matmul. GradScaler scales the loss by a
    large factor pre-backward, then unscales before the optimizer step.
    """
    # torch.amp.GradScaler in torch ≥ 2.4; fall back to the older path on
    # earlier torch versions.
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled, init_scale=init_scale)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled, init_scale=init_scale)


def _global_grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        total += float(p.grad.detach().float().norm().item()) ** 2
    return math.sqrt(total)


def train_epoch(
    *,
    unet: torch.nn.Module,
    scheduler,
    optimizer: AdamW,
    dataset: list[CachedStep],
    seq_a: torch.Tensor, pool_a: torch.Tensor,
    seq_b: torch.Tensor, pool_b: torch.Tensor,
    seq_e: torch.Tensor, pool_e: torch.Tensor,
    cfg: RunConfig,
    state: TrainerState,
    device: torch.device,
    train_dtype: torch.dtype,
    rng: torch.Generator,
    grad_scaler: "torch.cuda.amp.GradScaler | None" = None,
    logger_callback=None,
) -> bool:
    """One epoch = ``cfg.schedule.epoch_size`` optimizer steps over uniformly-sampled
    cached steps. Returns ``True`` if training should continue, ``False`` if
    a kill criterion triggered.
    """
    unet.train()
    if state.bucket_loss_running is None:
        state.bucket_loss_running = {"early": 0.0, "commit": 0.0, "late": 0.0}
        state.bucket_count_running = {"early": 0, "commit": 0, "late": 0}

    n_steps = int(cfg.schedule.epoch_size)
    grad_clip = float(cfg.optim.grad_clip)

    step_t0 = time.time()
    batch_K = int(cfg.schedule.train_batch_size)
    for _ in range(n_steps):
        idxs = torch.randint(0, len(dataset), (batch_K,), generator=rng).tolist()
        entries = [dataset[int(i)] for i in idxs]
        loss, info = _train_one_step(
            unet=unet, scheduler=scheduler, step_entries=entries,
            seq_a=seq_a, pool_a=pool_a,
            seq_b=seq_b, pool_b=pool_b,
            seq_e=seq_e, pool_e=pool_e,
            cfg=cfg, device=device, train_dtype=train_dtype,
        )

        optimizer.zero_grad(set_to_none=True)
        if grad_scaler is not None and grad_scaler.is_enabled():
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            lora_params = [p for n, p in unet.named_parameters() if "lora_" in n and p.requires_grad]
            grad_norm_pre = _global_grad_norm(lora_params)
            torch.nn.utils.clip_grad_norm_(lora_params, grad_clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            lora_params = [p for n, p in unet.named_parameters() if "lora_" in n and p.requires_grad]
            grad_norm_pre = _global_grad_norm(lora_params)
            torch.nn.utils.clip_grad_norm_(lora_params, grad_clip)
            optimizer.step()

        loss_val = float(loss.detach().item())
        # Attribute the loss to whichever bucket is most represented in the
        # batch (with batch=1 this is just the single entry's bucket).
        bucket_counts = {"early": 0, "commit": 0, "late": 0}
        for s_idx in info["step_indices"]:
            bucket_counts[_bucket(int(s_idx), commit_window=cfg.probe.commit_window)] += 1
        bucket = max(bucket_counts, key=bucket_counts.get)
        state.bucket_loss_running[bucket] = (
            0.95 * state.bucket_loss_running[bucket] + 0.05 * loss_val
            if state.bucket_count_running[bucket] > 0 else loss_val
        )
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
                    "train/delta_target_norm": info["delta_target_norm"],
                    "train/delta_hat_norm": info["delta_hat_norm"],
                }
            )

        # Kill criteria.
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
