"""Predicted clean latent x̂₀(t) used by the semantic-MDS pipeline.

For SDXL DDIM eps-prediction (the only path used in this repo) the per-step
predicted clean latent is

    x̂₀(t) = (z_t − √(1−ᾱ_t) · ε̂_t) / √ᾱ_t

which is exactly what ``tweedie_mean`` computes — this module just exposes
that formula under a name that matches the semantic-MDS plan (Plan 13) and
adds an ``alpha_bar_from_scheduler`` helper so callers do not have to
re-derive the lookup convention every time.

Used by ``scripts/build_lora_inspector_mds_semantic.py`` to reconstruct
x̂₀(t) at subsampled timesteps from a ``LatentTrajectoryCollector`` that
ran the actual sampler. The tracker's ``velocities[step]`` field holds the
**guided** eps used to step (e.g. eps_PoE for the PoE sampler, the guided
eps_t for CFG), so x̂₀ here is the model's *guided* prediction, which is
the right object to embed for the convergence claim.
"""
from __future__ import annotations

import torch

from .metrics import tweedie_mean


def alpha_bar_from_scheduler(
    scheduler,
    timestep: int | float | torch.Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``scheduler.alphas_cumprod[t]`` cast to ``device``/``dtype``.

    The diffusers ``DDIMScheduler.alphas_cumprod`` is indexed by integer
    timestep (1000-step Markov index), which matches what the
    ``LatentTrajectoryCollector`` stores in its ``timesteps`` array.
    """
    t = int(timestep.item() if isinstance(timestep, torch.Tensor) else timestep)
    ab = scheduler.alphas_cumprod[t]
    if device is not None or dtype is not None:
        ab = ab.to(device=device, dtype=dtype)
    return ab


def predicted_x0(
    latents: torch.Tensor,
    eps: torch.Tensor,
    alpha_bar_t: torch.Tensor,
) -> torch.Tensor:
    """x̂₀(t) under eps-prediction. Thin alias over ``tweedie_mean``."""
    return tweedie_mean(latents, alpha_bar_t, eps)
