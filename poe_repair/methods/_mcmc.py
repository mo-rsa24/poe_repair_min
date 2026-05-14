"""MCMC correctors on a composed PoE score (Du et al., ICML 2023).

Score-only ports of the AnnealedULA / AnnealedUHA samplers from
``compositions/reduce_reuse_recycle/anneal_samplers.py`` (lines 218-242
for ULA, 173-214 for UHA). The Metropolis-corrected variants (MALA,
CHA) need an energy and SDXL is score-parameterised, so they are not
ported.

The kernel takes a ``score_fn(x) -> ∇_x log p_t(x)`` callable. On
diffusion models trained with ε prediction,
``score = -ε(x_t) / sqrt(1 - α_bar_t)``. The PoE composed score is
``-(ε_a + ε_b - ε_∅) / σ_t`` evaluated at the *same* noise level σ_t.

Step-size schedule follows Song & Ermon's annealed Langevin recipe:
``η_t = step_size_base · σ_t^2``. The base coefficient is the only
tuning knob; defaults are loose and intended to be swept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class MCMCCorrectorConfig:
    method: str = "ula"                # "ula" | "uha" | "none"
    num_corrector_steps: int = 5
    step_size_base: float = 1.0e-3     # η_t = step_size_base · σ_t^2
    window: tuple[int, int] | None = None   # active step-index range, half-open
    damping: float = 0.5               # UHA only: partial momentum refresh
    num_leapfrog: int = 3              # UHA only
    mass_diag_sqrt: float = 1.0        # UHA only


def ula_step(
    x: torch.Tensor,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    step_size: float,
) -> torch.Tensor:
    """One unadjusted-Langevin step: x ← x + η · s(x) + √(2η) · ξ."""
    score = score_fn(x)
    noise = torch.randn_like(x) * (2.0 * step_size) ** 0.5
    return x + step_size * score + noise


def uha_step(
    x: torch.Tensor,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    step_size: float,
    *,
    damping: float,
    num_leapfrog: int,
    mass_diag_sqrt: float,
) -> torch.Tensor:
    """One underdamped-HMC step with partial momentum refresh.

    No Metropolis correction (UHA, not HMC). Velocity is refreshed at
    each outer call; leapfrog integrates the Hamiltonian.
    """
    M = mass_diag_sqrt
    v = torch.randn_like(x) * M
    eps = torch.randn_like(x)
    v = damping * v + ((1.0 - damping ** 2) ** 0.5) * eps * M

    x_k, v_k = x, v
    grad = score_fn(x_k)
    mass_diag = M ** 2
    for _ in range(num_leapfrog):
        v_k = v_k + 0.5 * step_size * grad
        x_k = x_k + step_size * v_k / mass_diag
        grad = score_fn(x_k)
        v_k = v_k + 0.5 * step_size * grad
    return x_k


def run_corrector(
    x: torch.Tensor,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    sigma_t: float,
    cfg: MCMCCorrectorConfig,
) -> torch.Tensor:
    """K corrector steps at noise level σ_t. ``score_fn`` evaluates the
    composed score at noise level σ_t."""
    if cfg.method == "none" or cfg.num_corrector_steps <= 0:
        return x
    step_size = cfg.step_size_base * (sigma_t ** 2)
    if cfg.method == "ula":
        for _ in range(cfg.num_corrector_steps):
            x = ula_step(x, score_fn, step_size)
        return x
    if cfg.method == "uha":
        for _ in range(cfg.num_corrector_steps):
            x = uha_step(
                x, score_fn, step_size,
                damping=cfg.damping,
                num_leapfrog=cfg.num_leapfrog,
                mass_diag_sqrt=cfg.mass_diag_sqrt,
            )
        return x
    raise ValueError(f"unknown corrector method {cfg.method!r}")


def step_in_window(step_index: int, window: tuple[int, int] | None) -> bool:
    if window is None:
        return True
    lo, hi = window
    return lo <= step_index < hi
