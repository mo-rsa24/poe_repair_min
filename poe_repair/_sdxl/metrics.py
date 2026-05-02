"""Score-space helpers (CFG, PoE, Tweedie mean, DDIM step).

Vendored from `prototypes/oracle_gap_sdxl/metrics.py`. Pruned to the four
small helpers used by the samplers.
"""

from __future__ import annotations

import torch


def guided_eps(eps_cond: torch.Tensor, eps_uncond: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)


def poe_eps(eps_a: torch.Tensor, eps_b: torch.Tensor, eps_uncond: torch.Tensor) -> torch.Tensor:
    return eps_a + eps_b - eps_uncond


def tweedie_mean(latents: torch.Tensor, alpha_bar_t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    sqrt_alpha = torch.sqrt(alpha_bar_t)
    sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)
    while sqrt_alpha.ndim < latents.ndim:
        sqrt_alpha = sqrt_alpha.unsqueeze(-1)
        sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
    return (latents - sqrt_one_minus * eps) / sqrt_alpha


def ddim_prev_from_x0_eps(
    *,
    scheduler,
    timestep,
    step_index: int,
    x0: torch.Tensor,
    eps: torch.Tensor,
) -> torch.Tensor:
    if step_index + 1 < len(scheduler.timesteps):
        prev_timestep = int(scheduler.timesteps[step_index + 1].item())
        alpha_prod_prev = scheduler.alphas_cumprod[prev_timestep].to(device=x0.device, dtype=x0.dtype)
    else:
        alpha_prod_prev = torch.as_tensor(scheduler.final_alpha_cumprod, device=x0.device, dtype=x0.dtype)
    beta_prod_prev = 1.0 - alpha_prod_prev
    while alpha_prod_prev.ndim < x0.ndim:
        alpha_prod_prev = alpha_prod_prev.unsqueeze(-1)
        beta_prod_prev = beta_prod_prev.unsqueeze(-1)
    return torch.sqrt(alpha_prod_prev) * x0 + torch.sqrt(beta_prod_prev) * eps
