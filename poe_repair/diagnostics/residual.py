"""Residual trajectory utilities.

The central object is the *guided* mono–PoE residual

    r_t = ε̃_J(x_t^PoE, t, e_J) - ε̃_PoE(x_t^PoE, t, e_A, e_B),

an empirical estimate of the guided interaction correction. Both arms are
CFG-guided; both are evaluated at the same latent x_t along the PoE
trajectory. See memory/residual_definition.md for framing.
"""

from __future__ import annotations

import torch


def residual_trajectory(
    eps_j_traj: torch.Tensor,
    eps_poe_traj: torch.Tensor,
) -> torch.Tensor:
    """r_t per step. Shape: [T, C, H, W]."""
    if eps_j_traj.shape != eps_poe_traj.shape:
        raise ValueError(
            f"shape mismatch: eps_j {tuple(eps_j_traj.shape)} vs "
            f"eps_poe {tuple(eps_poe_traj.shape)}"
        )
    return eps_j_traj - eps_poe_traj


def norm_trajectory(traj: torch.Tensor) -> torch.Tensor:
    """L2 norm of each per-step tensor. Shape: [T]."""
    return traj.flatten(1).norm(dim=1)


def relative_norm_trajectory(
    r_traj: torch.Tensor,
    eps_j_traj: torch.Tensor,
    *,
    eps_denom: float = 1e-12,
) -> torch.Tensor:
    """||r_t|| / ||ε̃_J(x_t^PoE)||. Shape: [T]."""
    return norm_trajectory(r_traj) / (norm_trajectory(eps_j_traj) + eps_denom)
