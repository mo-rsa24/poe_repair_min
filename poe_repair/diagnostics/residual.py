"""Residual / attention diagnostics (minimal).

This module previously housed ~12 diagnostic functions (residual trajectory
norms, SVD energy, in-plane fraction, direction stability, low-pass filtering,
CLIP scoring, etc.) used to characterise the PoE interaction term r_t for
the diagnostic phases. Those phases are deleted; only `attention_overlap` is
kept because the live samplers reference it.
"""

from __future__ import annotations

import torch


def attention_overlap(
    map_a: torch.Tensor,
    map_b: torch.Tensor,
    *,
    eps_denom: float = 1e-12,
) -> float:
    """Cosine similarity between two flattened attention maps. Returns scalar."""
    if map_a.shape != map_b.shape:
        raise ValueError(
            f"shape mismatch: {tuple(map_a.shape)} vs {tuple(map_b.shape)}"
        )
    a = map_a.flatten().float()
    b = map_b.flatten().float()
    num = float((a * b).sum())
    den = float(a.norm() * b.norm())
    if den < eps_denom:
        return 0.0
    return num / den
