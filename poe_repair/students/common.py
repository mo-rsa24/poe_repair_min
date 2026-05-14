"""Shared building blocks for Group A external score-space correctors.

Provides:
- ``sinusoidal_time_embedding``: timestep encoder (same form SDXL uses).
- ``FiLMConditioning``: pooled-text + time → per-block (γ, β) projector.
- ``FiLMConvBlock``: GroupNorm → SiLU → 3x3 Conv → FiLM(γ, β) → residual.
- ``mean_pool_sequence``: token-axis mean-pool of ``e_J_seq`` for conditioning.

All three Group A correctors (A1 latent CNN, A2 latent UNet, A3 frozen-feature
MLP head) take ``(z_t, t, e_J_seq, e_J_pool)`` and emit ``r̂_t`` with the same
spatial shape as ``z_t``. They differ only in their internal body.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Timestep + text conditioning
# ---------------------------------------------------------------------------


def sinusoidal_time_embedding(
    timesteps: torch.Tensor,
    dim: int,
    *,
    max_period: float = 10_000.0,
) -> torch.Tensor:
    """Standard DDPM/DiT sinusoidal positional embedding for an integer
    timestep tensor of shape (B,). Returns (B, dim).
    """
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal_time_embedding dim must be even, got {dim}")
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=device)
        / half
    )
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def mean_pool_sequence(seq: torch.Tensor) -> torch.Tensor:
    """Mean-pool over the token axis of (B, T, D) → (B, D)."""
    return seq.mean(dim=1)


class FiLMConditioning(nn.Module):
    """Project ``(t, e_J_pool, mean_pool(e_J_seq))`` to a fixed-width conditioning
    vector that downstream FiLM blocks consume.

    Output dimension is ``cond_dim``. Per-block FiLM layers learn their own
    ``(γ, β)`` from this shared vector.
    """

    def __init__(
        self,
        *,
        time_dim: int = 256,
        pool_dim: int = 1280,
        seq_dim: int = 2048,
        cond_dim: int = 256,
    ):
        super().__init__()
        self.time_dim = int(time_dim)
        self.cond_dim = int(cond_dim)
        # Time MLP: sinusoidal → MLP → time_dim.
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        # Text projection: concat(pool, seq_mean) → cond_dim, mixed with time.
        self.text_proj = nn.Linear(pool_dim + seq_dim, time_dim)
        self.merge = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(
        self,
        t: torch.Tensor,            # (B,) long
        e_J_pool: torch.Tensor,     # (B, 1280)
        e_J_seq: torch.Tensor,      # (B, 77, 2048)
    ) -> torch.Tensor:               # (B, cond_dim)
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        text = torch.cat([e_J_pool.float(), mean_pool_sequence(e_J_seq).float()], dim=-1)
        text_emb = self.text_proj(text)
        return self.merge(torch.cat([t_emb, text_emb], dim=-1))


class FiLMConvBlock(nn.Module):
    """GroupNorm → SiLU → 3×3 Conv → FiLM(γ, β) → residual.

    Channels in and out are equal. Use ``DownBlock`` / ``UpBlock`` wrappers
    (in the UNet student) for resolution changes.
    """

    def __init__(self, channels: int, *, cond_dim: int, groups: int = 32):
        super().__init__()
        groups = min(groups, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.film = nn.Linear(cond_dim, 2 * channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(h)
        h = self.conv(h)
        gamma_beta = self.film(cond).to(h.dtype)        # (B, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        h = (1.0 + gamma) * h + beta
        return x + h


class DownBlock(nn.Module):
    """Strided 3×3 conv ×2 downsample, channel widening."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(x))


class UpBlock(nn.Module):
    """Nearest-neighbour ×2 upsample + 3×3 conv, channel narrowing."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.proj(x)


__all__ = [
    "DownBlock",
    "FiLMConditioning",
    "FiLMConvBlock",
    "UpBlock",
    "mean_pool_sequence",
    "sinusoidal_time_embedding",
]
