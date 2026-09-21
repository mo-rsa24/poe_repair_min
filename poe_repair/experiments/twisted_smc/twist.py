"""The twist head: a small conv classifier on SDXL latents.

``forward(x_t, timestep, cond) -> log psi`` of shape ``(B,)``. Trained with
binary cross-entropy where label 1 means "this x_t came from the joint prompt's
distribution" and 0 means "from the PoE distribution". At the optimum the logit
equals ``log p_J,t(x_t) - log p_PoE,t(x_t)``, which is exactly the twist a
twisted-SMC sampler with the PoE base as proposal needs (CDM, arXiv 2605.23346,
sections 3 and 4; the continuous-diffusion special case).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timestep: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """(B,) timesteps in [0, 1000) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timestep.device, dtype=torch.float32) / half
    )
    args = timestep.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class FiLMBlock(nn.Module):
    """conv -> GroupNorm -> FiLM(scale, shift from the conditioning) -> SiLU -> conv, stride-2 downsample."""

    def __init__(self, cin: int, cout: int, emb_dim: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, cout)
        self.film = nn.Linear(emb_dim, 2 * cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(8, cout)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(cin, cout, 1, stride=2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(self.conv1(x))
        scale, shift = self.film(emb).chunk(2, dim=-1)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = F.silu(h)
        h = self.drop(h)
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class TwistHead(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 4,
        width: int = 64,
        cond_dim: int = 1280,
        time_dim: int = 128,
        emb_dim: int = 256,
        dropout: float = 0.1,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.stem = nn.Conv2d(in_channels, width, 3, padding=1)
        chans = [width, 2 * width, 4 * width, 4 * width, 4 * width, 4 * width]
        blocks = []
        for i in range(num_blocks):
            blocks.append(FiLMBlock(chans[i], chans[i + 1], emb_dim, dropout))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(
            nn.Linear(chans[num_blocks] + emb_dim, emb_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(emb_dim, 1),
        )

    def forward(self, x_t: torch.Tensor, timestep: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(sinusoidal_embedding(timestep, self.time_dim)) + self.cond_mlp(cond.float())
        h = self.stem(x_t.float())
        for blk in self.blocks:
            h = blk(h, emb)
        pooled = h.mean(dim=(2, 3))
        return self.head(torch.cat([pooled, emb], dim=-1)).squeeze(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
