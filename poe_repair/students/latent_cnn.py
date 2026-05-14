"""A1 — Latent-CNN corrector.

A flat convolutional network on the SDXL latent (4 × 128 × 128) with FiLM
conditioning on ``(t, e_J)``. Outputs an additive ε-correction with the
same spatial shape as the latent.

The cheapest plausible learner in Group A: single resolution, no skip
connections, ~1M parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from poe_repair.students.common import FiLMConditioning, FiLMConvBlock


class LatentCNNCorrector(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 4,
        out_channels: int = 4,
        body_channels: int = 128,
        n_blocks: int = 5,
        cond_dim: int = 256,
        pool_dim: int = 1280,
        seq_dim: int = 2048,
        time_dim: int = 256,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.body_channels = int(body_channels)
        self.n_blocks = int(n_blocks)
        self.cond_dim = int(cond_dim)

        self.cond = FiLMConditioning(
            time_dim=time_dim, pool_dim=pool_dim, seq_dim=seq_dim, cond_dim=cond_dim,
        )
        self.in_conv = nn.Conv2d(in_channels, body_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [FiLMConvBlock(body_channels, cond_dim=cond_dim) for _ in range(n_blocks)]
        )
        self.out_norm = nn.GroupNorm(min(32, body_channels), body_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(body_channels, out_channels, kernel_size=3, padding=1)
        # Zero-init the final conv so r̂_t starts at zero — the λ=0 canary
        # at epoch 0 then matches vanilla PoE exactly (modulo float noise).
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        z_t: torch.Tensor,           # (B, 4, H, W)
        t: torch.Tensor,             # (B,) long
        e_J_seq: torch.Tensor,       # (B, 77, 2048)
        e_J_pool: torch.Tensor,      # (B, 1280)
    ) -> torch.Tensor:                # (B, 4, H, W)
        # Keep the body in fp32 — fp16 cancellation through 5 conv blocks
        # crushes gradients during training.
        h = self.in_conv(z_t.float())
        cond = self.cond(t, e_J_pool, e_J_seq)
        for block in self.blocks:
            h = block(h, cond)
        h = self.out_norm(h)
        h = self.out_act(h)
        return self.out_conv(h)


def build_latent_cnn(
    *,
    body_channels: int = 128,
    n_blocks: int = 5,
    cond_dim: int = 256,
) -> LatentCNNCorrector:
    return LatentCNNCorrector(
        body_channels=body_channels, n_blocks=n_blocks, cond_dim=cond_dim,
    )


__all__ = ["LatentCNNCorrector", "build_latent_cnn"]
