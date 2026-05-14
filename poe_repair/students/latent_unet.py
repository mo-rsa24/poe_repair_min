"""A2 — Latent-UNet corrector.

A small 2-scale encoder–decoder on the SDXL latent (4 × 128 × 128) with
FiLM conditioning at every block. Skip connections give the network coarse
and fine views of the residual simultaneously.

Output shape and forward signature match A1 exactly; the trainer and the
sampler do not branch on the technique type.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from poe_repair.students.common import (
    DownBlock,
    FiLMConditioning,
    FiLMConvBlock,
    UpBlock,
)


class LatentUNetCorrector(nn.Module):
    """2-scale encoder–decoder.

    Default channel schedule: 96 → 192 → 384 → 192 → 96. Each level has two
    FiLM conv blocks. Skip connections concatenate same-scale encoder and
    decoder features along the channel axis before a 1×1 projection back
    to the level's width.
    """

    def __init__(
        self,
        *,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 96,
        blocks_per_level: int = 2,
        cond_dim: int = 256,
        pool_dim: int = 1280,
        seq_dim: int = 2048,
        time_dim: int = 256,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.base_channels = int(base_channels)
        self.blocks_per_level = int(blocks_per_level)
        self.cond_dim = int(cond_dim)

        C = base_channels
        self.cond = FiLMConditioning(
            time_dim=time_dim, pool_dim=pool_dim, seq_dim=seq_dim, cond_dim=cond_dim,
        )
        self.in_conv = nn.Conv2d(in_channels, C, kernel_size=3, padding=1)

        # Encoder level 1 — operates at full resolution.
        self.enc1 = nn.ModuleList(
            [FiLMConvBlock(C, cond_dim=cond_dim) for _ in range(blocks_per_level)]
        )
        self.down1 = DownBlock(C, 2 * C)

        # Encoder level 2 — half resolution.
        self.enc2 = nn.ModuleList(
            [FiLMConvBlock(2 * C, cond_dim=cond_dim) for _ in range(blocks_per_level)]
        )
        self.down2 = DownBlock(2 * C, 4 * C)

        # Bottleneck — quarter resolution.
        self.bottleneck = nn.ModuleList(
            [FiLMConvBlock(4 * C, cond_dim=cond_dim) for _ in range(blocks_per_level)]
        )

        # Decoder level 2 — back to half resolution, skip from enc2.
        self.up2 = UpBlock(4 * C, 2 * C)
        self.skip_proj2 = nn.Conv2d(4 * C, 2 * C, kernel_size=1)
        self.dec2 = nn.ModuleList(
            [FiLMConvBlock(2 * C, cond_dim=cond_dim) for _ in range(blocks_per_level)]
        )

        # Decoder level 1 — back to full resolution, skip from enc1.
        self.up1 = UpBlock(2 * C, C)
        self.skip_proj1 = nn.Conv2d(2 * C, C, kernel_size=1)
        self.dec1 = nn.ModuleList(
            [FiLMConvBlock(C, cond_dim=cond_dim) for _ in range(blocks_per_level)]
        )

        self.out_norm = nn.GroupNorm(min(32, C), C)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(C, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _run_blocks(self, blocks: nn.ModuleList, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for b in blocks:
            h = b(h, cond)
        return h

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        e_J_seq: torch.Tensor,
        e_J_pool: torch.Tensor,
    ) -> torch.Tensor:
        cond = self.cond(t, e_J_pool, e_J_seq)

        h = self.in_conv(z_t.float())
        h = self._run_blocks(self.enc1, h, cond)
        skip1 = h
        h = self.down1(h)

        h = self._run_blocks(self.enc2, h, cond)
        skip2 = h
        h = self.down2(h)

        h = self._run_blocks(self.bottleneck, h, cond)

        h = self.up2(h)
        h = self.skip_proj2(torch.cat([h, skip2], dim=1))
        h = self._run_blocks(self.dec2, h, cond)

        h = self.up1(h)
        h = self.skip_proj1(torch.cat([h, skip1], dim=1))
        h = self._run_blocks(self.dec1, h, cond)

        h = self.out_norm(h)
        h = self.out_act(h)
        return self.out_conv(h)


def build_latent_unet(
    *,
    base_channels: int = 96,
    blocks_per_level: int = 2,
    cond_dim: int = 256,
) -> LatentUNetCorrector:
    return LatentUNetCorrector(
        base_channels=base_channels,
        blocks_per_level=blocks_per_level,
        cond_dim=cond_dim,
    )


__all__ = ["LatentUNetCorrector", "build_latent_unet"]
