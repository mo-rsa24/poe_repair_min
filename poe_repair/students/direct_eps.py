"""Method 2b student — direct eps-space residual prediction.

A small CNN that takes ``(x_t, t, pool_a, pool_b, pool_uncond)`` and
outputs a noise-space correction tensor of the same shape as ``x_t``.
At inference it is added to vanilla guided PoE under a λ_t schedule::

    ε_t = ε_PoE_guided + λ_t · δ_θ(x_t, t, c_a, c_b, c_∅)

The student replaces the joint UNet branch — no extra UNet pass, no
soft-prompt detour. The frozen SDXL UNet is only used to compute the
PoE update; the student is what carries the corrective signal.

Default architecture (~5M params):
  - sinusoidal timestep embedding -> MLP (D_cond)
  - concat(pool_a, pool_b, pool_uncond) -> MLP (D_cond)
  - combined conditioning vector cond ∈ ℝ^{D_cond}
  - body: conv_in (4 → C), N × ResBlock(C, cond), conv_out (C → 4)
  - conv_out is zero-initialised so the student starts as a no-op
    (clean residual perturbation around vanilla PoE).

Target during training:
  ``Δ_t = ε_J_guided − ε_PoE_guided
        = w · (ε_J_raw + ε_∅ − ε_A_raw − ε_B_raw)``
  i.e. the same PMI residual Method 2a targets, computed directly from
  the cached raw eps via :func:`poe_repair.embeddings.cache_dataset.pmi_target`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def sinusoidal_time_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    """SDXL-style sinusoidal timestep embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(1, half)
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class FiLM(nn.Module):
    """Feature-wise linear modulation: scale and shift broadcast spatially."""

    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + scale) + shift


class ResBlock(nn.Module):
    """GroupNorm-SiLU-Conv ResNet block with FiLM conditioning."""

    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.film = FiLM(channels, cond_dim)
        self.norm2 = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # Zero-init the second conv so each block starts as identity.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = self.film(h, cond)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return x + h


# ---------------------------------------------------------------------------
# Student
# ---------------------------------------------------------------------------


class DirectEpsStudent(nn.Module):
    """Small CNN that predicts a noise-space correction for Method 2b.

    Args:
      in_channels: latent channels (SDXL = 4).
      body_channels: hidden channel width of the body. Default 128.
      cond_dim: conditioning vector width. Default 256.
      num_blocks: number of ResBlocks. Default 8.
      pooled_dim: SDXL pooled embedding width (1280).
    """

    def __init__(
        self,
        *,
        in_channels: int = 4,
        body_channels: int = 128,
        cond_dim: int = 256,
        num_blocks: int = 8,
        pooled_dim: int = 1280,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.body_channels = body_channels
        self.cond_dim = cond_dim
        self.num_blocks = num_blocks
        self.pooled_dim = pooled_dim

        # Time embedding: sinusoidal -> 2-layer MLP
        self.t_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Pooled embedding projection (3 prompts: a, b, uncond)
        self.p_mlp = nn.Sequential(
            nn.Linear(3 * pooled_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        # Combine time + pooled into a single conditioning vector.
        self.cond_combine = nn.Linear(2 * cond_dim, cond_dim)

        # Body
        self.conv_in = nn.Conv2d(in_channels, body_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            ResBlock(body_channels, cond_dim) for _ in range(num_blocks)
        ])
        self.norm_out = nn.GroupNorm(
            num_groups=min(32, body_channels), num_channels=body_channels
        )
        self.conv_out = nn.Conv2d(body_channels, in_channels, kernel_size=3, padding=1)
        # Zero the output projection so the student starts as a no-op.
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    # ----- conditioning ---------------------------------------------------

    def _build_cond(
        self,
        t: torch.Tensor,
        pool_a: torch.Tensor,
        pool_b: torch.Tensor,
        pool_uncond: torch.Tensor,
    ) -> torch.Tensor:
        t_sin = sinusoidal_time_emb(t.float(), self.cond_dim)
        t_emb = self.t_mlp(t_sin.to(self.t_mlp[0].weight.dtype))
        p = torch.cat([pool_a, pool_b, pool_uncond], dim=-1)
        p_emb = self.p_mlp(p)
        cond = self.cond_combine(torch.cat([t_emb, p_emb], dim=-1))
        return cond

    # ----- forward --------------------------------------------------------

    def forward(
        self,
        x_t: torch.Tensor,            # (B, in_channels, H, W)
        t: torch.Tensor,              # (B,) timesteps (long)
        pool_a: torch.Tensor,         # (B, pooled_dim)
        pool_b: torch.Tensor,
        pool_uncond: torch.Tensor,
    ) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        cond = self._build_cond(t, pool_a, pool_b, pool_uncond)

        h = self.conv_in(x_t)
        for block in self.blocks:
            h = block(h, cond)
        h = F.silu(self.norm_out(h))
        return self.conv_out(h)

    # ----- introspection --------------------------------------------------

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_direct_eps_student(
    *,
    body_channels: int = 128,
    cond_dim: int = 256,
    num_blocks: int = 8,
) -> DirectEpsStudent:
    """Build a Method 2b student with default SDXL-compatible widths."""
    return DirectEpsStudent(
        in_channels=4, body_channels=body_channels,
        cond_dim=cond_dim, num_blocks=num_blocks, pooled_dim=1280,
    )


# ---------------------------------------------------------------------------
# Hourglass variant — UNet-style encoder/decoder with skip connections.
#
# Motivation: the flat student has a small effective receptive field (~17 px
# at 128×128 latent for 8 conv-3x3 blocks). The residual r_t depends on the
# spatial layout of subjects across the whole image (a cat over here AND a
# dog over there), so a small receptive field caps how well a CNN can
# localise the two-subject correction. This variant adds two downsampling
# stages, giving an effectively global receptive field at the bottleneck,
# and skip connections so per-pixel detail returns intact on the up path.
# ---------------------------------------------------------------------------


class HourglassResBlock(nn.Module):
    """ResBlock that handles channel-count changes via a 1×1 residual proj.

    Used in the hourglass student where consecutive levels have different
    channel widths (e.g. 128 → 256 → 512). The base ``ResBlock`` above
    assumes in == out and is kept unchanged for backward compatibility
    with the flat student's checkpoints.
    """

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.film = FiLM(out_channels, cond_dim)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = self.film(h, cond)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return self.skip(x) + h


class Downsample(nn.Module):
    """2× downsample via stride-2 conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    """2× upsample via nearest-neighbour interpolation + 3×3 conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


class HourglassDirectEpsStudent(nn.Module):
    """UNet-style hourglass student for Method 2b.

    Args:
      in_channels: latent channels (SDXL = 4).
      base_channels: width of the top resolution level.
      channel_mults: per-level multipliers. ``len()`` defines the number of
        levels (one downsample between each consecutive pair). Default
        ``(1, 2, 4)`` ⇒ 3 levels with 2 downsamples ⇒ bottleneck at H/4 × W/4.
      num_blocks_per_level: ResBlocks at each resolution.
      cond_dim: conditioning vector width (time + pooled embeddings).
      pooled_dim: SDXL pooled embedding width (1280).
    """

    def __init__(
        self,
        *,
        in_channels: int = 4,
        base_channels: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        num_blocks_per_level: int = 2,
        cond_dim: int = 256,
        pooled_dim: int = 1280,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_mults = tuple(channel_mults)
        self.num_blocks_per_level = num_blocks_per_level
        self.cond_dim = cond_dim
        self.pooled_dim = pooled_dim

        chs = [base_channels * m for m in channel_mults]

        # Conditioning (identical to flat student so they share an interface).
        self.t_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.p_mlp = nn.Sequential(
            nn.Linear(3 * pooled_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.cond_combine = nn.Linear(2 * cond_dim, cond_dim)

        self.conv_in = nn.Conv2d(in_channels, chs[0], kernel_size=3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        prev_ch = chs[0]
        for i, ch in enumerate(chs):
            level_blocks = nn.ModuleList()
            for j in range(num_blocks_per_level):
                in_ch = prev_ch if j == 0 else ch
                level_blocks.append(HourglassResBlock(in_ch, ch, cond_dim))
            self.down_blocks.append(level_blocks)
            prev_ch = ch
            if i < len(chs) - 1:
                self.down_samples.append(Downsample(prev_ch))

        # Bottleneck
        self.mid_blocks = nn.ModuleList([
            HourglassResBlock(prev_ch, prev_ch, cond_dim),
            HourglassResBlock(prev_ch, prev_ch, cond_dim),
        ])

        # Decoder (symmetric, with skip-concat at each level's first block).
        self.up_samples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for level_index, ch in enumerate(reversed(chs)):
            if level_index > 0:
                self.up_samples.append(Upsample(prev_ch))
            level_blocks = nn.ModuleList()
            for j in range(num_blocks_per_level):
                # First block ingests the concatenated skip → in_ch = prev + ch.
                in_ch = (prev_ch + ch) if j == 0 else ch
                level_blocks.append(HourglassResBlock(in_ch, ch, cond_dim))
            self.up_blocks.append(level_blocks)
            prev_ch = ch

        self.norm_out = nn.GroupNorm(min(32, prev_ch), prev_ch)
        self.conv_out = nn.Conv2d(prev_ch, in_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    # ----- conditioning ---------------------------------------------------

    def _build_cond(
        self,
        t: torch.Tensor,
        pool_a: torch.Tensor,
        pool_b: torch.Tensor,
        pool_uncond: torch.Tensor,
    ) -> torch.Tensor:
        t_sin = sinusoidal_time_emb(t.float(), self.cond_dim)
        t_emb = self.t_mlp(t_sin.to(self.t_mlp[0].weight.dtype))
        p = torch.cat([pool_a, pool_b, pool_uncond], dim=-1)
        p_emb = self.p_mlp(p)
        return self.cond_combine(torch.cat([t_emb, p_emb], dim=-1))

    # ----- forward --------------------------------------------------------

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        pool_a: torch.Tensor,
        pool_b: torch.Tensor,
        pool_uncond: torch.Tensor,
    ) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        cond = self._build_cond(t, pool_a, pool_b, pool_uncond)

        h = self.conv_in(x_t)

        skips: list[torch.Tensor] = []
        for i, level_blocks in enumerate(self.down_blocks):
            for blk in level_blocks:
                h = blk(h, cond)
            skips.append(h)
            if i < len(self.down_samples):
                h = self.down_samples[i](h)

        for blk in self.mid_blocks:
            h = blk(h, cond)

        for level_index, level_blocks in enumerate(self.up_blocks):
            if level_index > 0:
                h = self.up_samples[level_index - 1](h)
            skip = skips[len(skips) - 1 - level_index]
            h = torch.cat([h, skip], dim=1)
            for blk in level_blocks:
                h = blk(h, cond)

        h = F.silu(self.norm_out(h))
        return self.conv_out(h)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_hourglass_student(
    *,
    base_channels: int = 128,
    channel_mults: tuple[int, ...] = (1, 2, 4),
    num_blocks_per_level: int = 2,
    cond_dim: int = 256,
) -> HourglassDirectEpsStudent:
    """Build the hourglass Method 2b student with sensible defaults.

    With ``base_channels=128`` and ``channel_mults=(1, 2, 4)`` the
    bottleneck sits at H/4 × W/4 with 512 channels — global receptive
    field at the SDXL latent's 32×32 bottleneck. ~25–30M params.
    """
    return HourglassDirectEpsStudent(
        in_channels=4,
        base_channels=base_channels,
        channel_mults=tuple(channel_mults),
        num_blocks_per_level=num_blocks_per_level,
        cond_dim=cond_dim,
        pooled_dim=1280,
    )
