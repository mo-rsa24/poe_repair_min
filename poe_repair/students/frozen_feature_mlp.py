"""A3 — Frozen-feature MLP head.

Runs the frozen SDXL UNet on ``(z_t, t, e_J)`` and reads the mid-block
feature map via a forward hook. A small conv/MLP head maps those features
to the additive ε-correction.

SDXL's weights are not modified. Only the head trains. The cost per
forward is one full frozen-UNet pass plus the cheap head.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from poe_repair.methods._sampling import add_time_ids
from poe_repair.students.common import (
    FiLMConditioning,
    FiLMConvBlock,
)


class _MidBlockHook:
    """Captures the mid-block output of a diffusers SDXL UNet.

    Registered on attach; ``.feature`` holds the most recent activation
    after each forward pass. The hook stores a tensor view on whichever
    device the forward ran on; the caller is responsible for moving /
    detaching as needed.
    """

    def __init__(self):
        self.feature: Optional[torch.Tensor] = None
        self._handle = None

    def attach(self, unet: nn.Module) -> "_MidBlockHook":
        target = unet.mid_block

        def _hook(module, args, output):
            # ``output`` from mid-block is the activation tensor itself.
            # Some diffusers versions wrap it in a tuple — handle both.
            if isinstance(output, tuple):
                self.feature = output[0]
            else:
                self.feature = output

        self._handle = target.register_forward_hook(_hook)
        return self

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class FrozenFeatureMLPHead(nn.Module):
    """Head that runs *outside* SDXL.

    ``self.unet`` is the (frozen) SDXL UNet provided by the caller. The
    hook captures the mid-block feature on every forward; the head reads
    it and upsamples back to latent resolution.

    The mid-block output for SDXL base is typically (B, 1280, H/4, W/4).
    H and W refer to the *latent* resolution (so 128 → 32 for SDXL-base).
    """

    def __init__(
        self,
        unet: nn.Module,
        *,
        feature_channels: int = 1280,
        head_channels: int = 256,
        n_blocks: int = 3,
        out_channels: int = 4,
        cond_dim: int = 256,
        pool_dim: int = 1280,
        seq_dim: int = 2048,
        time_dim: int = 256,
        height: int = 1024,
        width: int = 1024,
        unet_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.unet = unet  # frozen; not registered as a sub-module to avoid double-counting.
        # We deliberately bypass nn.Module child registration so the head's
        # state_dict / optimizer covers head params only.
        object.__setattr__(self, "_unet_ref", unet)
        # Remove from auto-registration if it somehow got in.
        if "unet" in self._modules:
            del self._modules["unet"]
        self.head_channels = int(head_channels)
        self.feature_channels = int(feature_channels)
        self.out_channels = int(out_channels)
        self.height = int(height)
        self.width = int(width)
        self.unet_dtype = unet_dtype

        # Channel-reduction over the SDXL feature map.
        self.feature_proj = nn.Conv2d(feature_channels, head_channels, kernel_size=1)
        # Mix in t/e_J via FiLM blocks operating at the upsampled latent resolution.
        self.cond = FiLMConditioning(
            time_dim=time_dim, pool_dim=pool_dim, seq_dim=seq_dim, cond_dim=cond_dim,
        )
        self.blocks = nn.ModuleList(
            [FiLMConvBlock(head_channels, cond_dim=cond_dim) for _ in range(n_blocks)]
        )
        self.out_norm = nn.GroupNorm(min(32, head_channels), head_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(head_channels, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

        self.hook = _MidBlockHook().attach(unet)

    def __del__(self):
        try:
            self.hook.detach()
        except Exception:
            pass

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _frozen_sdxl_forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        e_J_seq: torch.Tensor,
        e_J_pool: torch.Tensor,
    ) -> torch.Tensor:
        """Run SDXL with joint conditioning once; the hook captures the
        mid-block activation as a side effect. Returns the mid-block
        tensor on whatever device/dtype SDXL ran on.
        """
        unet = self._unet_ref
        device = next(unet.parameters()).device
        dtype = self.unet_dtype
        z_in = z_t.to(device=device, dtype=dtype)
        e_seq = e_J_seq.to(device=device, dtype=dtype)
        e_pool = e_J_pool.to(device=device, dtype=dtype)
        cond = {
            "text_embeds": e_pool,
            "time_ids": add_time_ids(
                height=self.height, width=self.width,
                batch_size=z_in.shape[0], device=device, dtype=dtype,
            ),
        }
        with torch.no_grad():
            unet(z_in, t.to(device=device), encoder_hidden_states=e_seq,
                 added_cond_kwargs=cond, timestep_cond=None)
        if self.hook.feature is None:
            raise RuntimeError(
                "FrozenFeatureMLPHead: mid-block hook returned no feature. "
                "Re-check that the hook is attached to the right module."
            )
        return self.hook.feature.detach()

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        e_J_seq: torch.Tensor,
        e_J_pool: torch.Tensor,
    ) -> torch.Tensor:
        feature = self._frozen_sdxl_forward(z_t, t, e_J_seq, e_J_pool).float()
        # feature shape: (B, 1280, H/4, W/4) typically.
        h = self.feature_proj(feature)
        # Upsample to latent resolution (z_t.shape[-2:]).
        target_h, target_w = z_t.shape[-2], z_t.shape[-1]
        h = F.interpolate(h, size=(target_h, target_w), mode="bilinear", align_corners=False)
        cond = self.cond(t, e_J_pool, e_J_seq)
        for block in self.blocks:
            h = block(h, cond)
        h = self.out_norm(h)
        h = self.out_act(h)
        return self.out_conv(h)


def build_frozen_feature_mlp(
    unet: nn.Module,
    *,
    feature_channels: int = 1280,
    head_channels: int = 256,
    n_blocks: int = 3,
    cond_dim: int = 256,
    height: int = 1024,
    width: int = 1024,
    unet_dtype: torch.dtype = torch.float16,
) -> FrozenFeatureMLPHead:
    return FrozenFeatureMLPHead(
        unet,
        feature_channels=feature_channels,
        head_channels=head_channels,
        n_blocks=n_blocks,
        cond_dim=cond_dim,
        height=height,
        width=width,
        unet_dtype=unet_dtype,
    )


__all__ = ["FrozenFeatureMLPHead", "build_frozen_feature_mlp"]
