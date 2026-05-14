"""Method 2b — direct eps-space residual student."""

from __future__ import annotations

from pathlib import Path

import torch

from poe_repair.students.direct_eps import (
    DirectEpsStudent,
    HourglassDirectEpsStudent,
    build_direct_eps_student,
    build_hourglass_student,
)


def _build_from_checkpoint_meta(sd: dict):
    """Construct the right student class from a checkpoint's metadata.

    Older flat-student checkpoints store ``arch = {body_channels, cond_dim,
    num_blocks}`` and no ``arch_kind``. Hourglass checkpoints set
    ``arch_kind = "hourglass"`` and ``arch = {base_channels, channel_mults,
    num_blocks_per_level, cond_dim}``. Defaults to flat for backward
    compatibility.
    """
    arch_kind = sd.get("arch_kind", "flat") if isinstance(sd, dict) else "flat"
    arch = sd.get("arch", {}) if isinstance(sd, dict) else {}
    if arch_kind == "hourglass":
        return build_hourglass_student(
            base_channels=int(arch.get("base_channels", 128)),
            channel_mults=tuple(arch.get("channel_mults", (1, 2, 4))),
            num_blocks_per_level=int(arch.get("num_blocks_per_level", 2)),
            cond_dim=int(arch.get("cond_dim", 256)),
        )
    if arch_kind == "flat":
        return build_direct_eps_student(
            body_channels=int(arch.get("body_channels", 128)),
            cond_dim=int(arch.get("cond_dim", 256)),
            num_blocks=int(arch.get("num_blocks", 8)),
        )
    raise ValueError(f"unknown arch_kind in checkpoint: {arch_kind!r}")


def load_direct_eps_student(
    ckpt_path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
):
    """Load a Method 2b student from a checkpoint produced by
    ``train_direct_eps.py``. Architecture (flat vs hourglass) and width
    knobs are read from the checkpoint's ``arch_kind`` / ``arch`` fields.
    Older checkpoints without ``arch_kind`` default to the flat student.
    """
    sd = torch.load(Path(ckpt_path), map_location=device)
    state = sd.get("student", sd) if isinstance(sd, dict) else sd
    student = _build_from_checkpoint_meta(sd if isinstance(sd, dict) else {})
    missing, unexpected = student.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(
            f"load_direct_eps_student: missing keys when loading {ckpt_path}: {missing}"
        )
    if unexpected:
        raise RuntimeError(
            f"load_direct_eps_student: unexpected keys when loading {ckpt_path}: {unexpected}"
        )
    student.to(device=device, dtype=dtype)
    student.eval()
    return student


from poe_repair.students.latent_cnn import LatentCNNCorrector, build_latent_cnn
from poe_repair.students.latent_unet import LatentUNetCorrector, build_latent_unet
from poe_repair.students.frozen_feature_mlp import (
    FrozenFeatureMLPHead,
    build_frozen_feature_mlp,
)


__all__ = [
    "DirectEpsStudent",
    "FrozenFeatureMLPHead",
    "HourglassDirectEpsStudent",
    "LatentCNNCorrector",
    "LatentUNetCorrector",
    "build_direct_eps_student",
    "build_frozen_feature_mlp",
    "build_hourglass_student",
    "build_latent_cnn",
    "build_latent_unet",
    "load_direct_eps_student",
]
