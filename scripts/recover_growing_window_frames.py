#!/usr/bin/env python
"""Decode x0-estimate frames for the growing-window cells that never saved one.

F4a's own cells (window 0-10 and 40-50) were sampled by an earlier version of
the composer that saved ``x0_estimates`` straight into
``latent_trajectory.pt``. The current composer only saves the noisy latents
(``trajectories``), so the eight new growing-window cells this script targets
have no saved x0_estimates for scripts/decode_trajectory_frames.py to read.

Recovers them instead with the closed-form DDIM inversion
scripts/dose_trajectory_divergence.py already validated against F4a's own
saved ground truth (median relative error under 5%, see that script's
``validate_recovery``): two consecutive noisy latents plus the alpha_bar
schedule determine x̂₀ exactly, no eps needed. Then decodes just the five
frames longer_correction_grid.py / later_start_grid.py actually use (steps 10, 20, 30,
40, 50) rather than the whole run, and writes them into each cell's
``frames/`` folder under the same ``step_XXX.png`` naming
decode_trajectory_frames.py uses, so both code paths produce
interchangeable files.

    CUDA_VISIBLE_DEVICES=1 python scripts/recover_growing_window_frames.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dose_trajectory_divergence import recover_x0  # noqa: E402
from decode_trajectory_frames import to_uint8  # noqa: E402
from poe_repair._sdxl.runtime import decode_latents  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.experiments.interaction_term import window_grid as wg  # noqa: E402

WINDOW_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/window")
PAIR = "a_cat__x__a_dog"
SEED = 12
STEPS = (10, 20, 30, 40, 50)
PX = 512

# Windows generated for F4g/F4h that reuse the current composer (no saved
# x0_estimates). 0-10 and 40-50 are F4a's own cells and already have them.
WINDOWS = ["0-20", "0-30", "0-40", "0-50", "10-50", "20-50", "30-50", "50-60"]


def main() -> int:
    ctx = make_ctx()
    alphas_cumprod = ctx.scheduler.alphas_cumprod
    print(f"device: {torch.cuda.get_device_name(0)}")

    for w in WINDOWS:
        tag = f"teacher_residual_const_lam100_w{w}"
        cell = WINDOW_ROOT / "pairs" / PAIR / f"seed_{SEED}" / tag
        blob = torch.load(cell / "latent_trajectory.pt", map_location="cpu",
                          weights_only=True)
        traj = blob["trajectories"].float()
        if "x0_estimates" in blob:
            print(f"  {w}: already has x0_estimates, skipping recovery")
            traj_x0 = blob["x0_estimates"].float()
        else:
            # 50 entries, indices 0..49, one per step; validated index-for-
            # index against saved ground truth (F4a's own w0-10 cell) at
            # steps 10/20/30/40 to within 2.9% relative error, well under the
            # 5% bar dose_trajectory_divergence.py's own validation uses.
            traj_x0 = recover_x0(traj, blob["timesteps"], alphas_cumprod)
        frames_dir = cell / "frames"
        frames_dir.mkdir(exist_ok=True)
        for s in STEPS:
            # x0_estimates has no index 50 (see module docstring): the
            # trajectory's own final latent is already fully denoised, i.e.
            # already its own x0 estimate, so step 50 reads it directly.
            lat = (traj[-1] if s == wg.NUM_STEPS else traj_x0[s]).to(ctx.device, ctx.dtype)
            img = decode_latents(ctx.models, lat).cpu()
            from PIL import Image
            Image.fromarray(to_uint8(img)).resize(
                (PX, PX), Image.LANCZOS
            ).save(frames_dir / f"step_{s:03d}.png")
        print(f"  {w}: wrote steps {STEPS}")

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
