#!/usr/bin/env python
"""Same pair, new starting noise: nothing about the correction carries over.

Candidate figure D2. One plot. The line is the direction agreement (cosine)
between the corrections of two runs of the SAME pair (cat x dog, seeds 9 and
13), compared at matched denoising steps. It sits on zero the whole run.
Floating above the line, the first run's pictures forming (seed 9, blue
border); below it, the second run's (seed 13, green border).

The cached trajectories are the uncorrected PoE paths (the decoded frames
match each cell's saved poe.png, one fused animal, not its mono.png), so the
frames show the failure happening two different ways: seed 9 fuses toward an
orange cat-dog hybrid, seed 13 toward a single white cat. The correction each
run needs is measured at every step, and the two share no direction at any
step. The fix for one run does not fit the other, even for the same pair.

Why it matters: a correction cannot be stored and reused. It has to be
produced fresh from the current image, which is exactly what the trained
adapter does.

The thumbnails float in otherwise-empty axis space; their vertical position
carries no data, and each one drops a connector to its step on the axis. The
frame at step k is the run's own estimate of where it is heading (Tweedie
from x_t and the PoE prediction that steered the path, then the VAE), no
sampling.

    CUDA_VISIBLE_DEVICES=0 python scripts/rt_reuse_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from poe_repair import paths

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.experiments.interaction_term.cache import (  # noqa: E402
    _alphas_cumprod, load_cell,
)

PAIR = "a_cat__x__a_dog"
SEED_A, SEED_B = 9, 13          # 13 = seed 9's wrong-seed donor in the dose sweep
SHOW_STEPS = (10, 20, 30, 40)
OUT_DIR = paths.resolve(paths.DIRECTION_WALL)
FRAME_DIR = OUT_DIR / "frames"
BLUE, GREEN = "#1f77b4", "#2ca02c"


def x0_estimate(cell, step: int) -> torch.Tensor:
    """Where this run is heading at this step (Tweedie).

    Uses the PoE prediction because the cached trajectory IS the PoE path:
    the estimate must come from the prediction that steered the path, or the
    frame shows a hybrid of two different runs.
    """
    ab = _alphas_cumprod()[int(cell.timesteps[step])]
    x_t = cell.x_t[step].float()
    eps = cell.eps_poe()[step].float()
    return (x_t - (1.0 - ab).sqrt() * eps) / ab.sqrt()


def decode_frames() -> dict[tuple[int, int], Path]:
    """Decode the shown steps for both seeds, skipping frames already on disk."""
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    want = {(s, k): FRAME_DIR / f"{PAIR}_seed{s}_step{k:03d}.png"
            for s in (SEED_A, SEED_B) for k in SHOW_STEPS}
    todo = {key: p for key, p in want.items() if not p.exists()}
    if not todo:
        return want

    from diffusers import AutoencoderKL
    from poe_repair.config import RunConfig
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(
        RunConfig().model_id, subfolder="vae",
        torch_dtype=torch.float32).to(device).eval()
    cells = {s: load_cell(PAIR, s) for s in (SEED_A, SEED_B)}
    for (s, k), path in todo.items():
        lat = x0_estimate(cells[s], k).to(device) / vae.config.scaling_factor
        with torch.no_grad():
            img = vae.decode(lat).sample[0]
        arr = ((img / 2 + 0.5).clamp(0, 1) * 255).byte().cpu()
        Image.fromarray(arr.permute(1, 2, 0).numpy()).resize((256, 256)).save(path)
        print(f"decoded seed {s} step {k}")
    del vae
    return want


def main() -> int:
    a = load_cell(PAIR, SEED_A).r_t().float().flatten(1)
    b = load_cell(PAIR, SEED_B).r_t().float().flatten(1)
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=1).numpy()
    steps = np.arange(len(cosine), dtype=float)
    frames = decode_frames()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
    })
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.axhline(0.0, color="0.75", lw=0.6, zorder=1)
    ax.plot(steps, cosine, color="0.2", lw=1.4, zorder=3)

    # Thumbnails float in empty axis space: vertical position carries no data,
    # and every frame drops a connector to its step on the axis.
    w = 0.16                     # thumbnail width, axes fraction
    for k in SHOW_STEPS:
        cx = k / steps[-1]
        for seed, cy, colour in ((SEED_A, 0.76, BLUE), (SEED_B, 0.24, GREEN)):
            ia = ax.inset_axes([cx - w / 2, cy - w * 5.5 / 3.4 / 2,
                                w, w * 5.5 / 3.4])
            ia.imshow(plt.imread(frames[(seed, k)]))
            ia.set_xticks([]); ia.set_yticks([])
            for sp in ia.spines.values():
                sp.set_linewidth(1.4); sp.set_color(colour)
        ax.plot([k, k], [-0.14, 0.14], color="0.55", lw=0.6, zorder=2)
        ax.annotate(f"{k}", (k, 0), ha="center", va="center", fontsize=6,
                    color="0.45",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

    ax.text(0.01, 0.985, f"uncorrected run A, seed {SEED_A}",
            transform=ax.transAxes, fontsize=7.5, color=BLUE, va="top")
    ax.text(0.01, 0.015, f"uncorrected run B, seed {SEED_B}",
            transform=ax.transAxes, fontsize=7.5, color=GREEN, va="bottom")
    ax.text(0.99, 0.53, "direction agreement of the two runs' corrections",
            transform=ax.transAxes, fontsize=7, color="0.35", ha="right")

    ax.set_xlim(-1, steps[-1] + 1)
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("denoising step")
    ax.set_ylabel("direction agreement (cosine)")
    ax.text(0.0, -0.19, "noise", transform=ax.transAxes, fontsize=7, color="0.4")
    ax.text(1.0, -0.19, "image", transform=ax.transAxes, fontsize=7, color="0.4",
            ha="right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()

    fig_dir = Path("paper/iclr/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "direction-agreement-between-two-seeds.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)

    (fig_dir / "direction-agreement-between-two-seeds.json").write_text(json.dumps({
        "pair": PAIR, "seeds": [SEED_A, SEED_B], "shown_steps": SHOW_STEPS,
        "cosine_median": float(np.median(cosine)),
        "cosine_range": [float(cosine.min()), float(cosine.max())],
        "frames": "the run's own x0 estimate per step: Tweedie from cached x_t "
                  "and the guided PoE prediction that steered the path, decoded "
                  "by the SDXL VAE, fp32. The cached trajectory is the PoE path "
                  "(decoded endpoint matches poe.png, not mono.png)",
    }, indent=2))
    print(f"cosine median {np.median(cosine):+.3f}, "
          f"range [{cosine.min():+.3f}, {cosine.max():+.3f}]")
    print(f"figure   {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
