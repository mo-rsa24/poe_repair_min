#!/usr/bin/env python
"""The same composition rule succeeding on one prompt pair
and failing on another, with the size of the correction it never applies plotted
underneath each one.

Left column is "a butterfly" composed with "a flower meadow". Right column is "a cat"
composed with "a dog". Top row is one real uncorrected sample from each. Bottom row is
the size of r_t at every denoising step, as a fraction of the product-of-experts
prediction it would correct, median over the three seeds with a min-to-max band.

Both bottom panels share one y axis, because the whole reason they sit side by side is
that the reader compares them.

What the figure shows: the model's joint-prompt prediction differs from its
product-of-experts prediction throughout both trajectories, and only one of the two
samples is wrong. What it does not show: anything about R_t, the ratio of true
co-occurrence probabilities, which is measured nowhere. r_t is a difference between two
of one network's noise predictions, and the step from one to the other is the step
paragraph 5.5 of the manuscript refuses to take. Two pairs also cannot support any
claim that correction size predicts which pairs fail, and on these two it does not
point that way.

    python scripts/correction_size_over_the_run.py                  # the two-row figure, seed 42 on top
    python scripts/correction_size_over_the_run.py --top-seed 4     # a different sample on top
    python scripts/correction_size_over_the_run.py --images-only    # all three seeds, images, no curves
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch

# These cells live under veracity, not under the dose tree the other figures read.
PAIRS_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/veracity/pairs")

# Beside the manuscript, not on /datasets: a paper figure is a few hundred KB and
# LaTeX has to reach it. Naming rule in paper/iclr/README.md.
FIG_DIR = Path("paper/iclr/figures")
FIG_NAME = "correction-size-over-the-denoising-run"

COLUMNS = [
    ("a_butterfly__x__a_flower_meadow", "a butterfly $\\times$ a flower meadow",
     "both concepts appear"),
    ("a_cat__x__a_dog", "a cat $\\times$ a dog",
     "one blended animal"),
]
SEEDS = (4, 42, 123)
STEPS = 50

PAPER = "#ffffff"   # white page: the figure sits on the manuscript, not on a tinted card
INK = "#222222"
BORDER = "#444444"
CURVE = "#3d6a99"
BAND = "#9db8d2"
TITLE_PT = 8.5
SUB_PT = 7.5
CHIP_PT = 6.5
AXIS_PT = 7.0

FIG_W = 5.50
GUTTER = 0.10
# Room on the left for the y tick labels and the axis title. Without it the numbers
# fall off the canvas and the curves cannot be read off, only compared.
SIDE_L = 0.66
SIDE_R = 0.02
HEADER = 0.42          # above the images, for the two column labels
CURVE_H = 1.05         # the plot panel itself
AXIS_PAD = 0.42        # tick labels and x title under the plot
ROW_GAP = 0.16         # between the image and the plot above it


def panel_png(pair: str, seed: int) -> Path:
    """The uncorrected sample for one cell. lambda zero is the only tag this reads."""
    d = PAIRS_ROOT / pair / f"seed_{seed}" / "teacher_residual_const_lam000"
    hits = sorted(d.glob("*.png"))
    if not hits:
        raise SystemExit(f"no image under {d}")
    return hits[0]


def curve(pair: str, seed: int) -> np.ndarray:
    """||r_t|| as a fraction of ||eps_PoE||, per denoising step.

    The cache is fp16. Norms are taken in float32, because a sum of 65k squared
    half-precision terms loses digits that matter at this scale.
    """
    d = PAIRS_ROOT / pair / f"seed_{seed}" / "teacher_residual_const_lam000" / "residuals"
    out = []
    for i in range(STEPS):
        f = torch.load(d / f"step_{i:03d}.pt", map_location="cpu", weights_only=False)
        r = f["delta"].float()
        e = f["eps_poe"].float()
        out.append((r.norm() / e.norm()).item())
    return np.asarray(out)


def draw_curve(ax, band_lo, band_hi, med, ylim, is_left: bool) -> None:
    x = np.arange(STEPS)
    ax.fill_between(x, band_lo, band_hi, color=BAND, alpha=0.55, linewidth=0)
    ax.plot(x, med, color=CURVE, linewidth=1.1)
    ax.set_xlim(0, STEPS - 1)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=AXIS_PT, length=2.4, width=0.6, colors=INK)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color(BORDER)
    ax.set_facecolor(PAPER)
    ax.set_xlabel("denoising steps", fontsize=AXIS_PT,
                  family="serif", color=INK, labelpad=2)
    if is_left:
        ax.set_ylabel("residual",
                      fontsize=AXIS_PT, family="serif", color=INK, labelpad=2)
    else:
        ax.set_yticklabels([])
    # The number named on the panel it belongs to, so no table is needed to read it.
    ax.text(0.985, 0.94, f"median {np.median(med):.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=CHIP_PT, family="serif", color=INK,
            bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.4, alpha=0.9))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top-seed", type=int, default=42,
                    help="which sample sits above the curves. The curves use all seeds.")
    ap.add_argument("--images-only", action="store_true",
                    help="all three seeds as images, no curves. The earlier layout.")
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    if args.top_seed not in SEEDS:
        raise SystemExit(f"seed {args.top_seed} is not one of {SEEDS}")

    panel_w = (FIG_W - SIDE_L - SIDE_R - GUTTER) / 2

    if args.images_only:
        fig_h = HEADER + panel_w * len(SEEDS)
    else:
        fig_h = HEADER + panel_w + ROW_GAP + CURVE_H + AXIS_PAD

    fig = plt.figure(figsize=(FIG_W, fig_h), facecolor=PAPER)

    curves = {}
    if not args.images_only:
        for pair, _, _ in COLUMNS:
            per_seed = np.stack([curve(pair, s) for s in SEEDS])
            curves[pair] = per_seed
            print(f"{pair}: median over steps "
                  f"{np.median(np.median(per_seed, axis=0)):.3f}")
        stacked = np.concatenate([v for v in curves.values()])
        # One y range for both panels. Two panels compared on different scales would
        # be a different figure making a claim this one does not.
        ylim = (0.0, float(stacked.max()) * 1.08)

    for ci, (pair, title, reading) in enumerate(COLUMNS):
        x0 = (SIDE_L + ci * (panel_w + GUTTER)) / FIG_W
        fig.text(x0 + (panel_w / FIG_W) / 2, 1 - 0.11 / fig_h, title,
                 ha="center", va="top", fontsize=TITLE_PT, family="serif", color=INK)
        fig.text(x0 + (panel_w / FIG_W) / 2, 1 - 0.25 / fig_h, reading,
                 ha="center", va="top", fontsize=SUB_PT, family="serif",
                 color=INK, style="italic")

        img_seeds = SEEDS if args.images_only else (args.top_seed,)
        for ri, seed in enumerate(img_seeds):
            src = panel_png(pair, seed)
            print(f"{pair} seed {seed}: {src}")
            y0 = (fig_h - HEADER - (ri + 1) * panel_w) / fig_h
            ax = fig.add_axes([x0, y0, panel_w / FIG_W, panel_w / fig_h])
            ax.imshow(mpimg.imread(src))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.6); sp.set_color(BORDER)
            ax.text(0.015, 0.985, f"seed {seed}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=CHIP_PT, family="serif", color=INK,
                    bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.4, alpha=0.88))

        if not args.images_only:
            per_seed = curves[pair]
            cy = AXIS_PAD / fig_h
            ax = fig.add_axes([x0, cy, panel_w / FIG_W, CURVE_H / fig_h])
            draw_curve(ax, per_seed.min(axis=0), per_seed.max(axis=0),
                       np.median(per_seed, axis=0), ylim, is_left=(ci == 0))

    name = args.name or (f"{FIG_NAME}-images" if args.images_only else FIG_NAME)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = args.out_dir / f"{name}.{ext}"
        fig.savefig(out, dpi=300, facecolor=PAPER)
        print(f"wrote {out}")
    plt.close(fig)

    if not args.images_only:
        side = args.out_dir / f"{name}.json"
        side.write_text(json.dumps({
            "measure": "||r_t|| / ||eps_PoE||, per denoising step, fp16 upcast to float32",
            "seeds": list(SEEDS),
            "steps": STEPS,
            "band": "min to max over seeds; line is the median over seeds",
            "top_row_seed": args.top_seed,
            "pairs": {p: {"per_seed": curves[p].tolist(),
                          "median_over_steps": float(np.median(np.median(curves[p], axis=0)))}
                      for p, _, _ in COLUMNS},
        }, indent=2))
        print(f"wrote {side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
