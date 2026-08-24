#!/usr/bin/env python
"""What a product of two densities asks for, beside what
plain PoE actually produced.

Left panel is drawn here and contains no animals. Right panel is one real sampler
output. Every animal a reader sees in this figure came out of the model, from one named
seed, at a path this script prints.

The left panel is drawn rather than generated. It is contour lines and six words, so
matplotlib sets it exactly, at a size that reads in an ICLR column, and the same
command reproduces it forever. An image model's own lettering measured 2.4pt once the
panel was placed in the column.

The geometry is honest about one thing in particular. The product of two Gaussians is a
Gaussian at their midpoint, so the blue region is drawn where the maths puts it, not
where a lens-shaped intersection would look tidier. The midpoint of "a cat" and "a dog"
is the whole argument.

    python scripts/what_the_product_misses.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Ellipse
import numpy as np

PAIRS_ROOT = Path("outputs/interaction_term/dose/pairs")

# Beside the manuscript, not on /datasets: a paper figure is a few hundred KB and
# LaTeX has to reach it. Naming rule in paper/iclr/README.md.
FIG_DIR = Path("paper/iclr/figures")
FIG_NAME = "what-the-product-misses-explainer"

# The drawn plane. Nothing here is measured; the ranges exist so the panel's physical
# aspect is fixed and the labels always land clear of the contours.
XLIM = (-3.75, 3.75)
YLIM = (-3.45, 2.05)
MU = 1.60              # each concept's centre, left and right of zero
MU_Y = 0.45
SIG_X, SIG_Y = 1.00, 0.75
LEVELS = (0.15, 0.28, 0.43, 0.60, 0.78, 0.93)
PROD_LEVEL = 0.75      # the product's own contour, kept tight so it reads as a peak

DASH_XY, DASH_W, DASH_H = (-2.02, -2.40), 3.05, 1.00

# Page geometry, in inches. ICLR 2027 is single column at 5.5in textwidth, so F1 is a
# `figure`, not a `figure*`. The panel height follows from the drawn plane's aspect, so
# the two panels are the same height without anything being squashed.
FIG_W = 5.50
SCHEM_W = 2.95
PANEL_H = SCHEM_W * (YLIM[1] - YLIM[0]) / (XLIM[1] - XLIM[0])
PHOTO_W = PANEL_H
GUTTER = FIG_W - SCHEM_W - PHOTO_W
VMARGIN = 0.06
FIG_H = PANEL_H + 2 * VMARGIN

PAPER = "#faf6ef"
WARM = "#c2572e"       # one concept on its own
COOL_FILL = "#9db8d2"  # what the product asks for
COOL_EDGE = "#3d6a99"
INK = "#222222"
BORDER = "#444444"     # both panels framed identically, drawn and measured alike
LABEL_PT = 8.0


def gauss(xx, yy, cx):
    return np.exp(-(((xx - cx) / SIG_X) ** 2 + ((yy - MU_Y) / SIG_Y) ** 2) / 2)


def draw_schematic(ax) -> tuple[float, float]:
    """The two concepts, their product, and the reading the product never reaches.
    Returns the product peak in data coordinates, for the leader line."""
    ax.set_facecolor(PAPER)
    xx, yy = np.meshgrid(np.linspace(*XLIM, 600), np.linspace(*YLIM, 420))

    for cx in (-MU, MU):
        ax.contour(xx, yy, gauss(xx, yy, cx), levels=LEVELS,
                   colors=WARM, linewidths=0.7)

    prod = gauss(xx, yy, -MU) * gauss(xx, yy, MU)
    prod = prod / prod.max()
    ax.contourf(xx, yy, prod, levels=[PROD_LEVEL, 1.0], colors=[COOL_FILL], zorder=3)
    ax.contour(xx, yy, prod, levels=[PROD_LEVEL], colors=[COOL_EDGE],
               linewidths=0.7, zorder=3)
    ax.plot([0], [MU_Y], marker="o", markersize=2.4, color=COOL_EDGE, zorder=4)

    ax.add_patch(Ellipse(DASH_XY, DASH_W, DASH_H, facecolor="none",
                         edgecolor="#555555", linewidth=0.7, linestyle=(0, (4, 3))))

    lab = dict(fontsize=LABEL_PT, family="serif", color=INK, ha="center")
    ax.text(-MU, -1.15, "a cat", va="top", **lab)
    ax.text(MU, -1.15, "a dog", va="top", **lab)
    # Punched through the contour rings it sits over, so the word stays legible
    # without moving the label away from the thing it names.
    ax.text(0.0, MU_Y + 0.50, "the product", va="bottom", zorder=5,
            bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.2), **lab)
    ax.text(*DASH_XY, "cat beside dog", va="center", **lab)

    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color(BORDER)
    return 0.0, MU_Y


def panel_png(pair: str, seed: int, lam: float) -> Path:
    tag = f"teacher_residual_const_lam{int(round(lam * 100)):03d}"
    d = PAIRS_ROOT / pair / f"seed_{seed}" / tag
    hits = sorted(d.glob("*.png"))
    if not hits:
        raise SystemExit(f"no image under {d}")
    return hits[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", default="a_cat__x__a_dog",
                    help="the paper's running example; every figure shows this cell")
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--lam", type=float, default=0.0,
                    help="0.0 is plain PoE, nothing injected. F1 makes no other claim.")
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--name", default=FIG_NAME)
    args = ap.parse_args()

    if args.lam != 0.0:
        print(f"warning: lambda={args.lam}, so the right panel is not plain PoE. "
              f"F1's claim is about the uncorrected sampler.")

    cell = panel_png(args.pair, args.seed, args.lam)
    print(f"right panel {cell}")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    y0, hh = VMARGIN / FIG_H, PANEL_H / FIG_H

    ax_s = fig.add_axes([0.0, y0, SCHEM_W / FIG_W, hh])
    peak = draw_schematic(ax_s)

    ax_p = fig.add_axes([(SCHEM_W + GUTTER) / FIG_W, y0, PHOTO_W / FIG_W, hh])
    ax_p.imshow(plt.imread(cell))
    ax_p.set_xticks([]); ax_p.set_yticks([])
    for sp in ax_p.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color("#444444")

    # From the peak the drawing marks, to the image the sampler actually produced
    # there. The line is the figure's whole argument, so it is the only arrow drawn.
    fig.add_artist(ConnectionPatch(
        xyA=peak, coordsA=ax_s.transData,
        xyB=(0.0, 0.5), coordsB=ax_p.transAxes,
        linewidth=0.7, color="#444444", zorder=6,
        arrowstyle="-|>", mutation_scale=7, shrinkA=4, shrinkB=1,
    ))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name}.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"panels     {SCHEM_W:.2f}in + {GUTTER:.2f}in gutter + {PHOTO_W:.2f}in, "
          f"both {PANEL_H:.2f}in tall")
    print(f"wrote      {out} and {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
