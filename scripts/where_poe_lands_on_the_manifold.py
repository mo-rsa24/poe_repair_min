#!/usr/bin/env python
"""Where each prompt lands on the manifold, and where the product lands instead.

A drawn surface with four real renders pinned to it. Height is how readily the model
produces an image at that point, over an illustrative two-dimensional plane. Three
peaks: a cat on its own, a dog on its own, and the two-animal scene the joint prompt
"a cat and a dog" reaches. Marked separately is the peak of the product of the two
single-animal densities, which is what plain PoE composition maximises. That peak lands
on the low saddle between the two single-animal mountains, and it is nowhere near the
two-animal one.

The surface is drawn; the four thumbnails are not. All four come from one cell,
a_cat__x__a_dog at seed 9, same starting latent, 50 steps, guidance 7.5, no adapter
attached, so the only difference between the four pictures is what the sampler was
conditioned on. The script prints every path it read.

Nothing about the surface is measured. The plane is not an embedding of anything, the
peaks are Gaussians placed by hand, the ripples exist so it reads as a landscape rather
than three balloons, and the three peaks are drawn at equal height because this figure
makes no claim about how often each occurs. The one geometric fact it honours is the
one F1 honours: the product of two Gaussians is a Gaussian at their midpoint, so the
product's peak is drawn where the maths puts it.

Companion to scripts/what_the_product_misses.py (F1), which argues the same thing in
flat contour lines.

    python scripts/where_poe_lands_on_the_manifold.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.mplot3d import proj3d
import numpy as np

FIG_DIR = Path("paper/iclr/figures")
FIG_NAME = "where-the-product-lands-on-the-manifold"

# One cell, four conditions, one starting latent. Written by
# scripts/showcase/where_each_condition_lands.py; the manifest beside these frames
# records init_latents, steps, guidance and adapter_attached_before_references=false.
CELL_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/"
                 "where_each_condition_lands/pairs/a_cat__x__a_dog")
SEED = 9
FINAL_FRAME = "step_050.png"

# The drawn plane. Coordinates mean nothing; its extent is a layout choice below.
CAT_XY, DOG_XY = (-2.05, 1.45), (2.05, 1.45)
SIG = (0.95, 0.92)
JOINT_XY, JOINT_SIG = (0.05, -2.30), (1.40, 1.02)
PEAK_H = 1.00          # all three prompts drawn the same height: no claim about mass

WARM = "#b2402a"       # one concept on its own
GREEN = "#1f6b45"      # the reading the prompt asks for and the product never reaches
COOL = "#26527f"       # what the product asks for
BORDER = "#555555"
LABEL_PT = 8.0
CHIP_PT = 6.4

# Page geometry, in inches. ICLR 2027 is single column at 5.5in textwidth.
#
# Two layouts, because the two are read at different sizes. `spacious` is the original
# balance, which reads on screen. `print` is measured against a 5.5in column: the four
# renders carry the argument, and at 0.95in they print at 142px, where the fused
# animal's dog muzzle on a cat's frame stops being visible. So `print` grows every
# thumbnail, trims the surface's empty skirt, and coarsens the mesh, whose fine white
# lines on a coloured ground are where print moire shows up.
#
# Each condition entry: the frame directory, the words beside its thumbnail, its
# colour, which corner the leader leaves from, and whether its words go above or below.
# The joint render sits low on the page, where a word above it would land on the
# surface, so its words go underneath.
CONDITIONS = [
    ("solo_a", '"a cat"',                     WARM,  "s", "above"),
    ("poe",    "PoE: the product of the two", COOL,  "s", "above"),
    ("solo_b", '"a dog"',                     WARM,  "s", "above"),
    ("joint",  '"a cat and a dog"',           GREEN, "e", "below"),
]

LAYOUTS = {
    "spacious": dict(
        fig=(5.50, 4.35), thumb=0.95, plane=4.00, edge=(3.70, 4), mesh=(150, 2),
        axes=(-0.10, -0.11, 1.20, 1.13), elev=36.0, zbox=0.62,
        slots={"solo_a": (0.108, 0.775), "poe": (0.500, 0.805),
               "solo_b": (0.892, 0.775), "joint": (0.150, 0.215)},
    ),
    # Sized so the float and its caption fit on the page that introduces it. At the
    # `print` layout's 4.35in the float plus caption fills two thirds of an ICLR page,
    # and LaTeX pushes it past the paragraph that says "as shown in Figure 1". This one
    # is 3.30in: a lower camera and a flatter surface buy the height back without
    # shrinking the renders below the size at which the fused animal reads.
    "column": dict(
        fig=(5.50, 3.62), thumb=1.04, plane=3.40, edge=(3.15, 8), mesh=(132, 3),
        axes=(-0.03, -0.085, 1.06, 1.15), elev=27.0, zbox=0.50,
        slots={"solo_a": (0.140, 0.775), "poe": (0.520, 0.808),
               "solo_b": (0.860, 0.775), "joint": (0.108, 0.218)},
        border=1.7,
        label_side={"joint": "below"},
    ),
    "print": dict(
        fig=(5.50, 4.35), thumb=1.26, plane=3.40, edge=(3.15, 8), mesh=(132, 3),
        axes=(0.005, -0.055, 0.99, 0.95), elev=36.0, zbox=0.62,
        slots={"solo_a": (0.128, 0.755), "poe": (0.500, 0.790),
               "solo_b": (0.872, 0.755), "joint": (0.145, 0.183)},
    ),
}
LAYOUT = LAYOUTS["spacious"]     # set per run by build()


def bump(xx, yy, cx, cy, sx, sy):
    return np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2)


def surface(xx, yy):
    """Three prompts as peaks, on a gently rolling floor."""
    e_scale, e_pow = LAYOUT["edge"]
    z = PEAK_H * (bump(xx, yy, *CAT_XY, *SIG)
                  + bump(xx, yy, *DOG_XY, *SIG)
                  + bump(xx, yy, *JOINT_XY, *JOINT_SIG))
    # Small enough never to rival a peak, large enough that the eye reads one
    # continuous landscape instead of three balloons on a table.
    ripple = (0.075 * np.sin(0.95 * xx + 0.35) * np.cos(0.85 * yy - 0.20)
              + 0.045 * np.sin(1.55 * yy + 1.1) * np.cos(0.70 * xx + 0.6))
    edge = np.exp(-((xx / e_scale) ** e_pow + (yy / e_scale) ** e_pow))
    return (z + ripple) * edge


def height_at(x, y):
    return float(surface(np.array([[x]]), np.array([[y]]))[0, 0])


def to_figure_xy(fig, ax, x, y, z):
    """A point on the surface, in figure fractions, so a flat artist can reach it."""
    px, py, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
    disp = ax.transData.transform((px, py))
    return fig.transFigure.inverted().transform(disp)


def draw_surface(ax) -> None:
    lim = (-LAYOUT["plane"], LAYOUT["plane"])
    n, stride = LAYOUT["mesh"]
    g = np.linspace(*lim, n)
    xx, yy = np.meshgrid(g, g)
    ax.plot_surface(xx, yy, surface(xx, yy), rstride=stride, cstride=stride,
                    cmap="coolwarm", linewidth=0.18, edgecolors=(1, 1, 1, 0.32),
                    antialiased=True, shade=False, vmin=-0.12, vmax=PEAK_H * 1.02)
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_zlim(-0.28, 1.30)
    ax.set_box_aspect((1.0, 1.0, LAYOUT["zbox"]))
    ax.set_axis_off()


def anchors():
    """Where each condition sits on the surface, in data coordinates."""
    sx, sy = (CAT_XY[0] + DOG_XY[0]) / 2, (CAT_XY[1] + DOG_XY[1]) / 2
    return {"solo_a": CAT_XY, "solo_b": DOG_XY, "joint": JOINT_XY, "poe": (sx, sy)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", choices=sorted(LAYOUTS), default="spacious",
                    help="spacious reads on screen; print is balanced for a 5.5in "
                         "column, with larger renders and a quieter mesh")
    ap.add_argument("--elev", type=float, default=None,
                    help="camera elevation; each layout carries its own default")
    ap.add_argument("--azim", type=float, default=-62.0)
    ap.add_argument("--seed", type=int, default=SEED,
                    help="which cell's four renders to pin on")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds; writes one file each, suffixed "
                         "-seedNN, for picking a cell by eye. The joint render is "
                         "two dogs on seeds 10 and 13 and ambiguous on 14 and 15, "
                         "so those cells cannot carry this figure.")
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--name", default=FIG_NAME)
    args = ap.parse_args()

    seeds = ([int(t) for t in args.seeds.split(",")] if args.seeds
             else [args.seed])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        suffix = f"-seed{seed:02d}" if args.seeds else ""
        if args.layout != "spacious":
            suffix += f"-{args.layout}"
        build(args, seed, args.out_dir / f"{args.name}{suffix}.png")
    return 0


def build(args, seed: int, out: Path) -> None:
    global LAYOUT
    LAYOUT = LAYOUTS[args.layout]
    fig_w, fig_h = LAYOUT["fig"]
    thumb = LAYOUT["thumb"]
    frames = {c: CELL_ROOT / c / f"seed_{seed}" / FINAL_FRAME
              for c, *_ in CONDITIONS}
    for cond, p in frames.items():
        if not p.exists():
            raise SystemExit(f"no render for {cond} at {p}")
        print(f"seed {seed:>2d}  {cond:8s} {p}")

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes(list(LAYOUT["axes"]), projection="3d")
    draw_surface(ax)
    elev = args.elev if args.elev is not None else LAYOUT["elev"]
    ax.view_init(elev=elev, azim=args.azim)
    fig.canvas.draw()          # the projection is only real once the canvas exists

    at = anchors()
    zoom = thumb * fig.dpi / 512.0      # renders are 512px square

    for cond, words, colour, side, where in CONDITIONS:
        slot = LAYOUT["slots"][cond]
        where = LAYOUT.get("label_side", {}).get(cond, where)
        x, y = at[cond]
        z = height_at(x, y)
        tip = to_figure_xy(fig, ax, x, y, z)

        # The mark on the surface itself, so the thumbnail is pinned to a place and
        # not merely floating near one.
        ax.plot([x], [y], [z], marker="X" if cond == "poe" else "o",
                markersize=7.5 if cond == "poe" else 5.0, linestyle="none",
                color=colour, markeredgecolor="white", markeredgewidth=0.9,
                zorder=10)

        img = OffsetImage(plt.imread(frames[cond]), zoom=zoom)
        # The image is drawn at `xybox` (its slot on the page) and the arrow runs from
        # it to `xy` (the mark on the surface), so the thumbnail is pinned to a place.
        box = AnnotationBbox(
            img, tip, xycoords="figure fraction",
            xybox=slot, boxcoords="figure fraction", box_alignment=(0.5, 0.5),
            frameon=True, pad=0.0,
            bboxprops=dict(edgecolor=colour, linewidth=0.9),
            arrowprops=dict(arrowstyle="-", linewidth=0.7, color=colour,
                            shrinkA=1.5, shrinkB=2.0,
                            relpos={"s": (0.5, 0.0), "w": (0.0, 0.5),
                                    "n": (0.5, 1.0), "e": (1.0, 0.5)}[side]))
        fig.add_artist(box)

        half = (thumb / fig_h) / 2
        fig.text(slot[0], slot[1] + (half + 0.012 if where == "above"
                                     else -half - 0.012), words,
                 color=colour, fontsize=LABEL_PT, family="serif",
                 ha="center", va="bottom" if where == "above" else "top")

    # The gap between where the product peaks and where the joint prompt lands is the
    # whole argument, so it is the only distance drawn on the surface.
    px, py = at["poe"]
    t = np.linspace(0, 1, 60)
    gx, gy = px + t * (JOINT_XY[0] - px), py + t * (JOINT_XY[1] - py)
    gz = np.array([height_at(a, b) for a, b in zip(gx, gy)]) + 0.045
    ax.plot(gx, gy, gz, linestyle=(0, (3, 2.5)), linewidth=1.0, color=BORDER, zorder=9)
    mid = to_figure_xy(fig, ax, gx[34], gy[34], gz[34])
    fig.text(mid[0] + 0.018, mid[1], "the product never gets here", color=BORDER,
             fontsize=CHIP_PT, family="serif", ha="left", va="center",
             bbox=dict(facecolor="white", edgecolor="none", pad=1.4, alpha=0.80))

    fig.savefig(out, dpi=400, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"         wrote {out} and {out.with_suffix('.pdf')}, "
          f"elev {elev:.0f} azim {args.azim:.0f}")


if __name__ == "__main__":
    raise SystemExit(main())
