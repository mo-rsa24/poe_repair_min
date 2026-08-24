#!/usr/bin/env python
"""When the correction arrives decides whether it works.

Every cat x dog cell in the window experiment, 9 windows by 4 seeds, drawn as a
grid of real images. Nothing is selected: 36 cells exist and 36 are shown, so the
grid cannot flatter the claim by leaving a cell out.

One thing varies down the rows. The correction is the same vector at the same
strength applied for the same ten steps in every cell; only the ten steps it is
applied during change. Columns are the four noise draws, so a reader can see the
effect is not one lucky seed.

No count is printed on the cells. The detector disagrees with the pictures on this
pair: for seed 12 it reports one animal for every window from step 20 onward, and
those frames plainly hold a dog with a cat beside it. A number the figure's own
images contradict is worse than no number, so the grid argues from the pictures
and the rate curve carries the counting, with its own caveat.

Restoring the chips is `--counts`, and the disagreement is why it is off by
default rather than deleted.

    python scripts/window_position_grid.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WINDOW_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/window")
CURVES = WINDOW_ROOT / "window_curves.json"
FIG_DIR = Path("paper/iclr/figures")
FIG_NAME = "samples-as-the-correction-window-moves"

PAIR = "a_cat__x__a_dog"

# Two greys, not a red and a green. The success cells are already obvious from the
# pictures; the chip is there to be read, not to colour the reader's verdict in
# before they have looked at the image.
HIT_FACE, HIT_TEXT = "#111111", "white"
MISS_FACE, MISS_TEXT = "none", "#222222"
INK = "#222222"

# Windows run left to right, seeds down. Time reads along the page the way a
# reader expects it to, and nine columns by four rows lands inside the 5.5in
# ICLR column where nine rows by four columns did not.
CELL = 0.545           # inches, one square panel
LEFT_GUTTER = 0.62     # row labels: which seed
TOP_BAND = 0.32        # column labels, plus a small margin above them
RIGHT_PAD = 0.06
BOTTOM_PAD = 0.08      # small margin, clear of the bottom row


def cell_png(seed: int, w0: int, w1: int) -> Path:
    tag = f"teacher_residual_const_lam100_w{w0}-{w1}"
    d = WINDOW_ROOT / "pairs" / PAIR / f"seed_{seed}" / tag
    hits = sorted(d.glob("*.png"))
    if not hits:
        raise SystemExit(f"no image under {d}")
    return hits[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--name", default=FIG_NAME)
    ap.add_argument("--counts", action="store_true",
                    help="print the detector's animal count on each cell. Off by "
                         "default: it disagrees with the images on seed 12.")
    args = ap.parse_args()

    doc = json.loads(CURVES.read_text())
    cells = [r for r in doc["scores"] if r["pair"] == PAIR]
    windows = sorted({tuple(r["window"]) for r in cells})
    seeds = sorted({r["seed"] for r in cells})
    count = {(r["seed"], tuple(r["window"])): int(r["n_instances"]) for r in cells}

    missing = [(s, w) for s in seeds for w in windows if (s, w) not in count]
    if missing:
        raise SystemExit(f"grid has holes, {len(missing)} cells missing: {missing[:5]}")
    print(f"grid       {len(windows)} windows x {len(seeds)} seeds = {len(cells)} cells, "
          f"all present")

    nrow, ncol = len(seeds), len(windows)
    fig_w = LEFT_GUTTER + ncol * CELL + RIGHT_PAD
    fig_h = TOP_BAND + nrow * CELL + BOTTOM_PAD
    fig = plt.figure(figsize=(fig_w, fig_h))

    for i, seed in enumerate(seeds):
        for j, win in enumerate(windows):
            ax = fig.add_axes([
                (LEFT_GUTTER + j * CELL) / fig_w,
                (BOTTOM_PAD + (nrow - 1 - i) * CELL) / fig_h,
                CELL / fig_w, CELL / fig_h,
            ])
            ax.imshow(plt.imread(cell_png(seed, *win)))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5); sp.set_color("#888888")

            if args.counts:
                n = count[(seed, tuple(win))]
                composed = n >= 2
                ax.text(0.13, 0.13, str(n), transform=ax.transAxes,
                        ha="center", va="center", fontsize=5.5, zorder=5,
                        color=HIT_TEXT if composed else MISS_TEXT,
                        bbox=dict(boxstyle="circle,pad=0.26",
                                  facecolor=HIT_FACE if composed else MISS_FACE,
                                  edgecolor="#111111", linewidth=0.5))

        ax_lab = fig.add_axes([0.0, (BOTTOM_PAD + (nrow - 1 - i) * CELL) / fig_h,
                               LEFT_GUTTER / fig_w, CELL / fig_h])
        ax_lab.axis("off")
        ax_lab.text(0.90, 0.5, f"seed {seed}", ha="right", va="center",
                    fontsize=7, family="serif", color=INK)

    for j, win in enumerate(windows):
        ax_top = fig.add_axes([(LEFT_GUTTER + j * CELL) / fig_w,
                               (BOTTOM_PAD + nrow * CELL) / fig_h,
                               CELL / fig_w, 0.26 / fig_h])
        ax_top.axis("off")
        ax_top.text(0.5, 0.05, f"{win[0]}–{win[1]}", ha="center", va="bottom",
                    fontsize=6.5, family="serif", color=INK)


    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name}.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"size       {fig_w:.2f} x {fig_h:.2f} in")
    print(f"wrote      {out} and {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
