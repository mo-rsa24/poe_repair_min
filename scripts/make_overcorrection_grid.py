#!/usr/bin/env python
"""Over-correction grid: does pushing lambda past 1.0 help, hurt, or do
nothing, and does that depend on which window the correction lands in?

Rows are 5 non-overlapping windows (0-10, 10-20, 20-30, 30-40, 40-50), each a
tenth of the run. Columns are 4 lambda values, 0.5 to 2.0 in steps of 0.5, so
1.0 (the normal, non-overdosed strength) sits in the grid as a reference
column, not an endpoint. Cat x dog, seed 9, matching every other qualitative
figure in this paper.

Different window definition from F4a/F4b/F4c/F4e's 9-window stride-5 grid
(poe_repair/experiments/interaction_term/window_grid.py): this one is
non-overlapping, by request, so a row reads as "which tenth of the run", not
"which sliding position".

Every panel is scored fresh with the validated instance-count scorer
(GroundingDINO, compose iff kept boxes >= 2), same rule as F2b, so the count
in the corner is checkable against the picture rather than asserted.

Needs a GPU: every panel is scored live.

    python scripts/make_overcorrection_grid.py
    python scripts/make_overcorrection_grid.py --seed 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAIR = "a_cat__x__a_dog"
WINDOWS = ((0, 10), (10, 20), (20, 30), (30, 40), (40, 50))
LAMBDAS = (0.5, 1.0, 1.5, 2.0)

IMAGE_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/"
                  "overcorrection_grid/pairs")
SCORER_CONTRACT = Path("outputs/compose_scorer/scorer_validated.json")
OUT_DIR = Path("paper/iclr/figures")

FIG_W = 5.5
GUTTER = 0.85
PANEL = 0.95
TOP_MARGIN = 0.24
BOTTOM_MARGIN = 0.06
FIG_H = TOP_MARGIN + 5 * PANEL + BOTTOM_MARGIN

BOX_COLOR = "#e6b800"


def cell_png(win: tuple[int, int], lam: float, seed: int) -> Path:
    lam_tag = f"{round(lam * 100):03d}"
    d = IMAGE_ROOT / PAIR / f"seed_{seed}" / f"teacher_residual_const_lam{lam_tag}_w{win[0]}-{win[1]}"
    hits = sorted(d.glob("*.png"))
    if not hits:
        raise SystemExit(f"no image under {d}")
    return hits[0]


def draw_chip(ax, count: int) -> None:
    composed = count >= 2
    ax.text(
        0.06, 0.06, str(count),
        transform=ax.transAxes, ha="center", va="center", fontsize=7,
        color="white" if composed else "#222222", zorder=5,
        bbox=dict(boxstyle="circle,pad=0.30",
                  facecolor="#111111" if composed else "none",
                  edgecolor="#111111", linewidth=0.7),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=9)
    args = ap.parse_args()

    contract = json.loads(SCORER_CONTRACT.read_text())
    if not contract.get("pass"):
        raise SystemExit(f"scorer at {SCORER_CONTRACT} is not marked validated: refusing.")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances

    panels = {}
    for win in WINDOWS:
        for lam in LAMBDAS:
            png = cell_png(win, lam, args.seed)
            n, boxes = count_instances(png)
            panels[(win, lam)] = (png, n, boxes)
            print(f"  window {win[0]:>2}-{win[1]:<2}  lambda {lam:.1f}  n={n}  {png.name}")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    x0 = GUTTER / FIG_W
    pw = PANEL / FIG_W
    ph = PANEL / FIG_H
    row_bottoms = [BOTTOM_MARGIN / FIG_H + (4 - i) * ph for i in range(5)]

    for i, win in enumerate(WINDOWS):
        for j, lam in enumerate(LAMBDAS):
            png, n, boxes = panels[(win, lam)]
            rect = [x0 + j * pw, row_bottoms[i], pw, ph]
            ax = fig.add_axes(rect)
            ax.imshow(plt.imread(png))
            for b in boxes:
                bx0, by0, bx1, by1 = b["box"]
                ax.add_patch(plt.Rectangle(
                    (bx0, by0), bx1 - bx0, by1 - by0, fill=False,
                    linewidth=1.1, edgecolor=BOX_COLOR, zorder=4))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.6); sp.set_color("#444444")
            draw_chip(ax, n)
            if lam == 1.0:
                for sp in ax.spines.values():
                    sp.set_linewidth(1.4); sp.set_color("#1f77b4")
        fig.text(x0 - 0.012, row_bottoms[i] + ph / 2, f"{win[0]}–{win[1]}",
                 ha="right", va="center", fontsize=8)

    for j, lam in enumerate(LAMBDAS):
        fig.text(x0 + j * pw + pw / 2, row_bottoms[0] + ph + 0.06 / FIG_H,
                 f"$\\lambda$={lam:.1f}", ha="center", va="bottom", fontsize=8)

    fig_name = ("F4g-overcorrection-grid" if args.seed == 9
               else f"F4g-overcorrection-grid-seed{args.seed}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{fig_name}.png"
    fig.savefig(out, dpi=300, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    print(f"\nwrote {out}")
    print(f"      {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
