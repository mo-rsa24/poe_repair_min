#!/usr/bin/env python
"""Compose register slot F2: the 4x5 grid of real cells above the four dose curves.

Rows: the pair's own r_t, then the three controls that each break exactly one
thing: wrong pair (another pair's r_t), wrong seed (same pair, another run's
r_t), wrong step (own r_t, step order deranged). The norm-matched random
vector is not a row: its number lives in the appendix, it answers only "does
any disturbance help", and the wrong-pair row already dominates it.

Two layouts, both spec'd by /pair-figure then /design-figure:

    --layout grid5    the full 4x5 grid, with the shared lambda=0 column boxed and
                      labelled. The paper figure.
    --layout shared0  lambda=0 drawn once on the second row, the rest of that
                      column empty. Better at half width (talk, poster).

Reads only artifacts that already exist. The per-panel animal counts come from
dose_curves.json's `n_instances`, not from a fresh detector pass, so this script
needs no GPU and cannot disagree with the curves it sits above.

Column centres land exactly on the curve axis ticks: the curves axis spans
lambda -0.125 to 1.125, half a column past each end, so column k sits over tick k.

    python scripts/make_f2.py --layout grid5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

DOSE = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose")
CURVES = DOSE / "dose_curves.json"
PAIRS_ROOT = Path("outputs/interaction_term/dose/pairs")

# The figure lands beside the manuscript, not on /datasets. A paper figure is a few
# hundred KB and LaTeX has to reach it, so the "large artifacts to /datasets" rule
# does not apply here. Names carry the register slot, so a file found on its own says
# which slot it fills.
FIG_DIR = Path("paper/iclr/figures")
FIG_NAME = {"grid5": "compose-rate-as-correction-rises", "shared0": "compose-rate-as-correction-rises-with-a-random-control"}

ROWS = ("oracle", "wrong_pair", "wrong_seed", "wrong_step")
ROW_LABEL = {"oracle": "real correction", "wrong_pair": "wrong pair",
             "wrong_seed": "wrong seed", "wrong_step": "wrong order"}
ROW_COLOR = {"oracle": "#1f77b4", "wrong_pair": "#ff7f0e",
             "wrong_seed": "#2ca02c", "wrong_step": "#9467bd"}
ROW_STYLE = {"oracle": "-", "wrong_pair": "--",
             "wrong_seed": "--", "wrong_step": "--"}
ROW_MARKER = {"oracle": "o", "wrong_pair": "^",
              "wrong_seed": "v", "wrong_step": "d"}

# Page geometry, in inches. ICLR 2027 is single column: iclr2027_conference.sty
# line 49 sets \textwidth 5.5 true in, so F2 is a `figure`, not a `figure*`.
FIG_W = 5.5
GUTTER = 0.92          # left band carrying the row labels, sized to the longest
PANEL = 0.92           # one image panel, square
RIGHT_MARGIN = 0.18
TOP_MARGIN = 0.06
NOTE_BAND = 0.24       # the one line under the grid
CURVE_H = 1.55
XLABEL_BAND = 0.42
FIG_H = TOP_MARGIN + 4 * PANEL + NOTE_BAND + CURVE_H + XLABEL_BAND


def run_dir(pair: str, seed: int, lam: float, row: str) -> Path:
    """Where the sampler wrote this cell. At lambda=0 nothing is injected, so all
    three rows are the same image by construction and share one directory."""
    tag = f"teacher_residual_const_lam{int(round(lam * 100)):03d}"
    if lam > 0 and row != "oracle":
        tag = f"{tag}_{row}"
    return PAIRS_ROOT / pair / f"seed_{seed}" / tag


def panel_png(pair: str, seed: int, lam: float, row: str) -> Path:
    d = run_dir(pair, seed, lam, row)
    hits = sorted(d.glob("*.png"))
    if not hits:
        raise SystemExit(f"no image under {d}")
    return hits[0]


def counts_for_cell(scores, pair: str, seed: int) -> dict[tuple[str, float], int]:
    out = {}
    for s in scores:
        if s["pair"] == pair and s["seed"] == seed:
            out[(s["row"], float(s["lambda"]))] = int(s["n_instances"])
    return out


def draw_chip(ax, count: int) -> None:
    """The verdict, as the detected count. A numeral rather than a word, so the
    lambda=1 oracle panel can honestly read 3: there are three animals in it, and a
    reader who counts three under a chip saying 2 would rightly distrust the figure.
    Filled background means the compose rule fired (count >= 2)."""
    composed = count >= 2
    ax.text(
        0.055, 0.055, str(count),
        transform=ax.transAxes, ha="center", va="center", fontsize=7,
        color="white" if composed else "#222222", zorder=5,
        bbox=dict(boxstyle="circle,pad=0.30",
                  facecolor="#111111" if composed else "none",
                  edgecolor="#111111", linewidth=0.7),
    )


def add_panel(fig, rect, png: Path, count: int) -> None:
    ax = fig.add_axes(rect)
    ax.imshow(plt.imread(png))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        # Every panel rendered identically. The control rows are NOT greyed or
        # faded: demoting them would tell the reader which rows are meant to lose
        # before they have looked, and a control's whole worth is that it was
        # given the same chance.
        sp.set_linewidth(0.6)
        sp.set_color("#444444")
    draw_chip(ax, count)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", choices=("grid5", "shared0"), default="grid5")
    # cat x dog carries the figure: strongest pair in the set (0% to 100%, AUC 0.562)
    # and the only one whose controls score exactly 0.000, so the grid and the curves
    # tell the same story without a caveat.
    ap.add_argument("--pair", default="a_cat__x__a_dog")
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--name", default=None,
                    help="output basename, overriding the layout's default. Use when "
                         "building a variant on a different pair.")
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    args = ap.parse_args()

    d = json.loads(CURVES.read_text())
    lams = [float(x) for x in d["lambdas"]]
    scores = d["scores"]
    counts = counts_for_cell(scores, args.pair, args.seed)
    missing = [(r, l) for r in ROWS for l in lams if (r, l) not in counts]
    if missing:
        raise SystemExit(f"no scores for {args.pair} seed {args.seed}: {missing}")

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    x0 = GUTTER / FIG_W
    pw = PANEL / FIG_W
    ph = PANEL / FIG_H
    grid_bottom = (XLABEL_BAND + CURVE_H + NOTE_BAND) / FIG_H
    row_bottoms = [grid_bottom + (3 - i) * ph for i in range(4)]   # row 0 on top

    for i, row in enumerate(ROWS):
        for j, lam in enumerate(lams):
            if args.layout == "shared0" and lam == 0.0 and row != "wrong_pair":
                continue    # lambda=0 drawn once, on the second row
            rect = [x0 + j * pw, row_bottoms[i], pw, ph]
            add_panel(fig, rect, panel_png(args.pair, args.seed, lam, row),
                      counts[(row, lam)])

        fig.text(x0 - 0.012, row_bottoms[i] + ph / 2, ROW_LABEL[row],
                 ha="right", va="center", fontsize=8)

    # The shared lambda=0 column, disclosed rather than styled away.
    if args.layout == "grid5":
        fig.patches.append(FancyBboxPatch(
            (x0 + 0.004, grid_bottom + 0.004), pw - 0.008, 4 * ph - 0.008,
            boxstyle="round,pad=0.004", transform=fig.transFigure,
            facecolor="none", edgecolor="#1f77b4", linewidth=1.0, zorder=6))
        note = "$\\lambda$=0: same image, nothing injected"
    else:
        # The rule sits on the column's RIGHT edge, dividing the one shared sample
        # from the four injected columns. Through the panel's centre it would read
        # as a crop mark.
        fig.patches.append(Rectangle(
            (x0 + pw, grid_bottom), 0.0, 4 * ph, transform=fig.transFigure,
            edgecolor="#999999", linewidth=0.8, zorder=7))
        note = "$\\lambda$=0: one sample, all rows"

    note_y = grid_bottom - NOTE_BAND / FIG_H * 0.55
    fig.text(x0, note_y, note, ha="left", va="center", fontsize=7, color="#1f77b4")
    fig.text(x0 + 5 * pw, note_y, "count = animals detected;  filled = composed",
             ha="right", va="center", fontsize=7, color="#333333")

    # The curves, on the same left and right bounds as the grid.
    ax = fig.add_axes([x0, XLABEL_BAND / FIG_H, 5 * pw, CURVE_H / FIG_H])
    for row in ROWS:
        rate = []
        for lam in lams:
            cells = [s["compose"] for s in scores
                     if s["row"] == row and float(s["lambda"]) == lam]
            n, k = len(cells), sum(cells)
            rate.append(k / n)
        ax.plot(lams, rate, ROW_STYLE[row], color=ROW_COLOR[row],
                marker=ROW_MARKER[row], markersize=4, linewidth=1.6,
                label=ROW_LABEL[row])

    ax.set_xlim(-0.125, 1.125)          # column k sits over tick k
    ax.set_ylim(0, 1.0)
    ax.set_xticks(lams)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("$\\lambda$", fontsize=9)
    ax.set_ylabel("compose rate", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name or FIG_NAME[args.layout]}.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"layout   {args.layout}")
    print(f"cell     {args.pair} seed {args.seed}")
    print(f"counts   " + "  ".join(
        f"{r}:{''.join(str(counts[(r, l)]) for l in lams)}" for r in ROWS))
    print(f"figure   {out}")
    print(f"         {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
