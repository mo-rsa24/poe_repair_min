"""Where a ten-step correction has to land: two figures, not one.

``--panel strip`` draws four seeds of one cat-and-dog pair against nine positions
of the window, at the full ICLR text width. That is what the failure looks like,
and it needs the width, because a reader has to count animals in each cell.

``--panel counts`` draws every pair against the same nine positions, as how many
of its four seeds composed. That is how often it happens, and it needs almost no
width, because a cell holds one digit. It is sized for a ``wrapfigure`` so the
prose runs beside it.

Holding them side by side in one figure cost the strip a fifth of its cell,
0.293 in against 0.561, to make room for a panel that did not need the space.
Splitting them gives each the width its content actually wants.

Both verdicts are the same validated instance-count scorer, from
window_curves.json. The two daggered cells in the counts figure are detector
error, one animal boxed twice, and they are marked rather than quietly dropped.

Placed with ``\\includegraphics[width=\\textwidth]`` for the strip and inside a
``wrapfigure`` at the width printed by ``--panel counts`` for the other.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from poe_repair import paths

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import later_start_grid as g

FIG_DIR = Path("paper/iclr/figures")
FIG_NAME = "where-the-correction-lands"
WINDOW = paths.resolve(paths.WINDOW)
PAIR = "a_cat__x__a_dog"
SEEDS = (9, 10, 11, 12)
STARTS = list(range(0, 45, 5))
DAGGER = {("a_frog__x__a_toad", 35), ("a_frog__x__a_toad", 40)}

TEXT_WIDTH = 5.50
GAP = 0.16            # between the two panels
PAD = 0.07
HEAT_COL = 0.145      # a heatmap column only holds a digit, so it is narrow
XTICK_BAND = 0.20     # the "first step" axis under (b)
NUM_BAND = 0.14       # the window ranges over (a)
CAPTION_BAND = 0.20
LEGEND_BAND = 0.17
BOTTOM_PAD = 0.04
FONT, TICK = 8, 6.5
INK = g.INK


def pretty(slug: str) -> str:
    return slug.replace("__x__", " × ").replace("a_", "").replace("an_", "")


def measure(s, **kw):
    f = plt.figure(figsize=(1, 1)); f.canvas.draw()
    t = f.text(0, 0, s, **kw)
    w = t.get_window_extent(renderer=f.canvas.get_renderer()).width / f.dpi
    plt.close(f)
    return w


def draw_strip(args, verdict, Ft):
    """Four seeds against nine window positions, at the full text width."""
    gut = measure("seed 10", fontsize=TICK, family="serif") + PAD
    n = len(STARTS)
    cell = (TEXT_WIDTH - gut - 0.02) / n
    fig_w = TEXT_WIDTH
    fig_h = BOTTOM_PAD + LEGEND_BAND + len(SEEDS) * cell + NUM_BAND
    y0 = BOTTOM_PAD + LEGEND_BAND
    top = y0 + len(SEEDS) * cell

    fig = plt.figure(figsize=(fig_w, fig_h)); fig.canvas.draw()
    for i, seed in enumerate(SEEDS):
        y = y0 + (len(SEEDS) - 1 - i) * cell
        for j, s in enumerate(STARTS):
            ax = fig.add_axes([(gut + j * cell) / fig_w, y / fig_h,
                               cell / fig_w, cell / fig_h])
            ax.imshow(plt.imread(g.final_png(seed, s, s + 10)))
            ax.set_xticks([]); ax.set_yticks([])
            on = verdict[(seed, s)]
            for sp in ax.spines.values():
                sp.set_linewidth(1.6 if on else 0.4)
                sp.set_color(g.CORRECTED_EDGE if on else g.PLAIN_EDGE)
        fig.text((gut - PAD / 2) / fig_w, (y + cell / 2) / fig_h, f"seed {seed}",
                 ha="right", va="center", **Ft)
    for j, s in enumerate(STARTS):
        fig.text((gut + (j + 0.5) * cell) / fig_w, (top + 0.03) / fig_h,
                 f"{s}\u2013{s + 10}", ha="center", va="bottom", **Ft)

    sw, ly = 0.10, BOTTOM_PAD + 0.02
    ax = fig.add_axes([gut / fig_w, ly / fig_h, sw / fig_w, sw / fig_h])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_linewidth(1.6); sp.set_color(g.CORRECTED_EDGE)
    fig.text((gut + sw + 0.05) / fig_w, (ly + sw / 2) / fig_h,
             "both animals in the picture",
             ha="left", va="center", **Ft)
    return fig, cell, fig_w, fig_h


def draw_counts(args, counts, pairs, Ft):
    """Every pair against the same nine positions, sized for a wrapfigure."""
    width = args.wrap_width * TEXT_WIDTH
    gut = max(measure(pretty(p_), fontsize=TICK, family="serif") for p_ in pairs) + PAD
    n = len(STARTS)
    col = (width - gut - 0.02) / n
    grid_h = len(pairs) * col                      # square cells
    fig_w = width
    fig_h = BOTTOM_PAD + XTICK_BAND + grid_h + 0.04
    y0 = BOTTOM_PAD + XTICK_BAND

    fig = plt.figure(figsize=(fig_w, fig_h)); fig.canvas.draw()
    axh = fig.add_axes([gut / fig_w, y0 / fig_h, (n * col) / fig_w, grid_h / fig_h])
    grid = np.array([[counts[(p_, s)] for s in STARTS] for p_ in pairs], dtype=float)
    axh.imshow(grid, cmap="Blues", vmin=0, vmax=len(SEEDS), aspect="auto")
    axh.set_xticks([]); axh.set_yticks([])
    for sp in axh.spines.values():
        sp.set_visible(False)
    for r, p_ in enumerate(pairs):
        for c, s in enumerate(STARTS):
            v = counts[(p_, s)]
            axh.text(c, r, f"{v}\u2020" if (p_, s) in DAGGER else str(v),
                     ha="center", va="center", fontsize=TICK, family="serif",
                     color="white" if v >= 3 else INK)
        fig.text((gut - PAD / 2) / fig_w, (y0 + grid_h - (r + 0.5) * col) / fig_h,
                 pretty(p_), ha="right", va="center", **Ft)
    for c, s in enumerate(STARTS):
        fig.text((gut + (c + 0.5) * col) / fig_w, (y0 - 0.09) / fig_h,
                 str(s), ha="center", va="bottom", **Ft)
    lab = "first step of the corrected window"
    wx = measure(lab, fontsize=TICK, family="serif")
    lx = min(max(gut + (n * col - wx) / 2, 0.02), fig_w - 0.02 - wx)
    fig.text(lx / fig_w, 0.03 / fig_h, lab, ha="left", va="bottom", **Ft)
    return fig, col, fig_w, fig_h


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--panel", choices=("strip", "counts"), default="strip")
    ap.add_argument("--name", default=None,
                    help="defaults to the panel's own figure name")
    ap.add_argument("--wrap-width", type=float, default=0.48,
                    help="counts panel only: width as a fraction of textwidth, "
                         "matching the wrapfigure it is placed in")
    ap.add_argument("--dpi", type=int, default=910)
    args = ap.parse_args()

    scores = json.loads((WINDOW / "window_curves.json").read_text())["scores"]
    Ft = dict(fontsize=TICK, family="serif", color=INK)

    if args.panel == "strip":
        verdict = {(s["seed"], s["window"][0]): s["compose"]
                   for s in scores if s["pair"] == PAIR}
        fig, unit, fw, fh = draw_strip(args, verdict, Ft)
        name = args.name or "where-the-correction-lands"
    else:
        counts = {}
        for s in scores:
            k = (s["pair"], s["window"][0])
            counts[k] = counts.get(k, 0) + s["compose"]
        pairs = sorted({s["pair"] for s in scores},
                       key=lambda p_: -sum(counts[(p_, st)] for st in STARTS))
        fig, unit, fw, fh = draw_counts(args, counts, pairs, Ft)
        name = args.name or "how-often-the-correction-lands"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{name}.png"
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(out.with_suffix(suffix), dpi=args.dpi)
    plt.close(fig)
    print(f"panel  {args.panel}: cell {unit:.3f} in")
    print(f"size   {fw:.2f} x {fh:.2f} in", end="")
    if args.panel == "counts":
        print(f"   -> wrapfigure width {args.wrap_width:g}\\textwidth", end="")
    print()
    print(f"wrote  {out.with_suffix('.pdf')} (+ .png, .svg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
