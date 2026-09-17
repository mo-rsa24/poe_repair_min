"""Both correction-window grids side by side, sharing one set of row numbers.

Panel (a) applies the correction early: each row corrects from step 0 out to the
step its row number names. Panel (b) applies it late: each row leaves the start
plain and corrects from its row number through to step 50.

The two panels share their row axis, so the numbers are printed once against the
left grid and read across. The column axis carries no numbers and no title at
all; the caption says the columns are decodes taken after 10, 20, 30, 40 and 50
steps, which is cheaper than printing five numbers and a heading on the figure.

An arrow on each outer edge says which way that panel's correction grows. It
points down on the left, because a lower row in (a) corrects more of the run, and
up on the right, because a higher row in (b) starts earlier and so also corrects
more of it. The two arrows pointing at each other is the whole difference between
the panels, stated without words.

Both carry the same label, "correction at step", naming the row numbers they sit
beside. The numbers themselves increase downward on both sides, so the upward
arrow is saying which way coverage grows rather than which way the numbers count.
The panel captions, "applied early" against "applied late", are what keep those
two readings apart.

The figure is drawn at the ICLR text width, so it is placed with
\\includegraphics[width=\\textwidth] and nothing else. Any scaling factor there
would shrink the labels off their stated point size.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import later_start_grid as g
from poe_repair.experiments.interaction_term import window_grid as wg

FIG_DIR = Path("paper/iclr/figures/when-the-correction-arrives")
FIG_NAME = "when-the-correction-arrives"

TEXT_WIDTH = 5.50
EDGE = 0.22           # an arrow and its rotated label, on each outer edge
ROW_NUMS = 0.24       # the shared row numbers, against the left grid only
GAP = 0.28            # between panels; also holds panel (b)'s one "off" label
TOP_PAD = 0.06
CAPTION_BAND = 0.22
LEGEND_BAND = 0.20
BOTTOM_PAD = 0.05
FONT = 8
INK = g.INK

ARROW_LABEL = "correction at step"
CAPTIONS = {"a": "correction steps applied early",
            "b": "correction steps applied late"}


def panel_rows(which):
    if which == "a":
        return [(w, str(w[1])) for w in wg.prefix_windows()]
    return [(w, "off" if w[0] >= wg.NUM_STEPS else str(w[0]))
            for w in wg.suffix_windows()]


def text_width(fig, s, **kw):
    t = fig.text(0, 0, s, **kw)
    w = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--name", default=FIG_NAME)
    ap.add_argument("--seed", type=int, default=g.SEED)
    ap.add_argument("--width", type=float, default=TEXT_WIDTH)
    ap.add_argument("--dpi", type=int, default=1130,
                    help="resolution the thumbnails are embedded at. The decode "
                         "frames are 512 px and a cell is 0.454 in, so 1130 "
                         "embeds all 512 and anything higher invents detail.")
    args = ap.parse_args()

    ncol, nrow = len(g.CHUNKS), len(wg.prefix_windows())
    cell = (args.width - 2 * EDGE - ROW_NUMS - GAP) / (2 * ncol)
    fig_w = args.width
    fig_h = BOTTOM_PAD + LEGEND_BAND + CAPTION_BAND + nrow * cell + TOP_PAD

    xa = EDGE + ROW_NUMS
    xb = xa + ncol * cell + GAP
    y0 = BOTTOM_PAD + LEGEND_BAND + CAPTION_BAND
    top = y0 + nrow * cell

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.canvas.draw()
    F = dict(fontsize=FONT, family="serif", color=INK)

    for which, x0 in (("a", xa), ("b", xb)):
        for i, (win, label) in enumerate(panel_rows(which)):
            y = y0 + (nrow - 1 - i) * cell
            for j, chunk in enumerate(g.CHUNKS):
                ax = fig.add_axes([(x0 + j * cell) / fig_w, y / fig_h,
                                   cell / fig_w, cell / fig_h])
                ax.imshow(plt.imread(g.frame_png(args.seed, *win, chunk[1])))
                ax.set_xticks([]); ax.set_yticks([])
                on = g.contains(win, chunk)
                for sp in ax.spines.values():
                    sp.set_linewidth(1.7 if on else 0.4)
                    sp.set_color(g.CORRECTED_EDGE if on else g.PLAIN_EDGE)
            if which == "a":
                fig.text((x0 - 0.05) / fig_w, (y + cell / 2) / fig_h, label,
                         ha="right", va="center", **F)
            elif label == "off":
                fig.text((x0 - GAP / 2) / fig_w, (y + cell / 2) / fig_h, label,
                         ha="center", va="center", **F)

        letter, phrase = f"({which})", f" {CAPTIONS[which]}"
        wl = text_width(fig, letter, **F, weight="bold")
        wp = text_width(fig, phrase, **F)
        left = x0 + (ncol * cell - (wl + wp)) / 2
        cy = (BOTTOM_PAD + LEGEND_BAND + 0.07) / fig_h
        fig.text(left / fig_w, cy, letter, ha="left", va="bottom", weight="bold", **F)
        fig.text((left + wl) / fig_w, cy, phrase, ha="left", va="bottom", **F)

    # one arrow per outer edge, pointing the way that panel's coverage grows
    for x_arrow, x_text, tip, tail in (
            (EDGE - 0.07, 0.07, y0 + 0.05, top - 0.05),          # (a): downward
            (fig_w - EDGE + 0.07, fig_w - 0.07, top - 0.05, y0 + 0.05)):  # (b): upward
        fig.add_artist(FancyArrowPatch(
            (x_arrow / fig_w, tail / fig_h), (x_arrow / fig_w, tip / fig_h),
            transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7,
            lw=0.7, color=INK))
        fig.text(x_text / fig_w, (y0 + nrow * cell / 2) / fig_h, ARROW_LABEL,
                 ha="center", va="center", rotation=90, **F)

    sw = 0.11
    ly = BOTTOM_PAD + 0.03
    lx = xa
    for colour, lwidth, words in ((g.CORRECTED_EDGE, 1.7, "correction on"),
                                  (g.PLAIN_EDGE, 0.4, "correction off")):
        ax = fig.add_axes([lx / fig_w, ly / fig_h, sw / fig_w, sw / fig_h])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("white")
        for sp in ax.spines.values():
            sp.set_linewidth(lwidth); sp.set_color(colour)
        fig.text((lx + sw + 0.05) / fig_w, (ly + sw / 2) / fig_h, words,
                 ha="left", va="center", **F)
        lx += sw + 0.05 + text_width(fig, words, **F) + 0.22

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name}.png"
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(out.with_suffix(suffix), dpi=args.dpi)
    plt.close(fig)

    print(f"cell       {cell:.3f} in  ({cell * args.dpi:.0f} px per thumbnail, source is 512)")
    print(f"size       {fig_w:.2f} x {fig_h:.2f} in, seed {args.seed}")
    print(f"wrote      {out.with_suffix('.pdf')} (+ .png, .svg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
