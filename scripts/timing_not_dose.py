"""Where the correction lands, and the control that rules out how much of it.

Panel (a) slides a ten-step correction window across the run, four seeds of the
same cat-and-dog pair. Composition survives only while the window sits at the
start.

That sweep confounds two things, because the correction's own size grows through
the run: the early window that composes also delivers less correction than the
late window that fails. Panel (b) crosses them. Its columns hold the delivered
total fixed, the sum over the window of lambda_t times the correction's size, and
its rows move the window. The lower-left cell is the one the experiment exists
for: it hands the late window exactly the total the early window succeeds with,
and it still fuses.

The number over a column of (b) is that delivered total, not lambda. Lambda is
different in every cell and is printed inside it. The decisive cell runs at 0.34,
a third of normal, and still delivers 3.6, because the correction is roughly
three times larger that late in the run.

Panel (b) carries no detector verdict. The detector disagrees with the pictures
on this pair, so that panel argues from what is in the frame and its caption says
so. Panel (a) keeps the frames, as the window map already does.

The figure is drawn at the ICLR text width, so it is placed with
\\includegraphics[width=\\textwidth] and nothing else.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from poe_repair import paths

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import later_start_grid as g

FIG_DIR = Path("paper/iclr/figures")
FIG_NAME = "timing-not-dose"

WINDOW = paths.resolve(paths.WINDOW)
SWAP = paths.resolve(paths.SAME_TOTAL_CORRECTION_DIFFERENT_WINDOW) / "swap_manifest.json"
PAIR = "a_cat__x__a_dog"
SEEDS = (9, 10, 11, 12)
STARTS = list(range(0, 45, 5))
EARLY, LATE = (0, 10), (40, 50)

TEXT_WIDTH = 5.50
GAP = 0.16
ROT_TITLE = 0.15      # the rotated "corrected window", tight against (b)
PAD = 0.08            # clearance between a label and what it labels
RIGHT_PAD = 0.02
NUM_BAND = 0.15
TITLE_BAND = 0.17
CAPTION_BAND = 0.20
LEGEND_BAND = 0.18
BOTTOM_PAD = 0.05
FONT, TICK = 8, 6.5   # titles and captions; labels sitting against a grid
INK = g.INK
DECISIVE = "#111111"


def measure(s, **kw):
    """Printed width of a string in inches, before the real figure exists."""
    f = plt.figure(figsize=(1, 1))
    f.canvas.draw()
    t = f.text(0, 0, s, **kw)
    w = t.get_window_extent(renderer=f.canvas.get_renderer()).width / f.dpi
    plt.close(f)
    return w


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
    ap.add_argument("--swap-seed", type=int, default=12)
    ap.add_argument("--dpi", type=int, default=1130)
    args = ap.parse_args()

    verdict = {(s["seed"], s["window"][0]): s["compose"]
               for s in json.loads((WINDOW / "window_curves.json").read_text())["scores"]
               if s["pair"] == PAIR}
    swap = {(c["seed"], c["tag"]): c
            for c in json.loads(SWAP.read_text())["cells"]}
    tag = lambda w, d: f"swap_w{w[0]}-{w[1]}_dose_of_w{d[0]}-{d[1]}"

    nL, nR = len(STARTS), 2
    M = dict(fontsize=TICK, family="serif")
    left_gut = measure("seed 10", **M) + PAD
    right_gut = measure("40–50", **M) + PAD
    cL = (TEXT_WIDTH - left_gut - GAP - ROT_TITLE - right_gut - RIGHT_PAD) / (nL + 2 * nR)
    cR = 2 * cL
    fig_w = TEXT_WIDTH
    fig_h = (BOTTOM_PAD + LEGEND_BAND + CAPTION_BAND + len(SEEDS) * cL
             + NUM_BAND + TITLE_BAND)

    xa = left_gut
    xb = xa + nL * cL + GAP + ROT_TITLE + right_gut
    y0 = BOTTOM_PAD + LEGEND_BAND + CAPTION_BAND
    top = y0 + len(SEEDS) * cL

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.canvas.draw()
    F = dict(fontsize=FONT, family="serif", color=INK)
    Ft = dict(fontsize=TICK, family="serif", color=INK)

    # panel (a): four seeds by nine window positions
    for i, seed in enumerate(SEEDS):
        y = y0 + (len(SEEDS) - 1 - i) * cL
        for j, s in enumerate(STARTS):
            ax = fig.add_axes([(xa + j * cL) / fig_w, y / fig_h, cL / fig_w, cL / fig_h])
            ax.imshow(plt.imread(g.final_png(seed, s, s + 10)))
            ax.set_xticks([]); ax.set_yticks([])
            on = verdict[(seed, s)]
            for sp in ax.spines.values():
                sp.set_linewidth(1.5 if on else 0.4)
                sp.set_color(g.CORRECTED_EDGE if on else g.PLAIN_EDGE)
        fig.text((xa - PAD / 2) / fig_w, (y + cL / 2) / fig_h, f"seed {seed}",
                 ha="right", va="center", **Ft)

    # panel (a) carries no axis title; the caption says what the ranges are
    for j, s in enumerate(STARTS):
        fig.text((xa + (j + 0.5) * cL) / fig_w, (top + 0.03) / fig_h,
                 f"{s}–{s + 10}", ha="center", va="bottom", **Ft)

    # panel (b): two windows by two delivered totals, lambda inside each cell
    totals = [swap[(args.swap_seed, tag(EARLY, d))]["delivered_total"] for d in (EARLY, LATE)]
    for i, win in enumerate((EARLY, LATE)):
        y = y0 + (1 - i) * cR
        for j, donor in enumerate((EARLY, LATE)):
            c = swap[(args.swap_seed, tag(win, donor))]
            ax = fig.add_axes([(xb + j * cR) / fig_w, y / fig_h, cR / fig_w, cR / fig_h])
            ax.imshow(plt.imread(c["image"]))
            ax.set_xticks([]); ax.set_yticks([])
            decisive = win == LATE and donor == EARLY
            for sp in ax.spines.values():
                sp.set_linewidth(1.5 if decisive else 0.4)
                sp.set_color(DECISIVE if decisive else g.PLAIN_EDGE)
            ax.text(0.5, 0.035, f"$\\lambda$ = {c['lambda_inside']:.2f}",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=TICK, family="serif", color="white",
                    path_effects=[pe.withStroke(linewidth=1.6, foreground="#111111")])
        fig.text((xb - PAD / 2) / fig_w, (y + cR / 2) / fig_h, f"{win[0]}–{win[1]}",
                 ha="right", va="center", **Ft)

    for j, t in enumerate(totals):
        fig.text((xb + (j + 0.5) * cR) / fig_w, (top + 0.03) / fig_h,
                 f"{t:.1f}", ha="center", va="bottom", **Ft)
    fig.text((xb + nR * cR / 2) / fig_w, (top + NUM_BAND + 0.02) / fig_h,
             "correction delivered", ha="center", va="bottom", **F)
    fig.text((xb - right_gut - ROT_TITLE / 2) / fig_w, (y0 + nR * cR / 2) / fig_h,
             "corrected window", ha="center", va="center", rotation=90, **Ft)

    # panel captions, letter bold
    for x0, span, letter, phrase in ((xa, nL * cL, "(a)", " where the correction lands"),
                                     (xb, nR * cR, "(b)", " same total, applied late")):
        wl = text_width(fig, letter, **F, weight="bold")
        wp = text_width(fig, phrase, **F)
        left = x0 + (span - (wl + wp)) / 2
        left = min(max(left, 0.02), fig_w - 0.02 - (wl + wp))
        cy = (BOTTOM_PAD + LEGEND_BAND + 0.05) / fig_h
        fig.text(left / fig_w, cy, letter, ha="left", va="bottom", weight="bold", **F)
        fig.text((left + wl) / fig_w, cy, phrase, ha="left", va="bottom", **F)

    # legend
    sw, ly, lx = 0.10, BOTTOM_PAD + 0.02, xa
    for colour, lw, words in ((g.CORRECTED_EDGE, 1.5, "detector scored two animals, in (a)"),
                              (DECISIVE, 1.5, "the cell the control turns on, in (b)")):
        ax = fig.add_axes([lx / fig_w, ly / fig_h, sw / fig_w, sw / fig_h])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("white")
        for sp in ax.spines.values():
            sp.set_linewidth(lw); sp.set_color(colour)
        fig.text((lx + sw + 0.04) / fig_w, (ly + sw / 2) / fig_h, words,
                 ha="left", va="center", **Ft)
        lx += sw + 0.04 + text_width(fig, words, **Ft) + 0.20

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name}.png"
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(out.with_suffix(suffix), dpi=args.dpi)
    plt.close(fig)
    print(f"(a) cell {cL:.3f} in   (b) cell {cR:.3f} in")
    print(f"size     {fig_w:.2f} x {fig_h:.2f} in")
    print(f"totals   {totals[0]:.2f} and {totals[1]:.2f}, swap seed {args.swap_seed}")
    print(f"wrote    {out.with_suffix('.pdf')} (+ .png, .svg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
