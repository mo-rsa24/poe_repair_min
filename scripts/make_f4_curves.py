#!/usr/bin/env python
"""Register slot F4.2: where the correction is large is not where it matters.

Two panels stacked on one denoising-step axis, for the pair the paper follows.
Read together they dissociate size from effect, which is the point: the
correction is roughly 2.7x larger over steps 20 to 40 than over steps 0 to 14
(0.48 against 1.33, averaged over the four seeds, every seed peaking between
steps 21 and 37), and the only window that composes is 0 to 10, where it is
smallest.

So a reader may not infer timing from size, and the paper may not argue that the
correction matters where it is biggest. It matters where the image is still
noise.

(a) How big the correction is at each step, one line per seed. Read straight from
    the cached trajectories, using the same measure as F3, imported rather than
    repeated so the two figures cannot disagree about what size means.

(b) How often the correction works, against the ten-step window it was applied
    during. One line per seed, each a real 0-or-1 outcome at every window, not a
    rate: n=1 per seed per window, so nothing here is averaged.

    Seed 12's line is drawn dashed with hollow markers from window (20,30)
    onward. The scorer reads compose=0 there (n_instances=1), but the actual
    frames (F4a's grid, seed-12 row) show a cat beside a dog through every
    window to (40,50). That disagreement is why F4a suppresses its count chips
    by default; here the wrong reading is drawn, not hidden, and flagged instead.

    python scripts/make_f4_curves.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from snr_collapse import curve_for  # noqa: E402  the single definition of the measure
from make_f3 import smooth          # noqa: E402  the same smoothing F3 draws with

WINDOW_CURVES = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/"
                     "window/window_curves.json")
FIG_DIR = Path("paper/iclr/figures")
FIG_NAME = "F4b-size-is-not-timing"

PAIR = "a_cat__x__a_dog"
SEEDS = (9, 10, 11, 12)
SEED_COLOURS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd")

INK = "#222222"

FIG_W = 5.50
PANEL_H = 1.55
GAP = 0.42
LEFT = 0.78
RIGHT = 0.14
TOP = 0.16
BOTTOM = 0.52
FIG_H = BOTTOM + 2 * PANEL_H + GAP + TOP

# The scorer disagrees with its own images here (see module docstring): drawn
# dashed and hollow rather than dropped, so the wrong reading is visible, not
# silently smoothed away.
SUSPECT_SEED = 12
SUSPECT_FROM_WINDOW = (20, 30)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--name", default=FIG_NAME)
    args = ap.parse_args()

    sizes = {}
    for seed in SEEDS:
        _, _, _, y = curve_for(PAIR, seed)
        sizes[seed] = np.asarray(y, dtype=float)
        print(f"size curve  seed {seed:<3d} {len(y)} steps")

    doc = json.loads(WINDOW_CURVES.read_text())
    centres = np.asarray(doc["window_centres"], dtype=float)

    # Panel a is this pair's four seeds, so panel b draws this pair too, one line
    # per seed: each point is a single real outcome (compose or not), not an
    # average, so nothing here is smoothed into looking like a rate.
    windows = sorted({tuple(r["window"]) for r in doc["scores"]})
    seed_compose = {}
    for seed in SEEDS:
        cells = {tuple(r["window"]): int(r["compose"]) for r in doc["scores"]
                 if r["pair"] == PAIR and r["seed"] == seed}
        missing = [w for w in windows if w not in cells]
        if missing:
            raise SystemExit(f"seed {seed} missing windows {missing}")
        seed_compose[seed] = np.asarray([cells[w] for w in windows], dtype=float)
        print(f"outcome     seed {seed:<3d} " +
              " ".join(f"{w[0]}-{w[1]}:{int(v)}"
                       for w, v in zip(windows, seed_compose[seed])))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    aw = (FIG_W - LEFT - RIGHT) / FIG_W
    ax_a = fig.add_axes([LEFT / FIG_W, (BOTTOM + PANEL_H + GAP) / FIG_H,
                         aw, PANEL_H / FIG_H])
    ax_b = fig.add_axes([LEFT / FIG_W, BOTTOM / FIG_H, aw, PANEL_H / FIG_H])

    # Smoothed exactly as F3 smooths it, imported rather than re-written. The same
    # measure drawn two ways in two figures makes a reader wonder which is real.
    for seed, colour in zip(SEEDS, SEED_COLOURS):
        y = smooth(sizes[seed])
        ax_a.plot(np.arange(len(y)), y, color=colour, lw=1.2, label=f"seed {seed}")
    ax_a.set_ylabel("correction size", fontsize=8, family="serif", color=INK)
    ax_a.legend(fontsize=6.5, frameon=False, ncol=4, loc="lower right",
                handlelength=1.4, columnspacing=1.1, borderaxespad=0.2)
    ax_a.set_title("a. how big the correction is at each step",
                   fontsize=8.5, family="serif", color=INK, loc="left", pad=4)

    # Step against window centre are the same axis only because every window is
    # the same width. Guarded above by the equal-cell-count check and here by
    # drawing the width, so a reader sees what a point covers.
    suspect_idx = windows.index(SUSPECT_FROM_WINDOW)
    for seed, colour in zip(SEEDS, SEED_COLOURS):
        y = seed_compose[seed]
        if seed == SUSPECT_SEED:
            # Solid and filled where the scorer's reading is trusted; dashed
            # and hollow where it disagrees with its own images (see module
            # docstring). Drawn, not dropped: hiding the point would claim a
            # cleaner result than the instrument actually delivered.
            ax_b.plot(centres[:suspect_idx + 1], y[:suspect_idx + 1],
                      color=colour, lw=1.6, marker="o", markersize=3.4)
            ax_b.plot(centres[suspect_idx:], y[suspect_idx:],
                      color=colour, lw=1.2, ls="--", marker="o", markersize=3.4,
                      markerfacecolor="white", markeredgecolor=colour)
        else:
            ax_b.plot(centres, y, color=colour, lw=1.6, marker="o", markersize=3.4)
    ax_b.text(centres[suspect_idx], 0.06,
              "seed 12 past here: scorer says 1, images show 2",
              transform=ax_b.transData, ha="left", va="bottom",
              fontsize=5.5, family="serif", color="#777777")
    ax_b.set_ylabel("outcome", fontsize=8, family="serif", color=INK)
    ax_b.set_ylim(-0.08, 1.08)
    ax_b.set_yticks([0.0, 1.0])
    ax_b.set_yticklabels(["blend", "compose"], fontsize=7, family="serif")
    ax_b.set_title(f"b. how often it works, {doc['width']}-step window, one seed per line",
                   fontsize=8.5, family="serif", color=INK, loc="left", pad=4)
    ax_b.set_xlabel("denoising step", fontsize=8, family="serif", color=INK)

    for ax in (ax_a, ax_b):
        ax.set_xlim(0, 49)
        ax.tick_params(labelsize=7, length=2.5)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_family("serif")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.6)
            ax.spines[side].set_color("#444444")

    fig.text(LEFT / FIG_W, 0.10 / FIG_H, "noise", ha="left", va="bottom",
             fontsize=7, family="serif", color="#666666")
    fig.text(1 - RIGHT / FIG_W, 0.10 / FIG_H, "image", ha="right", va="bottom",
             fontsize=7, family="serif", color="#666666")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name}.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"size        {FIG_W:.2f} x {FIG_H:.2f} in")
    print(f"wrote       {out} and {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
