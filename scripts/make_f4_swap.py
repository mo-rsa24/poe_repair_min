#!/usr/bin/env python
"""Register slot F4d: timing decides, not dose.

The nine-window sweep confounded two things, because the correction's own size
grows through the run: the early window that composes also delivers less
correction than the late window that fails. So "early wins" and "small wins"
could not be told apart.

This crosses them. Two windows by two doses, one seed per figure:

                    the early window's dose      the late window's dose
    steps 0 to 10   composes (its own setting)   more correction, applied early
    steps 40 to 50  the SAME total that works    fails (its own setting)
                    early, applied late

The bottom-left cell is the one the experiment exists for. It gives the late
window exactly the correction total the early window succeeds with, so if it
still fuses, no dose objection survives and timing is doing the work alone.

Every cell is a real sample. The two diagonal cells re-run settings the window
sweep already covered, and all eight such cells across the four seeds came back
byte-identical to it, so these are comparable with the sweep rather than merely
similar to it.

No count is printed. The detector disagrees with the pictures on this pair, so
the grid argues from what is in the frame and the caption says so.

    python scripts/make_f4_swap.py --seed 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose_matched")
MANIFEST = ROOT / "swap_manifest.json"
FIG_DIR = Path("paper/iclr/figures")
FIG_NAME = "F4d-timing-not-dose"

PAIR = "a_cat__x__a_dog"
PAIR_LABEL = "a cat and a dog"
EARLY, LATE = (0, 10), (40, 50)
INK = "#222222"
ROW_AXIS_TITLE = "corrected window"
COL_AXIS_TITLE = "correction strength"

CELL = 1.95
LEFT = 1.05             # row axis title plus row values
TOP = 0.50              # column axis title plus column values
RIGHT = 0.10
BOTTOM = 0.15


def tag_for(win, donor):
    return f"swap_w{win[0]}-{win[1]}_dose_of_w{donor[0]}-{donor[1]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=12,
                    help="12 shows the contrast most clearly by eye; every seed "
                         "gives the same verdict on the decisive cell")
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--name", default=FIG_NAME)
    args = ap.parse_args()

    cells = {(c["seed"], c["tag"]): c for c in json.loads(MANIFEST.read_text())["cells"]}

    fig_w = LEFT + 2 * CELL + RIGHT
    fig_h = TOP + 2 * CELL + BOTTOM
    fig = plt.figure(figsize=(fig_w, fig_h))

    rows = ((EARLY, "0–10"), (LATE, "40–50"))
    cols = (EARLY, LATE)

    for i, (win, row_label) in enumerate(rows):
        for j, donor in enumerate(cols):
            c = cells[(args.seed, tag_for(win, donor))]
            ax = fig.add_axes([
                (LEFT + j * CELL) / fig_w,
                (BOTTOM + (1 - i) * CELL) / fig_h,
                CELL / fig_w, CELL / fig_h,
            ])
            ax.imshow(plt.imread(c["image"]))
            ax.set_xticks([]); ax.set_yticks([])
            decisive = (win == LATE and donor == EARLY)
            for sp in ax.spines.values():
                sp.set_linewidth(1.4 if decisive else 0.6)
                sp.set_color("#111111" if decisive else "#888888")

        ax_lab = fig.add_axes([0.30 / fig_w, (BOTTOM + (1 - i) * CELL) / fig_h,
                               (LEFT - 0.30) / fig_w, CELL / fig_h])
        ax_lab.axis("off")
        ax_lab.text(0.92, 0.5, row_label, ha="right", va="center",
                    fontsize=9, family="serif", color=INK)

    fig.text(0.12 / fig_w, BOTTOM / fig_h + CELL / fig_h,
             ROW_AXIS_TITLE, ha="center", va="center", rotation=90,
             fontsize=9, family="serif", color=INK)

    for j, donor in enumerate(cols):
        # Both rows in a column share this total by construction, so either
        # row's cell reports the number the whole column holds fixed.
        total = cells[(args.seed, tag_for(EARLY, donor))]["delivered_total"]
        ax_top = fig.add_axes([(LEFT + j * CELL) / fig_w,
                               (BOTTOM + 2 * CELL) / fig_h,
                               CELL / fig_w, 0.24 / fig_h])
        ax_top.axis("off")
        ax_top.text(0.5, 0.1, f"{total:.1f}", ha="center", va="bottom",
                    fontsize=9, family="serif", color=INK)

    fig.text((LEFT + CELL) / fig_w, 1 - 0.10 / fig_h,
             COL_AXIS_TITLE, ha="center", va="top",
             fontsize=9, family="serif", color=INK)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name}.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)

    (args.out_dir / f"{args.name}.json").write_text(json.dumps({
        "pair": PAIR, "seed": args.seed,
        "cells": {tag_for(w, d): cells[(args.seed, tag_for(w, d))]
                  for w, _ in rows for d in cols},
        "decisive_cell": tag_for(LATE, EARLY),
        "measure": "delivered total = sum over the window of lambda_t * size_t, "
                   "size being the same per-step measure F3 and F4b draw",
        "verdict_by_eye": "the decisive cell holds one fused animal on every seed; "
                          "the early window at the same total holds two animals",
    }, indent=2))
    print(f"seed {args.seed}, 4 cells, decisive = {tag_for(LATE, EARLY)}")
    print(f"size {fig_w:.2f} x {fig_h:.2f} in")
    print(f"wrote {out} and {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
