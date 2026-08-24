#!/usr/bin/env python
"""The timing cliff survives holding the dose constant.

The nine-window sweep applied the correction at full strength in every window,
so the early window that composes also delivered less correction than the late
window that fails, because the correction's own size grows through the run.
Early and small were confounded.

This rescales the dose inside each window so all nine deliver the same total,
the amount the earliest window delivers at full strength, then reruns them. Only
the timing differs.

The result answers the objection twice over. The cliff keeps its shape, and the
windows at steps 5 to 15 and 10 to 20 still compose at lambda 0.52 and 0.37, so
a fraction-strength correction is not automatically too weak. It works early and
does nothing late at comparable strength.

Four cells per window, cat x dog only, so this is a rate over four runs and not a
population estimate. The eight-pair sweep is the population, drawn behind it.

    python scripts/timing_cliff_matched_dose.py
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import matplotlib
from poe_repair import paths
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = paths.resolve(paths.SAME_TOTAL_CORRECTION_DIFFERENT_WINDOW)
SCORES = ROOT / "matched_scores.json"
WINDOW_CURVES = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/"
                     "window/window_curves.json")
FIG_DIR = Path("paper/iclr/figures")
FIG_NAME = "compose-rate-as-the-window-moves-at-matched-total"

PAIR = "a_cat__x__a_dog"
PAIR_LABEL = "a cat and a dog"
INK = "#222222"
MATCHED_C = "#1f77b4"
UNMATCHED_C = "#d62728"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--name", default=FIG_NAME)
    args = ap.parse_args()

    rows = json.loads(SCORES.read_text())
    by = collections.defaultdict(list)
    for r in rows:
        by[tuple(r["window"])].append(r)
    wins = sorted(by)
    centres = np.array([(w[0] + w[1]) / 2 for w in wins], dtype=float)
    matched = np.array([sum(x["compose"] for x in by[w]) / len(by[w]) for w in wins])
    lam = np.array([by[w][0]["lambda_inside"] for w in wins])
    n_cells = len(by[wins[0]])

    # The same nine windows at full strength, this pair only, so the two lines
    # differ in dose and in nothing else.
    doc = json.loads(WINDOW_CURVES.read_text())
    unmatched = []
    for w in wins:
        cells = [r for r in doc["scores"]
                 if r["pair"] == PAIR and tuple(r["window"]) == w]
        unmatched.append(sum(c["compose"] for c in cells) / len(cells))
    unmatched = np.array(unmatched)

    print(f"{'window':>9} {'lambda':>7} {'matched':>8} {'full dose':>10}")
    for w, l, m, u in zip(wins, lam, matched, unmatched):
        print(f"{str(list(w)):>9} {l:>7.3f} {m:>8.2f} {u:>10.2f}")

    fig = plt.figure(figsize=(5.5, 3.05))
    ax = fig.add_axes([0.115, 0.33, 0.775, 0.50])

    ax.plot(centres, unmatched, color=UNMATCHED_C, lw=1.3, marker="s",
            markersize=3, linestyle=(0, (4, 2)),
            label="full strength in every window")
    ax.plot(centres, matched, color=MATCHED_C, lw=1.7, marker="o", markersize=3.6,
            label="every window delivers the same total")

    # The dose each matched window needed, on its own axis: it is the thing being
    # held constant, so a reader should be able to see what it cost.
    ax2 = ax.twinx()
    ax2.plot(centres, lam, color="0.65", lw=0.9, linestyle=":", zorder=0)
    ax2.set_ylabel("strength used", fontsize=7.5, family="serif", color="0.45")
    ax2.tick_params(labelsize=6.5, colors="0.45", length=2)
    ax2.set_ylim(0, 1.15)
    for lb in ax2.get_yticklabels():
        lb.set_family("serif")
    ax2.spines["top"].set_visible(False)

    ax.set_xlim(0, 49)
    ax.set_ylim(-0.04, 1.30)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "", "half", "", "all"], fontsize=7, family="serif")
    ax.set_xlabel("denoising steps the correction was applied during",
                  fontsize=8, family="serif", color=INK)
    ax.set_ylabel("composed", fontsize=8, family="serif", color=INK)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right", handlelength=2.0,
              borderaxespad=0.2)
    ax.set_title(f"{PAIR_LABEL}: the cliff is not a dose effect",
                 fontsize=9, family="serif", color=INK, loc="left", pad=5)
    ax.text(0.0, -0.30,
            f"{n_cells} cells per point. Dotted grey: the strength each matched\n"
            f"window needed. The windows at steps 5-15 and 10-20 still compose at\n"
            f"{lam[1]:.2f} and {lam[2]:.2f} strength, so a weak correction early beats "
            f"a strong one late.",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=6, family="serif", color="#777777", linespacing=1.6)
    ax.tick_params(labelsize=7, length=2.5)
    for lb in ax.get_xticklabels():
        lb.set_family("serif")
    for side in ("top",):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color("#444444")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name}.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)

    (args.out_dir / f"{args.name}.json").write_text(json.dumps({
        "pair": PAIR, "cells_per_point": n_cells,
        "windows": [list(w) for w in wins],
        "lambda_inside": lam.tolist(),
        "compose_rate_dose_matched": matched.tolist(),
        "compose_rate_full_strength": unmatched.tolist(),
        "delivered_total": by[wins[0]][0]["delivered_total"],
        "measure": "delivered total = sum over the window of lambda_t * size_t; "
                   "compose = detector counts >= 2 animals",
        "caveat": "four cells per point, one pair. The cliff survives matching, "
                  "and the windows at 5-15 and 10-20 compose at fraction strength, "
                  "so the late tail is not explained by the smaller lambda alone.",
    }, indent=2))
    print(f"wrote {out} and {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
