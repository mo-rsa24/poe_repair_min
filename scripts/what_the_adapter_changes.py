#!/usr/bin/env python
"""The adapter changes what a word paints.

For each attention map the probe measures two things: how much the adapter moved
where the model looks (weight pattern) and how much it moved what gets painted
there (content pattern). The number plotted is their ratio, so 1.0 means the
adapter moved both equally and above 1.0 means it moved content more.

One point per pair, eight seeds behind it. The data nests three deep, 384 rows
inside 64 cells inside 8 pairs, and rows within a cell are the same image read at
three steps and two tokens. They are not independent, so plotting rows as points
would inflate the sample 48-fold and turn a claim about eight pairs into an
apparent claim about hundreds of observations. The decision is recorded in the
hypothesis-01 review.

**What this figure may not say.** The ratio is above 1 on every pair measured,
including the control pair, so it describes what the adapter does to any pair it
touches. It is not evidence that content-change is why composition succeeds.
That cap is in the register and belongs in the caption.

The pre-registered bar was computed over rows (median 1.52, 97% above one) and is
cited as the bar, not as this figure's statistic.

    python scripts/what_the_adapter_changes.py
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as stats
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VERDICT = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/"
               "reprobe/verdict.json")
FIG_DIR = Path("paper/iclr/figures")
FIG_NAME = "content-change-relative-to-attention-change-under-lora"

RUNNING = "a_cat__x__a_dog"
DISSIMILAR = "an_elephant__x__a_penguin"
INK = "#222222"
PAIR_C = "#1f77b4"
SEED_C = "#a9c6e0"
HILITE = "#d62728"


def pretty(slug: str) -> str:
    a, b = slug.split("__x__")
    strip = lambda s: s.replace("_", " ").removeprefix("a ").removeprefix("an ")
    return f"{strip(a)} x {strip(b)}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--name", default=FIG_NAME)
    args = ap.parse_args()

    doc = json.loads(VERDICT.read_text())
    rows = doc["rows"]

    # rows -> one value per (pair, seed) cell -> one median per pair.
    by_cell = collections.defaultdict(list)
    for r in rows:
        by_cell[(r["pair"], r["seed"])].append(float(r["ratio"]))
    cell = {k: stats.median(v) for k, v in by_cell.items()}
    by_pair = collections.defaultdict(list)
    for (pair, _), v in cell.items():
        by_pair[pair].append(v)
    pair_median = {p: stats.median(v) for p, v in by_pair.items()}
    order = sorted(pair_median, key=lambda p: pair_median[p])

    print(f"{'pair':30s} {'median':>7} {'seeds':>6}  seed range")
    for p in reversed(order):
        v = sorted(by_pair[p])
        print(f"{pretty(p):30s} {pair_median[p]:>7.3f} {len(v):>6d}  "
              f"{v[0]:.2f} to {v[-1]:.2f}")
    overall = stats.median(list(pair_median.values()))
    print(f"\nmedian over the {len(order)} pairs: {overall:.3f}")
    print(f"pre-registered bar, over rows: median {doc['median_ratio']:.3f}, "
          f"{doc['fraction_above_one']:.1%} above one, {doc['n_rows']} rows")

    fig = plt.figure(figsize=(5.5, 3.35))
    ax = fig.add_axes([0.30, 0.325, 0.665, 0.545])

    ax.axvline(1.0, color="#555555", lw=0.8, linestyle=(0, (4, 3)), zorder=1)
    for i, p in enumerate(order):
        ax.plot(sorted(by_pair[p]), [i] * len(by_pair[p]), linestyle="none",
                marker="o", markersize=3.2, color=SEED_C, zorder=2)
        c = HILITE if p in (RUNNING, DISSIMILAR) else PAIR_C
        ax.plot([pair_median[p]], [i], linestyle="none", marker="D",
                markersize=5.0, color=c, zorder=3)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([pretty(p) for p in order], fontsize=7, family="serif")
    for lab, p in zip(ax.get_yticklabels(), order):
        if p in (RUNNING, DISSIMILAR):
            lab.set_color(HILITE)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlim(0.85, 2.6)
    ax.set_xlabel("content change relative to attention change",
                  fontsize=8, family="serif", color=INK)
    ax.tick_params(labelsize=7, length=2.5)
    for lb in ax.get_xticklabels():
        lb.set_family("serif")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color("#444444")

    ax.text(1.0, len(order) - 0.42, "  moves both equally", fontsize=6.5,
            family="serif", color="#666666", ha="left", va="center")
    ax.text(0.0, -0.30,
            f"Diamond: one pair, median of its {len(by_pair[order[0]])} seeds, shown behind "
            f"in pale blue. n = {len(order)} pairs.\nRed: the running example and the "
            f"dissimilar pair. Every pair sits above 1, including pairs the fix\ndoes not "
            f"help, so this is what the adapter does to any pair it touches.",
            transform=ax.transAxes, ha="left", va="top", fontsize=6,
            family="serif", color="#777777", linespacing=1.7)
    ax.set_title("what the adapter changes inside the model",
                 fontsize=9, family="serif", color=INK, loc="left", pad=6)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name}.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)

    (args.out_dir / f"{args.name}.json").write_text(json.dumps({
        "entity": "one point per pair, median over that pair's 8 seeds; each seed "
                  "is itself the median of its 6 rows (3 steps x 2 tokens)",
        "n_pairs": len(order), "n_cells": doc["n_cells"], "n_rows": doc["n_rows"],
        "pair_median": {p: pair_median[p] for p in order},
        "seed_values": {p: sorted(by_pair[p]) for p in order},
        "median_over_pairs": overall,
        "preregistered_bar": doc["bar"],
        "bar_as_measured_over_rows": {"median": doc["median_ratio"],
                                      "fraction_above_one": doc["fraction_above_one"]},
        "measure": doc["measure"],
        "caption_cap": "the ratio exceeds 1 on every pair measured, including the "
                       "control pair, so it describes what the adapter does to any "
                       "pair it touches and is not evidence for why the fix works",
        "owed": "the transfer table and the replication strip, both waiting on F8's "
                "training run",
    }, indent=2))
    print(f"wrote {out} and {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
