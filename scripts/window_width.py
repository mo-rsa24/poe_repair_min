#!/usr/bin/env python
"""Pick the sliding-window width for the timing sweep, from the cache.

The timing sweep slides a fixed-width window across the 50 denoising steps. The
width has to be chosen once, before any window runs, or the choice can be tuned
after seeing the answer.

The obvious rule does not work here, and why it fails is itself a result. The
rule would be: make the window as wide as the narrowest band of steps carrying
half the correction's total size (sum of ‖r_t‖ over steps). Run it and the
answer is 25 steps, half the trajectory, because ‖r_t‖ is nearly flat: each
tenth of the run carries 15-22% of the total and the largest step is only 1.8x
the smallest. A 25-wide window has six overlapping placements and every one of
them contains step 16, so the compose-rate curve would come out flat whatever
the truth is. Size cannot say when the correction is needed, because size barely
changes.

So the width is set by what the experiment has to be able to see, stated here
before the sweep runs, and lives in
poe_repair/experiments/interaction_term/window_grid.py. Width 10 at stride 5 is
small enough that sliding it resolves position: nine placements, each covering a
fifth of the run, with the fork step (16, from fork_curve.json) inside only two
of them. If timing matters, those two win and the rest do not. If nothing peaks,
timing does not matter, which is a finding rather than a failure.

This script reports both numbers: the energy band that would have been chosen,
and the flatness that disqualifies it. Writes
outputs/interaction_term/cache_analyses/window_width.json plus a figure of the
median ‖r_t‖-vs-step curve with both widths drawn on it.

Usage:
    python scripts/window_width.py
    python scripts/window_width.py --pairs a_cat__x__a_dog,a_frog__x__a_toad
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.experiments.interaction_term import window_grid as wg
from poe_repair.experiments.interaction_term.cache import CACHE_ROOT, load_cell

# The fraction of total correction size the window must contain. Fixed here,
# deliberately not a command-line flag: a threshold that can be passed in is a
# threshold that can be changed after seeing the curve.
ENERGY_FRACTION = 0.5
ROUND_TO = 5

OUT_DIR = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses")

# The eight pairs the dose sweep used, so the width is chosen on the same pairs
# the timing sweep will run on.
SWEEP_PAIRS = (
    "a_leopard__x__a_jaguar", "a_frog__x__a_toad", "an_eagle__x__a_hawk",
    "a_seal__x__a_walrus", "a_goose__x__a_swan", "a_cow__x__a_buffalo",
    "a_cat__x__a_dog", "an_elephant__x__a_penguin",
)


def narrowest_band(norms: np.ndarray, fraction: float) -> tuple[int, int, int]:
    """Narrowest contiguous [start, end) carrying `fraction` of the total.

    Returns (start, end, width). Linear scan over widths, so it reports the
    smallest width for which SOME placement clears the threshold.
    """
    total = float(norms.sum())
    target = fraction * total
    n = len(norms)
    csum = np.concatenate([[0.0], np.cumsum(norms)])
    for width in range(1, n + 1):
        sums = csum[width:] - csum[:-width]
        if sums.max() >= target:
            start = int(sums.argmax())
            return start, start + width, width
    return 0, n, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", help="comma-separated slugs (default: the 8 sweep pairs)")
    ap.add_argument("--seeds", default="9,10", help="comma-separated seeds")
    ap.add_argument("--min-steps", type=int, default=40,
                    help="skip cells shorter than this; short smoke runs must "
                         "not be pooled with 50-step cells")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    pairs = args.pairs.split(",") if args.pairs else list(SWEEP_PAIRS)
    seeds = [int(s) for s in args.seeds.split(",")]

    curves: list[np.ndarray] = []
    rows: list[dict] = []
    for pair in pairs:
        for seed in seeds:
            try:
                cell = load_cell(pair, seed, root=CACHE_ROOT)
            except FileNotFoundError as e:
                print(f"  skip {pair} seed {seed}: {e}", file=sys.stderr)
                continue
            if cell.n_steps < args.min_steps:
                print(f"  skip {pair} seed {seed}: {cell.n_steps} steps "
                      f"(under --min-steps {args.min_steps})", file=sys.stderr)
                continue
            with torch.no_grad():
                norms = cell.r_t_norm().numpy().astype(float)
            start, end, width = narrowest_band(norms, ENERGY_FRACTION)
            curves.append(norms)
            rows.append({"pair": pair, "seed": seed, "n_steps": cell.n_steps,
                         "band": [start, end], "width": width,
                         "peak_step": int(norms.argmax())})
            print(f"  {pair:<28} seed {seed}: {cell.n_steps} steps, "
                  f"half the correction sits in steps {start}-{end} "
                  f"(width {width}), peak at step {norms.argmax()}")

    if not curves:
        print("no cells loaded; nothing to choose a width from", file=sys.stderr)
        return 2

    lengths = {len(c) for c in curves}
    if len(lengths) > 1:
        print(f"cells disagree on step count: {sorted(lengths)}", file=sys.stderr)
        return 2
    n_steps = lengths.pop()

    widths = np.array([r["width"] for r in rows])
    median_width = float(np.median(widths))
    energy_width = int(np.ceil(median_width / ROUND_TO) * ROUND_TO)

    median_curve = np.median(np.stack(curves), axis=0)
    m_start, m_end, m_width = narrowest_band(median_curve, ENERGY_FRACTION)

    tenths = [median_curve[i:i + n_steps // 5].sum() / median_curve.sum()
              for i in range(0, n_steps, n_steps // 5)]
    flatness = float(median_curve.max() / median_curve.min())

    print(f"\n{len(rows)} cells, {n_steps} steps each")
    print(f"per-cell width carrying {ENERGY_FRACTION:.0%} of the correction: "
          f"median {median_width:.1f}, range {widths.min()}-{widths.max()}")
    print(f"on the median curve: steps {m_start}-{m_end} (width {m_width})")

    print(f"\nis the size curve concentrated enough to pick a width from?")
    for i, share in enumerate(tenths):
        lo = i * (n_steps // 5)
        print(f"  steps {lo:2d}-{lo + n_steps // 5 - 1:2d}: {share:.1%} of the total")
    print(f"  largest step / smallest step: {flatness:.2f}x")
    print(f"  NO. The energy rule gives width {energy_width}, which is "
          f"{energy_width / n_steps:.0%} of the run. Every placement of a window "
          f"that wide contains the fork step, so the curve would be flat "
          f"whatever the truth is.")

    windows = wg.windows()
    with_fork = [w for w in windows if wg.contains_fork(w)]
    print(f"\nWIDTH IN USE: {wg.WIDTH} steps at stride {wg.STRIDE}, from "
          f"window_grid.py, chosen to resolve position rather than to hold "
          f"energy.")
    print(f"  {len(windows)} placements: "
          f"{', '.join(f'{a}-{b}' for a, b in windows)}")
    print(f"  fork step {wg.FORK_STEP} is inside {len(with_fork)} of them: "
          f"{', '.join(f'{a}-{b}' for a, b in with_fork)}")
    print(f"  grid: {len(windows)} windows x {len(wg.PAIRS)} pairs x "
          f"{len(wg.SEEDS)} seeds = "
          f"{len(windows) * len(wg.PAIRS) * len(wg.SEEDS)} cells")

    result = {
        "energy_fraction": ENERGY_FRACTION,
        "round_to": ROUND_TO,
        "n_cells": len(rows),
        "n_steps": n_steps,
        "per_cell_width_median": median_width,
        "per_cell_width_min": int(widths.min()),
        "per_cell_width_max": int(widths.max()),
        "median_curve_band": [m_start, m_end],
        "median_curve_width": m_width,
        "energy_rule_width": energy_width,
        "energy_rule_rejected": True,
        "flatness_max_over_min": flatness,
        "fifth_shares": [float(x) for x in tenths],
        "width_in_use": wg.WIDTH,
        "stride_in_use": wg.STRIDE,
        "fork_step": wg.FORK_STEP,
        "windows": [list(w) for w in windows],
        "windows_containing_fork": [list(w) for w in with_fork],
        "median_curve": [float(x) for x in median_curve],
        "cells": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "window_width.json").write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out_dir / 'window_width.json'}")

    if not args.no_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for c in curves:
            ax.plot(c, color="0.8", lw=0.7)
        ax.plot(median_curve, color="tab:blue", lw=2.2, label="median over cells")
        ax.axvspan(m_start, m_end, color="tab:red", alpha=0.12,
                   label=f"energy rule: width {m_width}, rejected "
                         f"(covers the fork at every placement)")
        # The width actually swept, drawn at one placement so its size is
        # readable against the rejected one.
        ax.axvspan(wg.FORK_STEP - wg.WIDTH // 2, wg.FORK_STEP + wg.WIDTH // 2,
                   color="tab:green", alpha=0.22,
                   label=f"width in use: {wg.WIDTH}, slid at stride {wg.STRIDE}")
        ax.axvline(wg.FORK_STEP, color="k", ls="--", lw=1.2,
                   label=f"fork step {wg.FORK_STEP}")
        ax.set_xlabel("denoising step")
        ax.set_ylabel(r"$\|r_t\|$")
        ax.set_xlim(0, n_steps - 1)
        ax.set_title(f"The correction's size barely changes over the run "
                     f"({len(rows)} cells, {flatness:.2f}x max/min)\n"
                     f"so the sweep width is set by resolution, not by size")
        ax.legend(frameon=True, framealpha=0.9, edgecolor="none",
                  fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_dir / "window_width.png", dpi=150)
        print(f"figure: {args.out_dir / 'window_width.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
