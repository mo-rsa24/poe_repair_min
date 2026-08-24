#!/usr/bin/env python
"""Do the ‖r_t‖ curves collapse onto one shape in log-SNR?

The scope's universality claim is that the interaction term is not a per-pair
quirk: rescale each pair's correction-size curve and plot it against log-SNR
rather than step index, and the curves should lie on top of each other. This
script measures how tightly they do.

Headline number is the collapse spread: the median across log-SNR bins of the
interquartile range of the normalised curves, as a percentage. Small means the
curves agree in shape; large means each pair does its own thing.

Cache-only, no GPU.

Usage:
    python scripts/snr_collapse.py --pair a_cat__x__a_dog          # one pair, all seeds
    python scripts/snr_collapse.py --all --max-pairs 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from poe_repair import paths

# Running as `python scripts/foo.py` puts scripts/ on sys.path, not the repo
# root, so the package would not import. The plan's engagement instructions use
# that form, so make it work rather than requiring `-m` or PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.experiments.interaction_term.cache import CACHE_ROOT, load_cell  # noqa: E402
from poe_repair.experiments.interaction_term.pool import load_pool  # noqa: E402

OUT_DIR = paths.resolve(paths.CACHE_ANALYSES)


def iter_cells(root: Path, pairs: list[str] | None, max_seeds: int | None,
               *, min_steps: int = 2):
    """Yield (pair_slug, seed) for cached cells with a real trajectory.

    Two traps this guards against, both of which silently produced wrong
    numbers before they were caught:

    - **Eval stubs.** ``build_eval_cache.py`` writes cells holding a single
      ``step_000.pt`` with the eps tensors zeroed, because only ``x_t`` is read
      from them. They are correct for their purpose and useless for any curve.
      ``min_steps`` drops them. A one-step "trajectory" averaged into a
      log-SNR curve collapses the shared range to a point.
    - **The same pair under both splits.** A slug can appear in ``train`` with
      full cells and in ``heldout`` as stubs (a_wolf__x__a_husky does). Walking
      train first and de-duplicating means the full cells win.
    """
    seen: set[tuple[str, int]] = set()
    for split in ("train", "heldout"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for pair_dir in sorted(split_dir.iterdir()):
            if not pair_dir.is_dir():
                continue
            if pairs and pair_dir.name not in pairs:
                continue
            seeds = sorted(
                int(d.name.split("_")[1]) for d in pair_dir.iterdir()
                if d.is_dir() and d.name.startswith("seed_")
            )
            kept = 0
            for s in seeds:
                if max_seeds and kept >= max_seeds:
                    break
                if (pair_dir.name, s) in seen:
                    continue
                n = len(list((pair_dir / f"seed_{s}" / "residuals").glob("step_*.pt")))
                if n < min_steps:
                    continue          # eval stub, not a trajectory
                seen.add((pair_dir.name, s))
                kept += 1
                yield pair_dir.name, s


def curve_for(slug: str, seed: int, *, normalize: str = "prereg",
              root: Path = CACHE_ROOT):
    """One cell's correction-size curve: (slug, seed, log_snr, scaled size).

    The single definition of the measure. `correction_size_vs_run_position.py` draws named pairs from
    outside the pool and imports this rather than repeating the arithmetic,
    because two copies of a measure drift and the drift is invisible in the
    figure.
    """
    c = load_cell(slug, seed, root=root)
    n = c.r_t_norm().numpy()
    if normalize == "prereg":
        # The plan-01 committed measure: correction size relative to the
        # prediction being corrected. report/normalization_preregistration.md
        n = n / c.eps_poe().flatten(1).norm(dim=1).numpy()
    # Then scale each curve to its own median. The claim is about SHAPE over the
    # run, not about every pair needing the same correction size, so a per-curve
    # scale still has to come out. Median, not peak: dividing by the max makes
    # every curve hostage to one noisy point and visibly amplifies jitter.
    return slug, seed, c.log_snr().numpy(), n / max(np.median(n), 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", action="append", dest="pairs")
    ap.add_argument("--all", action="store_true",
                    help="every pair in the cache dir; mixes experiments, "
                         "prefer --pool")
    ap.add_argument("--pool", nargs="?", const=str(paths.resolve(paths.DOES_THE_FIX_REACH_UNSEEN_PAIRS) / "pair_pool.yaml"),
                    help="restrict to one experiment's declared pairs")
    ap.add_argument("--max-pairs", type=int, help="cap distinct pairs (smoke runs)")
    ap.add_argument("--max-seeds", type=int, default=2, help="seeds per pair")
    ap.add_argument("--bins", type=int, default=20, help="log-SNR bins")
    ap.add_argument("--x-axis", default="log-snr", choices=("log-snr", "step"),
                    help="step plots one point per real denoising step and needs "
                         "no interpolation, so it cannot stretch a few steps "
                         "across several bins the way the log-SNR grid does at "
                         "the noisy end. log-snr is the sampler-independent axis")
    ap.add_argument("--normalize", default="prereg",
                    choices=("prereg", "own-median"),
                    help="prereg = ||r_t||/||eps_PoE|| then scaled to its own "
                         "median (the plan-01 committed measure); own-median = "
                         "||r_t|| scaled to its own median only")
    ap.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    if not args.all and not args.pairs and not args.pool:
        ap.error("give --pool, --all, or --pair")

    pairs = args.pairs
    if args.pool:
        pool = load_pool(args.pool)
        print(pool.summary())
        pairs = pool.train + pool.heldout()

    cells = list(iter_cells(args.cache_root, pairs, args.max_seeds))
    if args.max_pairs:
        keep, out = set(), []
        for slug, seed in cells:
            if slug not in keep and len(keep) >= args.max_pairs:
                continue
            keep.add(slug)
            out.append((slug, seed))
        cells = out
    if not cells:
        print("no cached cells matched")
        return 2

    curves = [curve_for(slug, seed, normalize=args.normalize, root=args.cache_root)
              for slug, seed in cells]

    if args.x_axis == "step":
        # Every cell runs the same schedule, so the curves already share an
        # x-axis and stack directly. No interpolation means no chance of the
        # grid disagreeing with where the steps actually are.
        lengths = {len(y) for _, _, _, y in curves}
        if len(lengths) != 1:
            print(f"cells disagree on step count {sorted(lengths)}; "
                  f"the step axis needs one schedule")
            return 2
        grid = np.arange(lengths.pop(), dtype=float)
        stack = np.stack([y for _, _, _, y in curves])
    else:
        lo = max(x.min() for _, _, x, _ in curves)
        hi = min(x.max() for _, _, x, _ in curves)
        grid = np.linspace(lo, hi, args.bins)
        stack = np.stack([
            np.interp(grid, x, y) for _, _, x, y in curves   # log-SNR ascends with step
        ])

    q75, q25 = np.percentile(stack, [75, 25], axis=0)
    med_curve = np.median(stack, axis=0)
    iqr = q75 - q25
    # Spread as a fraction of the median curve's own height at each bin, so the
    # number does not depend on the normalisation choice above. Tight curves
    # give a small percentage; per-pair curves that go their own way give a
    # large one.
    spread = float(np.median(iqr / np.maximum(med_curve, 1e-12))) * 100.0

    n_pairs = len({s for s, _, _, _ in curves})
    axis_name = "log-SNR" if args.x_axis == "log-snr" else "step"
    print(f"collapse spread: {spread:.1f}%   ({n_pairs} pairs, {len(curves)} curves)")
    if args.x_axis == "log-snr":
        print(f"  log-SNR range {lo:.2f} to {hi:.2f} in {args.bins} bins")
    else:
        print(f"  steps 0 to {int(grid[-1])}, one point per real step, no interpolation")
    # The median spread hides how uneven it is, and on the log-SNR axis the two
    # ends differ by a factor of seven. Print the extremes so a figure caption
    # cannot quote the median as if it held everywhere.
    ratio = iqr / np.maximum(med_curve, 1e-12) * 100.0
    iw, it = int(np.argmax(ratio)), int(np.argmin(ratio))
    print(f"  widest {ratio[iw]:.1f}% at {axis_name} {grid[iw]:.2f}, "
          f"tightest {ratio[it]:.1f}% at {axis_name} {grid[it]:.2f}")
    # A maximum at the first or last bin is a truncation, not a peak: the grid
    # stops where the shortest cell stops, so the curve is still moving there.
    imax = int(np.argmax(med_curve))
    at_edge = imax in (0, len(med_curve) - 1)
    if at_edge:
        print(f"  median curve still rising at the {'left' if imax == 0 else 'right'} "
              f"edge ({axis_name} {grid[imax]:.2f}); no interior peak in this range")
    else:
        print(f"  peak of the median curve at {axis_name} {grid[imax]:.2f}")
    # State the reading rather than leaving the number to speak for itself.
    verdict = ("tight" if spread < 15 else
               "loose" if spread < 40 else "no collapse")
    print(f"  reading: {verdict}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_name = "snr_collapse.json" if args.x_axis == "log-snr" else "step_collapse.json"
    (args.out_dir / out_name).write_text(json.dumps({
        "collapse_spread_pct": spread,
        "normalize": args.normalize,
        "x_axis": args.x_axis,
        "n_pairs": n_pairs, "n_curves": len(curves),
        "log_snr_grid": grid.tolist(),
        "median_curve": med_curve.tolist(),
        "verdict": verdict,
        "peak_log_snr": float(grid[imax]),
        "peak_at_edge": bool(at_edge),
        "iqr": iqr.tolist(),
        "q25": q25.tolist(), "q75": q75.tolist(),
        "spread_pct_per_bin": ratio.tolist(),
        "widest": {"pct": float(ratio[iw]), "x": float(grid[iw])},
        "tightest": {"pct": float(ratio[it]), "x": float(grid[it])},
        "cells": [[s, int(sd)] for s, sd, _, _ in curves],
        # The per-curve values, so a figure can draw named curves instead of
        # only the median and the band. Row i belongs to cells[i].
        "curves": stack.tolist(),
    }, indent=2))

    if not args.no_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        for _, _, x, y in curves:
            ax.plot(x, y, color="0.75", lw=0.7, zorder=1)
        ax.fill_between(grid, q25, q75, color="tab:blue", alpha=0.25, zorder=2,
                        label="interquartile range")
        ax.plot(grid, med_curve, color="tab:blue", lw=2.2, zorder=3,
                label="median")
        ax.set_xlabel("log-SNR" if args.x_axis == "log-snr" else "denoising step")
        ax.set_ylabel("correction size, scaled to own median")
        # Report the number; do not assert the conclusion in the title.
        ax.set_title(
            f"Correction size vs {axis_name}: spread {spread:.0f}% "
            f"({n_pairs} pairs, {verdict})"
        )
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        png = args.out_dir / out_name.replace(".json", ".png")
        fig.savefig(png, dpi=150)
        print(f"  figure: {png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
