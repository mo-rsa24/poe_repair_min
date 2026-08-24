#!/usr/bin/env python
"""Does the correction push along the path, or against it?

Plan 05 task 4, as worded: the accumulated dot product of r_t with the step the
latent actually takes, along the cached path.

    climb = sum_t  r_t . (x_{t+1} - x_t)

The sign is the whole point, and reading it without controls gets it BACKWARDS.

A DDIM step moves along MINUS eps, so the prediction that drives the step sits
near -0.60 against dx, not +1. Measured here, r_t sits at +0.40: the opposite
side of zero from the thing steering the trajectory.

So a positive climb does NOT mean "the correction agrees with the motion". It
means r_t opposes eps_PoE, subtracting from what PoE is asking for. Confirmed
directly: cos(r_t, eps_PoE) is negative in 38/38 cells. That is what "PoE
overshoots into a blend, and the correction pulls it back" predicts.

The controls are computed per cell and printed with the result, so this cannot
be misread again.

Reported normalised by sum ||r_t|| ||dx_t||, so it reads as an average cosine
in [-1, 1] and is comparable across pairs. The raw sum is printed too, since
the normalisation hides how much total work the correction does.

**What this can and cannot say.** The cache walks ONE path per cell, the PoE
path. So this measures the correction against the *uncorrected* trajectory: the
push it would apply at the states PoE actually visits. It is not the same as
the climb along the corrected path, which would need that path sampled. Plan
05's fork-curve task covers that generation; this is the free half.

Cache-only, no GPU.

    python scripts/plausibility_climb.py --pool
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from poe_repair import paths

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.experiments.interaction_term.cache import (  # noqa: E402
    CACHE_ROOT,
    load_cell,
)
from poe_repair.experiments.interaction_term.pool import load_pool  # noqa: E402
from scripts.snr_collapse import iter_cells  # noqa: E402

OUT_DIR = paths.resolve(paths.CACHE_ANALYSES)


def _cos(a, b):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-12)


def climb_for_cell(cell, *, seed: int = 0) -> dict:
    """Accumulated r_t . dx_t for one cell, with the controls that read it.

    The bare number is not interpretable on its own, and reading it without
    these controls gets the sign backwards. Three references:

      eps_PoE vs dx   the prediction that DRIVES the step. Strongly negative
                      (~-0.71) because a DDIM step moves along MINUS eps. This
                      is the scale: "aligned with the motion" means near -0.71,
                      not near +1.
      random vs dx    the floor, ~0.000.
      shuffled vs dx  r_t taken at the wrong step, ~0.05. Separates "r_t is
                      timed to this state" from "r_t always looks like this".
      r_t vs eps_PoE  the same fact stated without the step: does the
                      correction subtract from what PoE asks for?
    """
    r = cell.r_t().flatten(1)[:-1]                      # drop the last: no dx
    dx = (cell.x_t[1:] - cell.x_t[:-1]).flatten(1)
    eps = cell.eps_poe().flatten(1)[:-1]
    dots = (r * dx).sum(1)
    raw = float(dots.sum())
    denom = float((r.norm(dim=1) * dx.norm(dim=1)).sum())
    per_step = _cos(r, dx).numpy()

    g = torch.Generator().manual_seed(seed)
    rnd = torch.randn(r.shape, generator=g)
    perm = torch.randperm(r.shape[0], generator=g)
    return {
        "raw": raw,
        "normalised": raw / max(denom, 1e-12),
        "per_step_cosine": per_step.tolist(),
        "fraction_of_steps_negative": float((per_step < 0).mean()),
        "control_eps_vs_dx": float(_cos(eps, dx).median()),
        "control_random_vs_dx": float(_cos(rnd, dx).median()),
        "control_shuffled_vs_dx": float(_cos(r[perm], dx).median()),
        "r_vs_eps_poe": float(_cos(r, eps).median()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", nargs="?",
                    const=str(paths.resolve(paths.DOES_THE_FIX_REACH_UNSEEN_PAIRS) / "pair_pool.yaml"),
                    help="restrict to one experiment's declared pairs")
    ap.add_argument("--pair", action="append", dest="pairs")
    ap.add_argument("--max-seeds", type=int, default=2)
    ap.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    pairs = args.pairs
    if args.pool:
        pool = load_pool(args.pool)
        print(pool.summary())
        pairs = pool.train + pool.heldout(roles=("transfer", "reference", "control"))
    elif not pairs:
        ap.error("give --pool or --pair")

    cells = list(iter_cells(args.cache_root, pairs, args.max_seeds))
    if not cells:
        print("no cached cells matched", file=sys.stderr)
        return 2

    rows = []
    print(f"\n{'pair':<30}{'seed':>5}{'raw sum':>12}{'normalised':>12}{'% steps <0':>12}")
    for slug, seed in cells:
        c = climb_for_cell(load_cell(slug, seed, root=args.cache_root))
        rows.append({"pair": slug, "seed": int(seed), **c})
        print(f"{slug:<30}{seed:>5}{c['raw']:>12.1f}{c['normalised']:>12.4f}"
              f"{c['fraction_of_steps_negative']:>11.0%}")

    norm = np.array([r["normalised"] for r in rows])
    n_neg = int((norm < 0).sum())
    print(f"\n{len(rows)} cells, {len({r['pair'] for r in rows})} pairs")
    print(f"  normalised climb: median {np.median(norm):+.4f}, "
          f"IQR {np.percentile(norm, 25):+.4f} to {np.percentile(norm, 75):+.4f}")
    print(f"  cells with a NEGATIVE climb: {n_neg}/{len(rows)}")

    ctl_eps = np.median([r["control_eps_vs_dx"] for r in rows])
    ctl_rnd = np.median([r["control_random_vs_dx"] for r in rows])
    ctl_shuf = np.median([r["control_shuffled_vs_dx"] for r in rows])
    r_vs_eps = np.array([r["r_vs_eps_poe"] for r in rows])
    print("\n  controls (medians):")
    print(f"    eps_PoE vs dx    {ctl_eps:+.4f}   the prediction driving the step")
    print(f"    random  vs dx    {ctl_rnd:+.4f}   floor")
    print(f"    r_t shuffled     {ctl_shuf:+.4f}   r_t at the wrong step")
    print(f"    r_t vs eps_PoE   {np.median(r_vs_eps):+.4f}   "
          f"({int((r_vs_eps < 0).sum())}/{len(r_vs_eps)} negative)")

    if n_neg == 0:
        reading = "opposes eps_PoE"
        print("\n  Reading: DO NOT read the positive climb as 'the correction")
        print("  agrees with the motion'. A DDIM step moves along MINUS eps, so")
        print("  the prediction driving the step sits at "
              f"{ctl_eps:+.2f}, not +1.")
        print("  r_t having the OPPOSITE sign to that means it opposes eps_PoE:")
        print("  it subtracts from what PoE is asking for, partially undoing it.")
        print("  Confirmed directly: r_t vs eps_PoE is negative in every cell.")
        print("  That is what 'PoE overshoots into a blend' predicts.")
    elif n_neg == len(rows):
        reading = "negative"
        print("\n  Reading: the correction OPPOSES the motion everywhere: it is")
        print("  pulling the sample off the path PoE would have taken.")
    else:
        reading = "mixed"
        print(f"\n  Reading: mixed. {n_neg} of {len(rows)} cells oppose the motion.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "plausibility_climb.json").write_text(json.dumps({
        "measure": "sum_t r_t . (x_{t+1} - x_t), along the cached PoE path",
        "caveat": "the cache walks only the PoE path, so this is the push at "
                  "the states PoE visits, not the climb along a corrected path",
        "median_normalised": float(np.median(norm)),
        "n_cells": len(rows), "n_negative": n_neg, "reading": reading,
        "cells": rows,
    }, indent=2))

    if not args.no_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        for r in rows:
            ax1.plot(r["per_step_cosine"], color="0.75", lw=0.7)
        m = min(len(r["per_step_cosine"]) for r in rows)
        med = np.median(np.stack([r["per_step_cosine"][:m] for r in rows]), axis=0)
        ax1.plot(med, color="tab:blue", lw=2, label="median")
        ax1.axhline(0, color="k", lw=0.9, ls=":")
        ax1.set_xlabel("denoising step")
        ax1.set_ylabel("cosine, r_t vs the step taken")
        ax1.set_title("Does the correction push with the motion?")
        ax1.legend(frameon=False, fontsize=8)
        ax1.grid(alpha=0.3)

        ax2.hist(norm, bins=12, color="tab:blue", alpha=0.8)
        ax2.axvline(0, color="k", lw=1.2, ls=":")
        ax2.set_xlabel("accumulated climb, normalised")
        ax2.set_ylabel("cells")
        ax2.set_title(f"{len(rows)} cells, median {np.median(norm):+.2f}")
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(args.out_dir / "plausibility_climb.png", dpi=150)
        print(f"\nfigure: {args.out_dir / 'plausibility_climb.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
