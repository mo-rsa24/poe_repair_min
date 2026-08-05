#!/usr/bin/env python
"""Does the correction push along the path, or against it?

Plan 05 task 4, as worded: the accumulated dot product of r_t with the step the
latent actually takes, along the cached path.

    climb = sum_t  r_t . (x_{t+1} - x_t)

The sign is the whole point, and it is not obvious which way it should go.

  positive  the correction agrees with where the trajectory is already headed:
            it accelerates the motion rather than redirecting it.
  negative  the correction opposes the motion: it is pulling the sample off the
            path PoE would have taken, which is what "prying the chimera apart"
            would look like.

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.experiments.interaction_term.cache import (  # noqa: E402
    CACHE_ROOT,
    load_cell,
)
from poe_repair.experiments.interaction_term.pool import load_pool  # noqa: E402
from scripts.snr_collapse import iter_cells  # noqa: E402

OUT_DIR = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses")


def climb_for_cell(cell) -> dict:
    """Accumulated r_t . dx_t for one cell, raw and normalised."""
    r = cell.r_t().flatten(1)[:-1]                      # drop the last: no dx
    dx = (cell.x_t[1:] - cell.x_t[:-1]).flatten(1)
    dots = (r * dx).sum(1)
    raw = float(dots.sum())
    denom = float((r.norm(dim=1) * dx.norm(dim=1)).sum())
    per_step = (dots / (r.norm(dim=1) * dx.norm(dim=1)).clamp_min(1e-12)).numpy()
    return {
        "raw": raw,
        "normalised": raw / max(denom, 1e-12),
        "per_step_cosine": per_step.tolist(),
        "fraction_of_steps_negative": float((per_step < 0).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", nargs="?",
                    const="outputs/animals_compose_transfer/pair_pool.yaml",
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

    if n_neg == 0:
        reading = "positive"
        print("\n  Reading: the correction points WITH the motion, everywhere.")
        print("  It accelerates the trajectory rather than redirecting it. That")
        print("  is not what 'prying the chimera apart' would look like, and it")
        print("  is worth taking seriously rather than filing as expected.")
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
