#!/usr/bin/env python
"""Is ANY part of the correction shared across runs of the same pair?

Two cache-only measurements behind D2 and its companions. No GPU, no sampling.

1. Every pair of runs. D2 compares seeds 9 and 13 and finds zero agreement.
   This compares every available pair of cached runs of one pair of animals,
   so "zero" is a property of the population rather than of the two runs that
   happened to be drawn. It is also an exhaustive hunt for a counter-example:
   if any two runs of the same pair DID agree, D2's claim would need
   rewriting.

2. How much of a run's correction the OTHER runs can explain. At each step,
   take the mean correction over every other cached run of the same pair
   (leave-one-out, so nothing of the target run leaks in), and measure what
   fraction of the target run's correction energy that mean accounts for. The
   same measurement on random norm-matched vectors gives the floor, because
   any set of vectors in 65536 dimensions has some accidental overlap and the
   raw fraction alone would not say whether the overlap means anything.

Writes outputs/interaction_term/direction_wall/shared_component.json.

    python scripts/rt_shared_component.py
    python scripts/rt_shared_component.py --pair an_eagle__x__a_hawk
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from poe_repair import paths

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.experiments.interaction_term.cache import (  # noqa: E402
    CACHE_ROOT, load_cell,
)

OUT = paths.resolve(paths.DIRECTION_WALL) / "shared_component.json"
# Reported as ranges rather than 50 numbers: the interesting structure is that
# the shared part dies within the first few steps, which these bands show.
BANDS = ((0, 3), (3, 10), (10, 20), (20, 35), (35, 50))


def seeds_for(pair: str) -> list[int]:
    for split in ("heldout", "train"):
        d = CACHE_ROOT / split / pair
        if d.is_dir():
            return sorted(int(p.name.split("_")[1]) for p in d.glob("seed_*"))
    raise SystemExit(f"no cached cells for {pair}")


def leave_one_out_fraction(X: torch.Tensor) -> torch.Tensor:
    """[N,T,D] -> [N,T]: fraction of run i's energy explained at step t by the
    mean of the other N-1 runs at that step."""
    total = X.sum(0)
    out = torch.zeros(X.shape[0], X.shape[1])
    for i in range(X.shape[0]):
        m = (total - X[i]) / (X.shape[0] - 1)
        mhat = m / m.norm(dim=1, keepdim=True).clamp_min(1e-12)
        proj = (X[i] * mhat).sum(1)
        out[i] = proj ** 2 / X[i].norm(dim=1).clamp_min(1e-12) ** 2
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", default="a_cat__x__a_dog")
    args = ap.parse_args()

    seeds = seeds_for(args.pair)
    runs, kept = [], []
    for s in seeds:
        try:
            runs.append(load_cell(args.pair, s).r_t().float().flatten(1))
            kept.append(s)
        except Exception as e:                    # a partially written cell
            print(f"  skipping seed {s}: {type(e).__name__}", file=sys.stderr)
    R = torch.stack(runs)
    N, T, D = R.shape
    print(f"{args.pair}: {N} cached runs, {T} steps, {D} dimensions")

    # 1. every pair of runs
    combos = []
    for a, b in combinations(range(N), 2):
        c = torch.nn.functional.cosine_similarity(R[a], R[b], dim=1)
        combos.append({"seed_a": kept[a], "seed_b": kept[b],
                       "median": float(c.median()),
                       "first3": float(c[:3].median()),
                       "late": float(c[10:].median())})
    meds = np.array([c["median"] for c in combos])
    best = max(combos, key=lambda c: c["median"])
    print(f"\n{len(combos)} run-pairs: median of medians {np.median(meds):+.4f}, "
          f"range [{meds.min():+.4f}, {meds.max():+.4f}]")
    print(f"  most-agreeing run-pair: seeds {best['seed_a']} vs "
          f"{best['seed_b']} at {best['median']:+.4f}. No counter-example "
          f"unless this is large.")

    # 2. what the other runs can explain, against the random floor
    real = leave_one_out_fraction(R)
    g = torch.Generator().manual_seed(0)
    floor = leave_one_out_fraction(torch.randn(R.shape, generator=g))
    bands = []
    print(f"\nfraction of a run's correction explained by the other {N-1} runs")
    print(f"{'steps':>10} {'real':>9} {'random floor':>14} {'ratio':>10}")
    for lo, hi in BANDS:
        r = float(real[:, lo:hi].median())
        f = float(floor[:, lo:hi].median())
        bands.append({"steps": f"{lo}-{hi-1}", "real": r, "floor": f,
                      "ratio": r / max(f, 1e-12)})
        print(f"{f'{lo}-{hi-1}':>10} {r:>8.2%} {f:>13.2%} {r/max(f,1e-12):>9.0f}x")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "pair": args.pair, "seeds": kept, "n_runs": N, "steps": T, "dim": D,
        "dtype": "fp32 upcast from the fp16 cache",
        "all_run_pairs": {
            "n": len(combos),
            "median_of_medians": float(np.median(meds)),
            "min": float(meds.min()), "max": float(meds.max()),
            "most_agreeing": best,
            "reading": "an exhaustive counter-example hunt for D2: if any two "
                       "runs of the same pair agreed, the max would be large",
        },
        "shared_component": {
            "method": "leave-one-out: fraction of run i's correction energy "
                      "explained by the mean of the other runs at that step",
            "bands": bands,
            "per_step_median_real": np.median(real.numpy(), 0).tolist(),
            "floor": "same measurement on random Gaussian vectors of identical "
                     "shape, generator seed 0",
        },
    }, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
