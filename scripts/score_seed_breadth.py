#!/usr/bin/env python
"""Score the 12-seed breadth run and answer its one question.

Is "a good seed" a property of the initial noise, or of the noise together
with the prompt? Scores every cell of the breadth run with the validated
compose scorer, prints the pair-by-seed table, and computes the two numbers
the sweep's readings were written against:

    per-pair rate      how often each pair composes over its 12 seeds
    cross-pair agreement   for every pair of pairs, the correlation between
                       their seed-outcome patterns. A shared "easy seed"
                       factor shows up here and nowhere else.

Bars, from the sweep script's header, repeated here in source so a later
change shows up in a diff:
    median |r| >= 0.30   some seeds are globally easier
    median |r| <  0.15   seed quality is an interaction with the prompt
    between              inconclusive at this n; do not round it

Reads images, does not sample.

    python scripts/score_seed_breadth.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from poe_repair import paths

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_ROOT = paths.resolve(paths.COMPOSE_RATE_IN_THE_FIRST_WINDOW_ACROSS_TWELVE_SEEDS) / "pairs"
OUT = paths.resolve(paths.COMPOSE_RATE_IN_THE_FIRST_WINDOW_ACROSS_TWELVE_SEEDS) / "seed_breadth.json"
SCORER_CONTRACT = paths.resolve(paths.COMPOSE_SCORER_VALIDATION) / "scorer_validated.json"
SHARED_FACTOR_BAR = 0.30
NO_FACTOR_BAR = 0.15


def pretty(slug: str) -> str:
    def strip(side: str) -> str:
        side = side.replace("_", " ")
        for art in ("an ", "a "):
            if side.startswith(art):
                return side[len(art):]
        return side
    return " x ".join(strip(s) for s in slug.split("__x__"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if not SCORER_CONTRACT.exists():
        raise SystemExit(f"no scorer contract at {SCORER_CONTRACT}")
    contract = json.loads(SCORER_CONTRACT.read_text())
    if not contract.get("pass"):
        raise SystemExit("scorer is not marked validated: refusing")
    print(f"scorer: {contract['method']} via {contract['detector']}")
    print(f"  rule: {contract['compose_rule']}\n")

    from poe_repair.experiments.compose_scorer_validation.detection_scorer import (
        count_instances,
    )
    device = None
    if args.device:
        import torch
        device = torch.device(args.device)

    got: dict[str, dict[int, int]] = defaultdict(dict)
    counts: dict[tuple[str, int], int] = {}
    for pair_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        for seed_dir in sorted(pair_dir.glob("seed_*")):
            seed = int(seed_dir.name.split("_")[1])
            pngs = sorted(seed_dir.rglob("*.png"))
            if not pngs:
                continue
            n, _ = count_instances(pngs[0], device=device)
            got[pair_dir.name][seed] = int(n >= 2)
            counts[(pair_dir.name, seed)] = int(n)

    pairs = sorted(got)
    seeds = sorted({s for v in got.values() for s in v})
    if not pairs:
        raise SystemExit(f"no images under {args.root}")

    print(f"composed (1) by pair and seed, window 0-10, own $r_t$, full dose\n")
    print(f"{'pair':26}" + "".join(f"{'s'+str(s):>4}" for s in seeds) + "   rate")
    rates = {}
    for p in pairs:
        row = [got[p].get(s) for s in seeds]
        have = [x for x in row if x is not None]
        rates[p] = float(np.mean(have)) if have else float("nan")
        print(f"{pretty(p):26}"
              + "".join(f"{('-' if x is None else x):>4}" for x in row)
              + f"   {rates[p]:.2f}")
    per_seed = [np.mean([got[p][s] for p in pairs if s in got[p]])
                for s in seeds]
    print(f"{'--- seed mean':26}" + "".join(f"{v:>4.1f}" for v in per_seed))

    # Do pairs agree about which seeds are easy?
    M = np.array([[got[p].get(s, np.nan) for s in seeds] for p in pairs],
                 dtype=float)
    cors = []
    for i, j in combinations(range(len(pairs)), 2):
        a, b = M[i], M[j]
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() < 4 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
            continue
        cors.append({"pair_a": pairs[i], "pair_b": pairs[j],
                     "r": float(np.corrcoef(a[ok], b[ok])[0, 1]),
                     "n": int(ok.sum())})
    med_abs = float(np.median([abs(c["r"]) for c in cors])) if cors else float("nan")
    med_signed = float(np.median([c["r"] for c in cors])) if cors else float("nan")

    # A shared "easy seed" factor means pairs agree, which is a POSITIVE
    # correlation. Judging on |r| was wrong: it counts a pair that
    # systematically disagrees as evidence of a shared factor. And |r| has a
    # large floor at this sample size, so it needs its own null rather than a
    # threshold picked by eye. The null shuffles each pair's seed labels
    # independently, which destroys any shared factor and keeps each pair's
    # own success rate.
    rng = np.random.default_rng(0)
    null = []
    for _ in range(400):
        P = np.array([rng.permutation(row) for row in M])
        vals = []
        for i, j in combinations(range(len(pairs)), 2):
            a, b = P[i], P[j]
            ok = ~(np.isnan(a) | np.isnan(b))
            if ok.sum() < 4 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
                continue
            vals.append(abs(np.corrcoef(a[ok], b[ok])[0, 1]))
        if vals:
            null.append(float(np.median(vals)))
    null_med = float(np.median(null)) if null else float("nan")
    null_p95 = float(np.percentile(null, 95)) if null else float("nan")

    print(f"\ncross-pair agreement over {len(cors)} pairings of pairs")
    print(f"  median |r| {med_abs:.3f}   median r {med_signed:+.3f}")
    print(f"  null from shuffled seed labels: median |r| {null_med:.3f}, "
          f"95th percentile {null_p95:.3f}")
    spread = max(rates.values()) - min(rates.values())
    print(f"  per-pair rate spans {min(rates.values()):.2f} to "
          f"{max(rates.values()):.2f} (spread {spread:.2f})")

    beats_null = med_abs > null_p95
    if med_signed >= SHARED_FACTOR_BAR and beats_null:
        verdict = ("SHARED FACTOR: pairs agree about which seeds are easy, so "
                   "seed quality is partly a property of the initial noise "
                   "alone and could in principle be read off x_T")
    elif med_signed < NO_FACTOR_BAR and not beats_null:
        verdict = ("INTERACTION: pairs do not agree about which seeds are "
                   "easy. The agreement is at or below what shuffled labels "
                   "produce, so seed quality is a property of the noise "
                   "together with the prompt. No per-seed predictor can "
                   "ignore the pair, and a noise map has to be drawn one pair "
                   "at a time")
    else:
        verdict = (f"INCONCLUSIVE at this n: median signed r {med_signed:+.3f} "
                   f"with median |r| {med_abs:.3f} against a shuffled-label "
                   f"null of {null_med:.3f}. Report it, do not round it")
    print(f"\n  {verdict}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scorer": contract["method"], "window": [0, 10],
        "pairs": pairs, "seeds": seeds,
        "composed": {p: {str(s): v for s, v in got[p].items()} for p in pairs},
        "instance_counts": {f"{p}|{s}": n for (p, s), n in counts.items()},
        "per_pair_rate": rates,
        "per_seed_mean": {str(s): float(v) for s, v in zip(seeds, per_seed)},
        "cross_pair_correlations": cors,
        "median_abs_r": med_abs, "median_signed_r": med_signed,
        "shuffled_label_null": {"median_abs_r": null_med, "p95_abs_r": null_p95,
                                "n_permutations": len(null),
                                "why": "|r| has a large floor at 12 binary "
                                       "points, so the observed value needs a "
                                       "null rather than a threshold"},
        "bars": {"shared_factor": SHARED_FACTOR_BAR,
                 "no_factor": NO_FACTOR_BAR},
        "verdict": verdict,
        "note": "with 12 binary points a single |r| below about 0.58 is not "
                "individually significant, so the verdict rests on the median "
                "across pairings, not on any one of them",
    }, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
