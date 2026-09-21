#!/usr/bin/env python
"""Plan 09 (experiment-b-rank-16-32), task 1.3: the matched-checkpoint table
across r8/r16/r32, scored per held-out seed so rank 8's seed-noise band
(the review file's threshold) can actually be computed.

`compose_rate.py` (does_the_fix_reach_unseen_pairs) scores the same inline
samples but discards which of the 8 held-out seeds each render came from,
keeping only the per-pair aggregate. The review's threshold needs the
per-seed spread, so this script re-scores directly from the sample PNGs
with the same validated instance-count scorer
(compose_scorer_validation.detection_scorer.score_output_instances), rather
than editing that script's output shape.

For each rank and each matched checkpoint on the 10k grid: for each of the
8 held-out seeds, average the out_out compose rate across all held-out
pairs rendered at that seed. That gives 8 numbers per (rank, step); rank
8's spread across those 8 numbers is the seed-noise band the review file's
question is judged against.

Usage:
    python -m scripts.showcase.rank_ablation \
        --ranks 8,16,32 --grid 10000,20000,...,100000
    # or, for a dry look at whatever checkpoints already exist:
    python -m scripts.showcase.rank_ablation --ranks 8,16,32
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from poe_repair import paths
from poe_repair.experiments.compose_scorer_validation.detection_scorer import (
    score_output_instances,
)

log = logging.getLogger("showcase.rank_ablation")

SCOPE = paths.resolve(paths.DOES_THE_FIX_REACH_UNSEEN_PAIRS)
POOLED_LORA = SCOPE / "pooled_lora"
DEFAULT_GRID = list(range(10_000, 100_001, 10_000))
OUT_OUT_RE = re.compile(r"^out_out__(.+)__seed(\d+)$")


def _load_prompts() -> dict:
    import yaml

    return yaml.safe_load((SCOPE / "pair_prompts.yaml").read_text())


def _score_step(
    run_dir: Path, step: int, prompts: dict, device: torch.device
) -> dict[int, dict[str, list[bool]]] | None:
    """Return {seed: {pair: [is_compose, ...]}} for one run's out_out
    renders at one step, or None if that checkpoint has no samples yet."""
    snap_dirs = list((run_dir / "samples" / "per_epoch").glob(f"epoch_*_step_{step:06d}"))
    if not snap_dirs:
        return None
    snap = snap_dirs[0]
    by_seed: dict[int, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for png in sorted(snap.glob("out_out__*__seed*.png")):
        m = OUT_OUT_RE.match(png.stem)
        if not m:
            continue
        pair, seed_str = m.group(1), m.group(2)
        if pair not in prompts:
            continue
        seed = int(seed_str)
        qa, qb = prompts[pair]["prompt_a"], prompts[pair]["prompt_b"]
        sc = score_output_instances(png, qa, qb, device=device)
        by_seed[seed][pair].append(sc.label == "compose")
    return {s: dict(v) for s, v in by_seed.items()}


def _per_seed_rates(by_seed: dict[int, dict[str, list[bool]]]) -> dict[int, float]:
    """Average compose rate across held-out pairs, one number per seed."""
    rates = {}
    for seed, pair_hits in by_seed.items():
        pair_rates = [sum(hits) / len(hits) for hits in pair_hits.values() if hits]
        if pair_rates:
            rates[seed] = sum(pair_rates) / len(pair_rates)
    return rates


def main() -> int:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                         datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", default="8,16,32")
    ap.add_argument("--grid", default=",".join(str(s) for s in DEFAULT_GRID),
                     help="matched checkpoint steps, comma-separated")
    ap.add_argument("--out", default=str(POOLED_LORA / "rank_ablation.json"))
    args = ap.parse_args()

    ranks = [int(r) for r in args.ranks.split(",")]
    grid = [int(s) for s in args.grid.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompts = _load_prompts()

    per_rank: dict[str, dict] = {}
    for rank in ranks:
        run_dir = POOLED_LORA / f"phase1_r{rank}_100k"
        if not run_dir.exists():
            log.warning("rank %d: no run dir at %s yet, skipping", rank, run_dir)
            continue
        per_step = {}
        for step in grid:
            by_seed = _score_step(run_dir, step, prompts, device)
            if by_seed is None:
                log.info("rank %d step %d: no checkpoint samples yet", rank, step)
                continue
            seed_rates = _per_seed_rates(by_seed)
            if not seed_rates:
                continue
            values = list(seed_rates.values())
            per_step[step] = {
                "per_seed_compose_rate": seed_rates,
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "n_seeds": len(values),
            }
            log.info("rank %d step %d: mean=%.3f over %d seeds (min=%.3f max=%.3f)",
                      rank, step, per_step[step]["mean"], len(values),
                      per_step[step]["min"], per_step[step]["max"])
        per_rank[str(rank)] = per_step

    # The threshold in the review file: at matched 100k, do r16/r32 sit
    # inside r8's held-out seed-noise band (the observed min-max range
    # across its 8 held-out seeds at that step)?
    verdict = {}
    r8_100k = per_rank.get("8", {}).get(100_000)
    if r8_100k is not None:
        band = (r8_100k["min"], r8_100k["max"])
        verdict["rank_8_band_at_100k"] = {"min": band[0], "max": band[1]}
        for rank in ranks:
            if rank == 8:
                continue
            step_data = per_rank.get(str(rank), {}).get(100_000)
            if step_data is None:
                continue
            inside = band[0] <= step_data["mean"] <= band[1]
            verdict[f"rank_{rank}_at_100k"] = {
                "mean": step_data["mean"],
                "inside_rank_8_band": inside,
            }
    else:
        log.warning("rank 8 has no scored samples at step 100000 yet; "
                    "threshold verdict left empty")

    out = {
        "matched_grid": grid,
        "per_rank": per_rank,
        "threshold_verdict_at_100k": verdict,
        "crispness_note": (
            "phase1_r8_100k predates plan 06's tracking-set extension, so it "
            "has no eval/tracking/* reads to compare r16/r32 against; the "
            "review question 'does the crispness read move with rank' can "
            "only be answered self-consistently for r16 vs r32, not against "
            "r8, unless r8 is re-read through the tracking hook."
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=float))
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
