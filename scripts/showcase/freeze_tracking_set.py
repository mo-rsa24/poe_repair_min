#!/usr/bin/env python
"""Freeze tracking_set.json (plan 06, 01-showcase-the-trained-lora, task 1.2).

Writes the fixed manifest naming every render and metric family experiments A
and B read while they run: the three curves already wired by instrument-02,
plus the four additions task 1.1 wires into train_pooled.py's eval hook.

Scope deliberately small: one in-pair, one out-pair, 2 seeds each (4 cells).
The first attempt sampled the full pool (8 seeds x 19 pairs = 152 cells) at
50 DDIM steps, measured at ~49s/cell — about 2 hours per checkpoint, ~20
hours added across a 100k-step run. This tracking set's job is watching
training stay healthy and tracing how the four new diagnostic curves move
across checkpoints; one pair on each side does that as well as nineteen do,
and it does not need statistical coverage. The full-coverage, full-statistics
compose-rate comparison stays a SEPARATE step (plan 08 task 1.3's
experiment_a_verdict_inputs.json), run once per checkpoint being judged, not
every 10k steps.

The render list is built from the SAME plan-construction logic
train_pooled.py uses at startup, restricted to --in-pairs/--out-pairs, so the
manifest names exactly what a run with matching flags will actually render.

Usage:
    /home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python \
        scripts/showcase/freeze_tracking_set.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
SCOPE = REPO_ROOT / "artifacts/results/does-the-fix-reach-unseen-pairs"
OUT_PATH = SCOPE / "pooled_lora" / "tracking_set.json"

ALREADY_WIRED = [
    "eval/direction_cosine/<quadrant>/<pair>/seed_<NN> (+ /mean)",
    "eval/frac_distance_reached/<quadrant>/<pair>/seed_<NN> (+ /mean)",
    "eval/compose_rate/<quadrant>/<pair>/seed_<NN> (+ /mean, anchors-covered pairs only)",
]
NEW_ADDITIONS = [
    "eval/tracking/learned_actual_cosine/bucket_{early,commit,late}/<quadrant>/<pair>/seed_<NN> (+ /mean)",
    "eval/tracking/learned_actual_cosine/step_{07,15,22}/<quadrant>/<pair>/seed_<NN> (+ /mean)",
    "eval/tracking/embedding_drift/{dino,clip}/<quadrant>/<pair>/seed_<NN> (+ /mean)",
    "eval/tracking/divergence_profile/step_{07,15,22}/<quadrant>/<pair>/seed_<NN> (+ /mean)",
    "eval/tracking/spectral_share/{top1_share,top3_share} (cross-cell, step 22 only)",
]
# Mirrors RunConfig.probe (one_pair_one_seed/config.py): commit_window=(5,25),
# where_applied_steps=(7,15,22) — the same window train/loss_bucket/* uses.
BUCKETS = {"early": [0, 5], "commit": [5, 25], "late": [25, 49]}
REP_STEPS = [7, 15, 22]
SPECTRAL_STEP = 22

DEFAULT_IN_PAIRS = ["a_wolf__x__a_husky"]
DEFAULT_OUT_PAIRS = ["a_cat__x__a_dog"]


def build_plan(seed_pool, in_pairs: list[str], out_pairs: list[str],
                n_in: int, n_out: int) -> list[dict]:
    """Mirror train_pooled.py's sample-plan construction exactly (same slice
    of seed_pool.train_pool / seed_pool.held_out, same quadrant labels),
    restricted to the named pairs rather than the whole pool."""
    in_seeds = list(seed_pool.train_pool[:n_in])
    out_seeds = list(seed_pool.held_out[:n_out])
    plan = []
    for pair in in_pairs:
        for s in in_seeds:
            plan.append({"quadrant": "in_in", "pair": pair, "seed": int(s)})
    for pair in out_pairs:
        for s in out_seeds:
            plan.append({"quadrant": "out_out", "pair": pair, "seed": int(s)})
    return plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-in", type=int, default=2,
                     help="seeds per in-pair; matches --sample-cells-per-train-pair")
    ap.add_argument("--n-out", type=int, default=2,
                     help="seeds per out-pair; matches --sample-cells-per-heldout-pair")
    ap.add_argument("--in-pairs", default=",".join(DEFAULT_IN_PAIRS),
                     help="comma-separated; matches --sample-train-pairs")
    ap.add_argument("--out-pairs", default=",".join(DEFAULT_OUT_PAIRS),
                     help="comma-separated; matches --sample-heldout-pairs")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    from poe_repair.experiments.cross_pair_lora_pooling.seed_pool import load_seed_pool

    seed_pool = load_seed_pool(SCOPE / "seed_pool.yaml")
    in_pairs = [p.strip() for p in args.in_pairs.split(",") if p.strip()]
    out_pairs = [p.strip() for p in args.out_pairs.split(",") if p.strip()]
    plan = build_plan(seed_pool, in_pairs, out_pairs, args.n_in, args.n_out)

    manifest = {
        "schema": "poe_repair_min/tracking_set/v1",
        "source": "plans/01-showcase-the-trained-lora/plans/tools/06-extend-the-tracking-set.md",
        "assumes_sample_num_inference_steps": 50,
        "renders": plan,
        "n_renders": len(plan),
        "metric_families": {
            "already_wired": ALREADY_WIRED,
            "new_additions": NEW_ADDITIONS,
        },
        "buckets": BUCKETS,
        "representative_steps": REP_STEPS,
        "spectral_share_step": SPECTRAL_STEP,
        "deviates_from_ledger": (
            "decisions-taken-here.md#the-shared-tracking-set-extends-"
            "instrument-02-and-nothing-else named 'the four F9 renders "
            "(held-out pairs, seed 9)' and 'the repaired render of cat x "
            "dog seed 1'. Measured cost at that scope (152 cells, 50 "
            "steps) was ~49s/cell, ~2h/checkpoint, ~20h added across a "
            "100k-step run — infeasible. Replaced with one in-pair "
            f"({in_pairs}) and one out-pair ({out_pairs}) at {args.n_in} "
            "seeds each, decided live during plan 06's execution. The "
            "full-coverage, full-statistics compose-rate comparison stays "
            "a separate one-time pass (plan 08 task 1.3), unaffected."
        ),
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["manifest_hash"] = hashlib.sha256(canonical.encode()).hexdigest()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {out_path} ({len(plan)} renders), hash={manifest['manifest_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
