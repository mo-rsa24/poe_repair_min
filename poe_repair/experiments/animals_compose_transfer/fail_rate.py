"""Plan 01 task 4: fails-by-default rate over 8 seeds, per pair.

For every pair in the pool, generate the DEFAULT (uncorrected) vanilla-PoE
composition at 8 seeds (fresh seeded init, no cache), score each with the
instance-count scorer, and record the blend-rate = fraction of seeds that
blend (instance-count < 2).

A train pair is kept iff it blends by default at a high rate (it is genuinely
broken). A control is kept iff it composes by default (blends rarely). Output:
fail_rate.json + fail_rate.md table.

Run:  $PY -m poe_repair.experiments.animals_compose_transfer.fail_rate
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml
import torch

from poe_repair.run import make_ctx, run_method
from poe_repair.experiments._eval_common import cell_for
from poe_repair.experiments.compose_scorer.detection_scorer import score_output_instances

log = logging.getLogger("animals_compose_transfer.fail_rate")

REPO = Path(__file__).resolve().parents[3]
SCOPE = REPO / "outputs" / "animals_compose_transfer"
POOL = SCOPE / "pair_pool.yaml"
PROMPTS = SCOPE / "pair_prompts.yaml"
SEEDS = list(range(1, 9))  # 8 seeds, paired with the eval seeds later

# fail-rate thresholds (plan 01): a train pair must blend by default; a control must not.
TRAIN_MIN_FAIL = 0.5   # keep as training pair iff >= this fraction of seeds blend
CONTROL_MAX_FAIL = 0.5  # a control is "compose-by-default" iff < this


def main() -> int:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    pool = yaml.safe_load(POOL.read_text())
    prompts = yaml.safe_load(PROMPTS.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ctx = make_ctx()  # SDXL loaded once
    all_pairs = list(pool["train"]) + list(pool["heldout"])
    controls = {"an_elephant__x__a_penguin", "a_giraffe__x__a_crab", "an_octopus__x__a_sparrow"}

    results = {}
    for slug in all_pairs:
        p = prompts[slug]
        per_seed = []
        n_blend = 0
        for s in SEEDS:
            cell = cell_for(p["prompt_a"], p["prompt_b"], s)
            png = run_method("poe", cell, ctx)  # idempotent; fresh seeded init
            sc = score_output_instances(Path(png), p["prompt_a"], p["prompt_b"], device=device)
            is_blend = (sc.label == "blend")
            n_blend += is_blend
            per_seed.append({"seed": s, "png": str(png), "n_instances": sc.n_instances,
                             "label": sc.label})
        fail_rate = n_blend / len(SEEDS)
        role = "control" if slug in controls else ("reference" if slug == "a_cat__x__a_dog" else "train")
        results[slug] = {"role": role, "fail_rate": fail_rate, "n_blend": n_blend,
                         "n_seeds": len(SEEDS), "per_seed": per_seed}
        log.info("%-30s role=%-9s fail_rate=%.2f (%d/%d blend)", slug, role, fail_rate, n_blend, len(SEEDS))

    # Decide keep/drop per the plan rule.
    train_keep, train_drop, control_ok, control_bad = [], [], [], []
    for slug, r in results.items():
        if r["role"] == "train":
            (train_keep if r["fail_rate"] >= TRAIN_MIN_FAIL else train_drop).append(slug)
        elif r["role"] == "control":
            (control_ok if r["fail_rate"] < CONTROL_MAX_FAIL else control_bad).append(slug)
    summary = {
        "seeds": SEEDS, "thresholds": {"train_min_fail": TRAIN_MIN_FAIL, "control_max_fail": CONTROL_MAX_FAIL},
        "train_keep": train_keep, "train_drop": train_drop,
        "control_ok": control_ok, "control_bad": control_bad,
        "results": results,
    }
    (SCOPE / "fail_rate.json").write_text(json.dumps(summary, indent=2))

    # Markdown table.
    lines = ["# Fail-rate (vanilla PoE, 8 seeds, instance-count scorer)", "",
             "| pair | role | fail-rate | verdict |", "|---|---|---|---|"]
    for slug, r in sorted(results.items(), key=lambda kv: (kv[1]["role"], -kv[1]["fail_rate"])):
        if r["role"] == "train":
            verdict = "keep (blends by default)" if r["fail_rate"] >= TRAIN_MIN_FAIL else "DROP (composes by default)"
        elif r["role"] == "control":
            verdict = "good control (composes)" if r["fail_rate"] < CONTROL_MAX_FAIL else "BAD control (blends)"
        else:
            verdict = "reference"
        lines.append(f"| {slug} | {r['role']} | {r['fail_rate']:.2f} ({r['n_blend']}/{r['n_seeds']}) | {verdict} |")
    lines += ["", f"train kept: {len(train_keep)} / {len(train_keep)+len(train_drop)}  (floor for leave-one-out: 10)",
              f"train dropped: {train_drop or '(none)'}",
              f"controls composing-by-default: {control_ok}  |  bad controls: {control_bad or '(none)'}"]
    (SCOPE / "fail_rate.md").write_text("\n".join(lines))
    log.info("wrote fail_rate.json + fail_rate.md | train_keep=%d drop=%s", len(train_keep), train_drop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
