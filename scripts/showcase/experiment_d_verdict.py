"""Experiment D verdict (scope 01 plan 15): does AdamW weight decay stop the rank-32 adapter's
late fidelity decay, and does it change the best checkpoint at all?

The bars live here, in source, so they cannot move after the answer is seen. They were written
on 2026-09-06 before the run started, from the rank-32 baseline numbers in
report/does-training-longer-help-the-pooled-lora/does-training-longer-keep-improving-the-held-out-fix.md.

Reads:
  * results.json of two lambda_boundary_probe runs on the new adapter (30k and 90k) and of the
    two baseline probes (figure_r32_030050, figure_r32_090050), taking the full-window, lambda 1.0
    cell: 8-seed mean DINOv2 drift (negative = nearer the joint-prompt image) and the compose count.
  * the Frobenius norm of lora_state in the new run's 90k checkpoint against the baseline's 86.9.

Usage:
  python scripts/showcase/experiment_d_verdict.py \
      --new-30k <out-root of the 30k probe> --new-90k <out-root of the 90k probe> \
      --new-90k-ckpt <lora_step_090000.pt> [--out verdict.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SHOWCASE = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase")
BASELINE_30K = SHOWCASE / "figure_r32_030050"
BASELINE_90K = SHOWCASE / "figure_r32_090050"

# --- bars, fixed before the run ------------------------------------------------------------
# Baseline rank 32, 8-seed DINOv2 drift at lambda 1.0, full window: -0.091 at 30,050,
# -0.060 at 60,050, -0.003 at 70,050, -0.013 at 90,050. Compose count 7 of 8 at every one.
BASELINE_DRIFT_30K = -0.091
BASELINE_DRIFT_90K = -0.013
BASELINE_NORM_90K = 86.9
# The smallest gap the baseline grids told apart between neighbouring checkpoints (30k against
# 60k) is 0.03 of drift; anything inside it is read as no change.
DRIFT_MARGIN = 0.03
# Question 1 (moves the plan): at 90k, drift at or below the baseline's 60k value means the haze
# never arrived; at or above -0.03 means the decay happened anyway.
DECAY_SUPPORT_AT_90K = -0.060
DECAY_NULL_AT_90K = -0.030
# The read is only valid if the adapter still composes; below this the decay crushed it instead.
MIN_COMPOSE_COUNT = 6
# Weight decay must have bitten for question 1 to be readable at all.
NORM_MUST_DROP_TO = 0.8 * BASELINE_NORM_90K


def _cell(results_json: Path, window: str = "full", lam: str = "1.0") -> dict:
    d = json.loads(results_json.read_text())
    try:
        return d["summary"][window][lam]
    except KeyError as exc:
        raise KeyError(f"no ({window}, {lam}) cell in {results_json}: {exc}") from exc


def _count(cell: dict) -> int:
    n = cell.get("n", 8)
    return int(round(cell["compose_rate"] * n))


def _fro_norm(ckpt: Path) -> float:
    import torch
    sd = torch.load(ckpt, map_location="cpu")["lora_state"]
    return math.sqrt(sum(float((v.float() ** 2).sum()) for v in sd.values()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-30k", required=True)
    ap.add_argument("--new-90k", required=True)
    ap.add_argument("--new-90k-ckpt", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    n30 = _cell(Path(a.new_30k) / "results.json")
    n90 = _cell(Path(a.new_90k) / "results.json")
    d30, d90 = n30["mean_dino_drift"], n90["mean_dino_drift"]
    c30, c90 = _count(n30), _count(n90)
    norm90 = _fro_norm(Path(a.new_90k_ckpt))

    v: dict = {
        "drift_30k": d30, "drift_90k": d90, "compose_30k": c30, "compose_90k": c90,
        "norm_90k": norm90, "baseline": {
            "drift_30k": BASELINE_DRIFT_30K, "drift_90k": BASELINE_DRIFT_90K,
            "norm_90k": BASELINE_NORM_90K},
    }
    # Q0: did weight decay bite?
    v["weight_decay_bit"] = norm90 <= NORM_MUST_DROP_TO
    # Q1: the late decay.
    if not v["weight_decay_bit"]:
        v["decay_verdict"] = "inconclusive: weight decay too weak to read (norm did not drop 20%)"
    elif c90 < MIN_COMPOSE_COUNT:
        v["decay_verdict"] = f"inconclusive: adapter crushed (composes {c90} of 8 at 90k)"
    elif d90 <= DECAY_SUPPORT_AT_90K:
        v["decay_verdict"] = "support: the haze did not arrive by 90k"
    elif d90 >= DECAY_NULL_AT_90K:
        v["decay_verdict"] = "null: fidelity decayed anyway, norm growth was not the cause"
    else:
        v["decay_verdict"] = "inconclusive: between the bars"
    # Q2: the best checkpoint.
    delta = d30 - BASELINE_DRIFT_30K
    if c30 < 7:
        v["best_checkpoint_verdict"] = f"worse: composes {c30} of 8 at 30k against 7"
    elif delta <= -DRIFT_MARGIN:
        v["best_checkpoint_verdict"] = "better: nearer the joint-prompt image by more than 0.03"
    elif delta >= DRIFT_MARGIN:
        v["best_checkpoint_verdict"] = "worse: further from the joint-prompt image by more than 0.03"
    else:
        v["best_checkpoint_verdict"] = "no change at the best checkpoint"
    print(json.dumps(v, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(v, indent=2))


if __name__ == "__main__":
    main()
