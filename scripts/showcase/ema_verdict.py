#!/usr/bin/env python
"""Plan 20 verdict (01-showcase-the-trained-lora): does an averaged copy of the adapter's weights
render nearer the joint-prompt image than the raw weights at the same training step?

Reads probe folders written by scripts/showcase/lambda_boundary_probe.py (--windows full
--lambdas 1.0), one per (checkpoint, weight key), and judges the EMA against the raw weights of
the same run at the same step. The bars are the constants below; they were fixed before the run
produced a checkpoint.

    cell: 8-seed mean DINOv2 drift (negative = nearer the joint-prompt image) and the compose
    count, from summary.full."1.0" of each folder's results.json.

Usage:
    python scripts/showcase/ema_verdict.py \
        --raw-30k <folder> --ema-30k <folder> [--raw-40k <folder> --ema-40k <folder>] \
        [--d-30k <experiment D's figure_r32_wd0.1_030000>] --out <verdict.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# The smallest drift gap the baseline grids told apart between neighbouring checkpoints
# (experiment D's DRIFT_MARGIN); a difference inside it is "the same".
DRIFT_MARGIN = 0.03
# The EMA may lose this many composing seeds against the raw weights and still support.
MAX_COMPOSE_LOSS = 1
# The raw weights of this run should replicate experiment D's trajectory at the same step
# (same flags, same torch seed); a gap beyond this margin is batch-shape nondeterminism and is
# reported, not judged.
REPLICATE_MARGIN = 0.03
LAMBDA_KEY = "1.0"
WINDOW = "full"


def read_cell(folder: Path) -> dict:
    r = json.loads((folder / "results.json").read_text())
    cell = r["summary"][WINDOW][LAMBDA_KEY]
    return {"folder": str(folder), "checkpoint": r.get("checkpoint"), "checkpoint_step": r.get("checkpoint_step"),
            "n": int(cell["n"]), "compose_n": int(round(cell["compose_rate"] * cell["n"])),
            "drift": float(cell["mean_dino_drift"])}


def judge(ema: dict, raw: dict) -> str:
    gap = ema["drift"] - raw["drift"]     # negative = the EMA is nearer the joint-prompt image
    loss = raw["compose_n"] - ema["compose_n"]
    if loss > MAX_COMPOSE_LOSS:
        return "null: the EMA loses composing seeds"
    if gap <= -DRIFT_MARGIN:
        return "support: the EMA renders nearer the joint-prompt image"
    if abs(gap) < DRIFT_MARGIN:
        return "null: the EMA renders the same as the raw weights"
    return "null: the EMA renders further from the joint-prompt image"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-30k", required=True); ap.add_argument("--ema-30k", required=True)
    ap.add_argument("--raw-40k"); ap.add_argument("--ema-40k")
    ap.add_argument("--d-30k", help="experiment D's 30k probe folder, for the replicate check")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = {"constants": {"DRIFT_MARGIN": DRIFT_MARGIN, "MAX_COMPOSE_LOSS": MAX_COMPOSE_LOSS,
                         "REPLICATE_MARGIN": REPLICATE_MARGIN, "window": WINDOW, "lambda": LAMBDA_KEY},
           "cells": {}, "verdicts": {}}
    pairs = [("30k", a.raw_30k, a.ema_30k)]
    if a.raw_40k and a.ema_40k:
        pairs.append(("40k", a.raw_40k, a.ema_40k))
    for tag, rawp, emap in pairs:
        raw, ema = read_cell(Path(rawp)), read_cell(Path(emap))
        out["cells"][f"raw_{tag}"], out["cells"][f"ema_{tag}"] = raw, ema
        out["verdicts"][tag] = judge(ema, raw)
    if a.d_30k:
        d = read_cell(Path(a.d_30k)); raw = out["cells"]["raw_30k"]
        gap = raw["drift"] - d["drift"]
        out["cells"]["experiment_d_30k"] = d
        out["replicate_check"] = {"raw_minus_d_drift": gap, "raw_compose_n": raw["compose_n"], "d_compose_n": d["compose_n"],
                                  "same_trajectory": abs(gap) < REPLICATE_MARGIN and abs(raw["compose_n"] - d["compose_n"]) <= 1}
    v30 = out["verdicts"]["30k"]
    out["plan_verdict"] = ("support" if v30.startswith("support")
                           else "null" if all(v.startswith("null") for v in out["verdicts"].values()) else "inconclusive")
    Path(a.out).write_text(json.dumps(out, indent=2))
    for tag, v in out["verdicts"].items():
        c = out["cells"]
        print(f"{tag}: raw drift {c[f'raw_{tag}']['drift']:+.3f} ({c[f'raw_{tag}']['compose_n']} of 8), "
              f"ema drift {c[f'ema_{tag}']['drift']:+.3f} ({c[f'ema_{tag}']['compose_n']} of 8) -> {v}")
    if "replicate_check" in out:
        print("replicate check against experiment D at 30k:", json.dumps(out["replicate_check"]))
    print("plan verdict:", out["plan_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
