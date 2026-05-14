"""E_teacher_residual — partial PoE→Mono blend on cat × dog × seeds.

For each seed, render six columns:

    PoE | Mono (e_J) | TR λ=0.25 | TR λ=0.50 | TR λ=1.00 (constant) | TR λ=1.00 early-only

Where TR is the teacher-residual sampler from
``methods/_sampling.py:run_teacher_residual``::

    ε_final = ε̃_PoE + λ_t · (ε̃_Mono − ε̃_PoE)

This experiment is the data-generator and the published ceiling for the
PoE-repair stack. λ=1 should approach Mono on every seed; λ=0 should equal
PoE. Intermediate λ traces the partial-correction curve.

Optional ``--regression`` flag runs a sanity check at λ=0 and λ=1 only,
comparing the resulting image to the cached PoE/Mono baselines and
reporting pixel L∞ / mean-L2 differences. Pass criterion: L∞ ≤ 6/255 and
mean-L2 ≤ 1/255 — anything looser than that suggests an algebra bug.

Outputs:
    outputs/e_teacher_residual/pairs/<slug>/seed_<n>/<method>/<method>.png
    outputs/e_teacher_residual/figures/aggregate__<slug>.png
    outputs/e_teacher_residual/summary.json
    outputs/e_teacher_residual/regression.json   (only with --regression)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from poe_repair.composers import mono as cmp_mono
from poe_repair.composers import poe as cmp_poe
from poe_repair.composers import teacher_residual as cmp_tr
from poe_repair.config import RunConfig
from poe_repair.experiments._eval_common import HEADLINE_PAIR, cell_for, slugify
from poe_repair.figures._common import image_grid
from poe_repair.run import MethodCtx, make_ctx
from poe_repair.runtime import ensure_dir, write_json


EXP_NAME = "e_teacher_residual"


def _load_image_array(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _diff_metrics(p1: Path, p2: Path) -> dict[str, float]:
    a = _load_image_array(p1)
    b = _load_image_array(p2)
    if a.shape != b.shape:
        return {"linf": float("nan"), "mean_l2": float("nan"), "shape_mismatch": True}
    diff = a - b
    linf = float(np.max(np.abs(diff)))
    mean_l2 = float(np.sqrt(np.mean(diff ** 2)))
    return {"linf": linf, "mean_l2": mean_l2}


def main() -> None:
    from poe_repair.experiments import _assert_env_ok
    _assert_env_ok()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1, 2, 3, 4])
    ap.add_argument(
        "--pair", type=str, default=None,
        help='Pair as "prompt_a|prompt_b". Default = HEADLINE_PAIR (cat × dog).',
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--regression", action="store_true",
        help="Run λ=0 and λ=1 sanity checks against cached PoE/Mono images.",
    )
    ap.add_argument(
        "--save-residuals", action="store_true",
        help="Save per-step Δ_t tensors for the λ=1 constant run only.",
    )
    args = ap.parse_args()

    if args.pair:
        a, _, b = args.pair.partition("|")
        if not a or not b:
            raise ValueError(f"--pair must be 'A|B', got {args.pair!r}")
        prompt_a, prompt_b = a, b
    else:
        prompt_a, prompt_b = HEADLINE_PAIR

    cfg = RunConfig()
    ctx: MethodCtx = make_ctx()
    fig_dir = ensure_dir(cfg.paths.output_root / EXP_NAME / "figures")
    slug = slugify(prompt_a, prompt_b)

    columns: list[tuple[str, dict]] = [
        ("PoE", {"kind": "poe"}),
        ("Mono (e_J)", {"kind": "mono"}),
        ("TR λ=0.25", {"kind": "tr", "lambda_schedule": "constant", "lambda_max": 0.25}),
        ("TR λ=0.50", {"kind": "tr", "lambda_schedule": "constant", "lambda_max": 0.50}),
        (
            "TR λ=1.00",
            {
                "kind": "tr", "lambda_schedule": "constant", "lambda_max": 1.0,
                "save_residuals": bool(args.save_residuals),
            },
        ),
        ("TR λ=1.00 early", {"kind": "tr", "lambda_schedule": "early_only", "lambda_max": 1.0}),
    ]

    rows: list[list[Path]] = []
    cell_records: list[dict] = []
    for seed in args.seeds:
        cell = cell_for(prompt_a, prompt_b, seed)
        print(f"[{EXP_NAME}] {slug} seed={seed}")
        row: list[Path] = []
        for label, spec in columns:
            kind = spec["kind"]
            if kind == "poe":
                p = cmp_poe.run(cell, ctx, exp_name=EXP_NAME, overwrite=args.overwrite)
            elif kind == "mono":
                p = cmp_mono.run(
                    cell, ctx, anchor_source="literal",
                    exp_name=EXP_NAME, overwrite=args.overwrite,
                )
            elif kind == "tr":
                p = cmp_tr.run(
                    cell, ctx,
                    lambda_schedule=spec["lambda_schedule"],
                    lambda_max=float(spec["lambda_max"]),
                    save_residuals=bool(spec.get("save_residuals", False)),
                    exp_name=EXP_NAME, overwrite=args.overwrite,
                )
            else:
                raise ValueError(kind)
            row.append(p)
        rows.append(row)
        cell_records.append({
            "seed": seed,
            "columns": [label for label, _ in columns],
            "paths": [str(p) for p in row],
        })

    agg_path = image_grid(
        rows, fig_dir / f"aggregate__{slug}.png",
        col_labels=[label for label, _ in columns],
        row_labels=[f"seed {s}" for s in args.seeds],
        title=(
            f"{EXP_NAME} — {prompt_a} × {prompt_b}\n"
            "PoE / Mono / TR λ blend"
        ),
        panel_size=2.2,
    )
    print(f"[{EXP_NAME}] wrote {agg_path}")

    write_json(
        ensure_dir(cfg.paths.output_root / EXP_NAME) / "summary.json",
        {
            "exp": EXP_NAME,
            "pair": [prompt_a, prompt_b],
            "seeds": args.seeds,
            "columns": [label for label, _ in columns],
            "cells": cell_records,
        },
    )

    if args.regression:
        print(f"[{EXP_NAME}] running λ=0 and λ=1 regression checks")
        regression_rows: list[dict] = []
        for seed in args.seeds:
            cell = cell_for(prompt_a, prompt_b, seed)
            poe_path = cmp_poe.run(
                cell, ctx, exp_name=EXP_NAME, overwrite=False,
            )
            mono_path = cmp_mono.run(
                cell, ctx, anchor_source="literal",
                exp_name=EXP_NAME, overwrite=False,
            )
            tr_zero = cmp_tr.run(
                cell, ctx,
                lambda_schedule="constant", lambda_max=0.0,
                exp_name=EXP_NAME, overwrite=args.overwrite,
            )
            tr_one = cmp_tr.run(
                cell, ctx,
                lambda_schedule="constant", lambda_max=1.0,
                exp_name=EXP_NAME, overwrite=False,
            )
            d_zero = _diff_metrics(tr_zero, poe_path)
            d_one = _diff_metrics(tr_one, mono_path)
            tol_linf = 6.0 / 255.0
            tol_mean = 1.0 / 255.0
            zero_pass = (
                d_zero.get("linf", float("inf")) <= tol_linf
                and d_zero.get("mean_l2", float("inf")) <= tol_mean
            )
            one_pass = (
                d_one.get("linf", float("inf")) <= tol_linf
                and d_one.get("mean_l2", float("inf")) <= tol_mean
            )
            row_rec = {
                "seed": seed,
                "lambda_zero_vs_poe": d_zero,
                "lambda_zero_pass": zero_pass,
                "lambda_one_vs_mono": d_one,
                "lambda_one_pass": one_pass,
                "tol_linf": tol_linf,
                "tol_mean_l2": tol_mean,
            }
            regression_rows.append(row_rec)
            print(
                f"[regression seed={seed}] "
                f"λ=0 vs PoE  linf={d_zero['linf']:.4f} "
                f"mean_l2={d_zero['mean_l2']:.4f} "
                f"{'PASS' if zero_pass else 'FAIL'}  |  "
                f"λ=1 vs Mono linf={d_one['linf']:.4f} "
                f"mean_l2={d_one['mean_l2']:.4f} "
                f"{'PASS' if one_pass else 'FAIL'}"
            )
        all_pass = all(r["lambda_zero_pass"] and r["lambda_one_pass"] for r in regression_rows)
        write_json(
            cfg.paths.output_root / EXP_NAME / "regression.json",
            {
                "exp": EXP_NAME,
                "pair": [prompt_a, prompt_b],
                "all_pass": all_pass,
                "rows": regression_rows,
            },
        )
        print(f"[{EXP_NAME}] regression: {'ALL PASS' if all_pass else 'SOME FAIL'}")


if __name__ == "__main__":
    main()
