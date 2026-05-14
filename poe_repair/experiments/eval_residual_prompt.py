"""Side-by-side eval: PoE | Mono | sched-M2(ê_J) | 2a (p*) | 2b (direct-eps).

Renders a 5-column grid for each (pair, seed). Default coverage is the
canonical held-out diagnostic — cat-dog, seeds 4 and 42 — but pairs and
seeds are CLI-overridable.

Method 2a (residual_prompt) — soft-prompt distillation against r_t. The
trained p* is fed as an extra UNet branch; its raw output is added under
λ_t. Checkpoint defaults to ``residual_mlp_pstar_v2/best.pt`` or
``$POE_REPAIR_PSTAR_CKPT``.

Method 2b (direct_eps) — small CNN trained against the same target. No
extra UNet branch; the CNN takes ``(x_t, t, pool_a, pool_b, pool_∅)``
and outputs the residual directly. Checkpoint defaults to
``direct_eps_v1/best.pt`` or ``$POE_REPAIR_DIRECT_EPS_CKPT``.

Usage:
    python -m poe_repair.experiments.eval_residual_prompt
    python -m poe_repair.experiments.eval_residual_prompt --pairs "a cat|a dog" --seeds 4 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

from poe_repair.composers import direct_eps as cmp_direct_eps
from poe_repair.composers import mono as cmp_mono
from poe_repair.composers import poe as cmp_poe
from poe_repair.composers import residual_prompt as cmp_residual_prompt
from poe_repair.composers import sched_m2 as cmp_schedm2
from poe_repair.config import RunConfig
from poe_repair.experiments._eval_common import cell_for, slugify
from poe_repair.figures._common import image_grid
from poe_repair.run import MethodCtx, make_ctx
from poe_repair.runtime import ensure_dir, write_json


EXP_NAME = "eval_residual_prompt"
COL_LABELS = [
    "PoE",
    "Mono (e_J)",
    "sched-M2 (ê_J)",
    "2a: residual_prompt (p*)",
    "2b: direct_eps (student)",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="*", type=int, default=[4, 42])
    ap.add_argument(
        "--pairs", nargs="*", type=str, default=["a cat|a dog"],
        help='Pairs as "prompt_a|prompt_b".',
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--pstar-ckpt", default=None,
        help="Path to Method 2a checkpoint. Defaults to "
             "checkpoints/residual_prompt/residual_mlp_pstar_v2/best.pt or "
             "$POE_REPAIR_PSTAR_CKPT.",
    )
    ap.add_argument(
        "--direct-eps-ckpt", default=None,
        help="Path to Method 2b checkpoint. Defaults to "
             "checkpoints/students/direct_eps_v1/best.pt or "
             "$POE_REPAIR_DIRECT_EPS_CKPT.",
    )
    ap.add_argument("--window-frac", type=float, default=0.4)
    ap.add_argument("--lambda-max", type=float, default=1.0)
    args = ap.parse_args()

    pairs = []
    for spec in args.pairs:
        a, _, b = spec.partition("|")
        if not a or not b:
            raise ValueError(f"--pairs entry {spec!r} must be 'A|B'")
        pairs.append((a, b))

    cfg = RunConfig()
    ctx: MethodCtx = make_ctx()
    fig_dir = ensure_dir(cfg.paths.output_root / EXP_NAME / "figures")
    summary_cells: list[dict] = []

    for prompt_a, prompt_b in pairs:
        slug = slugify(prompt_a, prompt_b)
        rows: list[list[Path]] = []
        for seed in args.seeds:
            cell = cell_for(prompt_a, prompt_b, seed)
            print(f"[eval-2ab] {slug} seed={seed}")
            row = [
                cmp_poe.run(cell, ctx, exp_name=EXP_NAME, overwrite=args.overwrite),
                cmp_mono.run(
                    cell, ctx, anchor_source="literal",
                    exp_name=EXP_NAME, overwrite=args.overwrite,
                ),
                cmp_schedm2.run(
                    cell, ctx, anchor_source="synth",
                    exp_name=EXP_NAME, overwrite=args.overwrite,
                ),
                cmp_residual_prompt.run(
                    cell, ctx,
                    window_frac=args.window_frac,
                    lambda_max=args.lambda_max,
                    pstar_ckpt=args.pstar_ckpt,
                    exp_name=EXP_NAME, overwrite=args.overwrite,
                ),
                cmp_direct_eps.run(
                    cell, ctx,
                    window_frac=args.window_frac,
                    lambda_max=args.lambda_max,
                    student_ckpt=args.direct_eps_ckpt,
                    exp_name=EXP_NAME, overwrite=args.overwrite,
                ),
            ]
            rows.append(row)
            summary_cells.append({
                "pair": [prompt_a, prompt_b], "seed": seed,
                "columns": list(COL_LABELS),
                "paths": [str(p) for p in row],
            })

        agg_path = image_grid(
            rows, fig_dir / f"aggregate__{slug}.png",
            col_labels=list(COL_LABELS),
            row_labels=[f"seed {s}" for s in args.seeds],
            title=(
                f"Method 2a vs 2b — {prompt_a} × {prompt_b}\n"
                "PoE / Mono(e_J) / sched-M2(ê_J) / residual_prompt(p*) / direct_eps(student)"
            ),
            panel_size=2.2,
        )
        print(f"[eval-2ab] wrote {agg_path}")

    write_json(
        ensure_dir(cfg.paths.output_root / EXP_NAME) / "summary.json",
        {
            "exp": EXP_NAME,
            "seeds": args.seeds,
            "pairs": [list(p) for p in pairs],
            "columns": list(COL_LABELS),
            "window_frac": args.window_frac,
            "lambda_max": args.lambda_max,
            "cells": summary_cells,
        },
    )


if __name__ == "__main__":
    main()
