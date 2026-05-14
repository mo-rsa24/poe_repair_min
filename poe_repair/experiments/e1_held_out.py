"""E1 — Headline result. PoE repair via sched-M2 + ê_J on held-out pairs.

For each (pair, seed) in held-out pairs × seeds, render five columns:

    PoE | Mono (literal e_J) | sched-M2 (literal e_J) | sched-M2 (ê_J) | CO3

Reading guide:
- **PoE**           : failure baseline. Hybrid / chimera output expected.
- **Mono (e_J)**    : oracle ceiling. Uses the literal joint prompt.
- **sched-M2 (e_J)**: oracle of our method (literal joint mixed with PoE).
- **sched-M2 (ê_J)**: **the deployed method**. Same sampler, but ê_J supplied
                      by the synthesiser from (e_A, e_B).
- **CO3**           : alternative published baseline (Dutta et al.).

Headline claim: ``sched-M2 (ê_J)`` approaches ``Mono (e_J)`` quality on held-out
pairs the synthesiser has not seen, demonstrating that PoE composition can be
repaired without a literal joint prompt.

Default coverage:
- 19 held-out pairs (``COLLISION_PAIRS + COOPERATIVE_PAIRS``).
- 8 seeds: ``[42, 1, 2, 3, 4, 5, 6, 7]``.

Outputs:
    outputs/e1_held_out/pairs/<slug>/seed_<n>/<method>/<method>.png
    outputs/e1_held_out/figures/aggregate__<slug>.png
    outputs/e1_held_out/summary.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from poe_repair.composers import co3 as cmp_co3
from poe_repair.composers import mono as cmp_mono
from poe_repair.composers import poe as cmp_poe
from poe_repair.composers import sched_m2 as cmp_schedm2
from poe_repair.config import RunConfig
from poe_repair.experiments._eval_common import HELD_OUT_PAIRS, cell_for, slugify
from poe_repair.figures._common import image_grid
from poe_repair.run import MethodCtx, make_ctx
from poe_repair.runtime import ensure_dir, write_json


EXP_NAME = "e1_held_out"

COL_LABELS = [
    "PoE",
    "Mono (e_J)",
    "sched-M2 (e_J)",
    "sched-M2 (ê_J)",
    "CO3",
]


def main() -> None:
    from poe_repair.experiments import _assert_env_ok
    _assert_env_ok()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="*", type=int,
                    default=[42, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument(
        "--pairs", nargs="*", type=str, default=None,
        help='Held-out pairs as "prompt_a|prompt_b". Default = ALL_HOLDOUT_PAIRS '
             '(19 pairs from holdout_pairs.py).',
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    seeds = args.seeds

    if args.pairs:
        pairs = []
        for spec in args.pairs:
            a, _, b = spec.partition("|")
            if not a or not b:
                raise ValueError(f"--pairs entry {spec!r} must be 'A|B'")
            pairs.append((a, b))
    else:
        pairs = list(HELD_OUT_PAIRS)

    cfg = RunConfig()
    ctx: MethodCtx = make_ctx()
    fig_dir = ensure_dir(cfg.paths.output_root / EXP_NAME / "figures")

    summary_cells: list[dict] = []
    for prompt_a, prompt_b in pairs:
        slug = slugify(prompt_a, prompt_b)
        rows: list[list[Path]] = []
        for seed in seeds:
            cell = cell_for(prompt_a, prompt_b, seed)
            print(f"[E1] {slug} seed={seed}")
            row = [
                cmp_poe.run(
                    cell, ctx, exp_name=EXP_NAME, overwrite=args.overwrite,
                ),
                cmp_mono.run(
                    cell, ctx, anchor_source="literal",
                    exp_name=EXP_NAME, overwrite=args.overwrite,
                ),
                cmp_schedm2.run(
                    cell, ctx, anchor_source="literal",
                    exp_name=EXP_NAME, overwrite=args.overwrite,
                ),
                cmp_schedm2.run(
                    cell, ctx, anchor_source="synth",
                    exp_name=EXP_NAME, overwrite=args.overwrite,
                ),
                cmp_co3.run(
                    cell, ctx, anchor_source="literal",
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
            row_labels=[f"seed {s}" for s in seeds],
            title=(
                f"E1 held-out — {prompt_a} × {prompt_b}\n"
                "PoE / Mono(e_J) / sched-M2(e_J) / sched-M2(ê_J) / CO3"
            ),
            panel_size=2.2,
        )
        print(f"[E1] wrote {agg_path}")

    write_json(
        ensure_dir(cfg.paths.output_root / EXP_NAME) / "summary.json",
        {
            "exp": EXP_NAME,
            "seeds": seeds,
            "pairs": [list(p) for p in pairs],
            "columns": list(COL_LABELS),
            "cells": summary_cells,
        },
    )


if __name__ == "__main__":
    main()
