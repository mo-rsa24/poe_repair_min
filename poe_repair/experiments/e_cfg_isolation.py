"""E-CFG-isolation — does real CO3's seed-42 failure come from CFG regime?

Real CO3 ships with ``guidance_scale=0.8`` and ``use_cfgpp=True`` while
our Mono baseline runs ``guidance_scale=7.5`` with standard CFG. So when
real CO3 fails on seed 42, we cannot tell whether it's:

  (a) the CO3 step-0 algorithm itself, or
  (b) the under-conditioned CFG regime CO3 uses.

This experiment runs real CO3 on a small grid of (guidance_scale, use_cfgpp)
overrides while keeping everything else at upstream defaults. If seed 42
recovers under standard CFG, the algorithm is fine and the regime was the
killer; if it still fails, the algorithm is over-perturbing on seeds where
Mono already had a good basin.

Default grid:

  - CO3 default:  guidance=0.8,  cfgpp=True
  - Mono CFG:     guidance=7.5,  cfgpp=False
  - Mid:          guidance=5.0,  cfgpp=False

Run:
    PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
    CUDA_VISIBLE_DEVICES=1 $PY -m poe_repair.experiments.e_cfg_isolation \\
        --seeds 4 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

from poe_repair.composers import co3 as cmp_co3
from poe_repair.composers import mono as cmp_mono
from poe_repair.config import RunConfig
from poe_repair.experiments._eval_common import HEADLINE_PAIR, cell_for, slugify
from poe_repair.figures._common import image_grid
from poe_repair.run import MethodCtx, make_ctx
from poe_repair.runtime import ensure_dir, write_json


EXP_NAME = "e_cfg_isolation"

# (label, overrides) — each tuple is one column in the output grid.
DEFAULT_GRID: list[tuple[str, dict]] = [
    ("co3_default__g08_cfgpp",  {"guidance_scale": 0.8, "use_cfgpp": True}),
    ("co3_mid__g50_stdcfg",     {"guidance_scale": 5.0, "use_cfgpp": False}),
    ("co3_monocfg__g75_stdcfg", {"guidance_scale": 7.5, "use_cfgpp": False}),
]


def main() -> None:
    from poe_repair.experiments import _assert_env_ok
    _assert_env_ok()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="*", type=int, default=[4, 42])
    ap.add_argument("--prompts", nargs=2, type=str, default=list(HEADLINE_PAIR),
                    metavar=("PROMPT_A", "PROMPT_B"))
    ap.add_argument("--include-mono", action="store_true",
                    help="Add a Mono baseline column for visual reference.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    prompt_a, prompt_b = args.prompts

    cfg = RunConfig()
    ctx: MethodCtx = make_ctx()
    fig_dir = ensure_dir(cfg.paths.output_root / EXP_NAME / "figures")
    summary_dir = ensure_dir(cfg.paths.output_root / EXP_NAME)
    slug = slugify(prompt_a, prompt_b)

    rows: list[list[Path]] = []
    summary_cells: list[dict] = []
    col_labels: list[str] = []
    if args.include_mono:
        col_labels.append("Mono (g=7.5, std CFG)")
    col_labels += [
        f"CO3 g={ov['guidance_scale']:.1f} cfgpp={ov['use_cfgpp']}"
        for _, ov in DEFAULT_GRID
    ]

    for seed in args.seeds:
        cell = cell_for(prompt_a, prompt_b, seed)
        row: list[Path] = []
        if args.include_mono:
            print(f"[cfg-iso] seed={seed} method=mono")
            row.append(cmp_mono.run(
                cell, ctx, anchor_source="literal",
                exp_name=EXP_NAME, overwrite=args.overwrite,
            ))
        per_variant: list[dict] = []
        for label, ov in DEFAULT_GRID:
            print(f"[cfg-iso] seed={seed} method=co3/{label}  overrides={ov}")
            path = cmp_co3.run(
                cell, ctx, anchor_source="literal",
                co3_overrides=ov,
                method_name_suffix=f"__{label}",
                exp_name=EXP_NAME, overwrite=args.overwrite,
            )
            row.append(path)
            per_variant.append({"label": label, "overrides": ov, "path": str(path)})
        rows.append(row)
        summary_cells.append({
            "seed": seed,
            "pair": [prompt_a, prompt_b],
            "include_mono": bool(args.include_mono),
            "variants": per_variant,
        })

    endpoint_path = image_grid(
        rows, fig_dir / f"endpoint__{slug}.png",
        col_labels=col_labels,
        row_labels=[f"seed {s}" for s in args.seeds],
        title=(
            f"E-CFG-isolation — real CO3 across CFG regimes — {prompt_a} × {prompt_b}\n"
            "Falsifies the CFG-confound hypothesis if seed-42 recovers under "
            "standard CFG."
        ),
        panel_size=2.6,
    )
    write_json(summary_dir / "summary.json", {
        "exp": EXP_NAME, "seeds": args.seeds, "pair": [prompt_a, prompt_b],
        "grid": [{"label": l, "overrides": ov} for l, ov in DEFAULT_GRID],
        "include_mono": bool(args.include_mono),
        "endpoint_path": str(endpoint_path),
        "cells": summary_cells,
    })
    print(f"[cfg-iso] wrote {endpoint_path}")


if __name__ == "__main__":
    main()
