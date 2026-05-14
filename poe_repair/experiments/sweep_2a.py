"""Sweep Method 2a (residual_prompt p*) over deployment knobs.

Purpose: the seed-42 pilot showed near-black images at λ_max=1.0, which is
either a gain problem (deployed too hot) or a magnitude mismatch between
training target and inference. This sweep separates those:

  A1  λ_max in {0.05, 0.1, 0.2, 0.3, 0.5, 1.0}    — turn the dial down
  A2  window_frac in {0.4, 0.2}                   — shorter correction window
  A3  correction_max_rel_norm in {None, 0.5}      — per-step magnitude cap

Layout: one figure per (pair, window_frac, max_rel_norm). Rows = seeds.
Columns = λ_max values. Each cell is one generated image, plus a left-most
PoE / Mono(e_J) reference column for orientation.

Usage::

    python -m poe_repair.experiments.sweep_2a \\
        --pairs "a cat|a dog" --seeds 4 42 \\
        --lambda-maxes 0.05 0.1 0.2 0.3 0.5 1.0 \\
        --window-fracs 0.4 0.2 \\
        --max-rel-norms none 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from poe_repair.composers import mono as cmp_mono
from poe_repair.composers import poe as cmp_poe
from poe_repair.composers import residual_prompt as cmp_residual_prompt
from poe_repair.config import RunConfig
from poe_repair.experiments._eval_common import cell_for, slugify
from poe_repair.figures._common import image_grid
from poe_repair.run import MethodCtx, make_ctx
from poe_repair.runtime import ensure_dir, write_json


EXP_NAME = "sweep_2a"


def _parse_cap(s: str) -> float | None:
    if s.lower() in {"none", "off", "null"}:
        return None
    return float(s)


def _config_slug(window_frac: float, max_rel_norm: float | None) -> str:
    cap_str = "off" if max_rel_norm is None else f"{max_rel_norm:.2f}".replace(".", "p")
    return f"win{window_frac:.2f}".replace(".", "p") + f"_cap{cap_str}"


def _lam_label(lam: float) -> str:
    return f"λ={lam:g}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="*", type=int, default=[4, 42])
    ap.add_argument(
        "--pairs", nargs="*", type=str, default=["a cat|a dog"],
        help='Pairs as "prompt_a|prompt_b".',
    )
    ap.add_argument(
        "--lambda-maxes", nargs="*", type=float,
        default=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
    )
    ap.add_argument(
        "--window-fracs", nargs="*", type=float, default=[0.4, 0.2],
    )
    ap.add_argument(
        "--max-rel-norms", nargs="*", type=str, default=["none"],
        help="Per-step correction cap as fraction of ||eps_PoE||. "
             "'none' disables. Pass space-separated list to sweep.",
    )
    ap.add_argument("--pstar-ckpt", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    pairs = []
    for spec in args.pairs:
        a, _, b = spec.partition("|")
        if not a or not b:
            raise ValueError(f"--pairs entry {spec!r} must be 'A|B'")
        pairs.append((a, b))

    caps = [_parse_cap(s) for s in args.max_rel_norms]

    cfg = RunConfig()
    ctx: MethodCtx = make_ctx()
    fig_dir = ensure_dir(cfg.paths.output_root / EXP_NAME / "figures")
    summaries: list[dict] = []

    for prompt_a, prompt_b in pairs:
        slug = slugify(prompt_a, prompt_b)
        for window_frac in args.window_fracs:
            for cap in caps:
                cfg_slug = _config_slug(window_frac, cap)
                cfg_exp = f"{EXP_NAME}/{cfg_slug}"
                rows: list[list[Path]] = []
                row_labels: list[str] = []
                for seed in args.seeds:
                    cell = cell_for(prompt_a, prompt_b, seed)
                    print(f"[sweep-2a] {slug} cfg={cfg_slug} seed={seed}")
                    ref_poe = cmp_poe.run(
                        cell, ctx, exp_name=cfg_exp, overwrite=args.overwrite,
                    )
                    ref_mono = cmp_mono.run(
                        cell, ctx, anchor_source="literal",
                        exp_name=cfg_exp, overwrite=args.overwrite,
                    )
                    cells: list[Path] = [ref_poe, ref_mono]
                    for lam in args.lambda_maxes:
                        cells.append(cmp_residual_prompt.run(
                            cell, ctx,
                            window_frac=float(window_frac),
                            lambda_max=float(lam),
                            pstar_ckpt=args.pstar_ckpt,
                            correction_max_rel_norm=cap,
                            exp_name=f"{cfg_exp}/lam{lam:g}".replace(".", "p"),
                            overwrite=args.overwrite,
                        ))
                    rows.append(cells)
                    row_labels.append(f"seed {seed}")
                    summaries.append({
                        "pair": [prompt_a, prompt_b],
                        "seed": seed,
                        "window_frac": float(window_frac),
                        "max_rel_norm": cap,
                        "lambda_maxes": list(args.lambda_maxes),
                        "paths": [str(p) for p in cells],
                    })

                col_labels = ["PoE", "Mono(e_J)"] + [
                    _lam_label(l) for l in args.lambda_maxes
                ]
                cap_str = "off" if cap is None else f"{cap:g}"
                fig_path = fig_dir / f"sweep_2a__{slug}__{cfg_slug}.png"
                image_grid(
                    rows, fig_path,
                    col_labels=col_labels,
                    row_labels=row_labels,
                    title=(
                        f"Method 2a sweep — {prompt_a} × {prompt_b}\n"
                        f"window_frac={window_frac:g}, "
                        f"correction_max_rel_norm={cap_str}"
                    ),
                    panel_size=2.0,
                )
                print(f"[sweep-2a] wrote {fig_path}")

    write_json(
        ensure_dir(cfg.paths.output_root / EXP_NAME) / "summary.json",
        {
            "exp": EXP_NAME,
            "pairs": [list(p) for p in pairs],
            "seeds": args.seeds,
            "lambda_maxes": list(args.lambda_maxes),
            "window_fracs": list(args.window_fracs),
            "max_rel_norms": [None if c is None else float(c) for c in caps],
            "cells": summaries,
        },
    )


if __name__ == "__main__":
    main()
