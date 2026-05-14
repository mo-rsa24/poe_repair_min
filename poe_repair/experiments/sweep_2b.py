"""Sweep Method 2b (direct_eps student) over deployment knobs.

Purpose: the seed-42 pilot showed a chimera at λ_max=1.0 — the student is
under-correcting. This sweep separates "needs more gain" from "needs a
different base":

  B1  λ_max in {1.0, 1.5, 2.0, 3.0}            — turn the dial up
  B2  window_frac in {0.4, 0.6}                — let the student keep firing
  B3  --on-schedm2 flag                        — base on sched-M2(ê_J)
                                                 instead of plain PoE

With ``--on-schedm2``, the trajectory base is
``(1 − β_t) · ε̃_PoE + β_t · ε̃_J`` (sched-M2 with synthesised ê_J), and
the student adds ``λ_t · δ_θ`` on top. This way the student only has to
close the gap from the joint basin rather than drag a chimera-bound
trajectory all the way over.

Layout: one figure per (pair, window_frac). Rows = seeds. Columns =
λ_max values, plus PoE / Mono(e_J) reference columns. With
``--on-schedm2`` an extra reference column for plain sched-M2(ê_J) is
also included.

Usage::

    # B1 + B2: gain + window sweep on plain PoE base
    python -m poe_repair.experiments.sweep_2b \\
        --pairs "a cat|a dog" --seeds 4 42 \\
        --lambda-maxes 1.0 1.5 2.0 3.0 \\
        --window-fracs 0.4 0.6

    # B3: stack student on sched-M2(ê_J) base
    python -m poe_repair.experiments.sweep_2b --on-schedm2 \\
        --pairs "a cat|a dog" --seeds 4 42 \\
        --lambda-maxes 0.25 0.5 1.0 1.5 \\
        --window-fracs 0.4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from poe_repair.composers import direct_eps as cmp_direct_eps
from poe_repair.composers import direct_eps_on_schedm2 as cmp_stack
from poe_repair.composers import mono as cmp_mono
from poe_repair.composers import poe as cmp_poe
from poe_repair.composers import sched_m2 as cmp_schedm2
from poe_repair.config import RunConfig
from poe_repair.experiments._eval_common import cell_for, slugify
from poe_repair.figures._common import image_grid
from poe_repair.run import MethodCtx, make_ctx
from poe_repair.runtime import ensure_dir, write_json


EXP_NAME_PLAIN = "sweep_2b"
EXP_NAME_STACK = "sweep_2b_on_schedm2"


def _config_slug(window_frac: float) -> str:
    return f"win{window_frac:.2f}".replace(".", "p")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="*", type=int, default=[4, 42])
    ap.add_argument(
        "--pairs", nargs="*", type=str, default=["a cat|a dog"],
        help='Pairs as "prompt_a|prompt_b".',
    )
    ap.add_argument(
        "--lambda-maxes", nargs="*", type=float,
        default=[1.0, 1.5, 2.0, 3.0],
    )
    ap.add_argument(
        "--window-fracs", nargs="*", type=float, default=[0.4, 0.6],
    )
    ap.add_argument(
        "--on-schedm2", action="store_true",
        help="Stack the student on a sched-M2(ê_J) base instead of plain PoE.",
    )
    ap.add_argument(
        "--beta-window-frac", type=float, default=0.4,
        help="(only with --on-schedm2) sched-M2 β window for the base.",
    )
    ap.add_argument(
        "--beta-max", type=float, default=1.0,
        help="(only with --on-schedm2) sched-M2 β peak for the base.",
    )
    ap.add_argument("--student-ckpt", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    pairs = []
    for spec in args.pairs:
        a, _, b = spec.partition("|")
        if not a or not b:
            raise ValueError(f"--pairs entry {spec!r} must be 'A|B'")
        pairs.append((a, b))

    exp_root = EXP_NAME_STACK if args.on_schedm2 else EXP_NAME_PLAIN

    cfg = RunConfig()
    ctx: MethodCtx = make_ctx()
    fig_dir = ensure_dir(cfg.paths.output_root / exp_root / "figures")
    summaries: list[dict] = []

    for prompt_a, prompt_b in pairs:
        slug = slugify(prompt_a, prompt_b)
        for window_frac in args.window_fracs:
            cfg_slug = _config_slug(window_frac)
            cfg_exp = f"{exp_root}/{cfg_slug}"
            rows: list[list[Path]] = []
            row_labels: list[str] = []
            for seed in args.seeds:
                cell = cell_for(prompt_a, prompt_b, seed)
                print(f"[sweep-2b] {slug} cfg={cfg_slug} seed={seed} "
                      f"on_schedm2={args.on_schedm2}")
                ref_poe = cmp_poe.run(
                    cell, ctx, exp_name=cfg_exp, overwrite=args.overwrite,
                )
                ref_mono = cmp_mono.run(
                    cell, ctx, anchor_source="literal",
                    exp_name=cfg_exp, overwrite=args.overwrite,
                )
                cells: list[Path] = [ref_poe, ref_mono]
                if args.on_schedm2:
                    cells.append(cmp_schedm2.run(
                        cell, ctx, anchor_source="synth",
                        exp_name=cfg_exp, overwrite=args.overwrite,
                    ))
                for lam in args.lambda_maxes:
                    lam_exp = f"{cfg_exp}/lam{lam:g}".replace(".", "p")
                    if args.on_schedm2:
                        out_path = cmp_stack.run(
                            cell, ctx,
                            beta_window_frac=float(args.beta_window_frac),
                            beta_max=float(args.beta_max),
                            student_window_frac=float(window_frac),
                            lambda_max=float(lam),
                            student_ckpt=args.student_ckpt,
                            exp_name=lam_exp,
                            overwrite=args.overwrite,
                        )
                    else:
                        out_path = cmp_direct_eps.run(
                            cell, ctx,
                            window_frac=float(window_frac),
                            lambda_max=float(lam),
                            student_ckpt=args.student_ckpt,
                            exp_name=lam_exp,
                            overwrite=args.overwrite,
                        )
                    cells.append(out_path)
                rows.append(cells)
                row_labels.append(f"seed {seed}")
                summaries.append({
                    "pair": [prompt_a, prompt_b],
                    "seed": seed,
                    "window_frac": float(window_frac),
                    "on_schedm2": bool(args.on_schedm2),
                    "lambda_maxes": list(args.lambda_maxes),
                    "paths": [str(p) for p in cells],
                })

            col_labels = ["PoE", "Mono(e_J)"]
            if args.on_schedm2:
                col_labels.append("sched-M2(ê_J)")
            col_labels += [f"λ={l:g}" for l in args.lambda_maxes]
            mode = "on schedm2(ê_J)" if args.on_schedm2 else "on plain PoE"
            fig_path = fig_dir / f"sweep_2b__{slug}__{cfg_slug}.png"
            image_grid(
                rows, fig_path,
                col_labels=col_labels,
                row_labels=row_labels,
                title=(
                    f"Method 2b sweep ({mode}) — {prompt_a} × {prompt_b}\n"
                    f"student_window_frac={window_frac:g}"
                ),
                panel_size=2.0,
            )
            print(f"[sweep-2b] wrote {fig_path}")

    write_json(
        ensure_dir(cfg.paths.output_root / exp_root) / "summary.json",
        {
            "exp": exp_root,
            "pairs": [list(p) for p in pairs],
            "seeds": args.seeds,
            "lambda_maxes": list(args.lambda_maxes),
            "window_fracs": list(args.window_fracs),
            "on_schedm2": bool(args.on_schedm2),
            "beta_window_frac": float(args.beta_window_frac),
            "beta_max": float(args.beta_max),
            "cells": summaries,
        },
    )


if __name__ == "__main__":
    main()
