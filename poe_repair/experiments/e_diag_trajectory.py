"""E-diag — per-step decoded x̂_0 + cat/dog cross-attention diagnostic.

Pure instrumentation. We diagnose *why* seeds 4 and 42 behave so differently
on cat × dog before throwing more solutions at the problem:

    seed 4 :  Mono fails  /  CO3 works
    seed 42:  Mono works  /  CO3 fails

For each (mode, seed) we run the trajectory and record at every step:
  - decoded Tweedie x̂_0
  - cat-token + dog-token cross-attention (J branch, 16²)
  - ‖ε_t‖ and ‖x̂_0(t) − x̂_0(t−1)‖

Modes (all run through ``run_diagnostic_trajectory`` so the instrumentation
is identical across them):

  - ``mono``           : pure Mono CFG with e_J
  - ``mono_co3step0``  : Mono CFG + CO3 step-0 contrastive latent correction
                         (a clean isolation of CO3's pre-step contribution
                         in our pipeline; not the actual CO3 pipeline,
                         which uses guidance=0.8 + CFG++)

For visual comparison against the *actual* CO3 pipeline (which loads its
own SDXL with CFG++), pass ``--include-co3`` and we additionally call
``cmp_co3.run`` and add its endpoint PNG to the final comparison grid —
endpoint only, no per-step trajectory there.

Outputs:
    outputs/e_diag_trajectory/figures/strip__<slug>__seed<n>__<mode>.png
    outputs/e_diag_trajectory/figures/curves__<slug>__seed<n>.png
    outputs/e_diag_trajectory/figures/endpoint__<slug>.png
    outputs/e_diag_trajectory/summary.json

Run:
    PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
    CUDA_VISIBLE_DEVICES=1 $PY -m poe_repair.experiments.e_diag_trajectory \\
        --seeds 4 42 --decode-stride 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from poe_repair.composers import co3 as cmp_co3
from poe_repair.composers import mono as cmp_mono
from poe_repair.composers._helpers import (
    compute_token_indices,
    encode_pair,
    get_joint_embeds,
    init_latents_for_cell,
)
from poe_repair.config import RunConfig
from poe_repair.experiments._eval_common import HEADLINE_PAIR, cell_for, slugify
from poe_repair.figures._common import image_grid, save_fig, stacked_line_plot
from poe_repair.methods._sampling import run_diagnostic_trajectory
from poe_repair.run import MethodCtx, make_ctx
from poe_repair.runtime import ensure_dir, write_json


EXP_NAME = "e_diag_trajectory"

MODES = {
    "mono":          {"co3_active": False},
    "mono_co3step0": {"co3_active": True},
}


def _attn_to_image(attn: torch.Tensor) -> torch.Tensor:
    """Normalise [H, W] map to [0, 1] RGB by per-map min-max + jet via heat triplet."""
    a = attn.float().detach()
    lo, hi = float(a.min().item()), float(a.max().item())
    if hi - lo < 1e-12:
        a = torch.zeros_like(a)
    else:
        a = (a - lo) / (hi - lo)
    # 3-channel pseudo-heat: red = a, green = a*0.5, blue = (1-a)*0.5
    rgb = torch.stack([a, a * 0.5, (1.0 - a) * 0.5], dim=0)
    return rgb.clamp(0.0, 1.0)


def _run_one(
    cell, ctx: MethodCtx, *, mode: str, decode_steps: list[int],
):
    emb = encode_pair(cell, ctx)
    seq_j, pool_j, _ = get_joint_embeds(cell, ctx, anchor_source="literal")
    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    token_a, token_b, _ = compute_token_indices(
        cell.prompt_a, cell.prompt_b, ctx.models["tokenizer"],
        template=ctx.joint_template,
    )
    out = run_diagnostic_trajectory(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_j=seq_j, pool_j=pool_j,
        seq_a=emb["seq_a"], pool_a=emb["pool_a"],
        seq_b=emb["seq_b"], pool_b=emb["pool_b"],
        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        token_index_a=token_a, token_index_b=token_b,
        guidance_scale=ctx.guidance_scale,
        num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width,
        euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
        decode_at_steps=decode_steps,
        attn_resolution=16,
        **MODES[mode],
    )
    return out


def _strip_figure(out, *, output_path: Path, decode_steps: list[int],
                  title: str) -> Path:
    """Three-row strip: decoded x̂_0 / cat-attn / dog-attn at each step in
    decode_steps. Each cell is a tensor in [0, 1]."""
    decoded = out.extras["decoded_by_step"]
    attn_a = out.extras["attn_a_by_step"]
    attn_b = out.extras["attn_b_by_step"]
    cells: list[list[torch.Tensor]] = [[], [], []]
    for s in decode_steps:
        if s in decoded:
            cells[0].append(decoded[s])
        else:
            cells[0].append(torch.zeros(3, 64, 64))
        cells[1].append(_attn_to_image(attn_a[s]) if s in attn_a else torch.zeros(3, 16, 16))
        cells[2].append(_attn_to_image(attn_b[s]) if s in attn_b else torch.zeros(3, 16, 16))
    return image_grid(
        cells, output_path,
        col_labels=[f"t={s}" for s in decode_steps],
        row_labels=["x̂_0", "A_cat", "A_dog"],
        title=title,
        panel_size=1.6,
    )


def _curves_figure(
    by_mode: dict, *, output_path: Path, title: str,
) -> Path:
    """Stacked curves: max(A_cat), max(A_dog), cos overlap, ‖eps‖, ‖Δx̂_0‖."""
    panels = []
    # A_max series
    series_a = {}
    series_b = {}
    series_overlap = {}
    series_eps = {}
    series_dx0 = {}
    for mode, out in by_mode.items():
        T = ctx_num_steps(out)
        steps = list(range(T))
        a_max, b_max, overlap = [], [], []
        attn_a = out.extras["attn_a_by_step"]
        attn_b = out.extras["attn_b_by_step"]
        for s in steps:
            ma = attn_a.get(s)
            mb = attn_b.get(s)
            a_max.append(float(ma.max().item()) if ma is not None else float("nan"))
            b_max.append(float(mb.max().item()) if mb is not None else float("nan"))
            if ma is not None and mb is not None:
                fa = ma.flatten()
                fb = mb.flatten()
                denom = float((fa.norm() * fb.norm()).item()) + 1e-12
                overlap.append(float((fa @ fb).item()) / denom)
            else:
                overlap.append(float("nan"))
        series_a[mode] = (steps, a_max)
        series_b[mode] = (steps, b_max)
        series_overlap[mode] = (steps, overlap)
        series_eps[mode] = (steps, list(out.extras["eps_norm_per_step"]))
        series_dx0[mode] = (steps, list(out.extras["x0_delta_per_step"]))
    panels = [
        {"series": series_a, "ylabel": "max A_cat", "title": "cat-token attention peak"},
        {"series": series_b, "ylabel": "max A_dog", "title": "dog-token attention peak"},
        {"series": series_overlap, "ylabel": "cos(A_cat, A_dog)",
         "title": "spatial overlap (1 = same blob, 0 = disjoint)"},
        {"series": series_eps, "ylabel": "‖ε_t‖", "title": "guided-eps norm"},
        {"series": series_dx0, "ylabel": "‖Δx̂_0‖",
         "title": "Tweedie x̂_0 step-to-step delta (commitment proxy)"},
    ]
    return stacked_line_plot(
        panels, output_path, xlabel="step index",
        title=title, panel_height=2.0, panel_width=8.5,
    )


def ctx_num_steps(out) -> int:
    return len(out.extras["eps_norm_per_step"])


def main() -> None:
    from poe_repair.experiments import _assert_env_ok
    _assert_env_ok()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="*", type=int, default=[4, 42])
    ap.add_argument("--prompts", nargs=2, type=str, default=list(HEADLINE_PAIR),
                    metavar=("PROMPT_A", "PROMPT_B"))
    ap.add_argument("--decode-stride", type=int, default=5,
                    help="Stride for decoded preview steps; uses [0, stride, 2*stride, ...].")
    ap.add_argument("--include-co3", action="store_true",
                    help="Also call the actual CO3 pipeline for endpoint comparison "
                         "(loads its own SDXL; ~7GB extra GPU).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    prompt_a, prompt_b = args.prompts

    cfg = RunConfig()
    ctx: MethodCtx = make_ctx()
    fig_dir = ensure_dir(cfg.paths.output_root / EXP_NAME / "figures")
    summary_dir = ensure_dir(cfg.paths.output_root / EXP_NAME)
    slug = slugify(prompt_a, prompt_b)

    T = ctx.num_inference_steps
    decode_steps = list(range(0, T, max(1, args.decode_stride)))
    if (T - 1) not in decode_steps:
        decode_steps.append(T - 1)

    # Per-(seed, mode) trajectory + per-seed curves figure.
    summary_cells: list[dict] = []
    endpoint_rows: list[list] = []
    endpoint_col_labels = ["Mono (instrumented)", "Mono+CO3-step0 (instrumented)"]
    if args.include_co3:
        endpoint_col_labels.append("CO3 (real, CFG++)")

    for seed in args.seeds:
        cell = cell_for(prompt_a, prompt_b, seed)
        per_mode_outs: dict = {}
        endpoint_row: list = []
        for mode in MODES.keys():
            print(f"[diag] seed={seed} mode={mode}")
            out = _run_one(cell, ctx, mode=mode, decode_steps=decode_steps)
            per_mode_outs[mode] = out
            strip_path = fig_dir / f"strip__{slug}__seed{seed}__{mode}.png"
            _strip_figure(
                out, output_path=strip_path, decode_steps=decode_steps,
                title=f"{mode} — seed {seed} — {prompt_a} × {prompt_b}",
            )
            endpoint_row.append(out.image)

        curves_path = fig_dir / f"curves__{slug}__seed{seed}.png"
        _curves_figure(
            per_mode_outs, output_path=curves_path,
            title=f"diagnostic curves — seed {seed} — {prompt_a} × {prompt_b}",
        )

        if args.include_co3:
            print(f"[diag] seed={seed} extra=co3")
            p = cmp_co3.run(
                cell, ctx, anchor_source="literal",
                exp_name=EXP_NAME, overwrite=args.overwrite,
            )
            endpoint_row.append(p)

        endpoint_rows.append(endpoint_row)
        summary_cells.append({
            "seed": seed,
            "pair": [prompt_a, prompt_b],
            "decode_steps": decode_steps,
            "strip_paths": {
                m: str(fig_dir / f"strip__{slug}__seed{seed}__{m}.png")
                for m in MODES.keys()
            },
            "curves_path": str(curves_path),
            "per_mode_diagnostics": {
                m: {
                    "eps_norm_per_step": list(per_mode_outs[m].extras["eps_norm_per_step"]),
                    "x0_delta_per_step": list(per_mode_outs[m].extras["x0_delta_per_step"]),
                    "co3_active_per_step": list(per_mode_outs[m].extras["co3_active_per_step"]),
                    "max_attn_a_per_step": [
                        float(per_mode_outs[m].extras["attn_a_by_step"][s].max().item())
                        if s in per_mode_outs[m].extras["attn_a_by_step"]
                        else float("nan")
                        for s in range(T)
                    ],
                    "max_attn_b_per_step": [
                        float(per_mode_outs[m].extras["attn_b_by_step"][s].max().item())
                        if s in per_mode_outs[m].extras["attn_b_by_step"]
                        else float("nan")
                        for s in range(T)
                    ],
                }
                for m in MODES.keys()
            },
        })

    endpoint_path = image_grid(
        endpoint_rows, fig_dir / f"endpoint__{slug}.png",
        col_labels=endpoint_col_labels,
        row_labels=[f"seed {s}" for s in args.seeds],
        title=f"E-diag endpoint — {prompt_a} × {prompt_b}",
        panel_size=2.6,
    )
    write_json(summary_dir / "summary.json", {
        "exp": EXP_NAME, "seeds": args.seeds, "pair": [prompt_a, prompt_b],
        "decode_steps": decode_steps,
        "modes": list(MODES.keys()),
        "include_co3": bool(args.include_co3),
        "endpoint_path": str(endpoint_path),
        "cells": summary_cells,
    })
    print(f"[diag] wrote {endpoint_path}")


if __name__ == "__main__":
    main()
