"""Method 2b stacked on Method 1.

    ε_base = (1 − β_t)·ε̃_PoE + β_t·ε̃_J          # sched-M2 base
    ε_t   = ε_base + λ_t · δ_θ(x_t, t, pools)    # plus student

Uses sched-M2 with the synthesised ê_J as the trajectory base, then adds
the trained Method 2b CNN's correction on top under a separate λ_t
schedule. With ``lambda_max=0`` this reduces to plain sched-M2(ê_J);
with ``beta_max=0`` it reduces to plain Method 2b on PoE.
"""

from __future__ import annotations

from pathlib import Path

import torch

from poe_repair.composers._helpers import (
    cell_output_dir,
    encode_pair,
    get_joint_embeds,
    init_latents_for_cell,
)
from poe_repair.composers.direct_eps import _resolve_student_ckpt, _cached_student
from poe_repair.methods._sampling import (
    run_schedm2_plus_student,
    write_decoded_image,
)
from poe_repair.run import MethodCtx
from poe_repair.runtime import PairSeedCell, write_json


METHOD_NAME = "direct_eps_on_schedm2"


def build_rect_schedule(
    num_steps: int, window_frac: float, peak: float,
) -> torch.Tensor:
    sched = torch.zeros(num_steps)
    sched[: int(round(window_frac * num_steps))] = float(peak)
    return sched


def run(
    cell: PairSeedCell,
    ctx: MethodCtx,
    *,
    beta_window_frac: float = 0.4,
    beta_max: float = 1.0,
    student_window_frac: float = 0.4,
    lambda_max: float = 1.0,
    student_ckpt: str | Path | None = None,
    correction_max_rel_norm: float | None = None,
    exp_name: str = "tmp",
    overwrite: bool = False,
) -> Path:
    out_dir = cell_output_dir(ctx, exp_name, METHOD_NAME, cell)
    image_path = out_dir / f"{METHOD_NAME}.png"
    summary_path = out_dir / f"summary_{METHOD_NAME}.json"
    if image_path.exists() and not overwrite:
        return image_path

    ckpt_path = _resolve_student_ckpt(student_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"direct_eps checkpoint not found at {ckpt_path}. "
            "Train it first with `python -m poe_repair.students.train_direct_eps`."
        )
    student = _cached_student(str(ckpt_path), str(ctx.device))

    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)
    seq_j, pool_j, _ = get_joint_embeds(cell, ctx, anchor_source="synth")

    beta_schedule = build_rect_schedule(
        ctx.num_inference_steps, beta_window_frac, beta_max,
    )
    student_schedule = build_rect_schedule(
        ctx.num_inference_steps, student_window_frac, lambda_max,
    )

    out = run_schedm2_plus_student(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_a=emb["seq_a"], pool_a=emb["pool_a"],
        seq_b=emb["seq_b"], pool_b=emb["pool_b"],
        seq_j=seq_j, pool_j=pool_j,
        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        student=student,
        guidance_scale=ctx.guidance_scale,
        num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width,
        euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
        beta_schedule=beta_schedule,
        student_lambda_schedule=student_schedule,
        correction_max_rel_norm=correction_max_rel_norm,
    )
    write_decoded_image(out.image, image_path)
    write_json(
        summary_path,
        {
            "method": METHOD_NAME,
            "pair_slug": cell.pair_slug,
            "seed": cell.seed,
            "image_path": str(image_path),
            "pair": [cell.prompt_a, cell.prompt_b],
            "guidance_scale": ctx.guidance_scale,
            "num_inference_steps": ctx.num_inference_steps,
            "beta_window_frac": float(beta_window_frac),
            "beta_max": float(beta_max),
            "student_window_frac": float(student_window_frac),
            "lambda_max": float(lambda_max),
            "student_ckpt": str(ckpt_path),
            "correction_max_rel_norm": (
                None if correction_max_rel_norm is None
                else float(correction_max_rel_norm)
            ),
            **out.extras,
        },
    )
    return image_path
