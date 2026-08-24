"""Method 2b composer — PoE plus a direct eps-space student correction.

At inference, a small CNN trained against the guided PMI residual is fed
``(x_t, t, pool_a, pool_b, pool_uncond)`` and its output is added to the
guided PoE score under a λ_t schedule:

    ε_t = ε̃_PoE + λ_t · δ_θ(x_t, t, c_a, c_b, c_∅)

The trained student lives at
``outputs/group_a_failure/checkpoints/direct_eps/<output_name>/best.pt``.
Pass the path via ``student_ckpt`` or set ``POE_REPAIR_DIRECT_EPS_CKPT``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

from poe_repair import paths
from poe_repair.composers._helpers import (
    cell_output_dir,
    encode_pair,
    init_latents_for_cell,
)
from poe_repair.methods._sampling import (
    run_direct_eps_inject,
    write_decoded_image,
)
from poe_repair.run import MethodCtx
from poe_repair.runtime import PairSeedCell, write_json
from poe_repair.students import load_direct_eps_student


METHOD_NAME = "direct_eps"


def build_rect_schedule(
    num_steps: int, window_frac: float = 0.4, lambda_max: float = 1.0,
) -> torch.Tensor:
    """Rectangular λ schedule: λ_max for the first ``window_frac`` of steps, 0 after."""
    sched = torch.zeros(num_steps)
    sched[: int(round(window_frac * num_steps))] = float(lambda_max)
    return sched


@lru_cache(maxsize=4)
def _cached_student(ckpt_path: str, device_str: str):
    return load_direct_eps_student(
        Path(ckpt_path),
        device=torch.device(device_str),
        dtype=torch.float32,
    )


def _resolve_student_ckpt(student_ckpt: str | Path | None) -> Path:
    if student_ckpt is not None:
        return Path(student_ckpt)
    env = os.environ.get("POE_REPAIR_DIRECT_EPS_CKPT")
    if env:
        return Path(env)
    return (
        paths.resolve(paths.CORRECTION_OUTSIDE_THE_UNET) / "checkpoints"
        / "direct_eps" / "direct_eps_v1" / "best.pt"
    )


def run(
    cell: PairSeedCell,
    ctx: MethodCtx,
    *,
    window_frac: float = 0.4,
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
    schedule = build_rect_schedule(ctx.num_inference_steps, window_frac, lambda_max)

    out = run_direct_eps_inject(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_a=emb["seq_a"], pool_a=emb["pool_a"],
        seq_b=emb["seq_b"], pool_b=emb["pool_b"],
        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        student=student,
        guidance_scale=ctx.guidance_scale,
        num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width,
        euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
        lambda_schedule=schedule,
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
            "window_frac": float(window_frac),
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
