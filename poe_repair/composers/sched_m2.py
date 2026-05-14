"""Sched-M2 composer: 4-branch β-mixed PoE/anchor.

    ε_t = (1 − β_t) · ε̃_PoE + β_t · ε̃_J

The anchor ε̃_J comes from either the literal e_J or the synthesised ê_J
(``anchor_source``). Default schedule is Phase-11 rectangular: β=1.0 for
the first 40% of inference steps, β=0 thereafter.
"""

from __future__ import annotations

from pathlib import Path

import torch

from poe_repair.composers._helpers import (
    AnchorSource,
    cell_output_dir,
    encode_pair,
    get_joint_embeds,
    init_latents_for_cell,
)
from poe_repair.methods._sampling import run_beta_inject_online, write_decoded_image
from poe_repair.run import MethodCtx
from poe_repair.runtime import PairSeedCell


METHOD_NAME_BY_SOURCE = {
    "literal": "schedm2_literal",
    "synth": "schedm2_synth",
}


def build_rect_schedule(
    num_steps: int, window_frac: float = 0.4, beta_max: float = 1.0,
) -> torch.Tensor:
    """Phase-11 rectangular schedule: β_max for first ``window_frac · T`` steps."""
    sched = torch.zeros(num_steps)
    sched[: int(round(window_frac * num_steps))] = float(beta_max)
    return sched


def run(
    cell: PairSeedCell,
    ctx: MethodCtx,
    *,
    anchor_source: AnchorSource = "literal",
    window_frac: float = 0.4,
    beta_max: float = 1.0,
    exp_name: str = "tmp",
    overwrite: bool = False,
) -> Path:
    method_name = METHOD_NAME_BY_SOURCE[anchor_source]
    out_dir = cell_output_dir(ctx, exp_name, method_name, cell)
    image_path = out_dir / f"{method_name}.png"
    if image_path.exists() and not overwrite:
        return image_path

    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)
    seq_j, pool_j, _ = get_joint_embeds(cell, ctx, anchor_source=anchor_source)
    schedule = build_rect_schedule(ctx.num_inference_steps, window_frac, beta_max)

    out = run_beta_inject_online(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_a=emb["seq_a"], pool_a=emb["pool_a"],
        seq_b=emb["seq_b"], pool_b=emb["pool_b"],
        seq_j=seq_j, pool_j=pool_j,
        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        guidance_scale=ctx.guidance_scale,
        num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width,
        euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
        beta_schedule=schedule,
    )
    write_decoded_image(out.image, image_path)
    return image_path
