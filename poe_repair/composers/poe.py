"""Vanilla PoE composer: ε̃ = ε̃_A + ε̃_B − ε_∅. The CI baseline / failure case."""

from __future__ import annotations

from pathlib import Path

from poe_repair.composers._helpers import (
    cell_output_dir,
    encode_pair,
    init_latents_for_cell,
)
from poe_repair.methods._sampling import run_vanilla_poe, write_decoded_image
from poe_repair.run import MethodCtx
from poe_repair.runtime import PairSeedCell


METHOD_NAME = "poe"


def run(
    cell: PairSeedCell, ctx: MethodCtx, *, exp_name: str = "tmp",
    overwrite: bool = False,
) -> Path:
    out_dir = cell_output_dir(ctx, exp_name, METHOD_NAME, cell)
    image_path = out_dir / f"{METHOD_NAME}.png"
    if image_path.exists() and not overwrite:
        return image_path
    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)
    out = run_vanilla_poe(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_a=emb["seq_a"], pool_a=emb["pool_a"],
        seq_b=emb["seq_b"], pool_b=emb["pool_b"],
        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        guidance_scale=ctx.guidance_scale,
        num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width,
        euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
    )
    write_decoded_image(out.image, image_path)
    return image_path
