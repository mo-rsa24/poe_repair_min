"""Mono composer — single guided CFG branch on the literal joint embedding.

Diagnostic ceiling: uses the literal e_J = encode("a cat and a dog") via the
joint_template. The synthesised ê_J variant is removed.
"""

from __future__ import annotations

from pathlib import Path

from poe_repair.composers._helpers import (
    cell_output_dir,
    get_joint_embeds,
    init_latents_for_cell,
)
from poe_repair.methods._sampling import run_cfg, write_decoded_image
from poe_repair.run import MethodCtx
from poe_repair.runtime import PairSeedCell


def run(
    cell: PairSeedCell,
    ctx: MethodCtx,
    *,
    exp_name: str = "tmp",
    overwrite: bool = False,
) -> Path:
    """Run Mono on ``cell`` with the literal joint embedding; return image path."""
    method_name = "mono_literal"
    out_dir = cell_output_dir(ctx, exp_name, method_name, cell)
    image_path = out_dir / f"{method_name}.png"
    if image_path.exists() and not overwrite:
        return image_path

    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    seq_j, pool_j = get_joint_embeds(cell, ctx)
    from poe_repair.runtime import encode_prompt_sdxl
    seq_e, pool_e = encode_prompt_sdxl(
        "", models=ctx.models, device=ctx.device, dtype=ctx.dtype,
    )
    out = run_cfg(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_cond=seq_j, pool_cond=pool_j, seq_e=seq_e, pool_e=pool_e,
        guidance_scale=ctx.guidance_scale,
        num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width,
        euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
    )
    write_decoded_image(out.image, image_path)
    return image_path
