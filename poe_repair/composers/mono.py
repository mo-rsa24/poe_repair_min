"""Mono composer — single guided CFG branch on a joint embedding.

Reference baseline (literal e_J) and the main repair candidate (synthesised
ê_J). The sampler is identical for both; only the joint embedding differs.
"""

from __future__ import annotations

from pathlib import Path

from poe_repair.composers._helpers import (
    AnchorSource,
    cell_output_dir,
    get_joint_embeds,
    init_latents_for_cell,
)
from poe_repair.methods._sampling import run_cfg, write_decoded_image
from poe_repair.run import MethodCtx
from poe_repair.runtime import PairSeedCell


METHOD_NAME_BY_SOURCE = {
    "literal": "mono_literal",
    "synth": "mono_synth",
}


def run(
    cell: PairSeedCell,
    ctx: MethodCtx,
    *,
    anchor_source: AnchorSource = "literal",
    exp_name: str = "tmp",
    overwrite: bool = False,
) -> Path:
    """Run Mono on (cell, anchor_source) and return the image path."""
    method_name = METHOD_NAME_BY_SOURCE[anchor_source]
    out_dir = cell_output_dir(ctx, exp_name, method_name, cell)
    image_path = out_dir / f"{method_name}.png"
    if image_path.exists() and not overwrite:
        return image_path

    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    seq_j, pool_j, _ = get_joint_embeds(cell, ctx, anchor_source=anchor_source)
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
