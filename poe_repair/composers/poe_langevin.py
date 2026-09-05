"""Langevin corrector composer (scope 06, step 25).

Wraps ``run_poe_langevin``. At every denoising step the sampler runs ``k``
Langevin corrector steps on the product-of-experts score, then takes the
ordinary DDIM step from the settled point. ``k = 0`` is plain PoE, byte for byte.

Method name format::

    poe_langevin_k<NNN>_c<NNN>              # NNN = k, and round(c * 1000)
    poe_langevin_k<NNN>_c<NNN>_w<s>-<e>     # with a corrector window [s, e)
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
from poe_repair.methods._poe_langevin import run_poe_langevin
from poe_repair.methods._sampling import write_decoded_image
from poe_repair.run import MethodCtx
from poe_repair.runtime import PairSeedCell, write_json


def method_name_for(
    *, k: int, c: float, corrector_window: tuple[int, int] | None = None,
) -> str:
    name = f"poe_langevin_k{int(k):03d}_c{int(round(float(c) * 1000)):03d}"
    if corrector_window is not None:
        name += f"_w{int(corrector_window[0])}-{int(corrector_window[1])}"
    return name


def run(
    cell: PairSeedCell,
    ctx: MethodCtx,
    *,
    k: int,
    c: float,
    corrector_window: tuple[int, int] | None = None,
    measure_residual: bool = False,
    probe_inner_counts: tuple[int, ...] = (),
    noise_seed: int | None = None,
    decode: bool = True,
    save_trajectory: bool = False,
    method_name_override: str | None = None,
    exp_name: str = "interaction_term/corrector",
    overwrite: bool = False,
    return_outputs: bool = False,
):
    """Run the corrector on (cell, k, c, window) and return the image path.

    With ``return_outputs=True`` the ``SamplerOutputs`` come back too, so a caller
    that only wants the per-step rows (no image on disk) can take them without a
    second run. ``noise_seed`` defaults to the cell's seed, so two runs of the same
    cell draw the same Langevin noise and differ only in what they were asked to vary.
    """
    method_name = method_name_override or method_name_for(
        k=k, c=c, corrector_window=corrector_window,
    )
    out_dir = cell_output_dir(ctx, exp_name, method_name, cell)
    image_path = out_dir / f"{method_name}.png"
    summary_path = out_dir / f"summary_{method_name}.json"
    if image_path.exists() and not overwrite and not return_outputs:
        return image_path

    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)
    seq_j = pool_j = None
    if measure_residual:
        seq_j, pool_j = get_joint_embeds(cell, ctx)

    out = run_poe_langevin(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_a=emb["seq_a"], pool_a=emb["pool_a"],
        seq_b=emb["seq_b"], pool_b=emb["pool_b"],
        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        guidance_scale=ctx.guidance_scale,
        num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width,
        euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
        k=int(k), c=float(c), corrector_window=corrector_window,
        noise_seed=int(cell.seed if noise_seed is None else noise_seed),
        seq_j=seq_j, pool_j=pool_j,
        measure_residual=measure_residual,
        probe_inner_counts=probe_inner_counts,
        decode=decode,
    )
    if decode and out.image is not None:
        write_decoded_image(out.image, image_path)
    if save_trajectory:
        torch.save(
            {
                "trajectories": out.tracker.trajectories.to(torch.float16),
                "sigmas": out.tracker.sigmas,
                "timesteps": out.tracker.timesteps,
                "num_steps": int(out.tracker.num_steps),
            },
            out_dir / "latent_trajectory.pt",
        )
    write_json(
        summary_path,
        {
            "method": method_name,
            "pair_slug": cell.pair_slug,
            "seed": cell.seed,
            "image_path": str(image_path) if decode else None,
            "pair": [cell.prompt_a, cell.prompt_b],
            "guidance_scale": ctx.guidance_scale,
            "num_inference_steps": ctx.num_inference_steps,
            "saved_trajectory": bool(save_trajectory),
            **out.extras,
        },
    )
    if return_outputs:
        return image_path, out
    return image_path
