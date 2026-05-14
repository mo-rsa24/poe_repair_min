"""Teacher-residual composer.

At every step we compute both PoE and Mono predictions in one batched UNet
call and step with::

    ε_final = ε̃_PoE + λ_t · (ε̃_Mono − ε̃_PoE)

This is *not* a deployable method — it requires the literal joint prompt
e_J at inference. Its job is to (a) define the upper-bound corrector that
fully closes the PoE→Mono gap, and (b) generate the per-step residuals
``Δ_t = ε̃_Mono − ε̃_PoE`` that the learned residual model (idea 2) is
supervised on.

Method-name encodes the schedule + lambda_max so multiple settings can
coexist on disk (e.g. ``teacher_residual_const_lam100``).
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
from poe_repair.methods._sampling import run_teacher_residual, write_decoded_image
from poe_repair.run import MethodCtx
from poe_repair.runtime import PairSeedCell, write_json


SCHEDULE_TAG = {
    "constant": "const",
    "linear_decay": "decay",
    "early_only": "early",
}


def method_name_for(
    *, lambda_schedule: str, lambda_max: float,
    correction_window: tuple[int, int] | None = None,
) -> str:
    tag = SCHEDULE_TAG.get(lambda_schedule, lambda_schedule)
    lam = int(round(float(lambda_max) * 100))
    name = f"teacher_residual_{tag}_lam{lam:03d}"
    if correction_window is not None:
        name += f"_w{int(correction_window[0])}-{int(correction_window[1])}"
    return name


def run(
    cell: PairSeedCell,
    ctx: MethodCtx,
    *,
    lambda_schedule: str = "constant",
    lambda_max: float = 1.0,
    correction_window: tuple[int, int] | None = None,
    save_residuals: bool = False,
    save_x0_estimates: bool = False,
    save_trajectory: bool = False,
    adaptive_schedule: object | None = None,
    method_name_override: str | None = None,
    exp_name: str = "tmp",
    overwrite: bool = False,
    record_cross_attention: bool = False,
    attn_token_indices: dict | None = None,
    attn_resolution: int = 32,
) -> Path:
    """Run teacher-residual on (cell, knobs) and return the image path.

    Optional persistence flags (off by default to preserve existing
    callers and disk usage):
      save_residuals: per-step .pt files with ``Δ_t`` (and optionally
        ``eps_poe`` / ``eps_j`` if ``save_x0_estimates`` is also set).
      save_x0_estimates: requires ``save_residuals``; adds the two guided
        eps tensors per step. Needed for Tweedie ``x̂_0`` panels.
      save_trajectory: write ``latent_trajectory.pt`` next to the image,
        containing ``{trajectories, sigmas, timesteps}`` from the
        ``LatentTrajectoryCollector`` for path-level analysis.
    """
    method_name = (
        method_name_override
        if method_name_override is not None
        else method_name_for(
            lambda_schedule=lambda_schedule,
            lambda_max=lambda_max,
            correction_window=correction_window,
        )
    )
    out_dir = cell_output_dir(ctx, exp_name, method_name, cell)
    image_path = out_dir / f"{method_name}.png"
    summary_path = out_dir / f"summary_{method_name}.json"
    if image_path.exists() and not overwrite:
        return image_path

    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)
    seq_j, pool_j, _ = get_joint_embeds(cell, ctx, anchor_source="literal")

    residuals_dir = (out_dir / "residuals") if save_residuals else None
    attn_dir = (out_dir / "attn_maps") if record_cross_attention else None

    out = run_teacher_residual(
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
        lambda_schedule=lambda_schedule,
        lambda_max=float(lambda_max),
        correction_window=correction_window,
        save_residuals_dir=residuals_dir,
        save_x0_estimates=save_x0_estimates,
        adaptive_schedule=adaptive_schedule,
        attn_capture_dir=attn_dir,
        attn_token_indices=attn_token_indices,
        attn_resolution=attn_resolution,
    )
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
            "image_path": str(image_path),
            "pair": [cell.prompt_a, cell.prompt_b],
            "guidance_scale": ctx.guidance_scale,
            "num_inference_steps": ctx.num_inference_steps,
            "saved_trajectory": bool(save_trajectory),
            **out.extras,
        },
    )
    return image_path
