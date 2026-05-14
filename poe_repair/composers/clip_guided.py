"""CLIP-guided composer (Idea 5b).

Wraps ``run_clip_guided_repair``. At each step in the configured
correction window, runs PoE's standard 3-branch UNet, then adds a
corrective term computed by backpropagating the cosine similarity of
the Tweedie x̂_0 against a CLIP text embedding back to the latent.

The diffusion model never sees the joint embedding ``e_J`` at inference;
only CLIP's text encoder is asked about ``"a cat and a dog"`` (or any
configurable target prompt). Mono is not invoked.

Method-name format::

    clip_guided_alpha<NNN>            # NNN = round(α/α₀ · 100)
    clip_guided_calibrated            # alpha multiplier ≡ 1
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from poe_repair.composers._helpers import (
    cell_output_dir,
    encode_pair,
    init_latents_for_cell,
)
from poe_repair.config import RunConfig
from poe_repair.experiments.veracity.metrics import _get_clip
from poe_repair.methods._clip_guided import run_clip_guided_repair
from poe_repair.methods._sampling import write_decoded_image
from poe_repair.run import MethodCtx
from poe_repair.runtime import PairSeedCell, write_json


_TEXT_EMBED_CACHE: dict[str, torch.Tensor] = {}


@torch.no_grad()
def _cache_text_embed(prompt: str, device) -> torch.Tensor:
    if prompt in _TEXT_EMBED_CACHE:
        return _TEXT_EMBED_CACHE[prompt]
    clip = _get_clip(device)
    inputs = clip.processor(
        text=[prompt], return_tensors="pt", padding=True, truncation=True,
    )
    feats = clip.model.get_text_features(
        input_ids=inputs["input_ids"].to(clip.device),
        attention_mask=inputs["attention_mask"].to(clip.device),
    )
    feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
    feats = feats.detach()
    _TEXT_EMBED_CACHE[prompt] = feats
    return feats


def method_name_for(
    *,
    alpha_multiplier: float | None = None,
    calibrated: bool = False,
    schedule: str = "constant",
    correction_window: tuple[int, int] | None = None,
) -> str:
    if calibrated:
        name = "clip_guided_calibrated"
    else:
        if alpha_multiplier is None:
            raise ValueError("alpha_multiplier required when calibrated=False")
        name = f"clip_guided_alpha{int(round(alpha_multiplier * 100)):03d}"
    if schedule != "constant":
        name += f"_sched-{schedule}"
    if correction_window is not None and correction_window != (10, 25):
        name += f"_w{int(correction_window[0])}-{int(correction_window[1])}"
    return name


def _load_basin_templates(
    *,
    pair_slug: str, seed: int, output_root: Path,
) -> dict | None:
    veracity_root = output_root / "veracity" / "pairs" / pair_slug / f"seed_{seed}"
    poe_path = (
        veracity_root / "teacher_residual_const_lam000" / "latent_trajectory.pt"
    )
    mono_path = (
        veracity_root / "teacher_residual_const_lam100" / "latent_trajectory.pt"
    )
    if not poe_path.exists() or not mono_path.exists():
        return None
    poe = torch.load(poe_path, map_location="cpu", weights_only=False)
    mono = torch.load(mono_path, map_location="cpu", weights_only=False)
    return {
        "x_t_poe": poe["trajectories"],
        "x_t_mono": mono["trajectories"],
    }


def run(
    cell: PairSeedCell,
    ctx: MethodCtx,
    *,
    force_scaler: float,
    alpha_multiplier: float | None = 1.0,
    calibrated: bool = False,
    schedule: str = "constant",
    schedule_max: float | None = None,
    correction_window: tuple[int, int] = (10, 25),
    closed_loop_threshold: float = 0.5,
    adaptive_schedule: object | None = None,
    method_name_override: str | None = None,
    target_prompt: str = "a cat and a dog",
    decode_strategy: str = "full_vae",
    grad_norm_clip: float | None = None,
    save_residuals: bool = False,
    save_trajectory: bool = False,
    exp_name: str = "tmp",
    overwrite: bool = False,
) -> Path:
    """Run CLIP-guided repair on (cell, knobs) and return the image path.

    The applied per-step strength is ``schedule(t) · force_scaler``. By
    default ``schedule_max == alpha_multiplier`` so passing
    ``alpha_multiplier=0.5`` runs at half of the calibrated ``α₀``
    (encoded in ``force_scaler``).
    """
    if schedule_max is None:
        schedule_max = float(alpha_multiplier) if alpha_multiplier is not None else 1.0

    method_name = (
        method_name_override
        if method_name_override is not None
        else method_name_for(
            alpha_multiplier=alpha_multiplier,
            calibrated=calibrated,
            schedule=schedule,
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

    clip = _get_clip(ctx.device)
    text_embed = _cache_text_embed(target_prompt, ctx.device)

    basin_templates = None
    if schedule == "closed_loop":
        cfg = RunConfig()
        basin_templates = _load_basin_templates(
            pair_slug=cell.pair_slug, seed=cell.seed,
            output_root=cfg.paths.output_root,
        )
        if basin_templates is None:
            raise RuntimeError(
                "closed_loop schedule requires veracity λ=0 and λ=1 trajectories "
                f"under outputs/veracity/pairs/{cell.pair_slug}/seed_{cell.seed}/."
            )

    residuals_dir = (out_dir / "residuals") if save_residuals else None

    out = run_clip_guided_repair(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_a=emb["seq_a"], pool_a=emb["pool_a"],
        seq_b=emb["seq_b"], pool_b=emb["pool_b"],
        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        guidance_scale=ctx.guidance_scale,
        num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width,
        euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
        clip_model=clip.model,
        clip_text_embed=text_embed,
        correction_window=correction_window,
        force_scaler=float(force_scaler),
        schedule=schedule,
        schedule_max=float(schedule_max),
        basin_templates=basin_templates,
        closed_loop_threshold=float(closed_loop_threshold),
        adaptive_schedule=adaptive_schedule,
        decode_strategy=decode_strategy,
        grad_norm_clip=grad_norm_clip,
        save_residuals_dir=residuals_dir,
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
            "alpha_multiplier": (
                None if calibrated else float(alpha_multiplier or 0.0)
            ),
            "calibrated": bool(calibrated),
            "target_prompt": target_prompt,
            "saved_trajectory": bool(save_trajectory),
            **out.extras,
        },
    )
    return image_path
