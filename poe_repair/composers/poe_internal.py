"""PoE-internal composer (Idea 1).

Wraps ``run_poe_internal_repair``. At every denoising step the sampler
runs a 3-branch UNet (A, B, ∅) and adds a corrective force computed from
PoE's own outputs. No 4th UNet branch on a joint embedding; no Mono call
at inference.

Method name format::

    poe_internal_<force_kind>_alpha<NNN>     # NNN = round(α/α₀ · 100)
    poe_internal_<force_kind>_calibrated     # alpha multiplier ≡ 1 (== α₀)
"""

from __future__ import annotations

from pathlib import Path

import torch

from poe_repair.composers._helpers import (
    cell_output_dir,
    encode_pair,
    init_latents_for_cell,
)
from poe_repair.config import RunConfig
from poe_repair.methods._poe_internal import run_poe_internal_repair
from poe_repair.methods._sampling import write_decoded_image
from poe_repair.run import MethodCtx
from poe_repair.runtime import PairSeedCell, write_json


def method_name_for(
    *,
    force_kind: str,
    alpha_multiplier: float | None = None,
    calibrated: bool = False,
    schedule: str = "constant",
    correction_window: tuple[int, int] | None = None,
) -> str:
    if calibrated:
        name = f"poe_internal_{force_kind}_calibrated"
    else:
        if alpha_multiplier is None:
            raise ValueError("alpha_multiplier required when calibrated=False")
        name = (
            f"poe_internal_{force_kind}_alpha{int(round(alpha_multiplier * 100)):03d}"
        )
    if schedule != "constant":
        name += f"_sched-{schedule}"
    if correction_window is not None:
        name += f"_w{int(correction_window[0])}-{int(correction_window[1])}"
    return name


def _solo_subject_token_index(prompt: str, tokenizer) -> int:
    """Index of the last "real" token (the subject) in a solo encoding.

    For "a cat" / "a dog" with the standard SDXL tokenizer, returns 2 (the
    "cat"/"dog" token, after BOS + "a"). Falls back to 2 on any failure.
    """
    try:
        ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            try:
                return max(0, ids.index(eos) - 1)
            except ValueError:
                pass
        return max(0, len(ids) - 1)
    except Exception:
        return 2


def _load_basin_templates(
    *,
    pair_slug: str, seed: int, output_root: Path,
) -> dict | None:
    """Load veracity λ=0 and λ=1 trajectories for closed-loop scheduling."""
    veracity_root = output_root / "veracity" / "pairs" / pair_slug / f"seed_{seed}"
    poe_path = (
        veracity_root / "teacher_residual_const_lam000" / "latent_trajectory.pt"
    )
    mono_path = (
        veracity_root / "teacher_residual_const_lam100" / "latent_trajectory.pt"
    )
    if not poe_path.exists() or not mono_path.exists():
        return None
    poe = torch.load(poe_path, map_location="cpu")
    mono = torch.load(mono_path, map_location="cpu")
    return {
        "x_t_poe": poe["trajectories"],     # (T+1, B, C, H, W)
        "x_t_mono": mono["trajectories"],
    }


def run(
    cell: PairSeedCell,
    ctx: MethodCtx,
    *,
    force_kind: str,
    force_scaler: float,
    alpha_multiplier: float | None = 1.0,
    calibrated: bool = False,
    schedule: str = "constant",
    schedule_max: float | None = None,
    correction_window: tuple[int, int] | None = None,
    closed_loop_threshold: float = 0.5,
    adaptive_schedule: object | None = None,
    method_name_override: str | None = None,
    save_residuals: bool = False,
    save_trajectory: bool = False,
    exp_name: str = "tmp",
    overwrite: bool = False,
) -> Path:
    """Run PoE-internal repair on (cell, knobs) and return the image path.

    The applied per-step strength is ``schedule(t) · force_scaler``. By
    default ``schedule_max == alpha_multiplier`` so passing
    ``alpha_multiplier=0.5`` runs at half of the calibrated ``α₀``
    (which is encoded in ``force_scaler``).
    """
    if force_kind not in {"overlap", "alignment"}:
        raise ValueError(f"force_kind must be 'overlap' or 'alignment', got {force_kind!r}")
    if schedule_max is None:
        schedule_max = float(alpha_multiplier) if alpha_multiplier is not None else 1.0

    method_name = (
        method_name_override
        if method_name_override is not None
        else method_name_for(
            force_kind=force_kind,
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

    tokenizer = ctx.models.get("tokenizer")
    idx_a_solo = _solo_subject_token_index(cell.prompt_a, tokenizer) if tokenizer else 2
    idx_b_solo = _solo_subject_token_index(cell.prompt_b, tokenizer) if tokenizer else 2

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
                f"under outputs/residual_diagnostics/existence/pairs/{cell.pair_slug}/seed_{cell.seed}/. "
                "Run veracity first."
            )

    residuals_dir = (out_dir / "residuals") if save_residuals else None

    out = run_poe_internal_repair(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_a=emb["seq_a"], pool_a=emb["pool_a"],
        seq_b=emb["seq_b"], pool_b=emb["pool_b"],
        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        guidance_scale=ctx.guidance_scale,
        num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width,
        euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
        force_kind=force_kind,
        force_scaler=float(force_scaler),
        schedule=schedule,
        schedule_max=float(schedule_max),
        correction_window=correction_window,
        token_index_a_solo=idx_a_solo,
        token_index_b_solo=idx_b_solo,
        basin_templates=basin_templates,
        closed_loop_threshold=float(closed_loop_threshold),
        adaptive_schedule=adaptive_schedule,
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
            "saved_trajectory": bool(save_trajectory),
            **out.extras,
        },
    )
    return image_path
