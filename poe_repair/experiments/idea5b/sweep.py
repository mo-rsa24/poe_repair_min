"""Stage 1 — capacity check + eleven-point α-multiplier sweep.

The barrier height read from veracity sets the calibration target
(``BARRIER_TARGET = 1500``). One capacity-check run at
``force_scaler = 1.0`` measures ``K_clip = Σ_t ‖σ_t · g_t‖`` summed
over the correction window, which sets ``α₀ = BARRIER_TARGET / K_clip``.
The sweep then varies ``alpha_multiplier ∈ ALPHA_GRID`` against α₀.
"""

from __future__ import annotations

import json
from pathlib import Path

from poe_repair.composers import clip_guided as cmp_cg
from poe_repair.experiments._eval_common import cell_for
from poe_repair.run import MethodCtx, run_method
from poe_repair.runtime import PairSeedCell


ALPHA_GRID: tuple[float, ...] = (
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5,
)
BARRIER_TARGET: float = 1500.0
DEFAULT_CORRECTION_WINDOW: tuple[int, int] = (10, 25)
DEFAULT_TARGET_PROMPT: str = "a cat and a dog"


def _summary_path_for(method_dir: Path, method_name: str) -> Path:
    return method_dir / f"summary_{method_name}.json"


def capacity_check(
    *,
    cell: PairSeedCell,
    ctx: MethodCtx,
    exp_name: str,
    correction_window: tuple[int, int] = DEFAULT_CORRECTION_WINDOW,
    target_prompt: str = DEFAULT_TARGET_PROMPT,
    decode_strategy: str = "full_vae",
    grad_norm_clip: float | None = None,
    overwrite: bool = False,
) -> tuple[float, float, Path]:
    """Run one ``force_scaler=1.0`` pass and return ``(K_clip, α₀, path)``."""
    cap_path = cmp_cg.run(
        cell, ctx,
        force_scaler=1.0,
        alpha_multiplier=1.0,
        calibrated=False,
        schedule="constant",
        correction_window=correction_window,
        target_prompt=target_prompt,
        decode_strategy=decode_strategy,
        grad_norm_clip=grad_norm_clip,
        save_residuals=True,
        save_trajectory=True,
        exp_name=exp_name,
        overwrite=overwrite,
    )
    method_name = cmp_cg.method_name_for(
        alpha_multiplier=1.0, calibrated=False,
        correction_window=correction_window,
    )
    summary = json.loads(
        _summary_path_for(cap_path.parent, method_name).read_text()
    )
    K = float(summary["K_clip"])
    alpha0 = (BARRIER_TARGET / K) if K > 1e-9 else float("inf")
    return K, alpha0, cap_path


def run_strength_sweep(
    *,
    cell: PairSeedCell,
    ctx: MethodCtx,
    exp_name: str,
    alpha0: float,
    alphas: tuple[float, ...] = ALPHA_GRID,
    correction_window: tuple[int, int] = DEFAULT_CORRECTION_WINDOW,
    target_prompt: str = DEFAULT_TARGET_PROMPT,
    decode_strategy: str = "full_vae",
    grad_norm_clip: float | None = None,
    overwrite: bool = False,
) -> dict[float, Path]:
    paths: dict[float, Path] = {}
    for mult in alphas:
        print(f"[idea5b] sweep α/α₀={mult:.2f}")
        path = cmp_cg.run(
            cell, ctx,
            force_scaler=float(alpha0),
            alpha_multiplier=float(mult),
            calibrated=False,
            schedule="constant",
            correction_window=correction_window,
            target_prompt=target_prompt,
            decode_strategy=decode_strategy,
            grad_norm_clip=grad_norm_clip,
            save_residuals=True,
            save_trajectory=True,
            exp_name=exp_name,
            overwrite=overwrite,
        )
        paths[float(mult)] = path
    return paths


def run_calibrated(
    *,
    cell: PairSeedCell,
    ctx: MethodCtx,
    exp_name: str,
    alpha0: float,
    schedule: str = "constant",
    correction_window: tuple[int, int] = DEFAULT_CORRECTION_WINDOW,
    closed_loop_threshold: float = 0.5,
    target_prompt: str = DEFAULT_TARGET_PROMPT,
    decode_strategy: str = "full_vae",
    grad_norm_clip: float | None = None,
    overwrite: bool = False,
) -> Path:
    return cmp_cg.run(
        cell, ctx,
        force_scaler=float(alpha0),
        alpha_multiplier=1.0,
        calibrated=True,
        schedule=schedule,
        correction_window=correction_window,
        closed_loop_threshold=float(closed_loop_threshold),
        target_prompt=target_prompt,
        decode_strategy=decode_strategy,
        grad_norm_clip=grad_norm_clip,
        save_residuals=True,
        save_trajectory=True,
        exp_name=exp_name,
        overwrite=overwrite,
    )


def run_reference_baselines(
    *,
    cell: PairSeedCell,
    ctx: MethodCtx,
    exp_name: str,
    overwrite: bool = False,
) -> dict[str, Path]:
    proxy = MethodCtx(
        models=ctx.models, scheduler=ctx.scheduler,
        output_root=ctx.output_root / exp_name,
        device=ctx.device, dtype=ctx.dtype,
        guidance_scale=ctx.guidance_scale,
        num_inference_steps=ctx.num_inference_steps,
        joint_template=ctx.joint_template, synth=ctx.synth,
    )
    poe_p = run_method("poe", cell, proxy, overwrite=overwrite)
    mono_p = run_method("mono", cell, proxy, overwrite=overwrite)
    return {"poe": poe_p, "mono": mono_p}


def make_cell(prompt_a: str, prompt_b: str, seed: int) -> PairSeedCell:
    return cell_for(prompt_a, prompt_b, seed)
