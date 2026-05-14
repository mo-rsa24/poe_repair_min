"""Stage 2 — trigger sweep + matched-budget constant runs.

For one ``force_source``, sweeps over (rule, theta, K) combinations.
For each smart run, also produces a constant-schedule comparator at
matched total injected budget.
"""

from __future__ import annotations

from pathlib import Path

from poe_repair.composers import adaptive_schedule as cmp_adapt
from poe_repair.composers import clip_guided as cmp_clip
from poe_repair.composers import poe_internal as cmp_poe_internal
from poe_repair.composers import teacher_residual as cmp_residual
from poe_repair.experiments._eval_common import cell_for
from poe_repair.run import MethodCtx, run_method
from poe_repair.runtime import PairSeedCell


THETA_GRID: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6)
PERSISTENCE_K: int = 3
VELOCITY_LOOKBACK: int = 2


def make_cell(prompt_a: str, prompt_b: str, seed: int) -> PairSeedCell:
    return cell_for(prompt_a, prompt_b, seed)


def run_trigger_sweep(
    *,
    cell: PairSeedCell,
    ctx: MethodCtx,
    exp_name: str,
    force_source: str,
    rules: tuple[str, ...] = ("threshold", "persistence", "velocity"),
    thetas: tuple[float, ...] = THETA_GRID,
    persistence_K: int = PERSISTENCE_K,
    velocity_lookback: int = VELOCITY_LOOKBACK,
    schedule_max: float = 1.0,
    force_scaler: float = 1.0,
    correction_window: tuple[int, int] | None = None,
    target_prompt: str = "a cat and a dog",
    decode_strategy: str = "full_vae",
    grad_norm_clip: float | None = None,
    overwrite: bool = False,
) -> dict[tuple[str, float], Path]:
    """Run one trigger sweep entry per (rule, theta).

    Returns ``{(rule, theta): image_path}``.
    """
    paths: dict[tuple[str, float], Path] = {}
    for rule in rules:
        for theta in thetas:
            print(f"[idea2] {force_source}  rule={rule}  θ={theta:.2f}")
            path = cmp_adapt.run(
                cell, ctx,
                force_source=force_source,
                rule=rule, theta=float(theta),
                persistence_K=persistence_K,
                velocity_lookback=velocity_lookback,
                schedule_max=float(schedule_max),
                force_scaler=float(force_scaler),
                correction_window=correction_window,
                target_prompt=target_prompt,
                decode_strategy=decode_strategy,
                grad_norm_clip=grad_norm_clip,
                save_residuals=False,
                save_trajectory=True,
                exp_name=exp_name,
                overwrite=overwrite,
            )
            paths[(rule, float(theta))] = path
    return paths


def run_constant_match(
    *,
    cell: PairSeedCell,
    ctx: MethodCtx,
    exp_name: str,
    force_source: str,
    alpha_multipliers: tuple[float, ...],
    schedule_max: float = 1.0,
    force_scaler: float = 1.0,
    correction_window: tuple[int, int] | None = None,
    target_prompt: str = "a cat and a dog",
    decode_strategy: str = "full_vae",
    grad_norm_clip: float | None = None,
    overwrite: bool = False,
) -> dict[float, Path]:
    """Run constant-schedule comparators at the requested α multipliers.

    For ``residual``: ``lambda_max = mult``.
    For ``force_a/force_b/clip``: ``alpha_multiplier = mult`` against the
    inherited ``force_scaler``.
    """
    paths: dict[float, Path] = {}
    for mult in alpha_multipliers:
        method_name_suffix = f"adaptive_{force_source}_constant_match_alpha{int(round(mult * 100)):03d}"
        print(f"[idea2] {force_source}  constant-match α/α₀={mult:.2f}")

        if force_source == "residual":
            p = cmp_residual.run(
                cell, ctx,
                lambda_schedule="constant",
                lambda_max=float(mult),
                correction_window=correction_window,
                save_residuals=False,
                save_trajectory=True,
                method_name_override=method_name_suffix,
                exp_name=exp_name,
                overwrite=overwrite,
            )
        elif force_source in ("force_a", "force_b"):
            kind = "overlap" if force_source == "force_a" else "alignment"
            p = cmp_poe_internal.run(
                cell, ctx,
                force_kind=kind,
                force_scaler=float(force_scaler),
                alpha_multiplier=float(mult),
                calibrated=False,
                schedule="constant",
                schedule_max=float(mult),
                correction_window=correction_window,
                method_name_override=method_name_suffix,
                save_residuals=False,
                save_trajectory=True,
                exp_name=exp_name,
                overwrite=overwrite,
            )
        elif force_source == "clip":
            p = cmp_clip.run(
                cell, ctx,
                force_scaler=float(force_scaler),
                alpha_multiplier=float(mult),
                calibrated=False,
                schedule="constant",
                schedule_max=float(mult),
                correction_window=correction_window or (10, 25),
                target_prompt=target_prompt,
                decode_strategy=decode_strategy,
                grad_norm_clip=grad_norm_clip,
                method_name_override=method_name_suffix,
                save_residuals=False,
                save_trajectory=True,
                exp_name=exp_name,
                overwrite=overwrite,
            )
        else:
            raise ValueError(f"unknown force_source {force_source!r}")
        paths[float(mult)] = p
    return paths


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
    return {
        "poe": run_method("poe", cell, proxy, overwrite=overwrite),
        "mono": run_method("mono", cell, proxy, overwrite=overwrite),
    }
