"""Stage 1 — capacity check + eleven-point strength sweep per force.

The barrier height read from veracity sets the calibration target
(`BARRIER_TARGET = 1500`). For each force variant we run one capacity
check at ``force_scaler = 1.0`` to measure
``K_force = Σ_t ‖F_t(α=1)‖``, then set ``α₀ = BARRIER_TARGET / K_force``
and sweep ``alpha_multiplier ∈ ALPHA_GRID``.

Method names encode the multiplier so eleven runs per force coexist on
disk under the same cell directory.
"""

from __future__ import annotations

import json
from pathlib import Path

from poe_repair.composers import poe_internal as cmp_pi
from poe_repair.experiments._eval_common import cell_for
from poe_repair.run import MethodCtx, run_method
from poe_repair.runtime import PairSeedCell


ALPHA_GRID: tuple[float, ...] = (
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5,
)
FORCE_KINDS: tuple[str, ...] = ("overlap", "alignment")
BARRIER_TARGET: float = 1500.0


def _summary_path_for(method_dir: Path, method_name: str) -> Path:
    return method_dir / f"summary_{method_name}.json"


def capacity_check(
    *,
    cell: PairSeedCell,
    ctx: MethodCtx,
    exp_name: str,
    force_kind: str,
    overwrite: bool = False,
) -> tuple[float, float, Path]:
    """Run one ``force_scaler=1.0`` pass and return ``(K_force, α₀, path)``.

    The capacity-check method-name is ``poe_internal_<force>_alpha100``.
    Idempotent: if the summary already exists, read K_force from it.
    """
    cap_path = cmp_pi.run(
        cell, ctx,
        force_kind=force_kind,
        force_scaler=1.0,
        alpha_multiplier=1.0,
        calibrated=False,
        schedule="constant",
        save_residuals=True,
        save_trajectory=True,
        exp_name=exp_name,
        overwrite=overwrite,
    )
    method_name = cmp_pi.method_name_for(
        force_kind=force_kind, alpha_multiplier=1.0, calibrated=False,
    )
    summary = json.loads(
        _summary_path_for(cap_path.parent, method_name).read_text()
    )
    K = float(summary["K_force"])
    alpha0 = (BARRIER_TARGET / K) if K > 1e-9 else float("inf")
    return K, alpha0, cap_path


def run_strength_sweep(
    *,
    cell: PairSeedCell,
    ctx: MethodCtx,
    exp_name: str,
    force_kind: str,
    alpha0: float,
    alphas: tuple[float, ...] = ALPHA_GRID,
    overwrite: bool = False,
) -> dict[float, Path]:
    """Run the eleven-point sweep for one force variant.

    ``alphas`` are *multipliers* on ``α₀``. The scaler passed to the
    sampler is ``force_scaler = α₀`` and ``schedule_max =
    alpha_multiplier``, so the final strength is ``alpha_multiplier ·
    α₀``.
    """
    paths: dict[float, Path] = {}
    for mult in alphas:
        print(f"[idea1] sweep {force_kind} α/α₀={mult:.2f}")
        path = cmp_pi.run(
            cell, ctx,
            force_kind=force_kind,
            force_scaler=float(alpha0),
            alpha_multiplier=float(mult),
            calibrated=False,
            schedule="constant",
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
    force_kind: str,
    alpha0: float,
    schedule: str = "constant",
    closed_loop_threshold: float = 0.5,
    overwrite: bool = False,
) -> Path:
    """Run a single calibrated (alpha == α₀) job under the given schedule."""
    return cmp_pi.run(
        cell, ctx,
        force_kind=force_kind,
        force_scaler=float(alpha0),
        alpha_multiplier=1.0,
        calibrated=True,
        schedule=schedule,
        closed_loop_threshold=float(closed_loop_threshold),
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
    """Render PoE and Mono baselines into ``outputs/<exp>/<method>/...``.

    Uses the cached dispatcher; idempotent.
    """
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
