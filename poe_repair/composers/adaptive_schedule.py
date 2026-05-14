"""Idea 2 — adaptive-schedule composer.

A *thin dispatcher* that:

  1. Builds an ``AdaptiveSchedule`` (basin monitor + trigger rule) for
     the cell.
  2. Routes to the appropriate inner composer based on
     ``force_source ∈ {residual, force_a, force_b, clip}``, passing the
     adaptive schedule through.
  3. Encodes the trigger config in the method-name so multiple sweep
     points coexist on disk.

No new sampling logic; only schedule glue.

Method-name format::

    adaptive_<force>_<rule><theta_pct>[_K<K>][_lb<lb>]

where::

    <force>  ∈ {residual, force_a, force_b, clip}
    <rule>   ∈ {thr, pers, vel}
    <theta_pct> = round(θ * 100), e.g. θ=0.30 → "thr030"
    K        only present for persistence rule
    lb       only present for velocity rule (lookback)
"""

from __future__ import annotations

from pathlib import Path

from poe_repair.composers import clip_guided as cmp_clip
from poe_repair.composers import poe_internal as cmp_poe_internal
from poe_repair.composers import teacher_residual as cmp_residual
from poe_repair.config import RunConfig
from poe_repair.methods._adaptive_schedule import (
    AdaptiveSchedule,
    build_adaptive_schedule,
)
from poe_repair.run import MethodCtx
from poe_repair.runtime import PairSeedCell


VALID_FORCES = ("residual", "force_a", "force_b", "clip")
VALID_RULES = ("threshold", "persistence", "velocity")

_RULE_TAG = {
    "threshold": "thr",
    "persistence": "pers",
    "velocity": "vel",
}


def method_name_for(
    *,
    force_source: str,
    rule: str,
    theta: float,
    persistence_K: int = 3,
    velocity_lookback: int = 2,
) -> str:
    if force_source not in VALID_FORCES:
        raise ValueError(f"force_source must be one of {VALID_FORCES}, got {force_source!r}")
    if rule not in VALID_RULES:
        raise ValueError(f"rule must be one of {VALID_RULES}, got {rule!r}")
    tag = _RULE_TAG[rule]
    name = f"adaptive_{force_source}_{tag}{int(round(theta * 100)):03d}"
    if rule == "persistence":
        name += f"_K{int(persistence_K)}"
    elif rule == "velocity":
        name += f"_lb{int(velocity_lookback)}"
    return name


def run(
    cell: PairSeedCell,
    ctx: MethodCtx,
    *,
    force_source: str,
    rule: str,
    theta: float,
    persistence_K: int = 3,
    velocity_lookback: int = 2,
    schedule_max: float = 1.0,
    # Force-source-specific knobs
    force_scaler: float = 1.0,                          # for force_a / force_b / clip
    correction_window: tuple[int, int] | None = None,    # for clip / poe_internal
    target_prompt: str = "a cat and a dog",              # for clip
    decode_strategy: str = "full_vae",                   # for clip
    grad_norm_clip: float | None = None,                 # for clip
    # Persistence
    save_residuals: bool = False,
    save_trajectory: bool = True,
    exp_name: str = "tmp",
    overwrite: bool = False,
) -> Path:
    """Run an adaptive-schedule wrapper on (cell, knobs); return image path.

    The per-fire intensity is ``schedule_max``. Total inference budget
    is ``schedule_max × number_of_fires``, where the number of fires is
    decided at runtime by the trigger rule on the basin monitor.
    """
    if force_source not in VALID_FORCES:
        raise ValueError(f"force_source must be one of {VALID_FORCES}, got {force_source!r}")

    cfg = RunConfig()
    sched: AdaptiveSchedule = build_adaptive_schedule(
        pair_slug=cell.pair_slug, seed=cell.seed,
        output_root=cfg.paths.output_root,
        rule=rule, theta=float(theta),
        persistence_K=int(persistence_K),
        velocity_lookback=int(velocity_lookback),
    )

    method_name = method_name_for(
        force_source=force_source, rule=rule, theta=theta,
        persistence_K=persistence_K, velocity_lookback=velocity_lookback,
    )

    if force_source == "residual":
        return cmp_residual.run(
            cell, ctx,
            lambda_schedule="constant",
            lambda_max=float(schedule_max),
            correction_window=correction_window,
            save_residuals=save_residuals,
            save_x0_estimates=False,
            save_trajectory=save_trajectory,
            adaptive_schedule=sched,
            method_name_override=method_name,
            exp_name=exp_name,
            overwrite=overwrite,
        )

    if force_source in ("force_a", "force_b"):
        force_kind = "overlap" if force_source == "force_a" else "alignment"
        return cmp_poe_internal.run(
            cell, ctx,
            force_kind=force_kind,
            force_scaler=float(force_scaler),
            alpha_multiplier=1.0,
            calibrated=False,
            schedule="constant",
            schedule_max=float(schedule_max),
            correction_window=correction_window,
            adaptive_schedule=sched,
            method_name_override=method_name,
            save_residuals=save_residuals,
            save_trajectory=save_trajectory,
            exp_name=exp_name,
            overwrite=overwrite,
        )

    if force_source == "clip":
        return cmp_clip.run(
            cell, ctx,
            force_scaler=float(force_scaler),
            alpha_multiplier=1.0,
            calibrated=False,
            schedule="constant",
            schedule_max=float(schedule_max),
            correction_window=correction_window or (10, 25),
            adaptive_schedule=sched,
            method_name_override=method_name,
            target_prompt=target_prompt,
            decode_strategy=decode_strategy,
            grad_norm_clip=grad_norm_clip,
            save_residuals=save_residuals,
            save_trajectory=save_trajectory,
            exp_name=exp_name,
            overwrite=overwrite,
        )

    raise AssertionError(f"unhandled force_source {force_source!r}")
