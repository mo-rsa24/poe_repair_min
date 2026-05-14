"""Stage 2 — distance tables, gradient stats, four-method comparison.

Reuses ``poe_repair.experiments.veracity.metrics`` for image / latent
distances; adds idea5b-specific helpers for CLIP-grad capacity, per-step
similarity, and a method-comparison table that overlays this
experiment's ``d_Mono`` curves alongside veracity Δ + idea1 Force-A +
idea1 Force-B.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from poe_repair.composers.clip_guided import method_name_for
from poe_repair.experiments.idea5b.sweep import (
    ALPHA_GRID,
    BARRIER_TARGET,
    DEFAULT_CORRECTION_WINDOW,
)
from poe_repair.experiments.veracity import metrics as VM


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def run_dir_for_alpha(seed_dir: Path, alpha_mult: float) -> Path:
    name = method_name_for(
        alpha_multiplier=alpha_mult, calibrated=False,
        correction_window=DEFAULT_CORRECTION_WINDOW,
    )
    return seed_dir / name


def run_dir_calibrated(
    seed_dir: Path, schedule: str = "constant",
    correction_window: tuple[int, int] | None = None,
) -> Path:
    name = method_name_for(
        alpha_multiplier=None, calibrated=True,
        schedule=schedule,
        correction_window=correction_window,
    )
    return seed_dir / name


def summary_json_for(run_dir: Path) -> Path:
    return run_dir / f"summary_{run_dir.name}.json"


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------


def load_clip_capacity(seed_dir: Path) -> dict:
    cap_dir = run_dir_for_alpha(seed_dir, 1.0)
    summary = json.loads(summary_json_for(cap_dir).read_text())
    K = float(summary["K_clip"])
    alpha0 = (BARRIER_TARGET / K) if K > 1e-9 else float("inf")
    return {
        "K_clip": K,
        "alpha_zero": alpha0,
        "barrier_target": BARRIER_TARGET,
        "force_norm_per_step": list(summary["force_norm_per_step"]),
        "grad_norm_per_step": list(summary["grad_norm_per_step"]),
        "clip_similarity_per_step": list(summary["clip_similarity_per_step"]),
        "alpha_per_step": list(summary["alpha_per_step"]),
        "in_correction_window_per_step": list(summary["in_correction_window_per_step"]),
        "schedule": summary["schedule"],
        "correction_window": list(summary["correction_window"]),
        "decode_strategy": summary["decode_strategy"],
        "passes_capacity_floor": K >= 500.0,
    }


# ---------------------------------------------------------------------------
# Distance tables
# ---------------------------------------------------------------------------


def _final_latent(run_dir: Path) -> torch.Tensor:
    return VM.load_final_latent(run_dir)


def compute_distance_table(
    *,
    seed_dir: Path,
    poe_image_path: Path,
    mono_image_path: Path,
    poe_run_dir: Path | None = None,
    mono_run_dir: Path | None = None,
    alphas: tuple[float, ...] = ALPHA_GRID,
    device: torch.device | None = None,
) -> dict:
    run_dirs = {mult: run_dir_for_alpha(seed_dir, mult) for mult in alphas}
    image_paths = [
        run_dirs[mult] / f"{run_dirs[mult].name}.png" for mult in alphas
    ]

    embeds = VM.clip_image_embed(
        [poe_image_path, mono_image_path, *image_paths], device=device,
    )
    poe_emb, mono_emb, sweep_embeds = embeds[0], embeds[1], embeds[2:]
    d_poe_clip = [
        VM.clip_cosine_distance(sweep_embeds[i], poe_emb)
        for i in range(len(alphas))
    ]
    d_mono_clip = [
        VM.clip_cosine_distance(sweep_embeds[i], mono_emb)
        for i in range(len(alphas))
    ]

    poe_latent = (
        _final_latent(poe_run_dir)
        if poe_run_dir is not None and (poe_run_dir / "latent_trajectory.pt").exists()
        else _final_latent(run_dirs[alphas[0]])
    )
    mono_latent = (
        _final_latent(mono_run_dir)
        if mono_run_dir is not None and (mono_run_dir / "latent_trajectory.pt").exists()
        else _final_latent(run_dirs[alphas[-1]])
    )

    d_poe_l2: list[float] = []
    d_mono_l2: list[float] = []
    for mult in alphas:
        x = _final_latent(run_dirs[mult])
        d_poe_l2.append(VM.latent_l2(x, poe_latent))
        d_mono_l2.append(VM.latent_l2(x, mono_latent))

    return {
        "alphas": list(alphas),
        "anchor_poe_path": str(poe_image_path),
        "anchor_mono_path": str(mono_image_path),
        "anchor_poe_run_dir": (str(poe_run_dir) if poe_run_dir else None),
        "anchor_mono_run_dir": (str(mono_run_dir) if mono_run_dir else None),
        "latent_l2": {"d_poe": d_poe_l2, "d_mono": d_mono_l2},
        "clip_image_cosine": {"d_poe": d_poe_clip, "d_mono": d_mono_clip},
    }


# ---------------------------------------------------------------------------
# Per-step gradient stats
# ---------------------------------------------------------------------------


def compute_grad_stats(
    *,
    seed_dir: Path,
    alphas: tuple[float, ...] = ALPHA_GRID,
) -> dict:
    run_dirs = {mult: run_dir_for_alpha(seed_dir, mult) for mult in alphas}
    cap_dir = run_dirs[1.0]
    cap_summary = json.loads(summary_json_for(cap_dir).read_text())

    grad_norm_per_step = list(cap_summary["grad_norm_per_step"])
    force_norm_per_step = list(cap_summary["force_norm_per_step"])
    clip_similarity_per_step = list(cap_summary["clip_similarity_per_step"])
    K = float(cap_summary["K_clip"])

    total_injected: list[float] = []
    for mult in alphas:
        s = json.loads(summary_json_for(run_dirs[mult]).read_text())
        total_injected.append(float(sum(s["force_norm_per_step"])))

    stab = _direction_stability_for_clip(cap_dir)

    return {
        "alphas": list(alphas),
        "K_clip": K,
        "grad_norm_per_step": grad_norm_per_step,
        "force_norm_per_step": force_norm_per_step,
        "clip_similarity_per_step": clip_similarity_per_step,
        "total_injected_per_alpha": total_injected,
        "direction_stability_matrix": stab.tolist(),
        "num_steps": stab.shape[0],
        "anchor_run_dir": str(cap_dir),
    }


def _direction_stability_for_clip(run_dir: Path) -> torch.Tensor:
    res_dir = run_dir / "residuals"
    files = sorted(res_dir.glob("step_*.pt"))
    flats: list[torch.Tensor] = []
    for f in files:
        payload = torch.load(f, map_location="cpu", weights_only=False)
        if "force" not in payload:
            continue
        flat = payload["force"].float().flatten()
        n = flat.norm()
        if float(n.item()) <= 1e-12:
            continue
        flat = flat / (n + 1e-12)
        flats.append(flat)
    if not flats:
        return torch.zeros(0, 0)
    mat = torch.stack(flats, dim=0)
    return (mat @ mat.t()).float()


# ---------------------------------------------------------------------------
# Four-method comparison
# ---------------------------------------------------------------------------


def compute_method_comparison(
    *,
    distances_idea5b: dict,
    grad_stats_idea5b: dict,
    veracity_distances_path: Path,
    veracity_residual_stats_path: Path,
    idea1_distances_path: Path | None = None,
    idea1_residual_stats_path: Path | None = None,
) -> dict:
    """Build the headline overlay: d_Mono vs total integrated correction
    for veracity Δ + (optional) idea1 Force-A / Force-B + idea5b CLIP.
    """
    series: dict[str, dict] = {}

    veracity_distances = json.loads(veracity_distances_path.read_text())
    veracity_residual = json.loads(veracity_residual_stats_path.read_text())
    series["veracity_residual"] = {
        "x_total_injected": list(veracity_residual["total_injected_per_lambda"]),
        "lambdas": list(veracity_distances["lambdas"]),
        "d_mono_l2": list(veracity_distances["latent_l2"]["d_mono"]),
        "d_mono_clip": list(veracity_distances["clip_image_cosine"]["d_mono"]),
    }

    if (
        idea1_distances_path is not None and idea1_distances_path.exists()
        and idea1_residual_stats_path is not None and idea1_residual_stats_path.exists()
    ):
        idea1_distances = json.loads(idea1_distances_path.read_text())
        idea1_residual = json.loads(idea1_residual_stats_path.read_text())
        for fk in ("overlap", "alignment"):
            if fk not in idea1_distances or fk not in idea1_residual:
                continue
            stats = idea1_residual[fk]
            dist = idea1_distances[fk]
            series[f"force_{fk}"] = {
                "x_total_injected": list(stats["total_injected_per_alpha"]),
                "alphas": list(dist["alphas"]),
                "d_mono_l2": list(dist["latent_l2"]["d_mono"]),
                "d_mono_clip": list(dist["clip_image_cosine"]["d_mono"]),
            }

    series["clip_guided"] = {
        "x_total_injected": list(grad_stats_idea5b["total_injected_per_alpha"]),
        "alphas": list(distances_idea5b["alphas"]),
        "d_mono_l2": list(distances_idea5b["latent_l2"]["d_mono"]),
        "d_mono_clip": list(distances_idea5b["clip_image_cosine"]["d_mono"]),
    }

    return {"series": series, "barrier_target": BARRIER_TARGET}


# ---------------------------------------------------------------------------
# Trajectory distance per step
# ---------------------------------------------------------------------------


def trajectory_distance_per_step(
    *,
    seed_dir: Path,
    mono_run_dir: Path,
    alphas: tuple[float, ...] = ALPHA_GRID,
) -> dict:
    mono_traj = torch.load(
        mono_run_dir / "latent_trajectory.pt", map_location="cpu", weights_only=False,
    )["trajectories"].float()

    per_alpha: dict[float, list[float]] = {}
    for mult in alphas:
        run_dir = run_dir_for_alpha(seed_dir, mult)
        traj = torch.load(
            run_dir / "latent_trajectory.pt", map_location="cpu", weights_only=False,
        )["trajectories"].float()
        if traj.shape != mono_traj.shape:
            raise ValueError(
                f"trajectory shape mismatch for α={mult}: {traj.shape} vs {mono_traj.shape}"
            )
        diff = (traj - mono_traj).flatten(start_dim=1)
        n_per = diff.shape[1]
        per_alpha[mult] = (diff.norm(dim=1) / math.sqrt(max(1, n_per))).tolist()

    return {
        "alphas": list(alphas),
        "num_steps_plus_one": int(mono_traj.shape[0]),
        "per_alpha_distance_to_mono": {
            f"{m:.2f}": per_alpha[m] for m in alphas
        },
    }
