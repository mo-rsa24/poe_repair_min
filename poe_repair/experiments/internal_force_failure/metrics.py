"""Stage 2 — distance tables, force stats, method comparison.

Reuses ``poe_repair.experiments.residual_between_mono_and_poe.metrics`` for image / latent
distances; adds idea1-specific helpers for force capacity, force-norm
stats, and a cross-method comparison table that overlays this
experiment's ``d_Mono`` curves with veracity's residual-Δ curves.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from poe_repair.composers.poe_internal import method_name_for
from poe_repair.experiments.internal_force_failure.sweep import ALPHA_GRID, BARRIER_TARGET
from poe_repair.experiments.residual_between_mono_and_poe import metrics as VM


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def run_dir_for_alpha(seed_dir: Path, force_kind: str, alpha_mult: float) -> Path:
    name = method_name_for(
        force_kind=force_kind,
        alpha_multiplier=alpha_mult,
        calibrated=False,
    )
    return seed_dir / name


def run_dir_calibrated(seed_dir: Path, force_kind: str, schedule: str = "constant") -> Path:
    name = method_name_for(
        force_kind=force_kind, alpha_multiplier=None, calibrated=True,
        schedule=schedule,
    )
    return seed_dir / name


def baseline_run_dir(idea1_root: Path, kind: str, pair_slug: str, seed: int) -> Path:
    """Layout: outputs/internal_force_failure/<kind>/pairs/<slug>/seed_<n>/<kind>.png."""
    return idea1_root / kind / "pairs" / pair_slug / f"seed_{seed}"


def summary_json_for(run_dir: Path) -> Path:
    return run_dir / f"summary_{run_dir.name}.json"


# ---------------------------------------------------------------------------
# Force capacity
# ---------------------------------------------------------------------------


def load_force_capacity(seed_dir: Path, force_kind: str) -> dict:
    """Read K_force and α₀ from the alpha=1.0 capacity-check run."""
    cap_dir = run_dir_for_alpha(seed_dir, force_kind, 1.0)
    summary = json.loads(summary_json_for(cap_dir).read_text())
    K = float(summary["K_force"])
    alpha0 = (BARRIER_TARGET / K) if K > 1e-9 else float("inf")
    return {
        "force_kind": force_kind,
        "K_force": K,
        "alpha_zero": alpha0,
        "barrier_target": BARRIER_TARGET,
        "force_norm_per_step": list(summary["force_norm_per_step"]),
        "scaled_force_norm_per_step": list(summary["scaled_force_norm_per_step"]),
        "alpha_per_step": list(summary["alpha_per_step"]),
        "schedule": summary["schedule"],
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
    force_kind: str,
    poe_image_path: Path,
    mono_image_path: Path,
    poe_run_dir: Path | None = None,
    mono_run_dir: Path | None = None,
    alphas: tuple[float, ...] = ALPHA_GRID,
    device: torch.device | None = None,
) -> dict:
    """For one force variant, compute d_PoE(α) and d_Mono(α).

    Latent-L2 anchors come from saved trajectories: if a veracity λ=0
    run is supplied via ``poe_run_dir``, its final latent is the PoE
    anchor; otherwise we fall back to the α=0 idea1 run (which should
    reproduce PoE since the force is multiplied by 0). Same for Mono.
    """
    run_dirs = {
        mult: run_dir_for_alpha(seed_dir, force_kind, mult) for mult in alphas
    }
    image_paths = [
        run_dirs[mult] / f"{run_dirs[mult].name}.png" for mult in alphas
    ]

    # CLIP image cosine — anchored on PoE/Mono PNGs (cached baselines).
    embeds = VM.clip_image_embed(
        [poe_image_path, mono_image_path, *image_paths], device=device,
    )
    poe_emb, mono_emb, sweep_embeds = embeds[0], embeds[1], embeds[2:]
    d_poe_clip: list[float] = [
        VM.clip_cosine_distance(sweep_embeds[i], poe_emb)
        for i in range(len(alphas))
    ]
    d_mono_clip: list[float] = [
        VM.clip_cosine_distance(sweep_embeds[i], mono_emb)
        for i in range(len(alphas))
    ]

    # Latent L2 — anchors from saved trajectories where available.
    if poe_run_dir is not None and (poe_run_dir / "latent_trajectory.pt").exists():
        poe_latent = _final_latent(poe_run_dir)
    else:
        poe_latent = _final_latent(run_dirs[alphas[0]])  # α=0 fallback

    if mono_run_dir is not None and (mono_run_dir / "latent_trajectory.pt").exists():
        mono_latent = _final_latent(mono_run_dir)
    else:
        # Fall back to the highest-α run as Mono surrogate.
        mono_latent = _final_latent(run_dirs[alphas[-1]])

    d_poe_l2: list[float] = []
    d_mono_l2: list[float] = []
    for mult in alphas:
        x = _final_latent(run_dirs[mult])
        d_poe_l2.append(VM.latent_l2(x, poe_latent))
        d_mono_l2.append(VM.latent_l2(x, mono_latent))

    return {
        "force_kind": force_kind,
        "alphas": list(alphas),
        "anchor_poe_path": str(poe_image_path),
        "anchor_mono_path": str(mono_image_path),
        "anchor_poe_run_dir": (str(poe_run_dir) if poe_run_dir else None),
        "anchor_mono_run_dir": (str(mono_run_dir) if mono_run_dir else None),
        "latent_l2": {"d_poe": d_poe_l2, "d_mono": d_mono_l2},
        "clip_image_cosine": {"d_poe": d_poe_clip, "d_mono": d_mono_clip},
    }


# ---------------------------------------------------------------------------
# Force-norm + direction-stability stats
# ---------------------------------------------------------------------------


def compute_force_stats(
    *,
    seed_dir: Path,
    force_kind: str,
    alphas: tuple[float, ...] = ALPHA_GRID,
) -> dict:
    """Aggregate per-step ‖F_t‖ trajectory + total injected force per α.

    Direction-stability matrix is the T×T cosine of saved ``force``
    tensors at the α=1.0 capacity-check run (force is α-independent up
    to the schedule, so any non-zero α gives the same direction matrix).
    """
    run_dirs = {
        mult: run_dir_for_alpha(seed_dir, force_kind, mult) for mult in alphas
    }
    cap_dir = run_dirs[1.0]
    cap_summary = json.loads(summary_json_for(cap_dir).read_text())

    force_norm_per_step = list(cap_summary["force_norm_per_step"])
    K = float(cap_summary["K_force"])

    total_injected: list[float] = []
    for mult in alphas:
        s = json.loads(summary_json_for(run_dirs[mult]).read_text())
        total_injected.append(float(sum(s["scaled_force_norm_per_step"])))

    stab = _direction_stability_for_force(cap_dir)

    return {
        "force_kind": force_kind,
        "alphas": list(alphas),
        "K_force": K,
        "force_norm_per_step": force_norm_per_step,
        "total_injected_per_alpha": total_injected,
        "direction_stability_matrix": stab.tolist(),
        "num_steps": stab.shape[0],
        "anchor_run_dir": str(cap_dir),
    }


def _direction_stability_for_force(run_dir: Path) -> torch.Tensor:
    """T × T cosine similarity of the per-step ``force`` tensors."""
    res_dir = run_dir / "residuals"
    files = sorted(res_dir.glob("step_*.pt"))
    flats: list[torch.Tensor] = []
    for f in files:
        payload = torch.load(f, map_location="cpu")
        flat = payload["force"].float().flatten()
        n = flat.norm()
        flat = flat / (n + 1e-12)
        flats.append(flat)
    if not flats:
        return torch.zeros(0, 0)
    mat = torch.stack(flats, dim=0)
    return (mat @ mat.t()).float()


# ---------------------------------------------------------------------------
# Method comparison vs veracity residual
# ---------------------------------------------------------------------------


def compute_method_comparison(
    *,
    distances_by_force: dict[str, dict],
    veracity_distances_path: Path,
    veracity_residual_stats_path: Path,
    force_stats_by_force: dict[str, dict],
) -> dict:
    """Build the headline overlay: d_Mono vs total integrated correction
    for veracity Δ + each idea1 force.
    """
    veracity_distances = json.loads(veracity_distances_path.read_text())
    veracity_residual = json.loads(veracity_residual_stats_path.read_text())

    veracity_lambdas = list(veracity_distances["lambdas"])
    veracity_total = list(veracity_residual["total_injected_per_lambda"])
    veracity_d_mono_l2 = list(veracity_distances["latent_l2"]["d_mono"])
    veracity_d_mono_clip = list(veracity_distances["clip_image_cosine"]["d_mono"])

    series: dict[str, dict] = {
        "veracity_residual": {
            "x_total_injected": veracity_total,
            "lambdas": veracity_lambdas,
            "d_mono_l2": veracity_d_mono_l2,
            "d_mono_clip": veracity_d_mono_clip,
        },
    }
    for force_kind, dist in distances_by_force.items():
        stats = force_stats_by_force[force_kind]
        series[f"force_{force_kind}"] = {
            "x_total_injected": list(stats["total_injected_per_alpha"]),
            "alphas": list(dist["alphas"]),
            "d_mono_l2": list(dist["latent_l2"]["d_mono"]),
            "d_mono_clip": list(dist["clip_image_cosine"]["d_mono"]),
        }
    return {"series": series, "barrier_target": BARRIER_TARGET}


# ---------------------------------------------------------------------------
# Trajectory distance per step (mirror of veracity helper)
# ---------------------------------------------------------------------------


def trajectory_distance_per_step(
    *,
    seed_dir: Path,
    force_kind: str,
    mono_run_dir: Path,
    alphas: tuple[float, ...] = ALPHA_GRID,
) -> dict:
    """For each α, return ‖x_t(α) − x_t(Mono)‖ per saved trajectory entry."""
    mono_traj = torch.load(
        mono_run_dir / "latent_trajectory.pt", map_location="cpu",
    )["trajectories"].float()

    per_alpha: dict[float, list[float]] = {}
    for mult in alphas:
        run_dir = run_dir_for_alpha(seed_dir, force_kind, mult)
        traj = torch.load(
            run_dir / "latent_trajectory.pt", map_location="cpu",
        )["trajectories"].float()
        if traj.shape != mono_traj.shape:
            raise ValueError(
                f"trajectory shape mismatch for α={mult}: {traj.shape} vs {mono_traj.shape}"
            )
        diff = (traj - mono_traj).flatten(start_dim=1)
        n_per = diff.shape[1]
        per_alpha[mult] = (diff.norm(dim=1) / math.sqrt(max(1, n_per))).tolist()

    return {
        "force_kind": force_kind,
        "alphas": list(alphas),
        "num_steps_plus_one": int(mono_traj.shape[0]),
        "per_alpha_distance_to_mono": {
            f"{m:.2f}": per_alpha[m] for m in alphas
        },
    }
