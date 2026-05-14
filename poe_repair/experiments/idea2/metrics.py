"""Stage 3 — schedule stats + budget-vs-quality table.

Reuses ``poe_repair.experiments.veracity.metrics`` for image / latent
distances. Adds:

  - ``compute_schedule_stats(run_dir)`` — fired-step count, total
    integrated injection, monitor trace.
  - ``compute_budget_quality_table(...)`` — gathers smart-schedule
    points and constant-schedule points on a shared
    (total_injection, d_mono) plane.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch

from poe_repair.composers.adaptive_schedule import method_name_for as _smart_name
from poe_repair.experiments.veracity import metrics as VM


def smart_run_dir(
    seed_dir: Path, *, force_source: str, rule: str, theta: float,
    persistence_K: int = 3, velocity_lookback: int = 2,
) -> Path:
    name = _smart_name(
        force_source=force_source, rule=rule, theta=theta,
        persistence_K=persistence_K, velocity_lookback=velocity_lookback,
    )
    return seed_dir / name


def constant_match_run_dir(
    seed_dir: Path, *, force_source: str, alpha_mult: float,
) -> Path:
    return (
        seed_dir
        / f"adaptive_{force_source}_constant_match_alpha{int(round(alpha_mult * 100)):03d}"
    )


def summary_json_for(run_dir: Path) -> Path:
    return run_dir / f"summary_{run_dir.name}.json"


def image_path_for(run_dir: Path) -> Path:
    return run_dir / f"{run_dir.name}.png"


def trajectory_path_for(run_dir: Path) -> Path:
    return run_dir / "latent_trajectory.pt"


# ---------------------------------------------------------------------------
# Schedule stats
# ---------------------------------------------------------------------------


def _force_norm_per_step(summary: dict) -> list[float]:
    """Return per-step injection magnitudes regardless of force source.

    Different inner samplers store this under different keys:
      - residual (teacher_residual): we approximate via lambda_per_step
        × delta_norm_per_step.
      - poe_internal (force_a/b): scaled_force_norm_per_step.
      - clip_guided: force_norm_per_step (already includes α × σ_t × ‖g_t‖).
    """
    if "scaled_force_norm_per_step" in summary:
        return list(summary["scaled_force_norm_per_step"])
    if (
        "force_norm_per_step" in summary
        and "alpha_per_step" not in summary
    ):
        # Idea 5b stores force_norm_per_step already scaled by α·σ.
        return list(summary["force_norm_per_step"])
    if (
        "force_norm_per_step" in summary
        and "alpha_per_step" in summary
        and "grad_norm_per_step" in summary
    ):
        # idea5b: use the scaled force_norm directly.
        return list(summary["force_norm_per_step"])
    if (
        "delta_norm_per_step" in summary
        and "lambda_per_step" in summary
    ):
        deltas = list(summary["delta_norm_per_step"])
        lams = list(summary["lambda_per_step"])
        n = min(len(deltas), len(lams))
        return [float(lams[i]) * float(deltas[i]) for i in range(n)]
    return []


def compute_schedule_stats(run_dir: Path) -> dict:
    summary = json.loads(summary_json_for(run_dir).read_text())
    force_per_step = _force_norm_per_step(summary)
    total_injection = float(sum(force_per_step))
    fired_steps = summary.get("fired_steps")
    if fired_steps is None:
        # Constant runs: every step where alpha > 0.
        alpha_per_step = summary.get("alpha_per_step") or summary.get("lambda_per_step") or []
        fired_steps = [i for i, a in enumerate(alpha_per_step) if float(a) > 0]
    return {
        "run_dir": str(run_dir),
        "method": summary.get("method", run_dir.name),
        "force_per_step": force_per_step,
        "total_injection": total_injection,
        "fired_steps": list(fired_steps),
        "firings_count": len(list(fired_steps)),
        "basin_projection_per_step": summary.get("basin_projection_per_step"),
        "alpha_per_step": (
            list(summary.get("alpha_per_step")) if "alpha_per_step" in summary
            else (
                list(summary.get("lambda_per_step"))
                if "lambda_per_step" in summary else []
            )
        ),
    }


# ---------------------------------------------------------------------------
# Distance to anchors (CLIP cosine + latent-L2)
# ---------------------------------------------------------------------------


def _final_latent_or_none(run_dir: Path) -> torch.Tensor | None:
    p = trajectory_path_for(run_dir)
    if not p.exists():
        return None
    payload = torch.load(p, map_location="cpu", weights_only=False)
    return payload["trajectories"][-1].float()


def compute_distance_pair(
    *,
    run_dir: Path,
    poe_image_path: Path,
    mono_image_path: Path,
    poe_latent: torch.Tensor | None,
    mono_latent: torch.Tensor | None,
    device: torch.device | None = None,
) -> dict[str, float]:
    """One run → its d_PoE and d_Mono in CLIP cosine and (optionally) latent-L2."""
    img = image_path_for(run_dir)
    embs = VM.clip_image_embed([poe_image_path, mono_image_path, img], device=device)
    poe_emb, mono_emb, run_emb = embs[0], embs[1], embs[2]
    out: dict[str, float] = {
        "d_poe_clip": VM.clip_cosine_distance(run_emb, poe_emb),
        "d_mono_clip": VM.clip_cosine_distance(run_emb, mono_emb),
    }
    run_latent = _final_latent_or_none(run_dir)
    if run_latent is not None and poe_latent is not None and mono_latent is not None:
        out["d_poe_l2"] = VM.latent_l2(run_latent, poe_latent)
        out["d_mono_l2"] = VM.latent_l2(run_latent, mono_latent)
    else:
        out["d_poe_l2"] = float("nan")
        out["d_mono_l2"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Budget-vs-quality table
# ---------------------------------------------------------------------------


def compute_budget_quality_table(
    *,
    smart_runs: Iterable[Path],
    constant_runs: Iterable[Path],
    poe_image_path: Path,
    mono_image_path: Path,
    poe_run_dir: Path | None,
    mono_run_dir: Path | None,
    device: torch.device | None = None,
) -> dict:
    poe_latent = _final_latent_or_none(poe_run_dir) if poe_run_dir else None
    mono_latent = _final_latent_or_none(mono_run_dir) if mono_run_dir else None

    rows_smart: list[dict] = []
    rows_constant: list[dict] = []

    for run_dir in smart_runs:
        if not image_path_for(run_dir).exists():
            continue
        sched = compute_schedule_stats(run_dir)
        dist = compute_distance_pair(
            run_dir=run_dir, poe_image_path=poe_image_path,
            mono_image_path=mono_image_path,
            poe_latent=poe_latent, mono_latent=mono_latent,
            device=device,
        )
        rows_smart.append({
            "method": sched["method"],
            "total_injection": sched["total_injection"],
            "firings_count": sched["firings_count"],
            **dist,
        })

    for run_dir in constant_runs:
        if not image_path_for(run_dir).exists():
            continue
        sched = compute_schedule_stats(run_dir)
        dist = compute_distance_pair(
            run_dir=run_dir, poe_image_path=poe_image_path,
            mono_image_path=mono_image_path,
            poe_latent=poe_latent, mono_latent=mono_latent,
            device=device,
        )
        rows_constant.append({
            "method": sched["method"],
            "total_injection": sched["total_injection"],
            "firings_count": sched["firings_count"],
            **dist,
        })

    return {
        "smart": rows_smart,
        "constant": rows_constant,
    }


# ---------------------------------------------------------------------------
# Helper: pick alpha multipliers that match smart-run budgets
# ---------------------------------------------------------------------------


def suggest_constant_match_alphas(
    smart_total_injections: list[float],
    *,
    force_source: str,
    seed_dir: Path | None = None,
    capacity_summary_path: Path | None = None,
) -> list[float]:
    """Return α multipliers that should approximately reproduce each
    smart-run's total budget under a constant schedule.

    For ``residual``: α multiplier ≈ total_inj / sum(delta_norms_at_lambda1).
    For force_a/b and clip: α multiplier ≈ total_inj / K_force (read
    from the underlying capacity summary).

    If we can't read the capacity summary, falls back to a small grid
    of α multipliers spanning [0.0, 1.0].
    """
    if not smart_total_injections:
        return [0.5, 1.0]

    K_unit: float | None = None
    if capacity_summary_path is not None and capacity_summary_path.exists():
        cap = json.loads(capacity_summary_path.read_text())
        if isinstance(cap, dict):
            if "K_clip" in cap:
                K_unit = float(cap["K_clip"])
            elif "K_force" in cap:
                K_unit = float(cap["K_force"])

    if K_unit is None or K_unit <= 1e-9:
        # Fallback: a coarse grid of multipliers.
        return [0.25, 0.5, 0.75, 1.0]

    return sorted({
        round(float(v) / K_unit, 2)
        for v in smart_total_injections
        if float(v) > 0.0
    })
