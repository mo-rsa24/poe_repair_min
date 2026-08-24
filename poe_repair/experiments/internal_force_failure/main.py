"""Internal-force failure case — CLI orchestrator.

Stages:
  1. Capacity check at α=1.0 per force variant → measure K_force, set α₀.
  2. Eleven-point strength sweep over alpha multipliers per force.
  3. Schedule comparison (constant / early_only / closed_loop) at α₀.
  4. Held-out: same calibrated runs on additional seeds.

Each stage is independently re-runnable via --skip-* flags. All stages
read / write under outputs/internal_force_failure/. Calibration reads
``outputs/residual_diagnostics/existence/`` for the basin-barrier
target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from poe_repair import paths
from poe_repair.config import RunConfig
from poe_repair.experiments._eval_common import HEADLINE_PAIR
from poe_repair.experiments.internal_force_failure import figures as F
from poe_repair.experiments.internal_force_failure import metrics as IM
from poe_repair.experiments.internal_force_failure import sweep as S
from poe_repair.run import MethodCtx, make_ctx
from poe_repair.runtime import ensure_dir, write_json


EXP_NAME = "internal_force_failure"
DEFAULT_SEED = 42


def _parse_alphas(arg: str | None) -> tuple[float, ...]:
    if not arg:
        return S.ALPHA_GRID
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    return tuple(round(float(p), 4) for p in parts)


def _parse_force_kinds(arg: str) -> tuple[str, ...]:
    if arg == "both":
        return S.FORCE_KINDS
    if arg in S.FORCE_KINDS:
        return (arg,)
    raise ValueError(f"--force-kind must be 'overlap', 'alignment', or 'both'; got {arg!r}")


def _parse_seeds(arg: str) -> tuple[int, ...]:
    return tuple(int(s.strip()) for s in arg.split(",") if s.strip())


def _stage_sweep(
    *,
    cell, ctx: MethodCtx, force_kinds: tuple[str, ...], alphas: tuple[float, ...],
    overwrite: bool,
) -> tuple[dict[str, dict[float, Path]], dict[str, float], dict[str, dict[str, Path]]]:
    print(f"[{EXP_NAME}] stage 1+2 — capacity check + α-sweep "
          f"(forces={list(force_kinds)})")
    paths_by_force: dict[str, dict[float, Path]] = {}
    alpha0_by_force: dict[str, float] = {}
    for force_kind in force_kinds:
        K, alpha0, _ = S.capacity_check(
            cell=cell, ctx=ctx, exp_name=EXP_NAME,
            force_kind=force_kind, overwrite=overwrite,
        )
        print(f"[{EXP_NAME}]   {force_kind}: K_force={K:.1f}, α₀={alpha0:.3f}")
        alpha0_by_force[force_kind] = alpha0

        sweep_paths = S.run_strength_sweep(
            cell=cell, ctx=ctx, exp_name=EXP_NAME,
            force_kind=force_kind, alpha0=alpha0, alphas=alphas,
            overwrite=overwrite,
        )
        paths_by_force[force_kind] = sweep_paths

    print(f"[{EXP_NAME}] stage 1+2 — calibrated runs")
    calibrated_by_force: dict[str, Path] = {}
    for force_kind in force_kinds:
        calibrated_by_force[force_kind] = S.run_calibrated(
            cell=cell, ctx=ctx, exp_name=EXP_NAME,
            force_kind=force_kind, alpha0=alpha0_by_force[force_kind],
            schedule="constant", overwrite=overwrite,
        )

    print(f"[{EXP_NAME}] stage 1+2 — PoE / Mono baselines")
    baselines = S.run_reference_baselines(
        cell=cell, ctx=ctx, exp_name=EXP_NAME, overwrite=overwrite,
    )

    by_force: dict[str, dict[str, Path]] = {
        force_kind: {"calibrated": calibrated_by_force[force_kind]}
        for force_kind in force_kinds
    }
    by_force["_baselines"] = baselines
    return paths_by_force, alpha0_by_force, by_force


def _stage_metrics(
    *,
    seed_dir: Path,
    force_kinds: tuple[str, ...],
    alphas: tuple[float, ...],
    poe_image_path: Path, mono_image_path: Path,
    cfg: RunConfig, cell, ctx: MethodCtx | None,
    device: torch.device,
) -> tuple[dict, dict, dict, dict, dict]:
    print(f"[{EXP_NAME}] stage 2 — distances + force stats + method comparison")
    distances_by_force: dict[str, dict] = {}
    force_stats_by_force: dict[str, dict] = {}
    trajectory_by_force: dict[str, dict] = {}

    veracity_seed_dir = (
        paths.resolve(paths.RESIDUAL_BETWEEN_MONO_AND_POE) / "existence" / "pairs"
        / cell.pair_slug / f"seed_{cell.seed}"
    )
    poe_run = veracity_seed_dir / "teacher_residual_const_lam000"
    mono_run = veracity_seed_dir / "teacher_residual_const_lam100"

    for force_kind in force_kinds:
        distances_by_force[force_kind] = IM.compute_distance_table(
            seed_dir=seed_dir,
            force_kind=force_kind,
            poe_image_path=poe_image_path,
            mono_image_path=mono_image_path,
            poe_run_dir=poe_run if poe_run.exists() else None,
            mono_run_dir=mono_run if mono_run.exists() else None,
            alphas=alphas,
            device=device,
        )
        force_stats_by_force[force_kind] = IM.compute_force_stats(
            seed_dir=seed_dir, force_kind=force_kind, alphas=alphas,
        )
        if mono_run.exists() and (mono_run / "latent_trajectory.pt").exists():
            trajectory_by_force[force_kind] = IM.trajectory_distance_per_step(
                seed_dir=seed_dir, force_kind=force_kind,
                mono_run_dir=mono_run, alphas=alphas,
            )

    capacity = {
        force_kind: IM.load_force_capacity(seed_dir, force_kind)
        for force_kind in force_kinds
    }

    method_comparison = IM.compute_method_comparison(
        distances_by_force=distances_by_force,
        veracity_distances_path=paths.resolve(paths.RESIDUAL_BETWEEN_MONO_AND_POE) / "existence" / "metrics" / "distances.json",
        veracity_residual_stats_path=paths.resolve(paths.RESIDUAL_BETWEEN_MONO_AND_POE) / "existence" / "metrics" / "residual_stats.json",
        force_stats_by_force=force_stats_by_force,
    )

    return distances_by_force, force_stats_by_force, capacity, method_comparison, trajectory_by_force


def _stage_figures(
    *,
    fig_dir: Path,
    cell, ctx: MethodCtx | None,
    seed_dir: Path,
    paths_by_force: dict[str, dict[float, Path]],
    calibrated_by_force: dict[str, Path],
    poe_image_path: Path, mono_image_path: Path,
    distances_by_force: dict,
    force_stats_by_force: dict,
    method_comparison: dict,
    trajectory_by_force: dict,
    heldout_cells: dict | None = None,
) -> dict[str, Path]:
    print(f"[{EXP_NAME}] stage 3 — render figures")
    title_suffix = f"{cell.prompt_a} × {cell.prompt_b}  |  seed {cell.seed}"

    out: dict[str, Path] = {}
    out["fig01"] = F.fig01_anchors(
        fig_dir=fig_dir,
        poe_path=poe_image_path,
        mono_path=mono_image_path,
        poe_internal_calibrated_paths=calibrated_by_force,
        title_suffix=title_suffix,
    )
    out["fig02"] = F.fig02_strength_sweep_grid(
        fig_dir=fig_dir, paths_by_force=paths_by_force,
        title_suffix=title_suffix,
    )
    out["fig03"] = F.fig03_distance_curves(
        fig_dir=fig_dir, distances_by_force=distances_by_force,
    )
    out["fig04"] = F.fig04_force_norm_trajectory(
        fig_dir=fig_dir, force_stats_by_force=force_stats_by_force,
    )
    out["fig05"] = F.fig05_force_vs_effect(
        fig_dir=fig_dir,
        distances_by_force=distances_by_force,
        force_stats_by_force=force_stats_by_force,
    )
    out["fig06"] = F.fig06_spatial_heatmap(
        fig_dir=fig_dir,
        force_stats_by_force=force_stats_by_force,
        seed_dir=seed_dir,
        ctx=ctx,
    )
    out["fig07"] = F.fig07_direction_stability(
        fig_dir=fig_dir, force_stats_by_force=force_stats_by_force,
    )
    if trajectory_by_force:
        out["fig09"] = F.fig09_latent_trajectory_distance(
            fig_dir=fig_dir, trajectory_by_force=trajectory_by_force,
        )
    if "overlap" in force_stats_by_force:
        out["figN1"] = F.figN1_overlap_traces(
            fig_dir=fig_dir, force_stats=force_stats_by_force["overlap"],
        )
    if "alignment" in force_stats_by_force:
        out["figN2"] = F.figN2_alignment_field(
            fig_dir=fig_dir, force_stats=force_stats_by_force["alignment"],
        )
    out["figN3"] = F.figN3_method_overlay(
        fig_dir=fig_dir, method_comparison=method_comparison,
    )
    if heldout_cells:
        out["figN4"] = F.figN4_heldout_grid(
            fig_dir=fig_dir, cells=heldout_cells,
        )

    for name, path in out.items():
        print(f"[{EXP_NAME}]   wrote {name}: {path}")
    return out


def main() -> None:
    from poe_repair.experiments import _assert_env_ok
    _assert_env_ok()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pair", type=str, default=None,
        help='Pair as "prompt_a|prompt_b". Default = HEADLINE_PAIR (a cat × a dog).',
    )
    ap.add_argument(
        "--seeds", type=str, default=str(DEFAULT_SEED),
        help="Comma-separated seeds. The first is the headline cell; any "
             "extras become held-out runs at calibrated α₀.",
    )
    ap.add_argument(
        "--force-kind", type=str, default="both",
        help="'overlap', 'alignment', or 'both' (default).",
    )
    ap.add_argument(
        "--only-alphas", type=str, default=None,
        help="Comma-separated subset of α multipliers (e.g. 0.0,0.5,1.0).",
    )
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--skip-metrics", action="store_true")
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.pair:
        a, _, b = args.pair.partition("|")
        if not a or not b:
            raise ValueError(f"--pair must be 'A|B', got {args.pair!r}")
        prompt_a, prompt_b = a, b
    else:
        prompt_a, prompt_b = HEADLINE_PAIR

    seeds = _parse_seeds(args.seeds)
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    headline_seed = seeds[0]
    heldout_seeds = seeds[1:]
    force_kinds = _parse_force_kinds(args.force_kind)
    alphas = _parse_alphas(args.only_alphas)

    cfg = RunConfig()
    cell = S.make_cell(prompt_a, prompt_b, headline_seed)
    out_root = paths.resolve(paths.INTERNAL_FORCE_FAILURE)
    seed_dir = out_root / "pairs" / cell.pair_slug / f"seed_{cell.seed}"
    metrics_dir = ensure_dir(out_root / "metrics")
    fig_dir = ensure_dir(out_root / "figures")

    ctx: MethodCtx | None = None
    if not (args.skip_sweep and args.skip_figures):
        ctx = make_ctx()

    # ---------- stage 1+2 ----------
    if not args.skip_sweep:
        assert ctx is not None
        paths_by_force, alpha0_by_force, by_force = _stage_sweep(
            cell=cell, ctx=ctx, force_kinds=force_kinds, alphas=alphas,
            overwrite=args.overwrite,
        )
        baselines = by_force.pop("_baselines")
        calibrated_by_force = {fk: by_force[fk]["calibrated"] for fk in force_kinds}

        # Held-out seeds: just calibrated runs at the headline α₀.
        heldout_data: dict[int, dict[str, Path]] = {}
        for hseed in heldout_seeds:
            hcell = S.make_cell(prompt_a, prompt_b, hseed)
            hbaselines = S.run_reference_baselines(
                cell=hcell, ctx=ctx, exp_name=EXP_NAME, overwrite=args.overwrite,
            )
            entry: dict[str, Path] = {
                "poe": hbaselines["poe"], "mono": hbaselines["mono"],
            }
            for fk in force_kinds:
                entry[fk] = S.run_calibrated(
                    cell=hcell, ctx=ctx, exp_name=EXP_NAME,
                    force_kind=fk, alpha0=alpha0_by_force[fk],
                    schedule="constant", overwrite=args.overwrite,
                )
            heldout_data[hseed] = entry
        # Also store the headline cell at the top of heldout_cells for fig N4.
        all_heldout = {headline_seed: {
            "poe": baselines["poe"], "mono": baselines["mono"],
            **calibrated_by_force,
        }, **heldout_data}
    else:
        # Reconstruct paths from disk.
        paths_by_force = {
            fk: {a: IM.run_dir_for_alpha(seed_dir, fk, a) /
                    f"{IM.run_dir_for_alpha(seed_dir, fk, a).name}.png"
                  for a in alphas}
            for fk in force_kinds
        }
        calibrated_by_force = {
            fk: IM.run_dir_calibrated(seed_dir, fk) /
                f"{IM.run_dir_calibrated(seed_dir, fk).name}.png"
            for fk in force_kinds
        }
        baselines = {
            "poe": (out_root / "poe" / "pairs" / cell.pair_slug
                    / f"seed_{cell.seed}" / "poe.png"),
            "mono": (out_root / "mono" / "pairs" / cell.pair_slug
                     / f"seed_{cell.seed}" / "mono.png"),
        }
        all_heldout = {headline_seed: {
            "poe": baselines["poe"], "mono": baselines["mono"],
            **calibrated_by_force,
        }}
        for hseed in heldout_seeds:
            hcell_dir = out_root / "pairs" / cell.pair_slug / f"seed_{hseed}"
            entry: dict[str, Path] = {
                "poe": (out_root / "poe" / "pairs" / cell.pair_slug
                        / f"seed_{hseed}" / "poe.png"),
                "mono": (out_root / "mono" / "pairs" / cell.pair_slug
                         / f"seed_{hseed}" / "mono.png"),
            }
            for fk in force_kinds:
                entry[fk] = IM.run_dir_calibrated(hcell_dir, fk) / \
                    f"{IM.run_dir_calibrated(hcell_dir, fk).name}.png"
            all_heldout[hseed] = entry

    poe_image_path = baselines["poe"]
    mono_image_path = baselines["mono"]

    # ---------- stage 2: metrics ----------
    if not args.skip_metrics:
        device = ctx.device if ctx is not None else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        distances_by_force, force_stats_by_force, capacity, method_comparison, trajectory_by_force = _stage_metrics(
            seed_dir=seed_dir,
            force_kinds=force_kinds,
            alphas=alphas,
            poe_image_path=poe_image_path,
            mono_image_path=mono_image_path,
            cfg=cfg, cell=cell, ctx=ctx,
            device=device,
        )
        write_json(metrics_dir / "distances.json", distances_by_force)
        write_json(metrics_dir / "force_capacity.json", capacity)
        write_json(metrics_dir / "residual_stats.json", force_stats_by_force)
        write_json(metrics_dir / "method_comparison.json", method_comparison)
        if trajectory_by_force:
            write_json(metrics_dir / "trajectory_distance.json", trajectory_by_force)
    else:
        distances_by_force = json.loads((metrics_dir / "distances.json").read_text())
        force_stats_by_force = json.loads((metrics_dir / "residual_stats.json").read_text())
        capacity = json.loads((metrics_dir / "force_capacity.json").read_text())
        method_comparison = json.loads((metrics_dir / "method_comparison.json").read_text())
        traj_path = metrics_dir / "trajectory_distance.json"
        trajectory_by_force = (
            json.loads(traj_path.read_text()) if traj_path.exists() else {}
        )

    # ---------- stage 3: figures ----------
    if not args.skip_figures:
        _stage_figures(
            fig_dir=fig_dir,
            cell=cell, ctx=ctx,
            seed_dir=seed_dir,
            paths_by_force=paths_by_force,
            calibrated_by_force=calibrated_by_force,
            poe_image_path=poe_image_path,
            mono_image_path=mono_image_path,
            distances_by_force=distances_by_force,
            force_stats_by_force=force_stats_by_force,
            method_comparison=method_comparison,
            trajectory_by_force=trajectory_by_force,
            heldout_cells=all_heldout if len(all_heldout) > 1 else None,
        )

    write_json(
        out_root / "summary.json",
        {
            "exp": EXP_NAME,
            "pair": [cell.prompt_a, cell.prompt_b],
            "pair_slug": cell.pair_slug,
            "headline_seed": headline_seed,
            "heldout_seeds": list(heldout_seeds),
            "force_kinds": list(force_kinds),
            "alphas": list(alphas),
            "barrier_target": S.BARRIER_TARGET,
            "alpha_zero_by_force": {
                fk: capacity[fk]["alpha_zero"] for fk in force_kinds
            } if not args.skip_metrics else None,
            "guidance_scale": ctx.guidance_scale if ctx is not None else None,
            "num_inference_steps": ctx.num_inference_steps if ctx is not None else None,
            "metrics": ["latent_l2", "clip_image_cosine"],
            "notes": (
                "Mono used only via veracity barrier number; not called at "
                "inference. 3-branch UNet (A, B, ∅), no e_J encoding."
            ),
        },
    )
    print(f"[{EXP_NAME}] done — outputs under {out_root}")


if __name__ == "__main__":
    main()
