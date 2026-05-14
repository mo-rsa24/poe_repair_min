"""Idea 5b — CLI orchestrator.

Stages:
  1. Capacity check at α=1.0 → measure K_clip, set α₀ = 1500 / K_clip.
  2. Eleven-point α sweep (α/α₀ ∈ {0, 0.1, …, 1.0, 1.5}).
  3. Calibrated runs at α₀ for the headline cell + held-out seeds.
  4. Figures, including the four-method overlay (N3) against
     veracity Δ + idea1 Force-A / Force-B if those experiments exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from poe_repair.config import RunConfig
from poe_repair.experiments._eval_common import HEADLINE_PAIR
from poe_repair.experiments.idea5b import figures as F
from poe_repair.experiments.idea5b import metrics as IM
from poe_repair.experiments.idea5b import sweep as S
from poe_repair.run import MethodCtx, make_ctx
from poe_repair.runtime import ensure_dir, write_json


EXP_NAME = "idea5b"
DEFAULT_SEED = 42


def _parse_alphas(arg: str | None) -> tuple[float, ...]:
    if not arg:
        return S.ALPHA_GRID
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    return tuple(round(float(p), 4) for p in parts)


def _parse_seeds(arg: str) -> tuple[int, ...]:
    return tuple(int(s.strip()) for s in arg.split(",") if s.strip())


def _parse_window(arg: str | None) -> tuple[int, int]:
    if not arg:
        return S.DEFAULT_CORRECTION_WINDOW
    a, b = arg.split(",")
    return (int(a.strip()), int(b.strip()))


def main() -> None:
    from poe_repair.experiments import _assert_env_ok
    _assert_env_ok()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", type=str, default=None,
                    help='Pair as "prompt_a|prompt_b". Default = HEADLINE_PAIR (a cat × a dog).')
    ap.add_argument("--seeds", type=str, default=str(DEFAULT_SEED),
                    help="Comma-separated seeds. The first is the headline cell; extras are held-out.")
    ap.add_argument("--only-alphas", type=str, default=None,
                    help="Comma-separated subset of α multipliers (e.g. 0.0,0.5,1.0).")
    ap.add_argument("--correction-window", type=str, default=None,
                    help="'start,end' step indices. Default 10,25.")
    ap.add_argument("--target-prompt", type=str, default=S.DEFAULT_TARGET_PROMPT)
    ap.add_argument("--decode-strategy", type=str, default="full_vae",
                    choices=["full_vae", "half_res", "taesd"])
    ap.add_argument("--grad-clip", type=float, default=None,
                    help="Clip CLIP gradient norm (latent-shape) to this value, if set.")
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
    alphas = _parse_alphas(args.only_alphas)
    correction_window = _parse_window(args.correction_window)

    cfg = RunConfig()
    cell = S.make_cell(prompt_a, prompt_b, headline_seed)
    out_root = cfg.paths.output_root / EXP_NAME
    seed_dir = out_root / "pairs" / cell.pair_slug / f"seed_{cell.seed}"
    metrics_dir = ensure_dir(out_root / "metrics")
    fig_dir = ensure_dir(out_root / "figures")

    ctx: MethodCtx | None = None
    if not args.skip_sweep:
        ctx = make_ctx()

    # ---------- stage 1+2 ----------
    paths_by_alpha: dict[float, Path] = {}
    calibrated_path: Path | None = None
    baselines: dict[str, Path] = {}
    all_heldout: dict[int, dict[str, Path]] = {}
    alpha0: float = 1.0

    if not args.skip_sweep:
        assert ctx is not None
        print(f"[{EXP_NAME}] stage 1 — capacity check")
        K, alpha0, _ = S.capacity_check(
            cell=cell, ctx=ctx, exp_name=EXP_NAME,
            correction_window=correction_window,
            target_prompt=args.target_prompt,
            decode_strategy=args.decode_strategy,
            grad_norm_clip=args.grad_clip,
            overwrite=args.overwrite,
        )
        print(f"[{EXP_NAME}]   K_clip = {K:.2f}, α₀ = {alpha0:.4f}")

        print(f"[{EXP_NAME}] stage 2 — α-multiplier sweep")
        paths_by_alpha = S.run_strength_sweep(
            cell=cell, ctx=ctx, exp_name=EXP_NAME,
            alpha0=alpha0, alphas=alphas,
            correction_window=correction_window,
            target_prompt=args.target_prompt,
            decode_strategy=args.decode_strategy,
            grad_norm_clip=args.grad_clip,
            overwrite=args.overwrite,
        )

        print(f"[{EXP_NAME}] stage 2 — calibrated run")
        calibrated_path = S.run_calibrated(
            cell=cell, ctx=ctx, exp_name=EXP_NAME,
            alpha0=alpha0,
            schedule="constant",
            correction_window=correction_window,
            target_prompt=args.target_prompt,
            decode_strategy=args.decode_strategy,
            grad_norm_clip=args.grad_clip,
            overwrite=args.overwrite,
        )

        print(f"[{EXP_NAME}] stage 2 — PoE / Mono baselines")
        baselines = S.run_reference_baselines(
            cell=cell, ctx=ctx, exp_name=EXP_NAME, overwrite=args.overwrite,
        )

        # Held-out seeds: calibrated runs only.
        for hseed in heldout_seeds:
            hcell = S.make_cell(prompt_a, prompt_b, hseed)
            print(f"[{EXP_NAME}] stage 4 — held-out seed {hseed}")
            hbaselines = S.run_reference_baselines(
                cell=hcell, ctx=ctx, exp_name=EXP_NAME, overwrite=args.overwrite,
            )
            hclip = S.run_calibrated(
                cell=hcell, ctx=ctx, exp_name=EXP_NAME,
                alpha0=alpha0, schedule="constant",
                correction_window=correction_window,
                target_prompt=args.target_prompt,
                decode_strategy=args.decode_strategy,
                grad_norm_clip=args.grad_clip,
                overwrite=args.overwrite,
            )
            sched_m2 = (
                cfg.paths.output_root / "e_teacher_residual" / "pairs"
                / cell.pair_slug / f"seed_{hseed}"
                / "teacher_residual_const_lam100"
                / "teacher_residual_const_lam100.png"
            )
            all_heldout[hseed] = {
                "poe": hbaselines["poe"], "mono": hbaselines["mono"],
                "sched_m2": (sched_m2 if sched_m2.exists() else hbaselines["mono"]),
                "clip_guided": hclip,
            }

        sched_m2_head = (
            cfg.paths.output_root / "e_teacher_residual" / "pairs"
            / cell.pair_slug / f"seed_{headline_seed}"
            / "teacher_residual_const_lam100"
            / "teacher_residual_const_lam100.png"
        )
        all_heldout = {
            headline_seed: {
                "poe": baselines["poe"], "mono": baselines["mono"],
                "sched_m2": (sched_m2_head if sched_m2_head.exists() else baselines["mono"]),
                "clip_guided": calibrated_path,
            },
            **all_heldout,
        }
    else:
        # Reconstruct from disk.
        cap_dir = IM.run_dir_for_alpha(seed_dir, 1.0)
        if cap_dir.exists():
            cap_summary = json.loads(IM.summary_json_for(cap_dir).read_text())
            K = float(cap_summary["K_clip"])
            alpha0 = (S.BARRIER_TARGET / K) if K > 1e-9 else 1.0

        paths_by_alpha = {
            a: IM.run_dir_for_alpha(seed_dir, a) /
                f"{IM.run_dir_for_alpha(seed_dir, a).name}.png"
            for a in alphas
        }
        calibrated_dir = IM.run_dir_calibrated(
            seed_dir, "constant", correction_window=correction_window,
        )
        calibrated_path = calibrated_dir / f"{calibrated_dir.name}.png"
        baselines = {
            "poe": (out_root / "poe" / "pairs" / cell.pair_slug
                    / f"seed_{cell.seed}" / "poe.png"),
            "mono": (out_root / "mono" / "pairs" / cell.pair_slug
                     / f"seed_{cell.seed}" / "mono.png"),
        }
        sched_m2_head = (
            cfg.paths.output_root / "e_teacher_residual" / "pairs"
            / cell.pair_slug / f"seed_{headline_seed}"
            / "teacher_residual_const_lam100"
            / "teacher_residual_const_lam100.png"
        )
        all_heldout = {
            headline_seed: {
                "poe": baselines["poe"], "mono": baselines["mono"],
                "sched_m2": (sched_m2_head if sched_m2_head.exists() else baselines["mono"]),
                "clip_guided": calibrated_path,
            }
        }
        for hseed in heldout_seeds:
            hcell_dir = out_root / "pairs" / cell.pair_slug / f"seed_{hseed}"
            hclip_dir = IM.run_dir_calibrated(
                hcell_dir, "constant", correction_window=correction_window,
            )
            sched_m2 = (
                cfg.paths.output_root / "e_teacher_residual" / "pairs"
                / cell.pair_slug / f"seed_{hseed}"
                / "teacher_residual_const_lam100"
                / "teacher_residual_const_lam100.png"
            )
            poe_p = (out_root / "poe" / "pairs" / cell.pair_slug
                     / f"seed_{hseed}" / "poe.png")
            mono_p = (out_root / "mono" / "pairs" / cell.pair_slug
                      / f"seed_{hseed}" / "mono.png")
            all_heldout[hseed] = {
                "poe": poe_p, "mono": mono_p,
                "sched_m2": (sched_m2 if sched_m2.exists() else mono_p),
                "clip_guided": hclip_dir / f"{hclip_dir.name}.png",
            }

    # ---------- stage 3: metrics ----------
    poe_image_path = baselines["poe"]
    mono_image_path = baselines["mono"]

    if not args.skip_metrics:
        device = ctx.device if ctx is not None else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        veracity_seed_dir = (
            cfg.paths.output_root / "veracity" / "pairs"
            / cell.pair_slug / f"seed_{cell.seed}"
        )
        poe_run = veracity_seed_dir / "teacher_residual_const_lam000"
        mono_run = veracity_seed_dir / "teacher_residual_const_lam100"

        print(f"[{EXP_NAME}] stage 3 — distances + grad stats + 4-method comparison")
        distances = IM.compute_distance_table(
            seed_dir=seed_dir,
            poe_image_path=poe_image_path,
            mono_image_path=mono_image_path,
            poe_run_dir=poe_run if poe_run.exists() else None,
            mono_run_dir=mono_run if mono_run.exists() else None,
            alphas=alphas, device=device,
        )
        grad_stats = IM.compute_grad_stats(
            seed_dir=seed_dir, alphas=alphas,
        )
        capacity = IM.load_clip_capacity(seed_dir)

        method_comparison = IM.compute_method_comparison(
            distances_idea5b=distances,
            grad_stats_idea5b=grad_stats,
            veracity_distances_path=cfg.paths.output_root / "veracity" / "metrics" / "distances.json",
            veracity_residual_stats_path=cfg.paths.output_root / "veracity" / "metrics" / "residual_stats.json",
            idea1_distances_path=cfg.paths.output_root / "idea1" / "metrics" / "distances.json",
            idea1_residual_stats_path=cfg.paths.output_root / "idea1" / "metrics" / "residual_stats.json",
        )

        trajectory: dict | None = None
        if mono_run.exists() and (mono_run / "latent_trajectory.pt").exists():
            trajectory = IM.trajectory_distance_per_step(
                seed_dir=seed_dir, mono_run_dir=mono_run, alphas=alphas,
            )

        write_json(metrics_dir / "distances.json", distances)
        write_json(metrics_dir / "residual_stats.json", grad_stats)
        write_json(metrics_dir / "capacity.json", capacity)
        write_json(metrics_dir / "method_comparison.json", method_comparison)
        if trajectory is not None:
            write_json(metrics_dir / "trajectory_distance.json", trajectory)
    else:
        distances = json.loads((metrics_dir / "distances.json").read_text())
        grad_stats = json.loads((metrics_dir / "residual_stats.json").read_text())
        capacity = json.loads((metrics_dir / "capacity.json").read_text())
        method_comparison = json.loads((metrics_dir / "method_comparison.json").read_text())
        traj_path = metrics_dir / "trajectory_distance.json"
        trajectory = json.loads(traj_path.read_text()) if traj_path.exists() else None

    # ---------- stage 4: figures ----------
    if not args.skip_figures:
        title_suffix = f"{cell.prompt_a} × {cell.prompt_b}  |  seed {cell.seed}"
        fig_paths: dict[str, Path] = {}
        fig_paths["fig01"] = F.fig01_anchors(
            fig_dir=fig_dir,
            poe_path=poe_image_path,
            mono_path=mono_image_path,
            clip_calibrated_path=calibrated_path,
            title_suffix=title_suffix,
        )
        fig_paths["fig02"] = F.fig02_strength_sweep_grid(
            fig_dir=fig_dir, paths_by_alpha=paths_by_alpha,
            title_suffix=title_suffix,
        )
        fig_paths["fig03"] = F.fig03_distance_curves(
            fig_dir=fig_dir, distances=distances,
        )
        fig_paths["fig04"] = F.fig04_grad_norm_trajectory(
            fig_dir=fig_dir, grad_stats=grad_stats,
        )
        fig_paths["fig05"] = F.fig05_force_vs_effect(
            fig_dir=fig_dir, distances=distances, grad_stats=grad_stats,
        )
        fig_paths["fig06"] = F.fig06_spatial_heatmap(
            fig_dir=fig_dir, grad_stats=grad_stats,
        )
        fig_paths["fig07"] = F.fig07_direction_stability(
            fig_dir=fig_dir, grad_stats=grad_stats,
        )
        if trajectory is not None:
            fig_paths["fig09"] = F.fig09_latent_trajectory_distance(
                fig_dir=fig_dir, trajectory=trajectory,
            )
        fig_paths["figN1"] = F.figN1_clip_grad_traces(
            fig_dir=fig_dir, grad_stats=grad_stats,
        )
        fig_paths["figN3"] = F.figN3_method_overlay(
            fig_dir=fig_dir, method_comparison=method_comparison,
        )
        if all_heldout and len(all_heldout) >= 1:
            fig_paths["figN4"] = F.figN4_heldout_grid(
                fig_dir=fig_dir, cells=all_heldout,
            )
        for name, p in fig_paths.items():
            print(f"[{EXP_NAME}]   wrote {name}: {p}")

    write_json(
        out_root / "summary.json",
        {
            "exp": EXP_NAME,
            "pair": [cell.prompt_a, cell.prompt_b],
            "pair_slug": cell.pair_slug,
            "headline_seed": headline_seed,
            "heldout_seeds": list(heldout_seeds),
            "alphas": list(alphas),
            "correction_window": list(correction_window),
            "target_prompt": args.target_prompt,
            "decode_strategy": args.decode_strategy,
            "barrier_target": S.BARRIER_TARGET,
            "alpha_zero": alpha0,
            "guidance_scale": ctx.guidance_scale if ctx is not None else None,
            "num_inference_steps": ctx.num_inference_steps if ctx is not None else None,
            "metrics": ["latent_l2", "clip_image_cosine"],
            "notes": (
                "CLIP-guided PoE repair. 3-branch UNet (no e_J at inference). "
                "Corrective gradient comes from CLIP image-text similarity."
            ),
        },
    )
    print(f"[{EXP_NAME}] done — outputs under {out_root}")


if __name__ == "__main__":
    main()
