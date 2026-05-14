"""Idea 2 — CLI orchestrator.

Stages:
  1. Verify basin templates exist (run veracity at λ=0,1 first if not).
  2. Trigger sweep: (rule × theta) on the headline cell.
  3. Constant-schedule controls at matched total injection.
  4. Held-out: best-(rule, theta) on additional seeds + matched-budget control.
  5. Figures A, B, C, D + summary JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from poe_repair.config import RunConfig
from poe_repair.experiments._eval_common import HEADLINE_PAIR
from poe_repair.experiments.idea2 import figures as F
from poe_repair.experiments.idea2 import metrics as IM
from poe_repair.experiments.idea2 import sweep as S
from poe_repair.run import MethodCtx, make_ctx
from poe_repair.runtime import ensure_dir, write_json


EXP_NAME = "idea2"
DEFAULT_SEED = 42


def _parse_floats(arg: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if not arg:
        return default
    return tuple(float(p.strip()) for p in arg.split(",") if p.strip())


def _parse_seeds(arg: str) -> tuple[int, ...]:
    return tuple(int(s.strip()) for s in arg.split(",") if s.strip())


def _parse_rules(arg: str | None) -> tuple[str, ...]:
    if not arg:
        return ("threshold", "persistence", "velocity")
    return tuple(s.strip() for s in arg.split(",") if s.strip())


def _parse_window(arg: str | None) -> tuple[int, int] | None:
    if not arg:
        return None
    a, b = arg.split(",")
    return (int(a.strip()), int(b.strip()))


def _veracity_templates_present(
    cfg: RunConfig, pair_slug: str, seed: int,
) -> bool:
    base = cfg.paths.output_root / "veracity" / "pairs" / pair_slug / f"seed_{seed}"
    return (
        (base / "teacher_residual_const_lam000" / "latent_trajectory.pt").exists()
        and (base / "teacher_residual_const_lam100" / "latent_trajectory.pt").exists()
    )


def _load_force_scaler(
    *, force_source: str, cfg: RunConfig, pair_slug: str, seed: int,
) -> tuple[float, Path | None]:
    """For force_a/force_b/clip, read α₀ from the underlying capacity JSON.

    Returns ``(force_scaler, capacity_summary_path)``. For ``residual``,
    ``force_scaler`` is unused (the inner sampler uses ``lambda_max``
    directly), so we return 1.0.
    """
    if force_source == "residual":
        return 1.0, None
    if force_source in ("force_a", "force_b"):
        cap = cfg.paths.output_root / "idea1" / "metrics" / "force_capacity.json"
        if not cap.exists():
            return 1.0, None
        d = json.loads(cap.read_text())
        kind = "overlap" if force_source == "force_a" else "alignment"
        if kind not in d:
            return 1.0, None
        return float(d[kind]["alpha_zero"]), cap
    if force_source == "clip":
        cap = cfg.paths.output_root / "idea5b" / "metrics" / "capacity.json"
        if not cap.exists():
            return 1.0, None
        d = json.loads(cap.read_text())
        return float(d["alpha_zero"]), cap
    return 1.0, None


def main() -> None:
    from poe_repair.experiments import _assert_env_ok
    _assert_env_ok()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", type=str, default=None,
                    help='Pair as "prompt_a|prompt_b". Default = HEADLINE_PAIR.')
    ap.add_argument("--seeds", type=str, default=str(DEFAULT_SEED),
                    help="Comma-separated seeds. First is headline; extras are held-out.")
    ap.add_argument("--force-source", type=str, default="clip",
                    choices=["residual", "force_a", "force_b", "clip"],
                    help="Underlying force to schedule. Default 'clip'.")
    ap.add_argument("--rules", type=str, default=None,
                    help="Comma-separated trigger rules. Default 'threshold,persistence,velocity'.")
    ap.add_argument("--thetas", type=str, default=None,
                    help="Comma-separated θ values. Default 0.2,0.3,0.4,0.5,0.6.")
    ap.add_argument("--persistence-K", type=int, default=S.PERSISTENCE_K)
    ap.add_argument("--velocity-lookback", type=int, default=S.VELOCITY_LOOKBACK)
    ap.add_argument("--correction-window", type=str, default=None,
                    help="'start,end' for clip / poe-internal. Default depends on force.")
    ap.add_argument("--target-prompt", type=str, default="a cat and a dog")
    ap.add_argument("--decode-strategy", type=str, default="full_vae",
                    choices=["full_vae", "half_res", "taesd"])
    ap.add_argument("--grad-clip", type=float, default=None)
    ap.add_argument("--schedule-max", type=float, default=1.0,
                    help="Per-fire intensity (= lambda_max for residual, multiplier on α₀ otherwise).")
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
    rules = _parse_rules(args.rules)
    thetas = _parse_floats(args.thetas, S.THETA_GRID)
    correction_window = _parse_window(args.correction_window)

    cfg = RunConfig()
    cell = S.make_cell(prompt_a, prompt_b, headline_seed)
    out_root = cfg.paths.output_root / EXP_NAME
    seed_dir = out_root / "pairs" / cell.pair_slug / f"seed_{cell.seed}"
    metrics_dir = ensure_dir(out_root / "metrics")
    fig_dir = ensure_dir(out_root / "figures")

    # ----- preflight: basin templates -----
    for s in seeds:
        if not _veracity_templates_present(cfg, cell.pair_slug, s):
            raise FileNotFoundError(
                f"Basin templates missing for seed {s}. Run "
                f"`python -m poe_repair.experiments.veracity --seed {s} --only-lambdas 0.0,1.0` first."
            )

    force_scaler, capacity_summary = _load_force_scaler(
        force_source=args.force_source,
        cfg=cfg, pair_slug=cell.pair_slug, seed=headline_seed,
    )
    print(f"[{EXP_NAME}] force_source={args.force_source}, force_scaler={force_scaler:.4f}")

    # ----- ctx -----
    ctx: MethodCtx | None = None
    if not args.skip_sweep:
        ctx = make_ctx()

    smart_paths: dict[tuple[str, float], Path] = {}
    constant_paths: dict[float, Path] = {}
    baselines: dict[str, Path] = {}
    heldout_data: dict[int, dict[str, Path]] = {}

    if not args.skip_sweep:
        assert ctx is not None
        print(f"[{EXP_NAME}] stage 1 — PoE / Mono baselines")
        baselines = S.run_reference_baselines(
            cell=cell, ctx=ctx, exp_name=EXP_NAME, overwrite=args.overwrite,
        )
        print(f"[{EXP_NAME}] stage 2 — trigger sweep")
        smart_paths = S.run_trigger_sweep(
            cell=cell, ctx=ctx, exp_name=EXP_NAME,
            force_source=args.force_source,
            rules=rules, thetas=thetas,
            persistence_K=args.persistence_K,
            velocity_lookback=args.velocity_lookback,
            schedule_max=args.schedule_max,
            force_scaler=force_scaler,
            correction_window=correction_window,
            target_prompt=args.target_prompt,
            decode_strategy=args.decode_strategy,
            grad_norm_clip=args.grad_clip,
            overwrite=args.overwrite,
        )

        # Pick alpha multipliers for matched-budget constant runs.
        smart_totals = []
        for run_path in smart_paths.values():
            run_dir = run_path.parent
            stats = IM.compute_schedule_stats(run_dir)
            smart_totals.append(stats["total_injection"])
        constant_alphas = IM.suggest_constant_match_alphas(
            smart_totals,
            force_source=args.force_source,
            seed_dir=seed_dir,
            capacity_summary_path=capacity_summary,
        )
        print(f"[{EXP_NAME}] stage 3 — constant-schedule comparators at α∈{constant_alphas}")
        constant_paths = S.run_constant_match(
            cell=cell, ctx=ctx, exp_name=EXP_NAME,
            force_source=args.force_source,
            alpha_multipliers=tuple(constant_alphas),
            schedule_max=args.schedule_max,
            force_scaler=force_scaler,
            correction_window=correction_window,
            target_prompt=args.target_prompt,
            decode_strategy=args.decode_strategy,
            grad_norm_clip=args.grad_clip,
            overwrite=args.overwrite,
        )

        # Held-out seeds: best (rule, theta) by d_Mono on the headline cell.
        if heldout_seeds:
            poe_image_path = baselines["poe"]
            mono_image_path = baselines["mono"]
            poe_run = (
                cfg.paths.output_root / "veracity" / "pairs"
                / cell.pair_slug / f"seed_{headline_seed}"
                / "teacher_residual_const_lam000"
            )
            mono_run = (
                cfg.paths.output_root / "veracity" / "pairs"
                / cell.pair_slug / f"seed_{headline_seed}"
                / "teacher_residual_const_lam100"
            )
            best = None
            best_key = None
            for (rule, theta), path in smart_paths.items():
                dist = IM.compute_distance_pair(
                    run_dir=path.parent,
                    poe_image_path=poe_image_path,
                    mono_image_path=mono_image_path,
                    poe_latent=IM._final_latent_or_none(poe_run),
                    mono_latent=IM._final_latent_or_none(mono_run),
                    device=ctx.device,
                )
                score = float(dist["d_mono_clip"])
                if best is None or score < best:
                    best = score; best_key = (rule, theta)
            print(f"[{EXP_NAME}] stage 4 — best knobs on headline: {best_key} (d_mono_clip={best:.3f})")
            best_rule, best_theta = best_key

            # Use the smart-run total as the matched-budget for constant.
            sched = IM.compute_schedule_stats(smart_paths[best_key].parent)
            best_total = sched["total_injection"]
            best_alpha_match = (
                IM.suggest_constant_match_alphas(
                    [best_total], force_source=args.force_source,
                    seed_dir=seed_dir, capacity_summary_path=capacity_summary,
                ) or [0.5]
            )[0]

            for hs in heldout_seeds:
                hcell = S.make_cell(prompt_a, prompt_b, hs)
                print(f"[{EXP_NAME}] held-out seed {hs}: smart + constant-match")
                hbase = S.run_reference_baselines(
                    cell=hcell, ctx=ctx, exp_name=EXP_NAME, overwrite=args.overwrite,
                )
                hsmart_paths = S.run_trigger_sweep(
                    cell=hcell, ctx=ctx, exp_name=EXP_NAME,
                    force_source=args.force_source,
                    rules=(best_rule,), thetas=(best_theta,),
                    persistence_K=args.persistence_K,
                    velocity_lookback=args.velocity_lookback,
                    schedule_max=args.schedule_max,
                    force_scaler=force_scaler,
                    correction_window=correction_window,
                    target_prompt=args.target_prompt,
                    decode_strategy=args.decode_strategy,
                    grad_norm_clip=args.grad_clip,
                    overwrite=args.overwrite,
                )
                hconstant_paths = S.run_constant_match(
                    cell=hcell, ctx=ctx, exp_name=EXP_NAME,
                    force_source=args.force_source,
                    alpha_multipliers=(best_alpha_match,),
                    schedule_max=args.schedule_max,
                    force_scaler=force_scaler,
                    correction_window=correction_window,
                    target_prompt=args.target_prompt,
                    decode_strategy=args.decode_strategy,
                    grad_norm_clip=args.grad_clip,
                    overwrite=args.overwrite,
                )
                heldout_data[hs] = {
                    "poe": hbase["poe"],
                    "mono": hbase["mono"],
                    "smart": list(hsmart_paths.values())[0],
                    "constant_match": list(hconstant_paths.values())[0],
                }
    else:
        # Reconstruct paths from disk (figures-only re-run).
        baselines = {
            "poe": (out_root / "poe" / "pairs" / cell.pair_slug
                    / f"seed_{cell.seed}" / "poe.png"),
            "mono": (out_root / "mono" / "pairs" / cell.pair_slug
                     / f"seed_{cell.seed}" / "mono.png"),
        }
        for rule in rules:
            for theta in thetas:
                rd = IM.smart_run_dir(
                    seed_dir, force_source=args.force_source,
                    rule=rule, theta=theta,
                    persistence_K=args.persistence_K,
                    velocity_lookback=args.velocity_lookback,
                )
                p = IM.image_path_for(rd)
                if p.exists():
                    smart_paths[(rule, float(theta))] = p

    # ----- stage 3: metrics -----
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

        smart_run_dirs = [p.parent for p in smart_paths.values()]
        constant_run_dirs = [p.parent for p in constant_paths.values()]

        table = IM.compute_budget_quality_table(
            smart_runs=smart_run_dirs,
            constant_runs=constant_run_dirs,
            poe_image_path=baselines["poe"],
            mono_image_path=baselines["mono"],
            poe_run_dir=poe_run if poe_run.exists() else None,
            mono_run_dir=mono_run if mono_run.exists() else None,
            device=device,
        )

        sched_stats: dict[str, dict] = {}
        for key, p in smart_paths.items():
            sched_stats[f"{key[0]}_θ={key[1]:.2f}"] = IM.compute_schedule_stats(p.parent)

        write_json(metrics_dir / "budget_quality_table.json", table)
        write_json(metrics_dir / "schedule_stats.json", sched_stats)
    else:
        table = json.loads((metrics_dir / "budget_quality_table.json").read_text())
        sched_stats = json.loads((metrics_dir / "schedule_stats.json").read_text())

    # ----- stage 4: figures -----
    if not args.skip_figures:
        title_suffix = (
            f"{cell.prompt_a} × {cell.prompt_b}  |  "
            f"seed {cell.seed}  |  force={args.force_source}"
        )

        # Pick a few representative traces for Fig A. Show one per rule
        # at θ=0.4 (or the closest theta in the sweep).
        chosen_for_trace: dict[str, dict] = {}
        for rule in rules:
            tgt = 0.4
            theta_choice = min(thetas, key=lambda t: abs(t - tgt))
            label = f"{rule}  θ={theta_choice:.2f}"
            stats = sched_stats.get(f"{rule}_θ={theta_choice:.2f}")
            if stats is not None:
                chosen_for_trace[label] = stats
        if chosen_for_trace:
            F.figA_trace(
                fig_dir=fig_dir,
                sched_stats_by_label=chosen_for_trace,
                threshold_marker=0.4,
                title_suffix=title_suffix,
            )

        F.figB_budget_quality(
            fig_dir=fig_dir, table=table,
            metric="d_mono_l2", metric_label="d_Mono (latent-L2)",
            title_suffix=title_suffix,
        )
        F.figB_budget_quality(
            fig_dir=fig_dir, table=table,
            metric="d_mono_clip", metric_label="d_Mono (CLIP image cosine)",
            title_suffix=title_suffix,
        )

        # Fig C — pick three smart runs spanning the budget range and
        # their three matched-budget constants.
        smart_runs_sorted = sorted(
            table.get("smart", []), key=lambda r: r.get("total_injection", 0.0),
        )
        constant_runs_sorted = sorted(
            table.get("constant", []), key=lambda r: r.get("total_injection", 0.0),
        )
        if smart_runs_sorted and constant_runs_sorted:
            picks = [
                smart_runs_sorted[0],
                smart_runs_sorted[len(smart_runs_sorted) // 2],
                smart_runs_sorted[-1],
            ]
            # For each smart pick, find the closest constant by total_injection.
            def _nearest_constant(target: float) -> dict | None:
                if not constant_runs_sorted:
                    return None
                return min(constant_runs_sorted, key=lambda r: abs(r["total_injection"] - target))
            constant_picks = [_nearest_constant(p["total_injection"]) for p in picks]
            if all(c is not None for c in constant_picks):
                rows = {
                    "constant": [Path(c["run_dir"] if "run_dir" in c
                                       else seed_dir / c["method"])
                                  / f"{c['method']}.png" for c in constant_picks],
                    "smart": [Path(p.get("run_dir", seed_dir / p["method"]))
                              / f"{p['method']}.png" for p in picks],
                }
                # Resolve run_dir → ensure it's a Path with the .png appended.
                rows["constant"] = [
                    (seed_dir / m["method"] / f"{m['method']}.png")
                    for m in constant_picks
                ]
                rows["smart"] = [
                    (seed_dir / p["method"] / f"{p['method']}.png")
                    for p in picks
                ]
                col_labels = [
                    f"≈ {p['total_injection']:.0f}" for p in picks
                ]
                F.figC_constant_vs_smart_grid(
                    fig_dir=fig_dir, rows=rows,
                    col_labels=col_labels,
                    title_suffix=title_suffix,
                )

        if heldout_data:
            all_cells = {
                cell.seed: {
                    "poe": baselines["poe"], "mono": baselines["mono"],
                    "constant_match": (
                        Path(constant_runs_sorted[len(constant_runs_sorted) // 2].get(
                            "run_dir", seed_dir / constant_runs_sorted[len(constant_runs_sorted) // 2]["method"]
                        ))
                        / f"{constant_runs_sorted[len(constant_runs_sorted) // 2]['method']}.png"
                    ) if constant_runs_sorted else baselines["mono"],
                    "smart": (
                        Path(smart_runs_sorted[len(smart_runs_sorted) // 2].get(
                            "run_dir", seed_dir / smart_runs_sorted[len(smart_runs_sorted) // 2]["method"]
                        ))
                        / f"{smart_runs_sorted[len(smart_runs_sorted) // 2]['method']}.png"
                    ) if smart_runs_sorted else baselines["mono"],
                },
                **heldout_data,
            }
            F.figD_heldout(
                fig_dir=fig_dir,
                cells_by_seed=all_cells,
                title_suffix=title_suffix,
            )

    write_json(
        out_root / "summary.json",
        {
            "exp": EXP_NAME,
            "pair": [cell.prompt_a, cell.prompt_b],
            "pair_slug": cell.pair_slug,
            "headline_seed": headline_seed,
            "heldout_seeds": list(heldout_seeds),
            "force_source": args.force_source,
            "force_scaler": force_scaler,
            "rules": list(rules),
            "thetas": list(thetas),
            "persistence_K": args.persistence_K,
            "velocity_lookback": args.velocity_lookback,
            "schedule_max": args.schedule_max,
            "correction_window": list(correction_window) if correction_window else None,
            "notes": (
                "Idea 2 — adaptive schedule. The corrective force is sourced from "
                f"{args.force_source}; idea 2 only decides when to fire."
            ),
        },
    )
    print(f"[{EXP_NAME}] done — outputs under {out_root}")


if __name__ == "__main__":
    main()
