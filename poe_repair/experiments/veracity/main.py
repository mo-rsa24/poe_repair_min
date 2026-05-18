"""Veracity — CLI orchestrator.

Three stages, each independently re-runnable:

    python -m poe_repair.experiments.veracity                  # all stages
    python -m poe_repair.experiments.veracity --skip-sweep      # metrics + figures
    python -m poe_repair.experiments.veracity --skip-sweep --skip-metrics
                                                                # figures only
    python -m poe_repair.experiments.veracity --only-lambdas 0.0,0.5,1.0
                                                                # smoke run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from poe_repair.config import RunConfig
from poe_repair.experiments._eval_common import HEADLINE_PAIR, slugify
from poe_repair.experiments.veracity import figures as F
from poe_repair.experiments.veracity import metrics as M
from poe_repair.experiments.veracity import sweep as S
from poe_repair.run import MethodCtx, make_ctx
from poe_repair.runtime import ensure_dir, write_json


DEFAULT_EXP_NAME = "veracity"
DEFAULT_SEED = 42


def _parse_lambdas(arg: str | None) -> tuple[float, ...]:
    if not arg:
        return S.LAMBDA_GRID
    parts = [s.strip() for s in arg.split(",") if s.strip()]
    return tuple(round(float(p), 4) for p in parts)


def _stage_sweep(
    *,
    cell, ctx: MethodCtx, lambdas: tuple[float, ...], overwrite: bool,
    exp_name: str,
    record_cross_attention: bool = False,
    attn_token_indices: dict | None = None,
) -> tuple[dict[float, Path], Path]:
    print(f"[{exp_name}] stage 1 — λ-sweep over {list(lambdas)}")
    if record_cross_attention:
        print(f"[{exp_name}] stage 1 — cross-attention capture enabled (λ=0 only)")
    paths_by_lambda = S.run_sweep(
        cell, ctx,
        exp_name=exp_name,
        lambdas=lambdas,
        overwrite=overwrite,
        save_x0_estimates=True,
        record_cross_attention=record_cross_attention,
        attn_token_indices=attn_token_indices,
    )
    print(f"[{exp_name}] stage 1 — Mono reference panel")
    reference_path = S.run_reference(
        cell, ctx, exp_name=exp_name, overwrite=overwrite,
    )
    return paths_by_lambda, reference_path


def _stage_metrics(
    *,
    seed_dir: Path, lambdas: tuple[float, ...], device: torch.device,
    exp_name: str,
) -> tuple[dict, dict, dict, dict]:
    print(f"[{exp_name}] stage 2 — distance + residual + PMI metrics")
    run_dirs = {lam: M.run_dir_for(seed_dir, lam) for lam in lambdas}
    distances = M.compute_distance_table(run_dirs, device=device)
    residual_stats = M.compute_residual_stats(run_dirs)
    pmi = M.compute_pmi_identity(run_dirs)
    trajectory = M.trajectory_distance_per_step(run_dirs)
    return distances, residual_stats, pmi, trajectory



def _stage_figures(
    *,
    fig_dir: Path,
    cell,
    ctx: MethodCtx | None,
    seed_dir: Path,
    paths_by_lambda: dict[float, Path],
    reference_path: Path,
    distances: dict,
    residual_stats: dict,
    pmi: dict,
    trajectory: dict,
    exp_name: str,
    cfg=None,
) -> dict[str, Path]:
    print(f"[{exp_name}] stage 3 — render Phase 0 Veracity figure set")
    title_suffix = f"{cell.prompt_a} × {cell.prompt_b}  |  seed {cell.seed}"
    poe_run_dir = M.run_dir_for(seed_dir, min(paths_by_lambda))
    mono_run_dir = M.run_dir_for(seed_dir, max(paths_by_lambda))

    out: dict[str, Path] = {}

    # --- Fig 1: Existence preflight (PMI identity-in-code) ---
    out["fig1"] = F.fig1_existence_pmi(
        fig_dir=fig_dir, pmi=pmi, title_suffix=title_suffix,
    )

    # --- Fig 4: Sufficiency HEADLINE ---
    # Detection queries: each marginal prompt (e.g. "a cat", "a dog"). Falls
    # back gracefully if GroundingDINO is unavailable.
    fig4_detection = (cell.prompt_a, cell.prompt_b)
    fig4_per_concept = (cell.prompt_a, cell.prompt_b)
    out["fig4"] = F.fig4_sufficiency(
        fig_dir=fig_dir,
        paths_by_lambda=paths_by_lambda,
        distances=distances,
        title_suffix=title_suffix,
        detection_queries=fig4_detection,
        per_concept_texts=fig4_per_concept,
    )

    # --- App-A: trajectory dependence across regimes (cache only) ---
    # Multi-seed mode: probe seeds {4, 42, 123}; include each seed whose
    # PoE-anchor and Mono-anchor residuals are both cached. Falls back to
    # single-seed mode if only one seed is available.
    def _seeded_run_dirs(slug: str, seeds: list[int]) -> tuple[list, list]:
        poe_dirs, mono_dirs = [], []
        for s in seeds:
            sd = (
                cfg.paths.output_root / exp_name / "pairs" / slug / f"seed_{s}"
            )
            p = sd / "teacher_residual_const_lam000"
            m = sd / "teacher_residual_const_lam100"
            if (M.residuals_dir_for(p).exists()
                    and M.residuals_dir_for(m).exists()):
                poe_dirs.append(p)
                mono_dirs.append(m)
        return poe_dirs, mono_dirs

    app_a_cells = []
    if cfg is not None:
        try:
            from pairs import CONTROL_SEEDS as _SEEDS
        except ImportError:
            _SEEDS = [cell.seed]
        # Ensure the contested cell's actually-requested seed is in the list.
        if cell.seed not in _SEEDS:
            _SEEDS = list(_SEEDS) + [cell.seed]

        contested_poe, contested_mono = _seeded_run_dirs(cell.pair_slug, _SEEDS)
        if contested_poe:
            app_a_cells.append({
                "label": f"{cell.prompt_a} × {cell.prompt_b}",
                "regime": "collision (PoE fails)",
                "poe_run_dirs": contested_poe,
                "mono_run_dirs": contested_mono,
            })
            print(
                f"[{exp_name}] App-A: contested cell using "
                f"{len(contested_poe)} seed(s)"
            )

        companion_slug = "a_butterfly__x__a_flower_meadow"
        comp_poe, comp_mono = _seeded_run_dirs(companion_slug, _SEEDS)
        if comp_poe:
            app_a_cells.append({
                "label": "a butterfly × a flower meadow",
                "regime": "cooperative (PoE works)",
                "poe_run_dirs": comp_poe,
                "mono_run_dirs": comp_mono,
            })
            print(
                f"[{exp_name}] App-A: companion cell using "
                f"{len(comp_poe)} seed(s)"
            )
        else:
            print(
                f"[{exp_name}] App-A: companion cell {companion_slug} not yet cached; "
                f"single-cell mode."
            )

    if not app_a_cells:
        # Fall back to single-seed single-cell mode using the explicit run dirs.
        app_a_cells = [{
            "label": f"{cell.prompt_a} × {cell.prompt_b}",
            "regime": "collision (PoE fails)",
            "poe_run_dir": poe_run_dir,
            "mono_run_dir": mono_run_dir,
        }]
    out["app_a"] = F.app_a_trajectory_independence(
        fig_dir=fig_dir,
        cells=app_a_cells,
        scheduler=(ctx.scheduler if ctx is not None else None),
        title_suffix=title_suffix,
    )

    # --- App-B': detection-based failure modes (NEW v3) ---
    # Discover each control-seed's λ=0 image for the contested cell.
    if cfg is not None:
        try:
            from pairs import CONTROL_SEEDS as _ABP_SEEDS
        except ImportError:
            _ABP_SEEDS = [cell.seed]
        ab_paths: dict[int, Path] = {}
        for s in _ABP_SEEDS:
            run = (
                cfg.paths.output_root / exp_name / "pairs" / cell.pair_slug
                / f"seed_{s}" / "teacher_residual_const_lam000"
                / "teacher_residual_const_lam000.png"
            )
            if run.exists():
                ab_paths[s] = run
        if ab_paths:
            # Chimera seed for cat × dog is 123 per the plan.
            chimera_seed = 123 if cell.pair_slug == "a_cat__x__a_dog" else None
            try:
                out["app_b_prime"] = F.app_b_detection_failure_modes(
                    fig_dir=fig_dir,
                    image_paths_by_seed=ab_paths,
                    detection_queries=(cell.prompt_a, cell.prompt_b),
                    chimera_seed=chimera_seed,
                    title_suffix=title_suffix,
                )
            except Exception as exc:
                print(f"[{exp_name}] App-B' failed: {exc}")
        else:
            print(
                f"[{exp_name}] App-B' skipped — no per-seed λ=0 images cached "
                f"for {cell.pair_slug}."
            )

    for name, path in out.items():
        print(f"[{exp_name}]   wrote {name}: {path}")
    return out


def main() -> None:
    from poe_repair.experiments import _assert_env_ok
    _assert_env_ok()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pair", type=str, default=None,
        help='Pair as "prompt_a|prompt_b". Default = HEADLINE_PAIR (a cat × a dog).',
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument(
        "--only-lambdas", type=str, default=None,
        help="Comma-separated subset of λ to run (e.g. 0.0,0.5,1.0). "
             "Defaults to the full eleven-point grid.",
    )
    ap.add_argument(
        "--guidance-scale", type=float, default=None,
        help="Override CFG guidance scale (default = config.DEFAULT_GUIDANCE = 7.5). "
             "Used by App-C CFG-robustness sweep.",
    )
    ap.add_argument(
        "--exp-name", type=str, default=DEFAULT_EXP_NAME,
        help='Experiment name (output subdir under output_root). Default = "veracity". '
             'Used to route control runs to "veracity_self_pair", "veracity_disjoint", '
             '"veracity_cfg_5p0", etc., without colliding with the main veracity outputs.',
    )
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--skip-metrics", action="store_true")
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    exp_name = args.exp_name

    if args.pair:
        a, _, b = args.pair.partition("|")
        if not a or not b:
            raise ValueError(f"--pair must be 'A|B', got {args.pair!r}")
        prompt_a, prompt_b = a, b
    else:
        prompt_a, prompt_b = HEADLINE_PAIR

    lambdas = _parse_lambdas(args.only_lambdas)
    cfg = RunConfig()
    cell = S.make_cell(prompt_a, prompt_b, args.seed)
    slug = cell.pair_slug

    out_root = cfg.paths.output_root / exp_name
    seed_dir = out_root / "pairs" / slug / f"seed_{args.seed}"
    metrics_dir = ensure_dir(out_root / "metrics")
    fig_dir = ensure_dir(out_root / "figures")

    ctx: MethodCtx | None = None
    if not (args.skip_sweep and args.skip_figures):
        # Sweep needs ctx; Figure 6 (in figures stage) also needs ctx for
        # Tweedie-decoded x̂_0 panels.  Skip ctx only if we're metrics-only.
        ctx = make_ctx(guidance_scale=args.guidance_scale)

    if not args.skip_sweep:
        assert ctx is not None
        paths_by_lambda, reference_path = _stage_sweep(
            cell=cell, ctx=ctx, lambdas=lambdas, overwrite=args.overwrite,
            exp_name=exp_name,
        )
    else:
        paths_by_lambda = {
            lam: M.image_path_for(M.run_dir_for(seed_dir, lam))
            for lam in lambdas
        }
        # Reference image written by run_method("mono", ...) under
        # ctx.output_root/<exp>/mono/pairs/<slug>/seed_<n>/mono.png
        reference_path = (
            out_root / "mono" / "pairs" / slug / f"seed_{args.seed}" / "mono.png"
        )

    if not args.skip_metrics:
        device = ctx.device if ctx is not None else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        distances, residual_stats, pmi, trajectory = _stage_metrics(
            seed_dir=seed_dir, lambdas=lambdas, device=device,
            exp_name=exp_name,
        )
        write_json(metrics_dir / "distances.json", distances)
        write_json(metrics_dir / "residual_stats.json", residual_stats)
        write_json(metrics_dir / "pmi_identity.json", pmi)
        write_json(metrics_dir / "trajectory_distance.json", trajectory)
    else:
        distances = json.loads((metrics_dir / "distances.json").read_text())
        residual_stats = json.loads((metrics_dir / "residual_stats.json").read_text())
        pmi = json.loads((metrics_dir / "pmi_identity.json").read_text())
        trajectory = json.loads((metrics_dir / "trajectory_distance.json").read_text())

    if not args.skip_figures:
        _stage_figures(
            fig_dir=fig_dir,
            cell=cell, ctx=ctx,
            seed_dir=seed_dir,
            paths_by_lambda=paths_by_lambda,
            reference_path=reference_path,
            distances=distances,
            residual_stats=residual_stats,
            pmi=pmi,
            trajectory=trajectory,
            exp_name=exp_name,
            cfg=cfg,
        )

    write_json(
        out_root / "summary.json",
        {
            "exp": exp_name,
            "pair": [cell.prompt_a, cell.prompt_b],
            "pair_slug": cell.pair_slug,
            "seed": cell.seed,
            "lambdas": list(lambdas),
            "guidance_scale": (ctx.guidance_scale if ctx is not None else None),
            "num_inference_steps": (
                ctx.num_inference_steps if ctx is not None else None
            ),
            "metrics": ["latent_l2", "clip_image_cosine"],
            "metric_floor_note": (
                "LPIPS not in repo; we ship two metrics, not three. "
                "See OBJECTIVE / plan: latent-L2 + CLIP cosine satisfy "
                "the protocol's two-metric floor."
            ),
        },
    )
    print(f"[{exp_name}] done — outputs under {out_root}")


if __name__ == "__main__":
    main()
