"""Thread C orchestrator — D1-A, D1-B, D1-C, D2, D3 over the training cache.

Usage examples::

    # Defaults: cat × dog, seeds {4, 42, 123}, all plots, no GroundingDINO overlay.
    python -m poe_repair.experiments.thread_c_structure

    # Add D2 with GroundingDINO box overlay (needs transformers + GPU/CPU for DINO):
    python -m poe_repair.experiments.thread_c_structure --d2-overlay

    # Add the cooperative-pair D3 (requires the butterfly×meadow cache to exist;
    # see scripts/build_training_cache.py to materialise it):
    python -m poe_repair.experiments.thread_c_structure \\
        --cooperative-pair-slug a_butterfly__x__a_flower_meadow

    # Run on a specific D2 representative seed only:
    python -m poe_repair.experiments.thread_c_structure --d2-seed 42

Outputs land under ``outputs/thread_c_structure/``::

    outputs/thread_c_structure/
        a_cat__x__a_dog/
            seed_4/
                d1a_direction_stability.png
                d1a_direction_stability.json
                d1b_svd_energy.png
                d1b_svd_energy.json
                d1c_basis_alignment.png
                d1c_basis_alignment.json
                d2_spatial_heatmaps.png        # (if --d2-seed matches)
            seed_42/ ...
            seed_123/ ...
            d3_cross_seed.png
            d3_cross_seed.json
            VERDICT.json
        a_butterfly__x__a_flower_meadow/       # only if cache exists
            d3_cross_seed.png ...
            VERDICT_cooperative.json
        OVERALL_VERDICT.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from poe_repair.experiments.thread_c_structure import figures as fig
from poe_repair.experiments.thread_c_structure import metrics as M
from poe_repair.experiments.thread_c_structure.loader import (
    DEFAULT_CACHE_ROOT, CellPath,
)
from poe_repair.experiments.thread_c_structure.verdict import (
    DEFAULT_THRESHOLDS, overall_verdict,
    verdict_for_d1a, verdict_for_d1b, verdict_for_d1c,
    verdict_for_d3_cooperative, verdict_for_d4a, verdict_for_d4a_t,
    verdict_for_d4b,
)
from poe_repair.runtime import ensure_dir, write_json


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "thread_c_structure"
DEFAULT_PAIR_SLUG = "a_cat__x__a_dog"
DEFAULT_SEEDS = (4, 42, 123)
DEFAULT_D2_STEP_INDICES = (5, 15, 25, 35)
DEFAULT_D4C_STEP_INDICES = (5, 15, 25, 35, 45)
DEFAULT_PCA_GRID_STEP_INDICES = (49, 39, 29, 19, 9, 1)
DEFAULT_VLM_GRID_PANEL_STEPS = (49, 39, 29, 19, 9, 1)
DEFAULT_DETECTION_PALETTE = {
    "cat": "#E07A5F",
    "dog": "#3D5A80",
    "butterfly": "#9D4EDD",
    "flower meadow": "#52B788",
}


# ---------------------------------------------------------------------------
# Per-cell runner (D1-A, D1-B, D1-C, optionally D2)
# ---------------------------------------------------------------------------


def _detection_queries_from_slug(pair_slug: str) -> list[str]:
    """``a_cat__x__a_dog`` -> ``["a cat", "a dog"]``."""
    a, b = pair_slug.split("__x__")
    return [a.replace("_", " "), b.replace("_", " ")]


def _run_d2_if_wanted(
    cell: CellPath,
    out_dir: Path,
    *,
    d2_step_indices: tuple[int, ...],
    do_overlay: bool,
) -> None:
    panels = M.spatial_heatmaps(cell, step_indices=d2_step_indices)
    overlay_path: Path | None = None
    detections: list[dict] | None = None
    if do_overlay:
        # Boxes from the *final decoded image*, scaled into latent coords inside
        # render_d2. The user's text: "Overlay the GroundingDINO bounding boxes
        # from the decoded x̂_0." For the training cache, poe.png is the cell's
        # final decoded x̂_0 under PoE.
        candidate_image = cell.root / "poe.png"
        if candidate_image.exists():
            overlay_path = candidate_image
            try:
                from poe_repair.experiments.veracity.metrics import detect_boxes
                queries = _detection_queries_from_slug(cell.pair_slug)
                detections = detect_boxes(
                    overlay_path, queries,
                    box_threshold=0.35, text_threshold=0.25,
                )
            except Exception as exc:    # GroundingDINO unavailable, etc.
                print(f"[D2] detection skipped for {cell.pair_slug} seed={cell.seed}: {exc}")
                detections = []
        else:
            print(f"[D2] no poe.png at {candidate_image} — skipping overlay")
    fig_path = fig.render_d2(
        panels,
        out_dir / "d2_spatial_heatmaps.png",
        pair_slug=cell.pair_slug,
        seed=cell.seed,
        overlay_image_path=overlay_path,
        detections=detections,
        detection_palette=DEFAULT_DETECTION_PALETTE,
    )
    print(f"  D2 → {fig_path}")
    write_json(out_dir / "d2_spatial_heatmaps.json", {
        "step_indices_requested": list(d2_step_indices),
        "step_indices_rendered": [p.step_index for p in panels],
        "timesteps_rendered": [p.timestep for p in panels],
        "overlay_image": str(overlay_path) if overlay_path else None,
        "detections": detections if detections is not None else [],
    })


def _run_per_cell(
    cell: CellPath,
    out_dir: Path,
    *,
    do_d2: bool,
    d2_step_indices: tuple[int, ...],
    d2_overlay: bool,
    thresholds=DEFAULT_THRESHOLDS,
) -> dict:
    ensure_dir(out_dir)
    verdicts: list[dict] = []

    # D1-A
    cos = M.consecutive_cosine(cell)
    fig.render_d1a(
        cos, out_dir / "d1a_direction_stability.png",
        pair_slug=cell.pair_slug, seed=cell.seed,
        threshold=thresholds.d1a_min_mean_cos, window=thresholds.d1a_window,
    )
    write_json(out_dir / "d1a_direction_stability.json", M.consecutive_to_dict(cos))
    v_d1a = verdict_for_d1a(cos.mean_over(*thresholds.d1a_window), thresholds)
    verdicts.append(v_d1a)
    print(f"  D1-A  mean cos = {v_d1a['measured']['mean_consecutive_cos_in_window']:.3f}  "
          f"[{'PASS' if v_d1a['passed'] else 'FAIL'}]")

    # D1-B
    svd = M.svd_energy(cell)
    fig.render_d1b(
        svd, out_dir / "d1b_svd_energy.png",
        pair_slug=cell.pair_slug, seed=cell.seed,
        threshold_top3=thresholds.d1b_min_top3_share,
    )
    write_json(out_dir / "d1b_svd_energy.json", M.svd_to_dict(svd))
    v_d1b = verdict_for_d1b(svd.cumulative_topk.get(3, float("nan")), thresholds)
    verdicts.append(v_d1b)
    print(f"  D1-B  top-3 share = {v_d1b['measured']['top3_variance_share']:.3f}  "
          f"[{'PASS' if v_d1b['passed'] else 'FAIL'}]")

    # D1-C
    basis = M.basis_alignment(cell)
    fig.render_d1c(
        basis, out_dir / "d1c_basis_alignment.png",
        pair_slug=cell.pair_slug, seed=cell.seed,
        threshold=thresholds.d1c_min_best_cos, window=thresholds.d1c_window,
    )
    write_json(out_dir / "d1c_basis_alignment.json", M.basis_to_dict(basis))
    cand_means = basis.max_window_cos(*thresholds.d1c_window)
    best = max(cand_means.values()) if cand_means else float("nan")
    v_d1c = verdict_for_d1c(best, cand_means, thresholds)
    verdicts.append(v_d1c)
    print(f"  D1-C  best candidate ⟨cos⟩ = {best:.3f}  "
          f"[{'PASS' if v_d1c['passed'] else 'FAIL'}]")

    # D2
    if do_d2:
        _run_d2_if_wanted(
            cell, out_dir,
            d2_step_indices=d2_step_indices, do_overlay=d2_overlay,
        )

    return {
        "pair_slug": cell.pair_slug,
        "seed": cell.seed,
        "verdicts": verdicts,
        "overall": overall_verdict(verdicts),
    }


# ---------------------------------------------------------------------------
# Cross-seed D3 runner
# ---------------------------------------------------------------------------


def _run_cross_seed_d3(
    cells: list[CellPath],
    out_dir: Path,
    *,
    is_cooperative: bool,
    thresholds=DEFAULT_THRESHOLDS,
) -> dict:
    ensure_dir(out_dir)
    cross = M.cross_seed_cosine(cells)
    fig.render_d3(
        cross, out_dir / "d3_cross_seed.png",
        pair_slug=cells[0].pair_slug,
        threshold=thresholds.d3_cooperative_min_mean_cos,
        window=thresholds.d3_cooperative_window,
        is_cooperative=is_cooperative,
    )
    write_json(out_dir / "d3_cross_seed.json", M.cross_seed_to_dict(cross))
    mean_in_window = cross.mean_over_window(*thresholds.d3_cooperative_window)
    if is_cooperative:
        v = verdict_for_d3_cooperative(mean_in_window, thresholds)
        print(f"  D3-coop  mean ⟨cos⟩ = {mean_in_window:.3f}  "
              f"[{'PASS' if v['passed'] else 'FAIL'}]")
    else:
        v = {
            "metric": "D3 cross-seed cosine (collision pair, reported only)",
            "measured": {"mean_cross_pair_mean_cos_in_window": float(mean_in_window)},
            "window_step_indices": list(thresholds.d3_cooperative_window),
            "passed": None,
            "note": "C1 says: do not gate the verdict on collision-pair D3.",
        }
        print(f"  D3-collision  mean ⟨cos⟩ = {mean_in_window:.3f}  "
              "(reported, not gating)")
    return v


# ---------------------------------------------------------------------------
# Cross-seed D4-* + §7c PCA grid (post-hoc on the cache)
# ---------------------------------------------------------------------------


def _run_post_hoc_cross_seed(
    cells: list[CellPath],
    out_dir: Path,
    *,
    is_cooperative: bool,
    d4c_step_indices: tuple[int, ...] = DEFAULT_D4C_STEP_INDICES,
    pca_grid_step_indices: tuple[int, ...] = DEFAULT_PCA_GRID_STEP_INDICES,
    thresholds=DEFAULT_THRESHOLDS,
    n_permutations: int = 16,
) -> dict:
    """Render D4-B / D4-C / D4-D / PCA-grid for the supplied seeds.

    Returns a verdict-fragment dict with the D4-B cooperative-pair mean cos
    (None when ``is_cooperative=False``); the figures live next to D3.
    """
    ensure_dir(out_dir)
    pair_slug = cells[0].pair_slug

    # --- D4-B
    split = M.direction_magnitude_split(cells, n_permutations=n_permutations)
    fig.render_d4b(
        split, out_dir / "d4b_direction_magnitude.png",
        pair_slug=pair_slug,
        threshold=thresholds.d4b_min_cooperative_mean_cos,
        window=thresholds.d4b_window,
    )
    write_json(
        out_dir / "d4b_direction_magnitude.json",
        M.direction_magnitude_to_dict(split),
    )
    cand = split.mean_cos_in_window(*thresholds.d4b_window)
    vals = [v for v in cand.values() if v == v]              # filter NaN
    d4b_mean = float(sum(vals) / len(vals)) if vals else None
    print(
        f"  D4-B  cross-seed ⟨cos vs LOO mean⟩ in {thresholds.d4b_window} = "
        f"{d4b_mean if d4b_mean is None else f'{d4b_mean:.3f}'}"
        f"  ({'cooperative' if is_cooperative else 'collision'})"
    )

    # --- D4-C
    panels = M.cluster_ordered_cosine_panels(
        cells, step_indices_to_render=d4c_step_indices,
    )
    fig.render_d4c(
        panels, out_dir / "d4c_cluster_ordered_cosine.png",
        pair_slug=pair_slug,
    )
    write_json(
        out_dir / "d4c_cluster_ordered_cosine.json",
        M.cluster_panels_to_dict(panels),
    )
    print(f"  D4-C  cluster-ordered panels: {len(panels.panels)} timesteps")

    # --- D4-D (PCA with guards)
    guards = M.pca_with_guards(cells, n_permutations=n_permutations)
    fig.render_d4d(
        guards, out_dir / "d4d_pca_guards.png",
        pair_slug=pair_slug,
    )
    write_json(out_dir / "d4d_pca_guards.json", M.pca_guards_to_dict(guards))
    print("  D4-D  PCA-with-guards rendered")

    # --- §7c PCA grid
    pca_grid = M.pca_projection_grid(
        cells, step_indices_to_render=pca_grid_step_indices,
    )
    fig.render_pca_grid(
        pca_grid, out_dir / "pca_grid_latent.png",
        pair_slug=pair_slug,
    )
    write_json(out_dir / "pca_grid_latent.json", M.pca_grid_to_dict(pca_grid))
    print(f"  §7c PCA grid: {len(pca_grid.step_indices)} timesteps")

    return {
        "d4b_cooperative_mean_cos": d4b_mean if is_cooperative else None,
    }


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="thread_c_structure",
        description=(
            "Thread C — is Δ_t structured or noise? Runs D-series diagnostics "
            "(D1-A direction stability, D1-B SVD energy, D1-C Mono-free basis "
            "alignment, D2 spatial heatmap, D3 cross-seed cosine) against the "
            "training-cache layout."
        ),
    )
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--pair-slug", type=str, default=DEFAULT_PAIR_SLUG)
    p.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
        help="Seeds to include in the collision-pair D-series.",
    )
    p.add_argument(
        "--cooperative-pair-slug", type=str, default=None,
        help=(
            "Pair slug for the cooperative-pair D3 (per addition C1). "
            "If unset or the cache is missing, only the collision pair is run."
        ),
    )
    p.add_argument(
        "--cooperative-seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
        help="Seeds to use for the cooperative-pair D3.",
    )
    p.add_argument(
        "--d2-seed", type=int, default=42,
        help="Seed used for the D2 spatial heatmap panel (one representative).",
    )
    p.add_argument(
        "--d2-step-indices", type=int, nargs="+", default=list(DEFAULT_D2_STEP_INDICES),
        help="Step indices at which to render D2 heatmaps.",
    )
    p.add_argument(
        "--d2-overlay", action="store_true",
        help="Run GroundingDINO on poe.png and overlay boxes on D2 panels.",
    )
    p.add_argument("--skip-d2", action="store_true",
                   help="Skip D2 (e.g. when GroundingDINO is unavailable).")
    p.add_argument(
        "--skip-cross-seed-d4", action="store_true",
        help="Skip the post-hoc D4-B / D4-C / D4-D / PCA-grid step.",
    )
    p.add_argument(
        "--n-permutations", type=int, default=16,
        help="Permutations for D4-B / D4-D null bands.",
    )
    p.add_argument(
        "--d4c-step-indices", type=int, nargs="+",
        default=list(DEFAULT_D4C_STEP_INDICES),
        help="Step indices rendered as D4-C panels.",
    )
    p.add_argument(
        "--pca-grid-step-indices", type=int, nargs="+",
        default=list(DEFAULT_PCA_GRID_STEP_INDICES),
        help="Step indices rendered as §7c PCA-grid panels.",
    )
    # ----- D4-A (GPU; opt-in) -----
    p.add_argument(
        "--run-d4a", action="store_true",
        help="Run the D4-A substitution test (GPU; ~6h on the headline cell).",
    )
    p.add_argument(
        "--run-d4a-t", action="store_true",
        help="Run the D4-A-t windowed variant (GPU; ~18–24h on the headline cell).",
    )
    p.add_argument(
        "--d4a-overwrite", action="store_true",
        help="Re-render cached D4-A images instead of reusing them.",
    )
    # ----- §7c VLM grid (GPU; opt-in) -----
    p.add_argument(
        "--run-vlm-grid", action="store_true",
        help="Run the §7c VLM-projection grid (GPU + LLaVA + GroundingDINO).",
    )
    p.add_argument(
        "--vlm-grid-panel-steps", type=int, nargs="+",
        default=list(DEFAULT_VLM_GRID_PANEL_STEPS),
    )
    p.add_argument(
        "--vlm-grid-alphas", type=float, nargs="+",
        default=[0.0, 0.5, 1.0],
    )
    p.add_argument(
        "--vlm-grid-n-reruns", type=int, default=1,
        help="Reruns per (seed, t, α) — ≥ 3 unlocks confidence ellipses.",
    )
    p.add_argument(
        "--vlm-grid-overwrite", action="store_true",
        help="Re-render cached VLM-grid images instead of reusing them.",
    )
    p.add_argument(
        "--vlm-calibration-step", type=int, default=15,
        help="Timestep used for the §7c calibration α-sweep.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_root = args.out_root

    # ---------- Collision pair ----------
    coll_root = out_root / args.pair_slug
    ensure_dir(coll_root)
    print(f"[thread_c] collision pair = {args.pair_slug}, seeds = {args.seeds}")
    coll_cells: list[CellPath] = []
    per_cell_results: list[dict] = []
    for seed in args.seeds:
        try:
            cell = CellPath.from_root(
                args.pair_slug, seed, cache_root=args.cache_root,
            )
        except FileNotFoundError as exc:
            print(f"  skipping seed={seed}: {exc}")
            continue
        coll_cells.append(cell)
        seed_dir = coll_root / f"seed_{seed}"
        print(f"[thread_c] cell {cell.split}/{args.pair_slug}/seed_{seed}")
        do_d2 = (not args.skip_d2) and (seed == args.d2_seed)
        result = _run_per_cell(
            cell, seed_dir,
            do_d2=do_d2,
            d2_step_indices=tuple(args.d2_step_indices),
            d2_overlay=args.d2_overlay,
        )
        per_cell_results.append(result)

    if len(coll_cells) < 2:
        print("[thread_c] need ≥2 seeds with caches for D3; aborting D3 (collision).")
        coll_d3_verdict: dict | None = None
        coll_post_hoc: dict | None = None
    else:
        print(f"[thread_c] D3 collision pair across seeds={[c.seed for c in coll_cells]}")
        coll_d3_verdict = _run_cross_seed_d3(
            coll_cells, coll_root, is_cooperative=False,
        )
        if args.skip_cross_seed_d4:
            coll_post_hoc = None
        else:
            print(f"[thread_c] D4-B/C/D + §7c PCA grid on collision pair")
            coll_post_hoc = _run_post_hoc_cross_seed(
                coll_cells, coll_root, is_cooperative=False,
                d4c_step_indices=tuple(args.d4c_step_indices),
                pca_grid_step_indices=tuple(args.pca_grid_step_indices),
                n_permutations=int(args.n_permutations),
            )

    # ---------- Cooperative pair (optional) ----------
    coop_results: dict | None = None
    if args.cooperative_pair_slug:
        coop_root = out_root / args.cooperative_pair_slug
        ensure_dir(coop_root)
        print(f"[thread_c] cooperative pair = {args.cooperative_pair_slug}")
        coop_cells: list[CellPath] = []
        for seed in args.cooperative_seeds:
            try:
                cell = CellPath.from_root(
                    args.cooperative_pair_slug, seed, cache_root=args.cache_root,
                )
                coop_cells.append(cell)
            except FileNotFoundError as exc:
                print(f"  cooperative seed={seed} missing: {exc}")
        if len(coop_cells) < 2:
            print(
                "[thread_c] cooperative-pair cache incomplete. Materialise via:\n"
                "  PYTHONPATH=. python scripts/build_training_cache.py "
                f"--pairs {args.cooperative_pair_slug} --seeds "
                f"{' '.join(str(s) for s in args.cooperative_seeds)}\n"
                "(requires dataset/cells/<slug>/seed_<N>/{mono.png,poe.png} to exist; "
                "symlink from outputs/veracity/{pairs,mono/pairs}/... if needed)."
            )
        else:
            coop_verdict = _run_cross_seed_d3(
                coop_cells, coop_root, is_cooperative=True,
            )
            coop_post_hoc: dict | None = None
            d4b_verdict: dict | None = None
            if not args.skip_cross_seed_d4:
                print(
                    f"[thread_c] D4-B/C/D + §7c PCA grid on cooperative pair"
                )
                coop_post_hoc = _run_post_hoc_cross_seed(
                    coop_cells, coop_root, is_cooperative=True,
                    d4c_step_indices=tuple(args.d4c_step_indices),
                    pca_grid_step_indices=tuple(args.pca_grid_step_indices),
                    n_permutations=int(args.n_permutations),
                )
                d4b_verdict = verdict_for_d4b(
                    coop_post_hoc.get("d4b_cooperative_mean_cos"),
                )
            coop_results = {
                "pair_slug": args.cooperative_pair_slug,
                "seeds": [c.seed for c in coop_cells],
                "verdict": coop_verdict,
                "post_hoc": coop_post_hoc,
                "d4b_verdict": d4b_verdict,
            }
            write_json(coop_root / "VERDICT_cooperative.json", coop_results)

    # ---------- D4-A / D4-A-t (GPU; opt-in) ----------
    d4a_verdict: dict | None = None
    d4a_t_verdict: dict | None = None
    if (args.run_d4a or args.run_d4a_t) and len(coll_cells) >= 2:
        from poe_repair.experiments.thread_c_structure.d4a import (
            DEFAULT_WINDOWS, render_d4a, render_d4a_t, run_d4a,
        )
        # Resolve prompt_a / prompt_b from the cache's meta.json.
        meta = coll_cells[0].meta
        prompt_a = meta.get("prompt_a", meta.get("a", "a cat"))
        prompt_b = meta.get("prompt_b", meta.get("b", "a dog"))
        # When --run-d4a is set without --run-d4a-t, restrict to the "all" window
        # to keep cost at ~6 GPU-hours instead of ~24.
        if args.run_d4a_t:
            windows = DEFAULT_WINDOWS
        else:
            windows = (("all", (0, coll_cells[0].num_steps() - 1)),)
        d4a_dir = ensure_dir(coll_root / "d4a")
        print(
            f"[thread_c] D4-A: seeds={[c.seed for c in coll_cells]}  "
            f"windows={[w[0] for w in windows]}"
        )
        d4a_result = run_d4a(
            cells=coll_cells, prompt_a=prompt_a, prompt_b=prompt_b,
            out_dir=d4a_dir, windows=windows, overwrite=args.d4a_overwrite,
        )
        render_d4a(
            d4a_result, d4a_dir / "d4a.png",
            pair_slug=args.pair_slug, window_label="all",
        )
        # Count shared-mean passes against per-seed thresholds.
        def _count_shared_mean_passes(window_label: str) -> tuple[int, int]:
            from poe_repair.experiments.thread_c_structure.d4a import Condition
            n_vqa = 0; n_det = 0
            for seed in d4a_result.seeds:
                oracle = next(
                    (r for r in d4a_result.rows
                     if r.seed == seed and r.condition is Condition.ORACLE
                     and r.window_label == window_label),
                    None,
                )
                shared = next(
                    (r for r in d4a_result.rows
                     if r.seed == seed and r.condition is Condition.SHARED_MEAN
                     and r.window_label == window_label),
                    None,
                )
                zero = next(
                    (r for r in d4a_result.rows
                     if r.seed == seed and r.condition is Condition.ZERO
                     and r.window_label == window_label),
                    None,
                )
                if oracle is None or shared is None or zero is None:
                    continue
                thr = zero.grade.vqa_min + DEFAULT_THRESHOLDS.d4a_oracle_zero_fraction * (
                    oracle.grade.vqa_min - zero.grade.vqa_min
                )
                if shared.grade.vqa_min >= thr:
                    n_vqa += 1
                if shared.grade.detection_regime == "both_distinct":
                    n_det += 1
            return n_vqa, n_det
        n_vqa, n_det = _count_shared_mean_passes("all")
        d4a_verdict = verdict_for_d4a(
            seeds=d4a_result.seeds,
            shared_mean_vqa_pass_count=n_vqa,
            shared_mean_detection_pass_count=n_det,
        )
        if args.run_d4a_t:
            render_d4a_t(
                d4a_result, d4a_dir / "d4a_t.png",
                pair_slug=args.pair_slug,
            )
            n_commit, _ = _count_shared_mean_passes(
                DEFAULT_THRESHOLDS.d4a_t_window_label,
            )
            d4a_t_verdict = verdict_for_d4a_t(
                seeds=d4a_result.seeds,
                commit_window_shared_mean_pass_count=n_commit,
            )

    # ---------- §7c VLM grid (GPU; opt-in) ----------
    vlm_grid_summary: dict | None = None
    if args.run_vlm_grid and len(coll_cells) >= 2:
        from poe_repair.experiments.thread_c_structure.vlm_grid import (
            render_vlm_calibration, render_vlm_grid, run_vlm_grid,
        )
        meta = coll_cells[0].meta
        prompt_a = meta.get("prompt_a", meta.get("a", "a cat"))
        prompt_b = meta.get("prompt_b", meta.get("b", "a dog"))
        vlm_dir = ensure_dir(coll_root / "vlm_grid")
        print(
            f"[thread_c] §7c VLM grid: seeds={[c.seed for c in coll_cells]}  "
            f"panels={args.vlm_grid_panel_steps}  reruns={args.vlm_grid_n_reruns}"
        )
        vlm_result = run_vlm_grid(
            cells=coll_cells, prompt_a=prompt_a, prompt_b=prompt_b,
            out_dir=vlm_dir,
            panel_steps=tuple(args.vlm_grid_panel_steps),
            alphas=tuple(args.vlm_grid_alphas),
            n_reruns=int(args.vlm_grid_n_reruns),
            calibration_step=int(args.vlm_calibration_step),
            overwrite=args.vlm_grid_overwrite,
        )
        render_vlm_grid(vlm_result, vlm_dir / "vlm_grid.png")
        render_vlm_calibration(vlm_result, vlm_dir / "vlm_calibration.png")
        vlm_grid_summary = {
            "pair_slug": vlm_result.pair_slug,
            "panel_steps": vlm_result.panel_steps,
            "alphas": vlm_result.alphas,
            "n_reruns": vlm_result.n_reruns,
            "calibration": vlm_result.calibration,
        }

    # ---------- Per-cell + collision-pair verdict file ----------
    coll_verdict_payload = {
        "pair_slug": args.pair_slug,
        "seeds": [c.seed for c in coll_cells],
        "per_cell": per_cell_results,
        "d3_collision_pair_reported": coll_d3_verdict,
        "post_hoc": coll_post_hoc,
        "d4a_verdict": d4a_verdict,
        "d4a_t_verdict": d4a_t_verdict,
        "vlm_grid_summary": vlm_grid_summary,
        "thresholds": asdict(DEFAULT_THRESHOLDS),
    }
    write_json(coll_root / "VERDICT.json", coll_verdict_payload)

    # ---------- Overall verdict ----------
    all_gating: list[dict] = []
    for per_cell in per_cell_results:
        all_gating.extend(per_cell["verdicts"])
    if coop_results is not None:
        all_gating.append(coop_results["verdict"])
        if coop_results.get("d4b_verdict") is not None:
            all_gating.append(coop_results["d4b_verdict"])
    if d4a_verdict is not None:
        all_gating.append(d4a_verdict)
    if d4a_t_verdict is not None:
        all_gating.append(d4a_t_verdict)
    overall = overall_verdict(all_gating)
    overall_payload = {
        "pair_slug_collision": args.pair_slug,
        "pair_slug_cooperative": args.cooperative_pair_slug,
        "overall": overall,
        "gating_verdicts": all_gating,
    }
    write_json(out_root / "OVERALL_VERDICT.json", overall_payload)
    print(
        f"[thread_c] OVERALL: {overall['label']}  "
        f"({overall['passed_count']}/{overall['total']} passed)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
