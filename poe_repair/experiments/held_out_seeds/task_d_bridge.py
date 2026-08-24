"""Task D — Δ̄_t bridge analysis.

Consumes the per-step ε records dumped by ``sample_heldout --record-eps``
and the training cache, and produces:

  - cos(v_lora, v_bar)         vs t        (one curve per held-out seed)
  - cos(v_lora, v_seed)        vs t        (held-out generalization view)
  - ‖v_lora‖ / ‖v_bar‖         vs t
  - ‖v_lora − α_t·v_bar‖ / ‖v_lora‖ vs t   (relative residual-of-residual,
                                            after best per-step scalar fit)

Plus a baseline:
  - cos(Δ_t^(s1), Δ_t^(s2))    vs t        (pairwise across train-pool seeds)

And a spatial-heatmap dump for one chosen seed at four steps in the
commit window, so the residual-of-residual can be eyeballed for
structure.

No SDXL needed — pure tensor math on cached eps + dumped LoRA eps.

Inputs
------
Expects pooled-LoRA samples with ``--record-eps`` already produced; the
default discovers the largest-k run under
``outputs/cross_seed_lora_pooling/task_b_learning_curve``.

Usage::

    python -m poe_repair.experiments.held_out_seeds.task_d_bridge \
        [--pooled-run NAME] [--out-dir ...] [--spatial-seed 9]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from poe_repair.experiments.held_out_seeds.seed_pool import load_seed_pool
from poe_repair.training_cache import (
    DEFAULT_CACHE_ROOT,
    CellPath,
    iter_cell_deltas,
    mean_delta_per_step,
    resolve_cells,
)


log = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.float().reshape(-1)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    fa, fb = _flat(a), _flat(b)
    na, nb = float(fa.norm()), float(fb.norm())
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(torch.dot(fa, fb) / (na * nb))


def _norm(t: torch.Tensor) -> float:
    return float(_flat(t).norm())


def _alpha_fit(v_lora: torch.Tensor, v_ref: torch.Tensor) -> float:
    fa, fb = _flat(v_lora), _flat(v_ref)
    denom = float((fb * fb).sum())
    if denom == 0.0:
        return 0.0
    return float(torch.dot(fa, fb) / denom)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _eps_records_for_seed(eps_dir: Path, seed: int) -> list[dict] | None:
    p = eps_dir / f"eps_seed_{int(seed):02d}.pt"
    if not p.exists():
        log.warning("no eps record for seed %d at %s", seed, p)
        return None
    return torch.load(p, map_location="cpu", weights_only=False)


def _cache_delta_by_step(cell: CellPath) -> dict[int, torch.Tensor]:
    return {int(e["step_index"]): e["delta"] for e in iter_cell_deltas(cell)}


# ---------------------------------------------------------------------------
# Compute curves for one seed
# ---------------------------------------------------------------------------


def compute_curves(
    eps_records: list[dict],
    r_bar_by_step: dict[int, torch.Tensor],
    r_seed_by_step: dict[int, torch.Tensor],
) -> dict:
    """For one held-out seed, compute per-step alignment of v_lora against
    (Δ̄_t, Δ_t^(s)). v_lora is taken as the *delta_hat* recorded by
    run_lora_residual_inject: ε_PoE_lora − ε_PoE_frozen.
    """
    step_indices, timesteps = [], []
    cos_bar, cos_seed = [], []
    norm_ratio_bar, norm_ratio_seed = [], []
    res_rel_bar, res_rel_seed = [], []
    alphas_bar, alphas_seed = [], []
    v_lora_norms, v_bar_norms, v_seed_norms = [], [], []
    for rec in eps_records:
        si = int(rec["step_index"])
        v_lora = rec["delta_hat"]
        step_indices.append(si)
        timesteps.append(int(rec["timestep"]))
        v_lora_norms.append(_norm(v_lora))

        v_bar = r_bar_by_step.get(si)
        if v_bar is None:
            cos_bar.append(float("nan"))
            norm_ratio_bar.append(float("nan"))
            res_rel_bar.append(float("nan"))
            alphas_bar.append(float("nan"))
            v_bar_norms.append(float("nan"))
        else:
            cos_bar.append(_cos(v_lora, v_bar))
            v_bar_norms.append(_norm(v_bar))
            denom = _norm(v_bar) or 1.0
            norm_ratio_bar.append(_norm(v_lora) / denom)
            a = _alpha_fit(v_lora, v_bar)
            alphas_bar.append(a)
            res = _flat(v_lora) - a * _flat(v_bar)
            res_rel_bar.append(float(res.norm()) /
                               max(1e-12, _norm(v_lora)))

        v_seed = r_seed_by_step.get(si)
        if v_seed is None:
            cos_seed.append(float("nan"))
            norm_ratio_seed.append(float("nan"))
            res_rel_seed.append(float("nan"))
            alphas_seed.append(float("nan"))
            v_seed_norms.append(float("nan"))
        else:
            cos_seed.append(_cos(v_lora, v_seed))
            v_seed_norms.append(_norm(v_seed))
            denom = _norm(v_seed) or 1.0
            norm_ratio_seed.append(_norm(v_lora) / denom)
            a = _alpha_fit(v_lora, v_seed)
            alphas_seed.append(a)
            res = _flat(v_lora) - a * _flat(v_seed)
            res_rel_seed.append(float(res.norm()) /
                                max(1e-12, _norm(v_lora)))

    return {
        "step_indices": step_indices,
        "timesteps": timesteps,
        "cos_vs_bar": cos_bar,
        "cos_vs_seed": cos_seed,
        "norm_ratio_vs_bar": norm_ratio_bar,
        "norm_ratio_vs_seed": norm_ratio_seed,
        "res_rel_vs_bar": res_rel_bar,
        "res_rel_vs_seed": res_rel_seed,
        "alpha_vs_bar": alphas_bar,
        "alpha_vs_seed": alphas_seed,
        "v_lora_norm": v_lora_norms,
        "v_bar_norm": v_bar_norms,
        "v_seed_norm": v_seed_norms,
    }


# ---------------------------------------------------------------------------
# Baseline: cross-seed Δ_t cosine
# ---------------------------------------------------------------------------


def seed_pair_cosines(cells: list[CellPath]) -> dict:
    """For each step, the mean and min of cos(Δ_t^(si), Δ_t^(sj)) over all
    distinct seed pairs in `cells`. Anchor for "what 'seed-invariant'
    looks like in the raw data."""
    per_seed = [_cache_delta_by_step(c) for c in cells]
    all_steps = sorted(set().union(*[set(d) for d in per_seed]))
    mean_cos: list[float] = []
    min_cos: list[float] = []
    for si in all_steps:
        cs: list[float] = []
        for i in range(len(per_seed)):
            for j in range(i + 1, len(per_seed)):
                a = per_seed[i].get(si)
                b = per_seed[j].get(si)
                if a is None or b is None:
                    continue
                cs.append(_cos(a, b))
        if cs:
            mean_cos.append(float(np.mean(cs)))
            min_cos.append(float(np.min(cs)))
        else:
            mean_cos.append(float("nan"))
            min_cos.append(float("nan"))
    return {
        "step_indices": all_steps,
        "mean_cos": mean_cos,
        "min_cos": min_cos,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_seed_curves(curves: dict, *, seed: int, out_dir: Path,
                     baseline: dict | None = None) -> list[Path]:
    out_paths: list[Path] = []
    si = curves["step_indices"]

    def _plot_one(metric_bar, metric_seed, ylabel, fname, *, ylim=None):
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        ax.plot(si, metric_bar, marker="o", ms=3, label="vs Δ̄_t (train-pool mean)")
        ax.plot(si, metric_seed, marker="s", ms=3, label="vs Δ_t^(s) (held-out own)")
        if baseline is not None and "cos" in fname:
            ax.plot(baseline["step_indices"], baseline["mean_cos"],
                    "--", color="grey", alpha=0.6, label="baseline mean cos(Δ^(si), Δ^(sj))")
        ax.set_xlabel("step_index")
        ax.set_ylabel(ylabel)
        ax.set_title(f"held-out seed {seed} — {ylabel}")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        path = out_dir / fname
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        out_paths.append(path)

    _plot_one(curves["cos_vs_bar"], curves["cos_vs_seed"],
              "cos(v_lora, ·)", f"seed_{seed:02d}__cos.png", ylim=(-0.1, 1.05))
    _plot_one(curves["norm_ratio_vs_bar"], curves["norm_ratio_vs_seed"],
              "‖v_lora‖ / ‖·‖", f"seed_{seed:02d}__norm_ratio.png")
    _plot_one(curves["res_rel_vs_bar"], curves["res_rel_vs_seed"],
              "‖v_lora − α·ref‖ / ‖v_lora‖", f"seed_{seed:02d}__res_rel.png",
              ylim=(0, 1.2))
    return out_paths


def plot_baseline(baseline: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.plot(baseline["step_indices"], baseline["mean_cos"], "o-",
            ms=3, label="mean cos over seed pairs")
    ax.plot(baseline["step_indices"], baseline["min_cos"], "s-",
            ms=3, alpha=0.7, label="min cos over seed pairs")
    ax.axhline(0.0, color="grey", lw=0.5)
    ax.set_xlabel("step_index")
    ax.set_ylabel("cos(Δ_t^(si), Δ_t^(sj))")
    ax.set_title("Baseline — cross-seed Δ_t alignment in the train pool")
    ax.set_ylim(-0.1, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = out_dir / "baseline_seed_pair_cosines.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_spatial_residual(
    eps_records: list[dict],
    r_bar_by_step: dict[int, torch.Tensor],
    *, seed: int, out_dir: Path,
    steps: list[int] = (5, 15, 25, 35),
) -> Path:
    """4-panel heatmap of ‖v_lora − α·Δ̄_t‖ aggregated over latent channels
    at four chosen steps in the commit window."""
    rec_by_step = {int(r["step_index"]): r for r in eps_records}
    fig, axes = plt.subplots(1, len(steps), figsize=(3.4 * len(steps), 3.6))
    if len(steps) == 1:
        axes = [axes]
    for ax, si in zip(axes, steps):
        rec = rec_by_step.get(int(si))
        v_bar = r_bar_by_step.get(int(si))
        if rec is None or v_bar is None:
            ax.text(0.5, 0.5, f"[missing step {si}]", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        v_lora = rec["delta_hat"]
        a = _alpha_fit(v_lora, v_bar)
        res = (v_lora - a * v_bar).abs().mean(dim=1)  # (1, H, W)
        h = res[0].numpy()
        im = ax.imshow(h, cmap="magma")
        ax.set_title(f"step {si}, α={a:+.2f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f"Residual-of-residual ‖v_lora − α·Δ̄_t‖ — seed {seed}", fontsize=11)
    path = out_dir / f"seed_{seed:02d}__residual_of_residual_spatial.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _autodiscover_pooled_run(task_b_root: Path) -> Path | None:
    """Pick the largest-k run by directory name (k08 > k04 > k01)."""
    if not task_b_root.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for d in task_b_root.iterdir():
        if not d.is_dir() or not d.name.startswith("k"):
            continue
        try:
            k_token = d.name[1:].split("_")[0]
            k = int(k_token)
        except ValueError:
            continue
        if (d / "samples" / "heldout" / "eps_records").exists():
            candidates.append((k, d))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level="INFO",
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooled-run", default=None,
                    help="path to a task_b_learning_curve/kNN__epEE run dir; "
                         "auto-discovers the largest-k run if omitted")
    ap.add_argument("--out-dir", default=None,
                    help="default: outputs/cross_seed_lora_pooling/task_d_bridge")
    ap.add_argument("--spatial-seed", type=int, default=None,
                    help="held-out seed to use for the residual-of-residual "
                         "heatmap (default: first held-out)")
    ap.add_argument("--spatial-steps", default="5,15,25,35")
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--seed-pool-path", default=None)
    args = ap.parse_args(argv)

    pool = load_seed_pool(args.seed_pool_path)
    cache_root = Path(args.cache_root) if args.cache_root else DEFAULT_CACHE_ROOT
    out_root = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "outputs" / "cross_seed_lora_pooling" / "task_d_bridge"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    curves_dir = out_root / "curves"; curves_dir.mkdir(exist_ok=True)
    spatial_dir = out_root / "residual_of_residual_spatial"; spatial_dir.mkdir(exist_ok=True)

    if args.pooled_run:
        pooled_run = Path(args.pooled_run)
    else:
        pooled_run = _autodiscover_pooled_run(
            REPO_ROOT / "outputs" / "cross_seed_lora_pooling"
            / "task_b_learning_curve"
        )
        if pooled_run is None:
            raise FileNotFoundError(
                "no pooled run with --record-eps found under "
                "outputs/cross_seed_lora_pooling/task_b_learning_curve. "
                "Pass --pooled-run explicitly."
            )
    eps_dir = pooled_run / "samples" / "heldout" / "eps_records"
    log.info("pooled run: %s", pooled_run)
    log.info("eps records: %s", eps_dir)

    log.info("computing Δ̄_t from train pool ...")
    train_cells = resolve_cells(pool.pair_slug, list(pool.train_pool),
                                cache_root=cache_root)
    r_bar_list, r_bar_si, r_bar_ts = mean_delta_per_step(train_cells)
    r_bar_by_step = {int(si): t for si, t in zip(r_bar_si, r_bar_list)}

    log.info("computing baseline cross-seed Δ_t cosines ...")
    baseline = seed_pair_cosines(train_cells)
    plot_baseline(baseline, curves_dir)

    per_seed_summary: list[dict] = []
    for s in pool.held_out:
        log.info("---- held-out seed %d ----", int(s))
        recs = _eps_records_for_seed(eps_dir, int(s))
        if recs is None:
            continue
        oracle_cell = CellPath.from_root(pool.pair_slug, int(s), cache_root=cache_root)
        r_seed_by_step = _cache_delta_by_step(oracle_cell)
        curves = compute_curves(recs, r_bar_by_step, r_seed_by_step)
        plot_seed_curves(curves, seed=int(s), out_dir=curves_dir,
                         baseline=baseline)
        per_seed_summary.append({
            "seed": int(s),
            "curves_json": str(curves_dir / f"seed_{int(s):02d}__curves.json"),
            "curves": curves,
        })
        (curves_dir / f"seed_{int(s):02d}__curves.json").write_text(
            json.dumps(curves, indent=2)
        )

    spatial_seed = args.spatial_seed if args.spatial_seed is not None else int(pool.held_out[0])
    spatial_recs = _eps_records_for_seed(eps_dir, spatial_seed)
    if spatial_recs is not None:
        steps = [int(x) for x in args.spatial_steps.split(",") if x.strip()]
        plot_spatial_residual(spatial_recs, r_bar_by_step,
                              seed=spatial_seed, out_dir=spatial_dir,
                              steps=steps)

    (out_root / "summary.json").write_text(json.dumps({
        "pooled_run": str(pooled_run),
        "train_pool": list(pool.train_pool),
        "held_out": list(pool.held_out),
        "spatial_seed": spatial_seed,
        "per_seed": [
            {k: v for k, v in entry.items() if k != "curves"}
            for entry in per_seed_summary
        ],
    }, indent=2))
    log.info("Task D analysis → %s", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
