"""Plan-15 Phase 1 (seed-42 only) — compute cheap kinematic trajectory
metrics from the cached seed-42 single-seed LoRA `.npy` arrays and
render a multi-panel figure.

Zero new sampling. Reads:
    artifacts/rung1-overfit/lora/a_cat__x__a_dog/seed_42/run__local/mds_cache/
        static/{mono,solo_a,solo_b}.npy           (51, 65536) float16
        cells/epoch_NNNN/lambda_LL.LL.npy         (51, 65536) float16

Emits:
    <out>/metrics.json   — per-condition scalars and curves
    <out>/figure.png     — 4-panel summary
        A. per-step curvature θ_t  vs t
        B. distance to mono z_t  vs t
        C. z-space LoRA-correction proxy ‖z_t^{λ=1} − z_t^{λ=0}‖  vs t
        D. terminal-endpoint distances to mono z_0  (bar)

Treat panel D as "n=1 point comparison", not a statistical claim.
Plan-15's n=12 statistical version requires the backfill in
``scripts/manifold/sample_with_trajectory.py``.

Usage::

    python scripts/manifold/seed42_phase1.py
    python scripts/manifold/seed42_phase1.py --epoch 1600 --lambdas 0.00,0.50,1.00
    python scripts/manifold/seed42_phase1.py --results-root <alt> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from poe_repair import paths as poe_paths

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RESULTS_ROOT = (
    poe_paths.resolve(poe_paths.ONE_PAIR_ONE_SEED) / "a_cat__x__a_dog" / "seed_42" / "run__local"
)
DEFAULT_OUT = REPO_ROOT / "outputs" / "manifold_cache" / "seed_42_phase1"
DEFAULT_EPOCH = 1600
DEFAULT_LAMBDAS = (0.00, 0.50, 1.00)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_traj(path: Path) -> np.ndarray:
    """Return float32 (51, D) array, where D = 4·128·128 = 65536."""
    z = np.load(path)
    return z.astype(np.float32)


def load_all(results_root: Path, epoch: int,
             lambdas: list[float]) -> dict[str, np.ndarray]:
    static = results_root / "mds_cache" / "static"
    cell = results_root / "mds_cache" / "cells" / f"epoch_{epoch:04d}"
    out: dict[str, np.ndarray] = {}
    out["mono"] = load_traj(static / "mono.npy")
    out["solo_a"] = load_traj(static / "solo_a.npy")
    out["solo_b"] = load_traj(static / "solo_b.npy")
    for lam in lambdas:
        out[f"lambda_{lam:.2f}"] = load_traj(cell / f"lambda_{lam:.2f}.npy")
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    arc_length: float
    curvature_per_t: np.ndarray            # (T-1,)
    dist_to_mono_per_t: np.ndarray         # (T+1,)
    terminal_dist_to_mono: float


def per_step_displacements(z: np.ndarray) -> np.ndarray:
    """Δ_t = z_{t+1} − z_t. Shape (T, D)."""
    return np.diff(z, axis=0)


def euclidean_arc_length(z: np.ndarray) -> float:
    d = per_step_displacements(z)
    return float(np.linalg.norm(d, axis=1).sum())


def curvature_per_step(z: np.ndarray) -> np.ndarray:
    """θ_t = ∠(Δ_{t-1}, Δ_t) for t = 1..T-1. Shape (T-1,)."""
    d = per_step_displacements(z)                 # (T, D)
    a = d[:-1]                                    # (T-1, D)
    b = d[1:]                                     # (T-1, D)
    a_n = np.linalg.norm(a, axis=1) + 1e-12
    b_n = np.linalg.norm(b, axis=1) + 1e-12
    cos_t = np.einsum("td,td->t", a, b) / (a_n * b_n)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    return np.arccos(cos_t)


def dist_to_path(z_cond: np.ndarray, z_ref: np.ndarray) -> np.ndarray:
    """‖z_t^{cond} − z_t^{ref}‖ per t. Shape (T+1,)."""
    return np.linalg.norm(z_cond - z_ref, axis=1)


def compute_metrics(z: np.ndarray, z_mono: np.ndarray) -> Metrics:
    return Metrics(
        arc_length=euclidean_arc_length(z),
        curvature_per_t=curvature_per_step(z),
        dist_to_mono_per_t=dist_to_path(z, z_mono),
        terminal_dist_to_mono=float(np.linalg.norm(z[-1] - z_mono[-1])),
    )


def correction_proxy(z_lam1: np.ndarray, z_lam0: np.ndarray) -> np.ndarray:
    """z-space proxy for the LoRA correction's path impact.

    ‖z_t^{λ=1} − z_t^{λ=0}‖ per t.

    Not the eps-space r_t — that would need cached velocities, which
    we don't have. This is the path-divergence between the full-LoRA
    and no-LoRA arms, integrated up to step t by the sampler.
    """
    return np.linalg.norm(z_lam1 - z_lam0, axis=1)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def render_figure(
    metrics_by_cond: dict[str, Metrics],
    proxy_curve: np.ndarray,
    lambdas: list[float],
    epoch: int,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    # Stable, condition-keyed colour scheme.
    colours = {
        "mono": "#000000",
        "solo_a": "#cc4444",
        "solo_b": "#3a7ccc",
    }
    cmap = plt.get_cmap("plasma")
    for i, lam in enumerate(lambdas):
        colours[f"lambda_{lam:.2f}"] = cmap(0.15 + 0.7 * i / max(1, len(lambdas) - 1))

    # Plot order — references first, then PoE+λR cells.
    plot_conditions = ["mono", "solo_a", "solo_b"] + [
        f"lambda_{lam:.2f}" for lam in lambdas
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Seed 42 — cheap kinematic trajectory metrics  (epoch {epoch})",
        fontsize=13,
    )

    # Panel A — curvature
    ax = axes[0, 0]
    for cond in plot_conditions:
        if cond not in metrics_by_cond:
            continue
        m = metrics_by_cond[cond]
        ax.plot(m.curvature_per_t, label=cond, color=colours[cond], linewidth=1.2)
    ax.set_title("A. Per-step curvature  θ_t = ∠(Δ_{t-1}, Δ_t)")
    ax.set_xlabel("step t")
    ax.set_ylabel("radians")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel B — distance to mono z_t
    ax = axes[0, 1]
    for cond in plot_conditions:
        if cond == "mono" or cond not in metrics_by_cond:
            continue
        m = metrics_by_cond[cond]
        ax.plot(m.dist_to_mono_per_t, label=cond, color=colours[cond], linewidth=1.2)
    ax.set_title("B. ‖z_t^{cond} − z_t^{mono}‖  vs t")
    ax.set_xlabel("step t")
    ax.set_ylabel("Euclidean distance")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel C — z-space LoRA correction proxy
    ax = axes[1, 0]
    ax.plot(proxy_curve, color="#8844aa", linewidth=1.5)
    ax.set_title("C. z-space proxy  ‖z_t^{λ=1.00} − z_t^{λ=0.00}‖  vs t")
    ax.set_xlabel("step t")
    ax.set_ylabel("Euclidean distance")
    ax.grid(True, alpha=0.3)

    # Panel D — terminal endpoint distance + arc length bar
    ax = axes[1, 1]
    conds_d = [c for c in plot_conditions if c != "mono" and c in metrics_by_cond]
    xs = np.arange(len(conds_d))
    bar_w = 0.4
    term = [metrics_by_cond[c].terminal_dist_to_mono for c in conds_d]
    arc = [metrics_by_cond[c].arc_length for c in conds_d]
    arc_norm = [a / max(arc) for a in arc]
    ax.bar(xs - bar_w/2, term, bar_w, label="terminal ‖z_0−z_0^mono‖",
           color=[colours[c] for c in conds_d])
    ax.bar(xs + bar_w/2, arc_norm, bar_w, label="arc length (normalised)",
           color="#888888", alpha=0.7)
    ax.set_title("D. Terminal distance to mono z_0  +  arc length (normalised)")
    ax.set_xticks(xs)
    ax.set_xticklabels(conds_d, rotation=20, ha="right", fontsize=8)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    ap.add_argument("--epoch", type=int, default=DEFAULT_EPOCH)
    ap.add_argument("--lambdas",
                    default=",".join(f"{x:.2f}" for x in DEFAULT_LAMBDAS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    results_root = Path(args.results_root)
    out_root = Path(args.out)
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]

    if not results_root.is_dir():
        print(f"results_root not found: {results_root}", file=sys.stderr)
        return 1

    print(f"reading cached trajectories from: {results_root}")
    print(f"  epoch = {args.epoch}, lambdas = {lambdas}")

    z = load_all(results_root, args.epoch, lambdas)
    z_mono = z["mono"]
    print(f"  z_t shape per cell = {z_mono.shape}  (51 steps × 65536 dims)")

    metrics_by_cond: dict[str, Metrics] = {}
    for cond, arr in z.items():
        metrics_by_cond[cond] = compute_metrics(arr, z_mono)

    if "lambda_0.00" in z and "lambda_1.00" in z:
        proxy = correction_proxy(z["lambda_1.00"], z["lambda_0.00"])
    else:
        proxy = np.zeros(z_mono.shape[0])

    # Summary to stdout for an eyeball.
    print()
    print(f"{'condition':<14} {'arc_length':>14}  {'term-d(mono)':>14}  "
          f"{'mean κ':>10}")
    print("-" * 60)
    plot_conditions = ["mono", "solo_a", "solo_b"] + [
        f"lambda_{lam:.2f}" for lam in lambdas
    ]
    for cond in plot_conditions:
        if cond not in metrics_by_cond:
            continue
        m = metrics_by_cond[cond]
        print(f"{cond:<14} {m.arc_length:>14.3f}  {m.terminal_dist_to_mono:>14.3f}  "
              f"{float(m.curvature_per_t.mean()):>10.4f}")

    if proxy.any():
        print()
        print(f"z-space proxy ‖z_t^{{λ=1}} − z_t^{{λ=0}}‖:")
        print(f"  max  at t = {int(proxy.argmax())}  (value {proxy.max():.3f})")
        print(f"  mean over all t = {proxy.mean():.3f}")
        print(f"  terminal (t = T) = {proxy[-1]:.3f}")

    # Write metrics JSON.
    out_root.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "results_root": str(results_root),
        "epoch": int(args.epoch),
        "lambdas": lambdas,
        "z_shape": list(z_mono.shape),
        "per_condition": {
            cond: {
                "arc_length": m.arc_length,
                "terminal_dist_to_mono": m.terminal_dist_to_mono,
                "mean_curvature": float(m.curvature_per_t.mean()),
                "curvature_per_t": m.curvature_per_t.tolist(),
                "dist_to_mono_per_t": m.dist_to_mono_per_t.tolist(),
            }
            for cond, m in metrics_by_cond.items()
        },
        "correction_proxy_per_t": proxy.tolist(),
    }
    json_path = out_root / f"metrics_epoch_{args.epoch:04d}.json"
    json_path.write_text(json.dumps(json_payload, indent=2))
    print()
    print(f"wrote metrics → {json_path}")

    fig_path = out_root / f"figure_epoch_{args.epoch:04d}.png"
    render_figure(metrics_by_cond, proxy, lambdas, args.epoch, fig_path)
    print(f"wrote figure  → {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
