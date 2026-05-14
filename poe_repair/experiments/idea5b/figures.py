"""Stage 3 — render Figures 1–9 + N1, N3, N4 from on-disk artefacts.

Mirrors veracity / idea1 figures with the labels swapped to "α / α₀"
and one new figure layout (N1) showing what CLIP "saw" at the early /
peak / late steps of the correction window.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from poe_repair.experiments.idea5b import metrics as IM
from poe_repair.figures._common import image_grid, line_plot, save_fig


def _to_hwc(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().float().clamp(0.0, 1.0)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.permute(1, 2, 0)
    return arr.cpu().numpy()


# ---------------------------------------------------------------------------
# Figure 1 — anchors
# ---------------------------------------------------------------------------


def fig01_anchors(
    *,
    fig_dir: Path,
    poe_path: Path,
    mono_path: Path,
    clip_calibrated_path: Path,
    title_suffix: str = "",
) -> Path:
    cells = [[poe_path, clip_calibrated_path, mono_path]]
    title = "Figure 1 — anchors: PoE → CLIP-guided(α₀) → Mono"
    if title_suffix:
        title += f"\n{title_suffix}"
    return image_grid(
        cells,
        fig_dir / "fig01_anchors.png",
        col_labels=[
            "PoE  (CI baseline)",
            "PoE + CLIP guidance (α₀)",
            "Mono  (literal e_J)",
        ],
        title=title,
        panel_size=3.0,
    )


# ---------------------------------------------------------------------------
# Figure 2 — α-sweep grid
# ---------------------------------------------------------------------------


def fig02_strength_sweep_grid(
    *,
    fig_dir: Path,
    paths_by_alpha: dict[float, Path],
    title_suffix: str = "",
) -> Path:
    alphas = sorted(paths_by_alpha.keys())
    cells = [[paths_by_alpha[a] for a in alphas]]
    title = "Figure 2 — α-sweep: gradual basin transition under CLIP guidance"
    if title_suffix:
        title += f"\n{title_suffix}"
    return image_grid(
        cells,
        fig_dir / "fig02_strength_sweep_grid.png",
        col_labels=[f"α/α₀={a:.1f}" for a in alphas],
        title=title,
        panel_size=2.0,
    )


# ---------------------------------------------------------------------------
# Figure 3 — distance-to-anchor curves
# ---------------------------------------------------------------------------


def fig03_distance_curves(
    *,
    fig_dir: Path,
    distances: dict,
) -> Path:
    alphas = list(distances["alphas"])
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    for ax, key, title in (
        (axes[0], "d_poe", "Distance to PoE  (α=0)"),
        (axes[1], "d_mono", "Distance to Mono  (calibrated)"),
    ):
        ax.plot(alphas, distances["latent_l2"][key], marker="o", label="latent-L2")
        ax.plot(alphas, distances["clip_image_cosine"][key], marker="s", label="CLIP image cosine")
        ax.set_xlabel("α / α₀")
        ax.set_ylabel("distance")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(
        "Figure 3 — gap-closing curves under CLIP guidance",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_fig(fig, fig_dir / "fig03_distance_curves.png")


# ---------------------------------------------------------------------------
# Figure 4 — per-step gradient norm trajectory
# ---------------------------------------------------------------------------


def fig04_grad_norm_trajectory(
    *,
    fig_dir: Path,
    grad_stats: dict,
) -> Path:
    grad_norms = list(grad_stats["grad_norm_per_step"])
    force_norms = list(grad_stats["force_norm_per_step"])
    sims = list(grad_stats["clip_similarity_per_step"])
    steps = list(range(len(grad_norms)))

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.5))
    axes[0].plot(steps, grad_norms, marker="o", color="tab:blue")
    axes[0].set_xlabel("denoising step t")
    axes[0].set_ylabel("‖g_t‖  (CLIP gradient)")
    axes[0].set_title("Per-step CLIP gradient norm")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, force_norms, marker="o", color="tab:orange")
    axes[1].set_xlabel("denoising step t")
    axes[1].set_ylabel("‖α · σ_t · g_t‖  (applied)")
    axes[1].set_title("Applied correction at α = α₀")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, sims, marker="o", color="tab:green")
    axes[2].set_xlabel("denoising step t")
    axes[2].set_ylabel('cos(image, "a cat and a dog")')
    axes[2].set_title("CLIP similarity along the trajectory")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        "Figure 4 — CLIP gradient + applied correction + similarity per step",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_fig(fig, fig_dir / "fig04_grad_norm_trajectory.png")


# ---------------------------------------------------------------------------
# Figure 5 — integrated correction vs gap-closing
# ---------------------------------------------------------------------------


def fig05_force_vs_effect(
    *,
    fig_dir: Path,
    distances: dict,
    grad_stats: dict,
) -> Path:
    alphas = list(distances["alphas"])
    total = list(grad_stats["total_injected_per_alpha"])
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(total, distances["latent_l2"]["d_poe"], marker="o", color="tab:red", label="d_PoE  (latent-L2)")
    ax.plot(total, distances["latent_l2"]["d_mono"], marker="s", color="tab:blue", label="d_Mono (latent-L2)")
    for x, y, a in zip(total, distances["latent_l2"]["d_poe"], alphas):
        ax.annotate(f"{a:.1f}", (x, y), fontsize=7, alpha=0.7)
    ax.set_xlabel("total injected correction  Σ_t α · σ_t · ‖g_t‖")
    ax.set_ylabel("distance (latent-L2)")
    ax.set_title("Figure 5 — integrated CLIP correction ↔ gap-closing")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save_fig(fig, fig_dir / "fig05_force_vs_effect.png")


# ---------------------------------------------------------------------------
# Figure 6 — spatial heatmap at peak step
# ---------------------------------------------------------------------------


def fig06_spatial_heatmap(
    *,
    fig_dir: Path,
    grad_stats: dict,
) -> Path:
    grad_norms = list(grad_stats["grad_norm_per_step"])
    if max(grad_norms) <= 0.0:
        return fig_dir / "fig06_spatial_heatmap.png"
    peak = int(np.argmax(grad_norms))
    anchor = Path(grad_stats["anchor_run_dir"])
    payload = torch.load(
        anchor / "residuals" / f"step_{peak:03d}.pt",
        map_location="cpu", weights_only=False,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    if "decoded_x_hat_0" in payload:
        axes[0].imshow(_to_hwc(payload["decoded_x_hat_0"]))
        axes[0].set_title(f"Tweedie x̂_0 (peak step {peak})")
    else:
        axes[0].text(0.5, 0.5, "(no decoded x̂_0)", ha="center", va="center")
    axes[0].axis("off")

    if "force" in payload:
        F_t = payload["force"].float()
        spatial = F_t.norm(dim=1).squeeze(0).cpu().numpy()
        im = axes[1].imshow(spatial, cmap="magma")
        axes[1].set_title(f"‖F_t(i,j)‖  (peak step {peak})")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        axes[1].text(0.5, 0.5, "(no force saved)", ha="center", va="center")
    axes[1].axis("off")

    fig.suptitle(
        "Figure 6 — spatial localisation of the CLIP-gradient correction",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_fig(fig, fig_dir / "fig06_spatial_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 7 — direction stability
# ---------------------------------------------------------------------------


def fig07_direction_stability(
    *,
    fig_dir: Path,
    grad_stats: dict,
) -> Path:
    mat = np.asarray(grad_stats["direction_stability_matrix"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mat, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xlabel("step t"); ax.set_ylabel("step s")
    ax.set_title("Figure 7 — direction stability  cos(F_s, F_t)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return save_fig(fig, fig_dir / "fig07_direction_stability.png")


# ---------------------------------------------------------------------------
# Figure 9 — trajectory distance to Mono per step
# ---------------------------------------------------------------------------


def fig09_latent_trajectory_distance(
    *,
    fig_dir: Path,
    trajectory: dict,
) -> Path:
    alphas = list(trajectory["alphas"])
    n_entries = int(trajectory["num_steps_plus_one"])
    steps = list(range(n_entries))
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    cmap = plt.get_cmap("viridis")
    for i, a in enumerate(alphas):
        ys = trajectory["per_alpha_distance_to_mono"][f"{a:.2f}"]
        ax.plot(
            steps, ys,
            color=cmap(i / max(1, len(alphas) - 1)),
            label=f"α/α₀={a:.1f}",
        )
    ax.set_xlabel("trajectory entry  (0 = x_T, last = x_0)")
    ax.set_ylabel("‖x_t(α) − x_t(Mono)‖")
    ax.set_title("Figure 9 — path-level convergence to Mono trajectory across α")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    return save_fig(fig, fig_dir / "fig09_latent_trajectory_distance.png")


# ---------------------------------------------------------------------------
# Figure N1 — what CLIP "saw" at three steps
# ---------------------------------------------------------------------------


def figN1_clip_grad_traces(
    *,
    fig_dir: Path,
    grad_stats: dict,
) -> Path:
    grad_norms = list(grad_stats["grad_norm_per_step"])
    sims = list(grad_stats["clip_similarity_per_step"])
    in_window = list(grad_stats.get("in_correction_window_per_step", []))
    if not in_window:
        in_window = [g > 0.0 for g in grad_norms]
    window_steps = [i for i, w in enumerate(in_window) if w]
    if not window_steps:
        return fig_dir / "figN1_clip_grad_traces.png"

    early = window_steps[0]
    late = window_steps[-1]
    if max(grad_norms) > 0.0:
        peak = int(np.argmax(grad_norms))
    else:
        peak = window_steps[len(window_steps) // 2]
    chosen = sorted({early, peak, late})
    while len(chosen) < 3:
        chosen.append(chosen[-1])

    anchor = Path(grad_stats["anchor_run_dir"])
    fig, axes = plt.subplots(2, len(chosen), figsize=(4.5 * len(chosen), 8.5))
    for col, step in enumerate(chosen):
        path = anchor / "residuals" / f"step_{step:03d}.pt"
        if not path.exists():
            for row in range(2):
                axes[row][col].text(0.5, 0.5, f"(step {step} missing)", ha="center", va="center")
                axes[row][col].axis("off")
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        ax = axes[0][col]
        if "decoded_x_hat_0" in payload:
            ax.imshow(_to_hwc(payload["decoded_x_hat_0"]))
        else:
            ax.text(0.5, 0.5, "(no decoded x̂_0)", ha="center", va="center")
        sim = float(payload.get("clip_similarity", 0.0))
        ax.set_title(f"step {step}  |  cos = {sim:+.3f}")
        ax.axis("off")

        ax = axes[1][col]
        if "force" in payload:
            F_t = payload["force"].float()
            spatial = F_t.norm(dim=1).squeeze(0).cpu().numpy()
            im = ax.imshow(spatial, cmap="magma")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, "(no force)", ha="center", va="center")
        ax.set_title(f"‖F_t(i,j)‖ at step {step}")
        ax.axis("off")

    fig.suptitle(
        "Figure N1 — what CLIP saw and where the gradient pushed (early / peak / late)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_fig(fig, fig_dir / "figN1_clip_grad_traces.png")


# ---------------------------------------------------------------------------
# Figure N3 — four-method overlay (HEADLINE)
# ---------------------------------------------------------------------------


def figN3_method_overlay(
    *,
    fig_dir: Path,
    method_comparison: dict,
) -> Path:
    series = method_comparison["series"]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    colours = {
        "veracity_residual": "tab:purple",
        "force_overlap": "tab:orange",
        "force_alignment": "tab:green",
        "clip_guided": "tab:cyan",
    }
    labels = {
        "veracity_residual": "veracity Δ (Mono pull)",
        "force_overlap": "Force-A (overlap repulsion)",
        "force_alignment": "Force-B (alignment damping)",
        "clip_guided": "CLIP-guided (idea 5b)",
    }
    for col, (key, title) in enumerate((
        ("d_mono_l2", "d_Mono (latent-L2)"),
        ("d_mono_clip", "d_Mono (CLIP image cosine)"),
    )):
        ax = axes[col]
        for sid, sdata in series.items():
            ax.plot(
                sdata["x_total_injected"], sdata[key],
                marker="o", color=colours.get(sid, "black"),
                label=labels.get(sid, sid),
            )
        ax.set_xlabel("total injected correction  Σ_t α · ‖force_t‖")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(
        "Figure N3 — basin is method-agnostic: four corrective forces, one barrier",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_fig(fig, fig_dir / "figN3_method_overlay.png")


# ---------------------------------------------------------------------------
# Figure N4 — held-out grid
# ---------------------------------------------------------------------------


def figN4_heldout_grid(
    *,
    fig_dir: Path,
    cells: dict[int, dict[str, Path]],
) -> Path:
    seeds = sorted(cells.keys())
    columns = ["poe", "mono", "sched_m2", "clip_guided"]
    rows: list[list[Path]] = []
    for seed in seeds:
        rows.append([cells[seed][c] for c in columns])
    return image_grid(
        rows,
        fig_dir / "figN4_heldout_grid.png",
        col_labels=["PoE", "Mono", "sched-M2 + ê_J", "PoE + CLIP guidance"],
        row_labels=[f"seed {s}" for s in seeds],
        title="Figure N4 — held-out: cat × dog at multiple seeds",
        panel_size=2.6,
    )
