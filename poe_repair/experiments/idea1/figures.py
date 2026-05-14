"""Stage 3 — render Figures 1-9 + N1-N4 from on-disk artefacts.

Figures 1, 2, 3, 4, 5, 6, 7, 9 mirror the veracity figures but for the
PoE-internal forces (axis swap λ → α, two force variants overlaid where
sensible). Figure 8 is omitted (no PMI identity here). Figures N1-N4 are
new and method-specific:

  - N1 — attention-overlap detector traces (Force A only).
  - N2 — score-alignment detector field (Force B only).
  - N3 — headline overlay: d_Mono curves for residual / Force A / Force B
         on a shared total-injected-correction axis.
  - N4 — held-out grid: rows = seeds, columns = methods.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from poe_repair.experiments.idea1 import metrics as IM
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
    poe_internal_calibrated_paths: dict[str, Path],   # {"overlap": ..., "alignment": ...}
    title_suffix: str = "",
) -> Path:
    cells = [[
        poe_path,
        poe_internal_calibrated_paths.get("overlap", poe_path),
        poe_internal_calibrated_paths.get("alignment", poe_path),
        mono_path,
    ]]
    title = "Figure 1 — anchors: PoE → PoE+force(A) → PoE+force(B) → Mono"
    if title_suffix:
        title += f"\n{title_suffix}"
    return image_grid(
        cells,
        fig_dir / "fig01_anchors.png",
        col_labels=[
            "PoE  (CI baseline)",
            "PoE + Force-A (overlap, α₀)",
            "PoE + Force-B (alignment, α₀)",
            "Mono  (literal e_J)",
        ],
        title=title,
        panel_size=2.7,
    )


# ---------------------------------------------------------------------------
# Figure 2 — α-sweep image grids, one row per force
# ---------------------------------------------------------------------------


def fig02_strength_sweep_grid(
    *,
    fig_dir: Path,
    paths_by_force: dict[str, dict[float, Path]],   # {"overlap": {α: path}, "alignment": {...}}
    title_suffix: str = "",
) -> Path:
    rows: list[list[Path]] = []
    row_labels: list[str] = []
    col_labels: list[str] | None = None
    for force_kind in ("overlap", "alignment"):
        if force_kind not in paths_by_force:
            continue
        alphas = sorted(paths_by_force[force_kind].keys())
        rows.append([paths_by_force[force_kind][a] for a in alphas])
        row_labels.append(f"Force-{force_kind[0].upper()}")
        if col_labels is None:
            col_labels = [f"α/α₀={a:.1f}" for a in alphas]

    if not rows:
        raise ValueError("paths_by_force must contain at least one force kind")

    title = "Figure 2 — strength sweep: gradual basin transition under PoE-internal forces"
    if title_suffix:
        title += f"\n{title_suffix}"
    return image_grid(
        rows,
        fig_dir / "fig02_strength_sweep_grid.png",
        col_labels=col_labels,
        row_labels=row_labels,
        title=title,
        panel_size=1.9,
    )


# ---------------------------------------------------------------------------
# Figure 3 — distance-to-anchor curves per force
# ---------------------------------------------------------------------------


def fig03_distance_curves(
    *,
    fig_dir: Path,
    distances_by_force: dict[str, dict],
) -> Path:
    n_forces = len(distances_by_force)
    fig, axes = plt.subplots(n_forces, 2, figsize=(12.0, 4.5 * n_forces), squeeze=False)
    for row, (force_kind, dist) in enumerate(distances_by_force.items()):
        alphas = list(dist["alphas"])
        for col, (label, key) in enumerate(
            (("Distance to PoE  (α=0)", "d_poe"), ("Distance to Mono  (α=1)", "d_mono"))
        ):
            ax = axes[row][col]
            ax.plot(alphas, dist["latent_l2"][key], marker="o", label="latent-L2")
            ax.plot(alphas, dist["clip_image_cosine"][key], marker="s", label="CLIP image cosine")
            ax.set_xlabel("α / α₀")
            ax.set_ylabel("distance")
            ax.set_title(f"Force-{force_kind[0].upper()} — {label}")
            ax.grid(True, alpha=0.3)
            ax.legend()
    fig.suptitle(
        "Figure 3 — gap-closing curves: d_PoE rises, d_Mono falls (per force)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_fig(fig, fig_dir / "fig03_distance_curves.png")


# ---------------------------------------------------------------------------
# Figure 4 — per-step force-norm trajectories
# ---------------------------------------------------------------------------


def fig04_force_norm_trajectory(
    *,
    fig_dir: Path,
    force_stats_by_force: dict[str, dict],
) -> Path:
    n_forces = len(force_stats_by_force)
    fig, axes = plt.subplots(n_forces, 2, figsize=(13.0, 4.5 * n_forces), squeeze=False)
    for row, (force_kind, stats) in enumerate(force_stats_by_force.items()):
        norms = list(stats["force_norm_per_step"])
        steps = list(range(len(norms)))
        alphas = list(stats["alphas"])

        axes[row][0].plot(steps, norms, marker="o", color="tab:blue")
        axes[row][0].set_xlabel("denoising step t")
        axes[row][0].set_ylabel("‖F_t‖")
        axes[row][0].set_title(f"Force-{force_kind[0].upper()} — unscaled per-step norm")
        axes[row][0].grid(True, alpha=0.3)

        cmap = plt.get_cmap("viridis")
        for i, a in enumerate(alphas):
            scaled = [a * v for v in norms]
            axes[row][1].plot(
                steps, scaled,
                color=cmap(i / max(1, len(alphas) - 1)),
                label=f"α/α₀={a:.1f}",
            )
        axes[row][1].set_xlabel("denoising step t")
        axes[row][1].set_ylabel("‖α · F_t‖")
        axes[row][1].set_title("Applied correction per step, fanned by α")
        axes[row][1].grid(True, alpha=0.3)
        axes[row][1].legend(fontsize=7, ncol=2)

    fig.suptitle("Figure 4 — force-norm trajectories per force variant", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_fig(fig, fig_dir / "fig04_force_norm_trajectory.png")


# ---------------------------------------------------------------------------
# Figure 5 — integrated force vs gap-closing
# ---------------------------------------------------------------------------


def fig05_force_vs_effect(
    *,
    fig_dir: Path,
    distances_by_force: dict[str, dict],
    force_stats_by_force: dict[str, dict],
) -> Path:
    n_forces = len(distances_by_force)
    fig, axes = plt.subplots(1, n_forces, figsize=(6.0 * n_forces, 5.0), squeeze=False)
    for col, (force_kind, dist) in enumerate(distances_by_force.items()):
        stats = force_stats_by_force[force_kind]
        total = list(stats["total_injected_per_alpha"])
        alphas = list(dist["alphas"])
        ax = axes[0][col]
        ax.plot(total, dist["latent_l2"]["d_poe"], marker="o", color="tab:red", label="d_PoE")
        ax.plot(total, dist["latent_l2"]["d_mono"], marker="s", color="tab:blue", label="d_Mono")
        for x, y, a in zip(total, dist["latent_l2"]["d_poe"], alphas):
            ax.annotate(f"{a:.1f}", (x, y), fontsize=7, alpha=0.7)
        ax.set_xlabel("total injected force  Σ_t α · ‖F_t‖")
        ax.set_ylabel("distance (latent-L2)")
        ax.set_title(f"Force-{force_kind[0].upper()}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(
        "Figure 5 — integrated correction ↔ gap-closing (per force)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_fig(fig, fig_dir / "fig05_force_vs_effect.png")


# ---------------------------------------------------------------------------
# Figure 6 — spatial heatmap of F_t at peak step
# ---------------------------------------------------------------------------


def _tweedie_x0(x_t: torch.Tensor, eps: torch.Tensor, alpha_bar: float) -> torch.Tensor:
    sa = float(alpha_bar) ** 0.5
    so = float(1.0 - alpha_bar) ** 0.5
    return (x_t - so * eps) / sa


def fig06_spatial_heatmap(
    *,
    fig_dir: Path,
    force_stats_by_force: dict[str, dict],
    seed_dir: Path,
    ctx=None,
) -> Path:
    n_forces = len(force_stats_by_force)
    fig, axes = plt.subplots(n_forces, 2, figsize=(11.0, 5.0 * n_forces), squeeze=False)
    for row, (force_kind, stats) in enumerate(force_stats_by_force.items()):
        norms = list(stats["force_norm_per_step"])
        peak = int(np.argmax(norms))
        anchor = Path(stats["anchor_run_dir"])
        payload = torch.load(anchor / "residuals" / f"step_{peak:03d}.pt", map_location="cpu")
        F_t = payload["force"].float()                               # [B, C, H, W]
        spatial = F_t.norm(dim=1).squeeze(0).cpu().numpy()           # [H, W]

        ax = axes[row][0]
        if ctx is not None and "eps_poe" in payload:
            timestep = int(payload["timestep"])
            alpha_bar_t = float(
                ctx.scheduler.alphas_cumprod[timestep].to(dtype=torch.float64).item()
            )
            x_t = payload["x_t"].float().to(ctx.device, ctx.dtype)
            eps_poe = payload["eps_poe"].float().to(ctx.device, ctx.dtype)
            x0_poe = _tweedie_x0(x_t, eps_poe, alpha_bar_t)
            from poe_repair.runtime import decode_latents
            img = decode_latents(ctx.models, x0_poe).cpu()
            ax.imshow(_to_hwc(img))
            ax.set_title(f"Tweedie x̂_0 from ε̃_PoE  (step {peak})")
            ax.axis("off")
        else:
            ax.text(0.5, 0.5, "(VAE not provided)", ha="center", va="center")
            ax.axis("off")

        ax = axes[row][1]
        im = ax.imshow(spatial, cmap="magma")
        ax.set_title(f"Force-{force_kind[0].upper()}: ‖F_t(i,j)‖ at peak step {peak}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Figure 6 — spatial localisation of the corrective force", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_fig(fig, fig_dir / "fig06_spatial_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 7 — direction stability (T×T cosine of F_t)
# ---------------------------------------------------------------------------


def fig07_direction_stability(
    *,
    fig_dir: Path,
    force_stats_by_force: dict[str, dict],
) -> Path:
    n_forces = len(force_stats_by_force)
    fig, axes = plt.subplots(1, n_forces, figsize=(6.5 * n_forces, 5.5), squeeze=False)
    for col, (force_kind, stats) in enumerate(force_stats_by_force.items()):
        mat = np.asarray(stats["direction_stability_matrix"], dtype=np.float64)
        ax = axes[0][col]
        im = ax.imshow(mat, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_xlabel("step t"); ax.set_ylabel("step s")
        ax.set_title(f"Force-{force_kind[0].upper()}: cos(F_s, F_t)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Figure 7 — direction stability of the per-step force", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_fig(fig, fig_dir / "fig07_direction_stability.png")


# ---------------------------------------------------------------------------
# Figure 9 — latent trajectory distance to Mono
# ---------------------------------------------------------------------------


def fig09_latent_trajectory_distance(
    *,
    fig_dir: Path,
    trajectory_by_force: dict[str, dict],
) -> Path:
    n_forces = len(trajectory_by_force)
    fig, axes = plt.subplots(1, n_forces, figsize=(8.0 * n_forces, 5.0), squeeze=False)
    for col, (force_kind, traj) in enumerate(trajectory_by_force.items()):
        alphas = list(traj["alphas"])
        n_entries = int(traj["num_steps_plus_one"])
        steps = list(range(n_entries))
        ax = axes[0][col]
        cmap = plt.get_cmap("viridis")
        for i, a in enumerate(alphas):
            ys = traj["per_alpha_distance_to_mono"][f"{a:.2f}"]
            ax.plot(
                steps, ys,
                color=cmap(i / max(1, len(alphas) - 1)),
                label=f"α/α₀={a:.1f}",
            )
        ax.set_xlabel("trajectory entry  (0 = x_T, last = x_0)")
        ax.set_ylabel("‖x_t(α) − x_t(Mono)‖")
        ax.set_title(f"Force-{force_kind[0].upper()}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(
        "Figure 9 — path-level convergence to Mono trajectory across α",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_fig(fig, fig_dir / "fig09_latent_trajectory_distance.png")


# ---------------------------------------------------------------------------
# Figure N1 — attention-overlap detector traces (Force A only)
# ---------------------------------------------------------------------------


def figN1_overlap_traces(
    *,
    fig_dir: Path,
    force_stats: dict,
) -> Path:
    """Three columns (early, peak, late steps) × three rows (M_cat, M_dog, O)."""
    norms = list(force_stats["force_norm_per_step"])
    T = len(norms)
    if T == 0:
        return fig_dir / "figN1_attention_overlap_traces.png"
    peak = int(np.argmax(norms))
    early = max(0, peak // 3)
    late = min(T - 1, peak + (T - peak) // 2)
    chosen = [early, peak, late]
    anchor = Path(force_stats["anchor_run_dir"])

    fig, axes = plt.subplots(3, 3, figsize=(11.0, 10.0))
    row_labels = ["M_cat (token A)", "M_dog (token B)", "O = M_cat · M_dog"]
    for col, step in enumerate(chosen):
        payload = torch.load(anchor / "residuals" / f"step_{step:03d}.pt", map_location="cpu")
        if "M_cat" not in payload or "M_dog" not in payload:
            for row in range(3):
                axes[row][col].text(0.5, 0.5, "(no attention saved)", ha="center", va="center")
                axes[row][col].axis("off")
            continue
        M_cat = payload["M_cat"].float().cpu().numpy()
        M_dog = payload["M_dog"].float().cpu().numpy()
        O = M_cat * M_dog
        for row, arr in enumerate([M_cat, M_dog, O]):
            ax = axes[row][col]
            im = ax.imshow(arr, cmap="magma")
            ax.set_title(f"step {step}" if row == 0 else "")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(row_labels[row])
                ax.text(
                    -0.08, 0.5, row_labels[row],
                    fontsize=10, ha="right", va="center", transform=ax.transAxes,
                )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Figure N1 — attention-overlap detector at three steps (Force A)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_fig(fig, fig_dir / "figN1_attention_overlap_traces.png")


# ---------------------------------------------------------------------------
# Figure N2 — score-alignment field (Force B only)
# ---------------------------------------------------------------------------


def figN2_alignment_field(
    *,
    fig_dir: Path,
    force_stats: dict,
) -> Path:
    norms = list(force_stats["force_norm_per_step"])
    T = len(norms)
    if T == 0:
        return fig_dir / "figN2_score_alignment_field.png"
    peak = int(np.argmax(norms))
    early = max(0, peak // 3)
    late = min(T - 1, peak + (T - peak) // 2)
    chosen = [early, peak, late]
    anchor = Path(force_stats["anchor_run_dir"])

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5))
    for col, step in enumerate(chosen):
        payload = torch.load(anchor / "residuals" / f"step_{step:03d}.pt", map_location="cpu")
        ax = axes[col]
        if "alignment_field" not in payload:
            ax.text(0.5, 0.5, "(no alignment saved)", ha="center", va="center")
            ax.axis("off")
            continue
        a_field = payload["alignment_field"].float().squeeze().cpu().numpy()
        if a_field.ndim == 3:
            a_field = a_field[0]
        im = ax.imshow(a_field, cmap="coolwarm")
        ax.set_title(f"step {step}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Figure N2 — score-alignment field at three steps (Force B)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save_fig(fig, fig_dir / "figN2_score_alignment_field.png")


# ---------------------------------------------------------------------------
# Figure N3 — method overlay (HEADLINE)
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
    }
    labels = {
        "veracity_residual": "veracity Δ (Mono-anchor pull)",
        "force_overlap": "Force-A (overlap repulsion)",
        "force_alignment": "Force-B (alignment damping)",
    }
    for col, (key, title) in enumerate(
        (("d_mono_l2", "d_Mono (latent-L2)"),
         ("d_mono_clip", "d_Mono (CLIP image cosine)"))
    ):
        ax = axes[col]
        for sid, sdata in series.items():
            ax.plot(
                sdata["x_total_injected"],
                sdata[key],
                marker="o",
                color=colours.get(sid, "black"),
                label=labels.get(sid, sid),
            )
        ax.set_xlabel("total injected correction  Σ_t α · ‖F_t‖")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(
        "Figure N3 — basin is method-agnostic: three corrective forces, one barrier",
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
    cells: dict[int, dict[str, Path]],   # seed -> {"poe": ..., "mono": ..., "overlap": ..., "alignment": ...}
) -> Path:
    if not cells:
        raise ValueError("cells must contain at least one seed")
    seeds = sorted(cells.keys())
    columns = ["poe", "mono", "overlap", "alignment"]
    rows: list[list[Path]] = []
    for seed in seeds:
        row = [cells[seed][col] for col in columns]
        rows.append(row)
    return image_grid(
        rows,
        fig_dir / "figN4_heldout_grid.png",
        col_labels=["PoE", "Mono", "PoE + Force-A", "PoE + Force-B"],
        row_labels=[f"seed {s}" for s in seeds],
        title="Figure N4 — held-out: cat × dog at multiple seeds",
        panel_size=2.6,
    )
