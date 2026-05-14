"""D-series plotting. Pure rendering: input is the metric objects, output is
PNGs under the configured ``fig_dir``."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from poe_repair.figures._common import overlay_boxes, save_fig
from poe_repair.experiments.thread_c_structure.metrics import (
    BasisAlignment, ClusterOrderedPanels, ConsecutiveCosine, CrossSeedCosine,
    DirectionMagnitudeSplit, PcaGrid, PcaWithGuards, SpatialPanel, SvdEnergy,
)


# Distinct per-seed colour palette (D4-B, D4-C, PCA grid). Picked so the three
# default seeds (4, 42, 123) are easily distinguished from PALETTE colours.
SEED_PALETTE = {
    4: "#E07A5F",
    7: "#9D4EDD",
    42: "#3D5A80",
    123: "#52B788",
}


def _seed_colour(seed: int, fallback_index: int = 0) -> str:
    if seed in SEED_PALETTE:
        return SEED_PALETTE[seed]
    pool = ["#274C77", "#E07A5F", "#3D5A80", "#52B788", "#9D4EDD", "#F4A261"]
    return pool[fallback_index % len(pool)]


PALETTE = {
    # Reused from veracity figure style; conservative greens / reds.
    "delta": "#274C77",
    "j_minus_null": "#E07A5F",
    "a_minus_b": "#3D5A80",
    "threshold": "#888888",
    "good": "#2E8B57",
    "bad": "#B23A48",
}


# ---------------------------------------------------------------------------
# D1-A
# ---------------------------------------------------------------------------


def render_d1a(
    cos: ConsecutiveCosine,
    fig_path: Path,
    *,
    pair_slug: str,
    seed: int,
    threshold: float = 0.85,
    window: tuple[int, int] = (5, 25),
) -> Path:
    mean_in_window = cos.mean_over(*window)
    passed = mean_in_window >= threshold

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(cos.step_indices, cos.cosines, color=PALETTE["delta"], lw=1.8)
    ax.axhline(threshold, color=PALETTE["threshold"], ls="--", lw=1.0,
               label=f"pass ≥ {threshold:.2f}")
    ax.axvspan(window[0], window[1], color=PALETTE["good"], alpha=0.08,
               label=f"window [{window[0]}, {window[1]}]")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("denoising step t")
    ax.set_ylabel("cos(Δ_t, Δ_{t+1})")
    verdict = "PASS" if passed else "FAIL"
    color = PALETTE["good"] if passed else PALETTE["bad"]
    ax.set_title(
        f"D1-A — direction stability  |  {pair_slug} seed={seed}  |  "
        f"⟨cos⟩_{{{window[0]}-{window[1]}}} = {mean_in_window:.3f}  [{verdict}]",
        color=color, fontsize=10,
    )
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    return save_fig(fig, fig_path)


# ---------------------------------------------------------------------------
# D1-B
# ---------------------------------------------------------------------------


def render_d1b(
    svd: SvdEnergy,
    fig_path: Path,
    *,
    pair_slug: str,
    seed: int,
    threshold_top3: float = 0.8,
    top_ks: tuple[int, ...] = (1, 2, 3, 5, 10),
) -> Path:
    cum = [svd.cumulative_topk[k] for k in top_ks]
    top3 = svd.cumulative_topk.get(3, float("nan"))
    passed = top3 >= threshold_top3

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    xs = np.arange(len(top_ks))
    bars = ax.bar(xs, cum, color=PALETTE["delta"])
    # Mark the top-3 bar.
    for k, bar in zip(top_ks, bars):
        if k == 3:
            bar.set_edgecolor(PALETTE["good"] if passed else PALETTE["bad"])
            bar.set_linewidth(2.2)
    ax.axhline(threshold_top3, color=PALETTE["threshold"], ls="--", lw=1.0,
               label=f"pass top-3 ≥ {threshold_top3:.2f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"top-{k}" for k in top_ks])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("cumulative variance share")
    verdict = "PASS" if passed else "FAIL"
    color = PALETTE["good"] if passed else PALETTE["bad"]
    ax.set_title(
        f"D1-B — low-rank energy  |  {pair_slug} seed={seed}  |  "
        f"top-3 = {top3:.3f}  [{verdict}]",
        color=color, fontsize=10,
    )
    for x, v in zip(xs, cum):
        ax.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    return save_fig(fig, fig_path)


# ---------------------------------------------------------------------------
# D1-C
# ---------------------------------------------------------------------------


def render_d1c(
    basis: BasisAlignment,
    fig_path: Path,
    *,
    pair_slug: str,
    seed: int,
    threshold: float = 0.5,
    window: tuple[int, int] = (5, 25),
) -> Path:
    win_max = basis.max_window_cos(*window)
    best = max(win_max.values()) if win_max else float("nan")
    passed = best >= threshold

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(basis.step_indices, basis.cos_delta_vs_j_null,
            color=PALETTE["j_minus_null"], lw=1.6,
            label=f"ε̃_J − ε̃_∅  (⟨cos⟩={win_max['j_minus_null']:.3f})")
    ax.plot(basis.step_indices, basis.cos_delta_vs_a_minus_b,
            color=PALETTE["a_minus_b"], lw=1.6,
            label=f"ε̃_A − ε̃_B  (⟨cos⟩={win_max['a_minus_b']:.3f})")
    ax.axhline(threshold, color=PALETTE["threshold"], ls="--", lw=1.0,
               label=f"pass ≥ {threshold:.2f}")
    ax.axhline(0.0, color="black", lw=0.5)
    ax.axvspan(window[0], window[1], color=PALETTE["good"], alpha=0.08)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("denoising step t")
    ax.set_ylabel("cos(Δ_t, candidate_t)")
    verdict = "PASS" if passed else "FAIL"
    color = PALETTE["good"] if passed else PALETTE["bad"]
    ax.set_title(
        f"D1-C — Mono-free basis alignment  |  {pair_slug} seed={seed}  |  "
        f"best ⟨cos⟩_{{{window[0]}-{window[1]}}} = {best:.3f}  [{verdict}]",
        color=color, fontsize=10,
    )
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    return save_fig(fig, fig_path)


# ---------------------------------------------------------------------------
# D2
# ---------------------------------------------------------------------------


def render_d2(
    panels: list[SpatialPanel],
    fig_path: Path,
    *,
    pair_slug: str,
    seed: int,
    overlay_image_path: Path | None = None,
    detections: list[dict] | None = None,
    detection_palette: dict[str, str] | None = None,
) -> Path:
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), squeeze=False)
    # Shared colour scale across panels for honesty.
    vmax = max(float(p.heatmap.max()) for p in panels) if panels else 1.0
    for ax, panel in zip(axes[0], panels):
        im = ax.imshow(panel.heatmap.cpu().numpy(), cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(f"t={panel.step_index} (timestep {panel.timestep})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if detections is not None and overlay_image_path is not None:
            # Detection boxes are in *image-pixel* coordinates (e.g. 1024×1024);
            # the heatmap is in latent pixels (e.g. 128×128). Rescale boxes so
            # they overlay correctly on the latent grid.
            heat_h, heat_w = panel.heatmap.shape
            from PIL import Image as _PILImage
            with _PILImage.open(overlay_image_path) as img:
                img_w, img_h = img.size
            scale_x = heat_w / float(img_w)
            scale_y = heat_h / float(img_h)
            scaled = []
            for d in detections:
                x0, y0, x1, y1 = d["box"]
                scaled.append({
                    "box": (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y),
                    "confidence": d.get("confidence", 0.0),
                    "label": d.get("label", ""),
                })
            overlay_boxes(
                ax,
                scaled,
                palette=detection_palette or {},
                missing_threshold=0.35,
                linewidth=1.5,
                label_fontsize=7,
            )
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02,
                 label="‖Δ_t‖₂ (per-pixel, over 4 channels)")
    fig.suptitle(
        f"D2 — spatial localisation of Δ_t  |  {pair_slug} seed={seed}",
        fontsize=11,
    )
    return save_fig(fig, fig_path)


# ---------------------------------------------------------------------------
# D3
# ---------------------------------------------------------------------------


def render_d3(
    cross: CrossSeedCosine,
    fig_path: Path,
    *,
    pair_slug: str,
    threshold: float = 0.5,
    window: tuple[int, int] = (5, 25),
    is_cooperative: bool = False,
) -> Path:
    pair_labels = sorted(cross.per_pair_cosines.keys())
    mat = np.array([cross.per_pair_cosines[p] for p in pair_labels])  # (n_pairs, T)
    mean_window = cross.mean_over_window(*window)
    # Threshold only enforced on cooperative pair per C2.
    passed = (mean_window >= threshold) if is_cooperative else None

    fig, (ax_heat, ax_mean) = plt.subplots(
        2, 1, figsize=(8.0, 4.2), sharex=True,
        gridspec_kw={"height_ratios": [len(pair_labels) + 0.5, 1.4]},
    )
    im = ax_heat.imshow(
        mat, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0,
        extent=(cross.step_indices[0] - 0.5, cross.step_indices[-1] + 0.5,
                len(pair_labels) - 0.5, -0.5),
    )
    ax_heat.set_yticks(np.arange(len(pair_labels)))
    ax_heat.set_yticklabels(pair_labels, fontsize=8)
    ax_heat.set_ylabel("seed pair")
    fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.02, label="cos")

    ax_mean.plot(cross.step_indices, cross.mean_across_pairs,
                 color=PALETTE["delta"], lw=1.8, label="mean across pairs")
    ax_mean.axhline(threshold, color=PALETTE["threshold"], ls="--", lw=1.0,
                    label=f"pass ≥ {threshold:.2f}")
    ax_mean.axhline(0.0, color="black", lw=0.5)
    ax_mean.axvspan(window[0], window[1], color=PALETTE["good"], alpha=0.08)
    ax_mean.set_ylim(-1.05, 1.05)
    ax_mean.set_xlabel("denoising step t")
    ax_mean.set_ylabel("mean cos")
    ax_mean.legend(loc="lower left", fontsize=8, frameon=False)

    if is_cooperative:
        verdict = "PASS" if passed else "FAIL"
        color = PALETTE["good"] if passed else PALETTE["bad"]
        suffix = f"  |  ⟨mean cos⟩_{{{window[0]}-{window[1]}}} = {mean_window:.3f}  [{verdict}]"
    else:
        suffix = (
            f"  |  ⟨mean cos⟩_{{{window[0]}-{window[1]}}} = {mean_window:.3f}"
            "  (collision pair, no pass/fail per C1)"
        )
        color = "black"
    fig.suptitle(
        f"D3 — cross-seed cosine of Δ_t  |  {pair_slug}{suffix}",
        fontsize=10, color=color,
    )
    return save_fig(fig, fig_path)


# ---------------------------------------------------------------------------
# D4-B — direction vs magnitude (two-panel: norms + cos-to-LOO with null band)
# ---------------------------------------------------------------------------


def render_d4b(
    split: DirectionMagnitudeSplit,
    fig_path: Path,
    *,
    pair_slug: str,
    threshold: float = 0.5,
    window: tuple[int, int] = (5, 25),
) -> Path:
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.0, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.2]},
    )
    xs = split.step_indices

    # Top panel — per-seed ‖Δ_t‖.
    all_norms = []
    for i, seed in enumerate(split.seeds):
        norms = split.norms_per_seed[seed]
        all_norms.extend(norms)
        ax_top.plot(xs, norms, color=_seed_colour(seed, i), lw=1.4, alpha=0.85,
                    label=f"seed {seed}")
    # Median + IQR band over seeds (per step).
    arr = np.array([split.norms_per_seed[s] for s in split.seeds])
    if arr.shape[0] >= 2:
        median = np.median(arr, axis=0)
        q1 = np.percentile(arr, 25, axis=0)
        q3 = np.percentile(arr, 75, axis=0)
        ax_top.fill_between(xs, q1, q3, color="#888888", alpha=0.18,
                            label="IQR (across seeds)")
        ax_top.plot(xs, median, color="black", lw=1.0, ls=":",
                    label="median (across seeds)")
    ax_top.set_ylabel("‖Δ_t‖")
    ax_top.set_title(
        f"D4-B — direction vs magnitude  |  {pair_slug}",
        fontsize=10,
    )
    ax_top.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
    ax_top.axvspan(window[0], window[1], color=PALETTE["good"], alpha=0.05)

    # Bottom panel — per-seed cos vs LOO mean, plus null band.
    null_med = np.array(split.null_median)
    null_lo = np.array(split.null_lo)
    null_hi = np.array(split.null_hi)
    ax_bot.fill_between(xs, null_lo, null_hi, color="#bbbbbb", alpha=0.45,
                        label="permutation null (95%)")
    ax_bot.plot(xs, null_med, color="#666666", lw=0.8, ls=":")
    cand_means = split.mean_cos_in_window(*window)
    for i, seed in enumerate(split.seeds):
        cos = split.cos_vs_loo_mean_per_seed[seed]
        ax_bot.plot(xs, cos, color=_seed_colour(seed, i), lw=1.6,
                    label=f"seed {seed}  (⟨cos⟩={cand_means.get(seed, float('nan')):.2f})")
    ax_bot.axhline(threshold, color=PALETTE["threshold"], ls="--", lw=1.0,
                   label=f"pass ≥ {threshold:.2f}")
    ax_bot.axhline(0.0, color="black", lw=0.5)
    ax_bot.axvspan(window[0], window[1], color=PALETTE["good"], alpha=0.08)
    ax_bot.set_ylim(-1.05, 1.05)
    ax_bot.set_xlabel("denoising step t")
    ax_bot.set_ylabel("cos(Δ_t^(s), mean_{s' ≠ s} Δ_t^(s'))")

    # Pass criterion: across seeds, mean cos-vs-loo in window ≥ threshold.
    vals = [v for v in cand_means.values() if not np.isnan(v)]
    overall = float(np.mean(vals)) if vals else float("nan")
    passed = bool(vals) and overall >= threshold
    verdict = "PASS" if passed else "FAIL"
    color = PALETTE["good"] if passed else PALETTE["bad"]
    ax_bot.set_title(
        f"⟨cos vs LOO-mean⟩_{{{window[0]}-{window[1]}}} = {overall:.3f}  [{verdict}]",
        color=color, fontsize=9, loc="left",
    )
    ax_bot.legend(loc="lower left", fontsize=7, frameon=False, ncol=2)
    return save_fig(fig, fig_path)


# ---------------------------------------------------------------------------
# D4-C — cluster-ordered N×N cosine grid (one panel per timestep)
# ---------------------------------------------------------------------------


def render_d4c(
    panels: ClusterOrderedPanels,
    fig_path: Path,
    *,
    pair_slug: str,
) -> Path:
    n = len(panels.panels)
    if n == 0:
        # Render a blank-ish placeholder so callers always have a PNG.
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "no panels available", ha="center", va="center")
        ax.set_axis_off()
        return save_fig(fig, fig_path)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n + 1.0, 2.8), squeeze=False)
    n_seeds = len(panels.seeds_ordered)
    last_im = None
    for ax, mat, step_idx, ts in zip(
        axes[0], panels.panels, panels.step_indices, panels.timesteps,
    ):
        im = ax.imshow(mat.cpu().numpy(), cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        ax.set_title(f"t={step_idx}  (ts={ts})", fontsize=9)
        ax.set_xticks(np.arange(n_seeds))
        ax.set_xticklabels(panels.seeds_ordered, fontsize=7, rotation=0)
        ax.set_yticks(np.arange(n_seeds))
        ax.set_yticklabels(panels.seeds_ordered, fontsize=7)
        for i in range(n_seeds):
            for j in range(n_seeds):
                ax.text(j, i, f"{float(mat[i, j]):.2f}",
                        ha="center", va="center", fontsize=6,
                        color="black" if abs(float(mat[i, j])) < 0.6 else "white")
        last_im = im
    fig.colorbar(last_im, ax=axes[0].tolist(), fraction=0.025, pad=0.02, label="cos")
    fig.suptitle(
        f"D4-C — cluster-ordered cross-seed cosine  |  {pair_slug}  "
        f"(seeds reordered by clustering at panel 1)",
        fontsize=10,
    )
    return save_fig(fig, fig_path)


# ---------------------------------------------------------------------------
# D4-D — PCA with guards (3 curves over t)
# ---------------------------------------------------------------------------


def render_d4d(
    g: PcaWithGuards,
    fig_path: Path,
    *,
    pair_slug: str,
    window: tuple[int, int] = (5, 25),
) -> Path:
    fig, ax_left = plt.subplots(figsize=(7.6, 3.8))
    ax_right = ax_left.twinx()
    xs = g.step_indices

    # Left axis: top-1 variance share + null band.
    null_lo = np.array(g.null_top1_lo)
    null_hi = np.array(g.null_top1_hi)
    ax_left.fill_between(xs, null_lo, null_hi, color="#bbbbbb", alpha=0.4,
                         label="permutation null (95%)")
    ax_left.plot(xs, g.null_top1_median, color="#666666", lw=0.8, ls=":")
    ax_left.plot(xs, g.uncentred_top1_share, color=PALETTE["delta"], lw=1.8,
                 label="uncentred top-1 share")
    ax_left.axvspan(window[0], window[1], color=PALETTE["good"], alpha=0.06)
    ax_left.set_ylim(0.0, 1.05)
    ax_left.set_ylabel("top-1 variance share", color=PALETTE["delta"])
    ax_left.tick_params(axis="y", colors=PALETTE["delta"])
    ax_left.set_xlabel("denoising step t")

    # Right axis: angles (deg).
    ax_right.plot(xs, g.angle_pc1_vs_mean_deg, color=PALETTE["j_minus_null"],
                  lw=1.4, label="∠(PC1 uncentred, mean row)")
    ax_right.plot(xs, g.angle_centred_vs_uncentred_pc1_deg,
                  color=PALETTE["a_minus_b"], lw=1.4,
                  label="∠(PC1 centred, PC1 uncentred)")
    ax_right.set_ylabel("angle (deg)")
    ax_right.set_ylim(0.0, 90.0)

    lines_l, labels_l = ax_left.get_legend_handles_labels()
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_l + lines_r, labels_l + labels_r,
                   loc="upper right", fontsize=7, frameon=False)
    fig.suptitle(
        f"D4-D — PCA with guards  |  {pair_slug}  "
        f"(only trust top-1 share when it exits the null band; "
        f"only trust centred PCA when angle ≪ 90°)",
        fontsize=9,
    )
    return save_fig(fig, fig_path)


# ---------------------------------------------------------------------------
# §7c — PCA grid (one panel per timestep; seeds projected onto top-2 PCs)
# ---------------------------------------------------------------------------


def render_pca_grid(
    grid: PcaGrid,
    fig_path: Path,
    *,
    pair_slug: str,
) -> Path:
    n = len(grid.step_indices)
    if n == 0:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "no panels available", ha="center", va="center")
        ax.set_axis_off()
        return save_fig(fig, fig_path)
    # Standard 6-panel layout: 2 rows × 3 cols if n ≥ 6, else 1 row.
    if n >= 6:
        nrows, ncols = 2, 3
    elif n >= 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 1, n
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False,
    )
    flat_axes = [ax for row in axes for ax in row]
    for ax, step_idx in zip(flat_axes, grid.step_indices):
        coords = grid.coords_per_step[step_idx].cpu().numpy()
        share1, share2 = grid.variance_share_per_step[step_idx]
        for i, seed in enumerate(grid.seeds):
            ax.scatter(
                coords[i, 0], coords[i, 1],
                color=_seed_colour(seed, i), s=80, edgecolor="black", lw=0.6,
                label=f"seed {seed}",
            )
        ax.axhline(0.0, color="#bbbbbb", lw=0.5)
        ax.axvline(0.0, color="#bbbbbb", lw=0.5)
        ts = grid.timesteps[grid.step_indices.index(step_idx)]
        ax.set_title(
            f"t={step_idx}  (ts={ts})\n"
            f"PC1={share1*100:.0f}% · PC2={share2*100:.0f}%",
            fontsize=8,
        )
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.tick_params(labelsize=7)
    # Hide unused axes.
    for ax in flat_axes[len(grid.step_indices):]:
        ax.set_axis_off()
    flat_axes[0].legend(loc="upper right", fontsize=7, frameon=False)
    fig.suptitle(
        f"§7c — PCA grid (latent space)  |  {pair_slug}  "
        f"(seeds projected onto top-2 uncentred PCs at each timestep)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return save_fig(fig, fig_path)
