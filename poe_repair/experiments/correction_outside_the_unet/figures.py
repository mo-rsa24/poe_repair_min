"""Group A figures — thumbnail strip, cumulative grid, r̂_t-norm curve,
where-applied overlay.

Mirrors lora.figures with two differences:
 - the curve plot is ``‖r̂_t‖_sum vs epoch`` (qualitative MVP — no VQA gate).
 - the cumulative grid frame colour falls back to neutral if scoring was
   skipped (regime == "skipped").

All figures are regenerated each probe; the cumulative grid is the
headline deliverable, showing both training progress (rows) and λ-sweep
(columns) in one image.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn.functional as F
from PIL import Image

from poe_repair.experiments.correction_outside_the_unet.config import RunConfig
from poe_repair.figures._common import save_fig


log = logging.getLogger(__name__)


REGIME_COLORS = {
    "both_distinct": "#2C8F4A",
    "both_overlapping": "#D49B2A",
    "single": "#B33A3A",
    "none": "#888888",
    "skipped": "#444444",
    "error": "#888888",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_probe_summaries(probes_root: Path) -> list[dict]:
    out: list[dict] = []
    for epoch_dir in sorted(probes_root.glob("epoch_*")):
        path = epoch_dir / "summary.json"
        if path.exists():
            out.append(json.loads(path.read_text()))
    return out


def _epoch_color(i: int, n: int):
    cmap = plt.get_cmap("viridis")
    if n <= 1:
        return cmap(1.0)
    return cmap(i / (n - 1))


# ---------------------------------------------------------------------------
# Curve panel: ‖r̂_t‖_sum vs λ across probes
# ---------------------------------------------------------------------------


def render_curve_r_hat_norm(
    probes_root: Path,
    output_path: Path,
) -> Path:
    summaries = _read_probe_summaries(probes_root)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    if not summaries:
        ax.text(0.5, 0.5, "(no probes yet)", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("λ")
        ax.set_ylabel("‖r̂_t‖_sum")
        return save_fig(fig, output_path)

    n = len(summaries)
    for i, s in enumerate(summaries):
        rows = sorted(s["results"], key=lambda r: r["lambda"])
        xs = [r["lambda"] for r in rows]
        ys = [r.get("r_hat_norm_sum", 0.0) for r in rows]
        ax.plot(
            xs, ys,
            color=_epoch_color(i, n),
            marker="o", markersize=4, linewidth=1.6,
            label=f"epoch {int(s['epoch'])}",
        )

    ax.set_xlabel("λ (correction gain)")
    ax.set_ylabel("Σ_t ‖r̂_t‖ across 50 steps")
    sm = plt.cm.ScalarMappable(
        cmap=plt.get_cmap("viridis"),
        norm=plt.Normalize(vmin=0, vmax=max(1, n - 1)),
    )
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label("probe index", fontsize=8)
    ax.set_title("Σ ‖r̂_t‖ vs λ across probes")
    return save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Thumbnail strip — one row per epoch's λ-sweep
# ---------------------------------------------------------------------------


def render_thumbnail_strip(
    probes_root: Path,
    epoch: int,
    output_path: Path,
) -> Path:
    epoch_dir = probes_root / f"epoch_{epoch:04d}"
    summary_path = epoch_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text())
    rows = sorted(summary["results"], key=lambda r: r["lambda"])

    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.8), squeeze=False)
    for i, r in enumerate(rows):
        lam = float(r["lambda"])
        img_path = epoch_dir / f"lambda_{lam:.2f}" / "decoded.png"
        ax = axes[0][i]
        if img_path.exists():
            ax.imshow(Image.open(img_path))
        ax.axis("off")
        ax.set_title(
            f"λ={lam:.2f}\n‖r̂‖={r.get('r_hat_norm_sum', 0.0):.1f}\n{r.get('regime','')}",
            fontsize=8,
        )
        border = REGIME_COLORS.get(r.get("regime", "skipped"), "#888888")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border)
            spine.set_linewidth(2.2)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Group A — epoch {epoch:04d} probe", fontsize=10)
    return save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Cumulative grid: rows = probe epoch, cols = λ. Headline deliverable.
# ---------------------------------------------------------------------------


def render_cumulative_grid(
    probes_root: Path,
    output_path: Path,
    *,
    panel_size: float = 1.7,
    technique_label: str = "Group A",
) -> Path:
    summaries = _read_probe_summaries(probes_root)
    if not summaries:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "(no probes yet)", ha="center", va="center", transform=ax.transAxes)
        return save_fig(fig, output_path)

    cols = sorted({float(r["lambda"]) for r in summaries[0]["results"]})
    rows = summaries
    n_rows = len(rows)
    n_cols = len(cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_size * n_cols + 0.6, panel_size * n_rows + 0.5),
        squeeze=False,
    )

    for ri, s in enumerate(rows):
        epoch = int(s["epoch"])
        epoch_dir = probes_root / f"epoch_{epoch:04d}"
        lookup = {float(r["lambda"]): r for r in s["results"]}
        for ci, lam in enumerate(cols):
            ax = axes[ri][ci]
            r = lookup.get(float(lam))
            img_path = epoch_dir / f"lambda_{lam:.2f}" / "decoded.png"
            if img_path.exists():
                ax.imshow(Image.open(img_path))
            ax.set_xticks([])
            ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"λ={lam:.2f}", fontsize=9)
            if ci == 0:
                ax.set_ylabel(
                    f"epoch {epoch:04d}", fontsize=8, rotation=0,
                    labelpad=24, ha="right", va="center",
                )
            if r is not None:
                ax.text(
                    0.04, 0.96,
                    f"‖r̂‖={r.get('r_hat_norm_sum', 0.0):.1f}",
                    transform=ax.transAxes,
                    fontsize=6.5, va="top", ha="left",
                    color="white",
                    bbox=dict(
                        facecolor=REGIME_COLORS.get(r.get("regime", "skipped"), "#444"),
                        edgecolor="none", alpha=0.85,
                        boxstyle="round,pad=0.15",
                    ),
                )
                border = REGIME_COLORS.get(r.get("regime", "skipped"), "#888888")
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor(border)
                    spine.set_linewidth(2.0)

    fig.suptitle(
        f"{technique_label} cumulative grid — rows = probe epoch, cols = λ",
        fontsize=10,
    )
    handles = [mpatches.Patch(color=v, label=k) for k, v in REGIME_COLORS.items()]
    fig.legend(
        handles=handles, loc="lower center",
        ncol=min(4, len(REGIME_COLORS)), bbox_to_anchor=(0.5, -0.01),
        fontsize=8, frameon=False,
    )
    return save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Where-applied overlay
# ---------------------------------------------------------------------------


def _spatial_l2_heatmap(delta_hat: torch.Tensor) -> torch.Tensor:
    t = delta_hat.detach().float()
    if t.ndim == 4:
        t = t[0]
    norm = t.pow(2).sum(dim=0).sqrt()
    mx = float(norm.max().item())
    if mx <= 1e-12:
        return torch.zeros_like(norm)
    return norm / mx


def render_where_applied(
    *,
    cfg: RunConfig,
    probes_root: Path,
    epoch: int,
    lam: float,
    output_path: Path,
    image_size: tuple[int, int] = (1024, 1024),
) -> Path:
    overlays_dir = probes_root / f"epoch_{epoch:04d}" / f"lambda_{lam:.2f}" / "delta_overlays"
    decoded_path = probes_root / f"epoch_{epoch:04d}" / f"lambda_{lam:.2f}" / "decoded.png"

    refs = list(cfg.probe.where_applied_steps)
    fig, axes = plt.subplots(1, len(refs), figsize=(3.6 * len(refs), 3.8), squeeze=False)
    H, W = image_size

    for i, step_index in enumerate(refs):
        ax = axes[0][i]
        payload_path = overlays_dir / f"step_{step_index:02d}.pt"
        if not payload_path.exists():
            ax.text(0.5, 0.5, f"(no step {step_index})", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")
            continue
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        delta_hat = payload["delta_hat"]
        heatmap = _spatial_l2_heatmap(delta_hat)
        heatmap_up = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode="bilinear", align_corners=False,
        ).squeeze().numpy()
        if decoded_path.exists():
            ax.imshow(Image.open(decoded_path), alpha=1.0)
        ax.imshow(heatmap_up, cmap="magma", alpha=0.55, extent=[0, W, H, 0])
        ax.set_title(
            f"step {step_index} (t={int(payload.get('timestep', -1))})\n"
            f"‖r̂‖ over decoded (λ={lam:.2f})",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Group A where-applied — epoch {epoch:04d}, λ={lam:.2f}\n"
        f"reference steps: {tuple(refs)}",
        fontsize=10,
    )
    return save_fig(fig, output_path)
