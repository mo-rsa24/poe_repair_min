"""LoRA figures: curve panel, thumbnail strip, cumulative grid, where-applied overlay.

All four are regenerated each probe; the cumulative grid + curve panel are
overwritten in place and the headline deliverables for the run.

Style: shares ``apply_veracity_style`` with the Veracity figure set so the
LoRA panels read as one family with the diagnostic figures.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from poe_repair.experiments.one_pair_one_seed.config import RunConfig
from poe_repair.experiments.residual_between_mono_and_poe.metrics import detect_boxes
from poe_repair.figures._common import overlay_boxes, save_fig
from poe_repair.figures._veracity_style import (
    apply_veracity_style,
    HEATMAP_EPS_CMAP,
    LAMBDA_CMAP,
)


log = logging.getLogger(__name__)


REGIME_COLORS = {
    "both_distinct": "#2C8F4A",       # green
    "both_overlapping": "#D49B2A",    # amber
    "single": "#B33A3A",              # red
    "none": "#888888",                # gray
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
    cmap = plt.get_cmap(LAMBDA_CMAP)
    if n <= 1:
        return cmap(1.0)
    return cmap(i / (n - 1))


# ---------------------------------------------------------------------------
# Curve panel: VQAScore min vs λ, one line per probe (time-coloured)
# ---------------------------------------------------------------------------


def render_curve_vqa_vs_lambda(
    probes_root: Path,
    output_path: Path,
    *,
    threshold: float = 0.5,
) -> Path:
    apply_veracity_style()
    summaries = _read_probe_summaries(probes_root)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    if not summaries:
        ax.text(0.5, 0.5, "(no probes yet)", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("λ")
        ax.set_ylabel("VQAScore min")
        return save_fig(fig, output_path)

    n = len(summaries)
    for i, s in enumerate(summaries):
        rows = sorted(s["results"], key=lambda r: r["lambda"])
        xs = [r["lambda"] for r in rows]
        ys = [r["vqa_min"] for r in rows]
        ax.plot(
            xs, ys,
            color=_epoch_color(i, n),
            marker="o", markersize=4, linewidth=1.6,
            label=f"epoch {int(s['epoch'])}",
        )

    ax.axhline(threshold, linestyle="--", color="gray", linewidth=0.9)
    ax.text(
        max(xs) - 0.02 if xs else 1.0, threshold + 0.02,
        f"§4 pass = {threshold:.2f}",
        ha="right", va="bottom", fontsize=8, color="gray",
    )
    ax.set_xlabel("λ (residual gain)")
    ax.set_ylabel("VQAScore min")
    ax.set_ylim(0.0, 1.02)

    # Time-colour colourbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.get_cmap(LAMBDA_CMAP),
        norm=plt.Normalize(vmin=0, vmax=max(1, n - 1)),
    )
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label("probe index", fontsize=8)

    ax.set_title("VQAScore min vs λ across probes")
    return save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Thumbnail strip for this epoch
# ---------------------------------------------------------------------------


def render_thumbnail_strip(
    probes_root: Path,
    epoch: int,
    output_path: Path,
) -> Path:
    apply_veracity_style()
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
            f"λ={lam:.2f}\nvqa={r['vqa_min']:.2f}\n{r['regime']}",
            fontsize=8,
        )
        # Regime-coloured border
        border = REGIME_COLORS.get(r["regime"], "#888888")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border)
            spine.set_linewidth(2.2)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"LoRA — epoch {epoch:04d} probe", fontsize=10)
    return save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Cumulative grid: rows = epoch, cols = λ. Headline.
# ---------------------------------------------------------------------------


def render_cumulative_grid(
    probes_root: Path,
    output_path: Path,
    *,
    panel_size: float = 1.7,
) -> Path:
    apply_veracity_style()
    summaries = _read_probe_summaries(probes_root)
    if not summaries:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "(no probes yet)", ha="center", va="center", transform=ax.transAxes)
        return save_fig(fig, output_path)

    # Determine λ columns from the first summary; assume consistent across probes.
    cols = sorted({float(r["lambda"]) for r in summaries[0]["results"]})
    rows = summaries  # one per probe

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
            # Title only on top row.
            if ri == 0:
                ax.set_title(f"λ={lam:.2f}", fontsize=9)
            # Row label only on left col.
            if ci == 0:
                ax.set_ylabel(f"epoch {epoch:04d}", fontsize=8, rotation=0,
                              labelpad=24, ha="right", va="center")
            # Per-cell text overlay
            if r is not None:
                ax.text(
                    0.04, 0.96,
                    f"vqa={r['vqa_min']:.2f}\n{r['regime']}",
                    transform=ax.transAxes,
                    fontsize=6.5, va="top", ha="left",
                    color="white",
                    bbox=dict(
                        facecolor=REGIME_COLORS.get(r["regime"], "#444"),
                        edgecolor="none", alpha=0.85,
                        boxstyle="round,pad=0.15",
                    ),
                )
                # Regime-coloured frame.
                border = REGIME_COLORS.get(r["regime"], "#888888")
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor(border)
                    spine.set_linewidth(2.0)

    fig.suptitle(
        "LoRA cumulative grid — rows = probe, cols = λ\n"
        "frame colour = §4 detection regime",
        fontsize=10,
    )

    # Legend strip.
    handles = [
        mpatches.Patch(color=v, label=k) for k, v in REGIME_COLORS.items()
    ]
    fig.legend(
        handles=handles, loc="lower center",
        ncol=len(REGIME_COLORS), bbox_to_anchor=(0.5, -0.01),
        fontsize=8, frameon=False,
    )

    return save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Where-applied overlay
# ---------------------------------------------------------------------------


def _spatial_l2_heatmap(delta_hat: torch.Tensor) -> torch.Tensor:
    """``||Δ̂||_2`` across channels → (H, W) heatmap, normalised to [0, 1]."""
    t = delta_hat.detach().float()
    if t.ndim == 4:
        t = t[0]
    norm = t.pow(2).sum(dim=0).sqrt()           # (H, W)
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
    """Overlay the spatial L2 norm of Δ̂ on the decoded Tweedie x̂_0 at the
    chosen reference steps in the basin-commit window.
    """
    apply_veracity_style()
    overlays_dir = probes_root / f"epoch_{epoch:04d}" / f"lambda_{lam:.2f}" / "delta_overlays"
    decoded_path = probes_root / f"epoch_{epoch:04d}" / f"lambda_{lam:.2f}" / "decoded.png"

    refs = list(cfg.probe.where_applied_steps)
    fig, axes = plt.subplots(1, len(refs), figsize=(3.6 * len(refs), 3.8), squeeze=False)
    palette = {
        cfg.cell.prompt_a.strip().lower(): "#2C6FBB",
        cfg.cell.prompt_b.strip().lower(): "#B36464",
    }

    # Detection boxes from the final decoded image (proxy for box localisation
    # at the commit moment — boxes are time-stable enough for an overlay).
    dets: list[dict] = []
    if decoded_path.exists():
        try:
            dets = detect_boxes(
                decoded_path,
                [cfg.cell.prompt_a, cfg.cell.prompt_b],
                box_threshold=0.35,
                text_threshold=0.25,
            )
        except Exception as exc:  # pragma: no cover
            log.warning("detect_boxes failed for where-applied: %s", exc)

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

        # Decoded final image is a *proxy* for the commit-step x̂_0; we lack
        # a decoded x̂_0 at intermediate steps without an extra VAE call.
        # The figure-caption discipline: state explicitly in the title.
        if decoded_path.exists():
            ax.imshow(Image.open(decoded_path), alpha=1.0)
        ax.imshow(heatmap_up, cmap=HEATMAP_EPS_CMAP, alpha=0.55,
                  extent=[0, W, H, 0])
        overlay_boxes(ax, dets, palette, missing_threshold=0.35, linewidth=1.6,
                      label_fontsize=7)
        ax.set_title(
            f"step {step_index} (t={int(payload.get('timestep', -1))})\n"
            f"‖Δ̂‖ over decoded x_0  (λ={lam:.2f})",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"LoRA where-applied — epoch {epoch:04d}, λ={lam:.2f}\n"
        f"basin-commit reference steps: {tuple(refs)}",
        fontsize=10,
    )
    return save_fig(fig, output_path)
