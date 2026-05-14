"""Shared helpers for figure scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import matplotlib.pyplot as plt
import torch
from PIL import Image

ImageCell = Union[Path, torch.Tensor]


def _to_imshow_array(item: ImageCell):
    """Accept Path | float-tensor and return something matplotlib's imshow can plot."""
    if isinstance(item, Path):
        return Image.open(item)
    if isinstance(item, torch.Tensor):
        arr = item.detach().float().clamp(0.0, 1.0)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.permute(1, 2, 0)
        return arr.cpu().numpy()
    raise TypeError(f"image cell must be Path or torch.Tensor, got {type(item).__name__}")


def save_fig(fig: plt.Figure, path: Path, *, dpi: int = 200) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def overlay_boxes(
    ax,
    detections: list[dict],
    palette: dict[str, str],
    *,
    missing_threshold: float = 0.35,
    default_color: str = "#444444",
    linewidth: float = 2.0,
    label_fontsize: int = 7,
) -> None:
    """Draw bounding boxes from ``detections`` on top of an already-imshow'd
    axis. Each detection is ``{"box": (x0, y0, x1, y1), "confidence": float,
    "label": str}``. The label is looked up case-insensitively in ``palette``;
    unknown labels use ``default_color``.

    If ``detections`` is empty, an explicit "no detection (≥{threshold})"
    annotation is drawn in the top-left corner so the absence isn't
    silent. The honesty constraint from the plan: "If no detection above
    threshold: explicit annotation; do not silently omit."
    """
    from matplotlib.patches import Rectangle

    if not detections:
        ax.text(
            0.02, 0.98,
            f"no detection (≥ {missing_threshold:.2f})",
            ha="left", va="top", transform=ax.transAxes,
            fontsize=label_fontsize, color="#B36464",
            bbox=dict(facecolor="white", edgecolor="#B36464",
                      boxstyle="round,pad=0.2", linewidth=0.8),
        )
        return

    for d in detections:
        x0, y0, x1, y1 = d["box"]
        label = str(d.get("label", "")).strip().lower()
        conf = float(d.get("confidence", 0.0))
        color = palette.get(label, default_color)
        rect = Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, edgecolor=color, linewidth=linewidth,
        )
        ax.add_patch(rect)
        ax.text(
            x0, max(0.0, y0 - 2),
            f"{label} {conf:.2f}",
            color="white", fontsize=label_fontsize,
            bbox=dict(facecolor=color, edgecolor="none",
                      boxstyle="round,pad=0.15", alpha=0.85),
            va="bottom", ha="left",
        )


def image_grid(
    cells: list[list[ImageCell]],
    output_path: Path,
    *,
    col_labels: list[str] | None = None,
    row_labels: list[str] | None = None,
    title: str | None = None,
    panel_size: float = 3.5,
) -> Path:
    """Render `cells[row][col] -> Path | Tensor` as a labelled grid and save.

    Each cell may be a Path (loaded via PIL) or a torch.Tensor in [0, 1] of
    shape [3, H, W] / [H, W, 3] / [1, 3, H, W].
    """
    n_rows = len(cells)
    n_cols = len(cells[0]) if n_rows else 0
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_size * n_cols, panel_size * n_rows),
        squeeze=False,
    )
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.imshow(_to_imshow_array(cells[r][c]))
            ax.axis("off")
            if r == 0 and col_labels and c < len(col_labels):
                ax.set_title(col_labels[c], fontsize=11)
        if row_labels and r < len(row_labels):
            axes[r][0].text(
                -0.05,
                0.5,
                row_labels[r],
                fontsize=9,
                ha="right",
                va="center",
                transform=axes[r][0].transAxes,
            )
    if title:
        fig.suptitle(title, fontsize=12)
    return save_fig(fig, output_path)


def line_plot(
    series: dict[str, tuple[Sequence[float], Sequence[float]]],
    output_path: Path,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (6.0, 4.0),
    invert_x: bool = False,
    log_y: bool = False,
    log_x: bool = False,
    line_styles: dict[str, dict] | None = None,
) -> Path:
    """Render `{label: (x, y)}` as a single line plot.

    `line_styles` maps a label to kwargs forwarded to `ax.plot` (e.g.
    `{"chance": {"linestyle": "--", "color": "gray"}}`).
    """
    fig, ax = plt.subplots(figsize=figsize)
    style_overrides = line_styles or {}
    for label, (x, y) in series.items():
        ax.plot(x, y, label=label, **style_overrides.get(label, {}))
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if invert_x:
        ax.invert_xaxis()
    if log_y:
        ax.set_yscale("log")
    if log_x:
        ax.set_xscale("log")
    if series:
        ax.legend()
    return save_fig(fig, output_path)


def bar_plot(
    categories: Sequence[str],
    values_by_series: dict[str, Sequence[float]],
    output_path: Path,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8.0, 4.0),
    log_y: bool = False,
    rotation: int = 30,
) -> Path:
    """Grouped bar chart. `values_by_series[label] = list per category`."""
    fig, ax = plt.subplots(figsize=figsize)
    n_cats = len(categories)
    n_series = len(values_by_series)
    if n_series == 0 or n_cats == 0:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes)
        return save_fig(fig, output_path)
    width = 0.8 / max(1, n_series)
    import numpy as np

    base_x = np.arange(n_cats)
    for i, (label, vals) in enumerate(values_by_series.items()):
        offsets = base_x + (i - (n_series - 1) / 2) * width
        ax.bar(offsets, list(vals), width=width, label=label)
    ax.set_xticks(base_x.tolist())
    ax.set_xticklabels(list(categories), rotation=rotation, ha="right", fontsize=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    if values_by_series:
        ax.legend()
    return save_fig(fig, output_path)


def histogram_plot(
    series: dict[str, Sequence[float]],
    output_path: Path,
    *,
    bins: int = 20,
    xlabel: str | None = None,
    ylabel: str | None = "count",
    title: str | None = None,
    figsize: tuple[float, float] = (6.0, 4.0),
    vlines: dict[str, float] | None = None,
) -> Path:
    """Overlay histograms of `{label: values}` series. Optional vertical
    reference lines via `vlines = {"label": x_value}`.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for label, vals in series.items():
        ax.hist(list(vals), bins=bins, alpha=0.5, label=label)
    if vlines:
        for label, x_val in vlines.items():
            ax.axvline(x_val, linestyle="--", linewidth=1.2, label=label)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if series or vlines:
        ax.legend()
    return save_fig(fig, output_path)


def compose_decoded_strip_with_curves(
    strip_rows: list[list[ImageCell]],
    curves_panels: list[dict],
    output_path: Path,
    *,
    strip_col_labels: list[str] | None = None,
    strip_row_labels: list[str] | None = None,
    title: str | None = None,
    strip_panel_size: float = 2.4,
    curves_panel_height: float = 2.6,
    curves_panel_width: float = 9.0,
    invert_curves_x: bool = False,
) -> Path:
    """Combined figure: top half = decoded image strip, bottom half = stacked curves.

    `strip_rows` is the same list-of-lists that `image_grid` accepts.
    `curves_panels` is the same list of panel dicts that `stacked_line_plot`
    accepts. The two are stacked vertically into a single figure for paper-grade
    presentation.
    """
    n_rows = len(strip_rows)
    n_cols = len(strip_rows[0]) if n_rows else 0
    n_curves = len(curves_panels)

    strip_h = max(1, n_rows) * strip_panel_size
    curves_h = max(1, n_curves) * curves_panel_height
    total_h = strip_h + curves_h
    width = max(curves_panel_width, n_cols * strip_panel_size)
    height_ratios = [strip_h, curves_h]

    fig = plt.figure(figsize=(width, total_h))
    outer = fig.add_gridspec(2, 1, height_ratios=height_ratios, hspace=0.3)

    # Top: image strip.
    if n_rows > 0:
        strip_grid = outer[0].subgridspec(n_rows, max(1, n_cols))
        for r in range(n_rows):
            for c in range(n_cols):
                ax = fig.add_subplot(strip_grid[r, c])
                ax.imshow(_to_imshow_array(strip_rows[r][c]))
                ax.axis("off")
                if r == 0 and strip_col_labels and c < len(strip_col_labels):
                    ax.set_title(strip_col_labels[c], fontsize=10)
            if strip_row_labels and r < len(strip_row_labels):
                first_ax = fig.add_subplot(strip_grid[r, 0])
                first_ax.axis("off")  # already added above; this is a placeholder
                # better: re-fetch the first subplot
                # but that requires bookkeeping; instead rely on a y-axis text
        if strip_row_labels:
            # Add row labels to the leftmost subplot of each row.
            for r in range(min(n_rows, len(strip_row_labels))):
                ax = fig.add_subplot(strip_grid[r, 0])
                ax.text(
                    -0.08, 0.5, strip_row_labels[r],
                    fontsize=9, ha="right", va="center",
                    transform=ax.transAxes,
                )
                ax.axis("off")

    # Bottom: stacked curve panels.
    if n_curves > 0:
        curves_grid = outer[1].subgridspec(n_curves, 1, hspace=0.45)
        for i, panel in enumerate(curves_panels):
            ax = fig.add_subplot(curves_grid[i, 0])
            for label, (x, y) in panel.get("series", {}).items():
                ax.plot(x, y, label=label)
            if "axhline" in panel:
                ax.axhline(panel["axhline"], color="gray", linestyle="--", linewidth=0.7)
            if panel.get("ylabel"):
                ax.set_ylabel(panel["ylabel"])
            if panel.get("title"):
                ax.set_title(panel["title"], fontsize=10)
            if panel.get("series"):
                ax.legend(fontsize=8)
            if invert_curves_x:
                ax.invert_xaxis()
            if i == n_curves - 1 and panel.get("xlabel"):
                ax.set_xlabel(panel["xlabel"])

    if title:
        fig.suptitle(title, fontsize=12)
    return save_fig(fig, output_path)


def stacked_line_plot(
    panels: list[dict],
    output_path: Path,
    *,
    xlabel: str | None = None,
    title: str | None = None,
    panel_height: float = 3.0,
    panel_width: float = 7.0,
    invert_x: bool = False,
) -> Path:
    """Render N stacked line plots sharing an x axis.

    Each panel is a dict with keys:
        "series": dict[label, (x, y)]
        "ylabel": optional str
        "title":  optional str
        "axhline": optional float (e.g. 0.0 for cos θ baseline)
    """
    n = len(panels)
    fig, axes = plt.subplots(
        n, 1, figsize=(panel_width, panel_height * n), sharex=True, squeeze=False
    )
    axes_flat = [axes[i][0] for i in range(n)]
    for ax, panel in zip(axes_flat, panels):
        for label, (x, y) in panel.get("series", {}).items():
            ax.plot(x, y, label=label)
        if "axhline" in panel:
            ax.axhline(panel["axhline"], color="gray", linestyle="--", linewidth=0.7)
        if panel.get("ylabel"):
            ax.set_ylabel(panel["ylabel"])
        if panel.get("title"):
            ax.set_title(panel["title"])
        if panel.get("series"):
            ax.legend()
    if xlabel:
        axes_flat[-1].set_xlabel(xlabel)
    if invert_x:
        for ax in axes_flat:
            ax.invert_xaxis()
    if title:
        fig.suptitle(title)
    return save_fig(fig, output_path)
