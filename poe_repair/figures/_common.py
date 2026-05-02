"""Shared helpers for figure scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from PIL import Image


def save_fig(fig: plt.Figure, path: Path, *, dpi: int = 200) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def image_grid(
    cells: list[list[Path]],
    output_path: Path,
    *,
    col_labels: list[str] | None = None,
    row_labels: list[str] | None = None,
    title: str | None = None,
    panel_size: float = 3.5,
) -> Path:
    """Render `cells[row][col] -> Path` as a labelled grid and save."""
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
            ax.imshow(Image.open(cells[r][c]))
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
) -> Path:
    """Render `{label: (x, y)}` as a single line plot."""
    fig, ax = plt.subplots(figsize=figsize)
    for label, (x, y) in series.items():
        ax.plot(x, y, label=label)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if invert_x:
        ax.invert_xaxis()
    if series:
        ax.legend()
    return save_fig(fig, output_path)
