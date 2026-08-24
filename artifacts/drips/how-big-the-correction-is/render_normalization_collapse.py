#!/usr/bin/env python
"""Render the toy normalization-collapse figure for the F3 worked example."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/poe-repair-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "figures" / "normalization-collapse.png"

BG = "#080b12"
FG = "#eee9dc"
MUTED = "#a7acb8"
AXIS = "#3d4556"
BLUE = "#61d8ff"
GOLD = "#f5b14c"


def smooth_arc(x_points: np.ndarray, y_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quadratic through the three toy points."""
    xs = np.linspace(x_points.min(), x_points.max(), 240)
    coeff = np.polyfit(x_points, y_points, deg=2)
    return xs, np.polyval(coeff, xs)


def setup_axis(ax: plt.Axes, y_lim: tuple[float, float]) -> None:
    ax.set_xlim(-0.15, 2.15)
    ax.set_ylim(*y_lim)
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axvline(0, color=AXIS, lw=1.0, alpha=0.75, zorder=0)
    ax.axhline(y_lim[0] + 0.06 * (y_lim[1] - y_lim[0]), color=AXIS, lw=1.0, alpha=0.55, zorder=0)


def draw_curve(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    *,
    lw: float = 4.4,
    z: int = 3,
    marker_size: float = 58,
) -> None:
    xs, ys = smooth_arc(x, y)
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", zorder=z)
    ax.scatter(x, y, s=marker_size, color=BG, edgecolor=color, lw=2.4, zorder=z + 1)


def label_steps(ax: plt.Axes, *, y: float) -> None:
    labels = ["near noise", "mid", "near image"]
    for xi, label in zip([0, 1, 2], labels):
        ax.text(
            xi,
            y,
            label,
            ha="center",
            va="top",
            color=FG,
            fontsize=16,
            fontfamily="DejaVu Serif",
        )


def main() -> int:
    x = np.array([0.0, 1.0, 2.0])
    pair_1_raw = np.array([0.6, 1.0, 0.8])
    pair_2_raw = np.array([1.2, 2.0, 1.6])
    collapsed = np.array([0.75, 1.25, 1.0])

    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "dejavuserif",
        "text.color": FG,
        "axes.edgecolor": AXIS,
        "savefig.facecolor": BG,
    })

    fig = plt.figure(figsize=(9, 7.2), facecolor=BG)
    top = fig.add_axes((0.105, 0.565, 0.82, 0.335))
    bottom = fig.add_axes((0.105, 0.125, 0.82, 0.255))

    setup_axis(top, (0.34, 2.22))
    draw_curve(top, x, pair_2_raw, BLUE, lw=4.8, z=2)
    draw_curve(top, x, pair_1_raw, GOLD, lw=4.8, z=4)
    label_steps(top, y=0.305)

    setup_axis(bottom, (0.52, 1.42))
    # Same coordinates, different stroke widths: the blue underlay remains
    # visible around the gold curve without moving either curve off the shared path.
    draw_curve(bottom, x, collapsed, BLUE, lw=7.2, z=2, marker_size=72)
    draw_curve(bottom, x, collapsed, GOLD, lw=3.6, z=4, marker_size=34)
    label_steps(bottom, y=0.49)

    arrow = plt.annotate(
        "",
        xy=(0.5, 0.405),
        xytext=(0.5, 0.525),
        xycoords="figure fraction",
        textcoords="figure fraction",
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 1.8,
            "color": MUTED,
            "mutation_scale": 18,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    arrow.set_clip_on(False)
    fig.text(
        0.5,
        0.465,
        r"$\div$  divide by own median",
        ha="center",
        va="center",
        fontsize=17,
        fontfamily="DejaVu Serif",
        color=FG,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
