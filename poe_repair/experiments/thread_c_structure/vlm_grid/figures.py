"""§7c VLM-projection grid renderer.

Two figures:

* ``render_vlm_grid``: the main 6-panel grid. Each panel is one injected
  timestep; per seed, three points (α=0, α_partial, α=1) are connected by
  an arrow. 95% ellipses are drawn around each (seed, t, α) point when
  ``n_reruns >= 3``. Arrowheads carry the route tag — filled if the seed
  has a clean A or B route at α=1, hollow if ambiguous.
* ``render_vlm_calibration``: a side car showing the calibration α-sweep.
  Two curves (x-axis vs α, y-axis vs α); a flag is set in the verdict if
  either curve is non-monotone.

Both renderers read ``VlmGridResult`` directly.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from poe_repair.experiments.thread_c_structure.vlm_grid.runner import (
    VlmGridResult, VlmGridSample,
)
from poe_repair.figures._common import save_fig


SEED_PALETTE = {
    4:   "#E07A5F",
    7:   "#9D4EDD",
    42:  "#3D5A80",
    123: "#52B788",
}


def _seed_colour(seed: int, fallback_index: int = 0) -> str:
    if seed in SEED_PALETTE:
        return SEED_PALETTE[seed]
    pool = ["#274C77", "#E07A5F", "#3D5A80", "#52B788", "#9D4EDD", "#F4A261"]
    return pool[fallback_index % len(pool)]


def _group_samples(
    result: VlmGridResult,
) -> dict[tuple[int, int, float], list[VlmGridSample]]:
    """Group samples by (seed, step_index, alpha)."""
    out: dict[tuple[int, int, float], list[VlmGridSample]] = defaultdict(list)
    for s in result.samples:
        out[(s.seed, s.step_index, float(s.alpha))].append(s)
    return out


def _cov_ellipse(
    xs: list[float], ys: list[float],
    *, n_std: float = 1.96,    # ~95% for a 2D Gaussian
) -> tuple[float, float, float, float, float] | None:
    """Return (cx, cy, width, height, angle_deg) for the cov ellipse, or
    None if fewer than 3 points."""
    if len(xs) < 3:
        return None
    arr = np.array([xs, ys])
    cx, cy = float(arr[0].mean()), float(arr[1].mean())
    cov = np.cov(arr)
    if cov.shape != (2, 2):
        return None
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0.0, None)
    order = np.argsort(vals)[::-1]
    vals = vals[order]; vecs = vecs[:, order]
    width = float(2.0 * n_std * np.sqrt(vals[0]))
    height = float(2.0 * n_std * np.sqrt(vals[1]))
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    return cx, cy, width, height, angle


def render_vlm_grid(
    result: VlmGridResult,
    fig_path: Path,
) -> Path:
    panel_steps = sorted(set(result.panel_steps), reverse=True)
    n = len(panel_steps) or 1
    if n >= 6:
        nrows, ncols = 2, 3
    elif n >= 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 1, n
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows), squeeze=False,
    )
    flat_axes = [ax for row in axes for ax in row]

    grouped = _group_samples(result)
    alphas_sorted = sorted({float(a) for a in result.alphas})
    for ax, step_idx in zip(flat_axes, panel_steps):
        for i, seed in enumerate(result.seeds):
            colour = _seed_colour(seed, i)
            xs_path: list[float] = []
            ys_path: list[float] = []
            for alpha in alphas_sorted:
                samples = grouped.get((seed, step_idx, float(alpha)), [])
                if not samples:
                    continue
                xs = [s.cooccurrence_score for s in samples]
                ys = [s.separation_confidence for s in samples]
                xs_clean = [x for x in xs if not np.isnan(x)]
                ys_clean = [y for y in ys if not np.isnan(y)]
                if not xs_clean or not ys_clean:
                    continue
                mx, my = float(np.mean(xs_clean)), float(np.mean(ys_clean))
                xs_path.append(mx); ys_path.append(my)
                # Ellipse if enough reruns.
                ellipse_params = _cov_ellipse(xs_clean, ys_clean)
                if ellipse_params is not None:
                    cx, cy, w, h, angle = ellipse_params
                    e = Ellipse(
                        (cx, cy), w, h, angle=angle,
                        facecolor=colour, alpha=0.12, edgecolor=colour, linewidth=0.6,
                    )
                    ax.add_patch(e)
                # Marker per α: open at α=0, half at intermediate, filled at α=1.
                fill = "white" if alpha == 0.0 else colour
                ax.scatter(
                    mx, my, s=44, color=fill,
                    edgecolor=colour, linewidths=1.2, zorder=3,
                )
            # Arrow path baseline → partial → oracle.
            if len(xs_path) >= 2:
                ax.plot(
                    xs_path, ys_path,
                    color=colour, lw=1.2, alpha=0.9, zorder=2,
                )
                # Arrowhead on the final segment, route-tag aware.
                tip = grouped.get((seed, step_idx, alphas_sorted[-1]), [])
                route_tag = tip[0].route_tag if tip else "ambiguous"
                # "ambiguous" → hollow; otherwise filled.
                head_kwargs = dict(
                    color=colour, length_includes_head=True,
                    head_width=0.025, head_length=0.025,
                )
                if route_tag == "ambiguous":
                    head_kwargs["fill"] = False
                ax.arrow(
                    xs_path[-2], ys_path[-2],
                    xs_path[-1] - xs_path[-2], ys_path[-1] - ys_path[-2],
                    **head_kwargs,
                )
        ax.set_xlim(0.0, 1.05)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("co-occurrence (VQA P(yes))")
        ax.set_ylabel("separation conf. (P_A · P_B)")
        ax.axhline(0.5, color="#eeeeee", lw=0.5, zorder=0)
        ax.axvline(0.5, color="#eeeeee", lw=0.5, zorder=0)
        ax.set_title(f"injected step t={step_idx}", fontsize=9)

    for ax in flat_axes[len(panel_steps):]:
        ax.set_axis_off()

    # Seed legend on the first axis only.
    handles = []
    for i, seed in enumerate(result.seeds):
        handles.append(plt.Line2D(
            [], [], marker="o", linestyle="-",
            color=_seed_colour(seed, i), label=f"seed {seed}",
        ))
    flat_axes[0].legend(
        handles=handles, loc="upper left", fontsize=7, frameon=False,
    )

    n_pts = result.n_reruns
    fig.suptitle(
        f"§7c — VLM-projection grid  |  {result.pair_slug}  "
        f"|  α ∈ {result.alphas}  |  reruns/cell = {n_pts}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return save_fig(fig, fig_path)


def render_vlm_calibration(
    result: VlmGridResult,
    fig_path: Path,
) -> Path:
    calib = result.calibration
    alphas = list(calib.get("alphas", []))
    xs = list(calib.get("axis_x_values", []))
    ys = list(calib.get("axis_y_values", []))
    step_idx = calib.get("step_index")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    if not alphas:
        ax.text(0.5, 0.5, "no calibration data", ha="center", va="center")
        ax.set_axis_off()
        return save_fig(fig, fig_path)
    ax.plot(alphas, xs, "-o", color="#274C77", label="co-occurrence (x-axis)")
    ax.plot(alphas, ys, "-o", color="#E07A5F", label="separation (y-axis)")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("α")
    ax.set_ylabel("grader score")
    x_mono = calib.get("x_monotone")
    y_mono = calib.get("y_monotone")
    warn = calib.get("warning")
    flags = []
    if x_mono is False:
        flags.append("x non-monotone")
    if y_mono is False:
        flags.append("y non-monotone")
    if not flags:
        flags = ["both monotone"]
    if warn:
        flags.append(f"warning: {warn}")
    ax.set_title(
        f"§7c calibration  |  {result.pair_slug}  step={step_idx}  "
        f"|  {' · '.join(flags)}",
        fontsize=9,
    )
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    return save_fig(fig, fig_path)
