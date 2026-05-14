"""Stage 4 — Figures A, B, C, D.

  - **A — trace.** Per-step monitor reading + trigger firing markers.
  - **B — budget-vs-quality scatter.** Smart cloud vs constant cloud.
  - **C — constant-vs-smart image grid** at matched budgets.
  - **D — held-out seed comparison.**
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from poe_repair.figures._common import image_grid, save_fig


# ---------------------------------------------------------------------------
# Figure A — monitor trace
# ---------------------------------------------------------------------------


def figA_trace(
    *,
    fig_dir: Path,
    sched_stats_by_label: dict[str, dict],
    threshold_marker: float | None = None,
    title_suffix: str = "",
) -> Path:
    """For each run, plot the monitor reading per step + fired-step bars.

    ``sched_stats_by_label`` maps a short label → ``compute_schedule_stats``
    output dict.
    """
    n = len(sched_stats_by_label)
    if n == 0:
        return fig_dir / "figA_trace.png"
    fig, axes = plt.subplots(n, 1, figsize=(10.0, 3.0 * n), sharex=True, squeeze=False)
    for row, (label, stats) in enumerate(sched_stats_by_label.items()):
        proj = stats.get("basin_projection_per_step") or []
        fired = stats.get("fired_steps") or []
        ax = axes[row][0]
        if proj:
            steps = list(range(len(proj)))
            ax.plot(steps, proj, marker="o", color="tab:blue",
                    label="basin projection")
            if threshold_marker is not None:
                ax.axhline(
                    float(threshold_marker), color="gray", linestyle="--",
                    linewidth=0.8, label=f"θ = {threshold_marker:.2f}",
                )
            for t in fired:
                ax.axvline(int(t), color="tab:red", alpha=0.35, linewidth=1.2)
            ax.set_ylim(-0.1, 1.1)
        else:
            ax.text(0.5, 0.5, "(no projection trace saved)",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("projection")
        ax.set_title(f"{label}  |  fires = {len(fired)}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[-1][0].set_xlabel("denoising step  (0 = noisiest)")
    title = "Figure A — basin-axis projection per step (red bars = trigger fires)"
    if title_suffix:
        title += f"\n{title_suffix}"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_fig(fig, fig_dir / "figA_trace.png")


# ---------------------------------------------------------------------------
# Figure B — budget-vs-quality scatter
# ---------------------------------------------------------------------------


def figB_budget_quality(
    *,
    fig_dir: Path,
    table: dict,
    metric: str = "d_mono_l2",
    metric_label: str | None = None,
    title_suffix: str = "",
) -> Path:
    metric_label = metric_label or metric
    fig, ax = plt.subplots(figsize=(9.0, 6.0))

    constant_pts = [
        (r["total_injection"], r.get(metric, float("nan")))
        for r in table.get("constant", [])
    ]
    smart_pts = [
        (r["total_injection"], r.get(metric, float("nan")))
        for r in table.get("smart", [])
    ]

    if constant_pts:
        xs, ys = zip(*[(x, y) for x, y in constant_pts if not np.isnan(y)])
        ax.scatter(xs, ys, c="tab:gray", marker="s", s=70,
                   label="constant schedule")
    if smart_pts:
        xs, ys = zip(*[(x, y) for x, y in smart_pts if not np.isnan(y)])
        ax.scatter(xs, ys, c="tab:orange", marker="o", s=70,
                   label="smart (idea 2)")

    # Annotate smart points with their method-name suffix.
    for r in table.get("smart", []):
        x = r["total_injection"]; y = r.get(metric)
        if y is None or np.isnan(y):
            continue
        suffix = r["method"].split("_")[-1]   # last name token, e.g. thr030
        ax.annotate(suffix, (x, y), fontsize=7, alpha=0.7)

    ax.set_xlabel("total injected correction  Σ_t α(t) · ‖force_t‖")
    ax.set_ylabel(metric_label)
    title = f"Figure B — budget vs quality  ({metric_label})"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save_fig(fig, fig_dir / f"figB_budget_quality_{metric}.png")


# ---------------------------------------------------------------------------
# Figure C — constant vs smart at matched budgets
# ---------------------------------------------------------------------------


def figC_constant_vs_smart_grid(
    *,
    fig_dir: Path,
    rows: dict[str, list[Path]],   # {"constant": [...], "smart": [...]}
    col_labels: list[str],
    title_suffix: str = "",
) -> Path:
    cells = [rows.get("constant", []), rows.get("smart", [])]
    title = "Figure C — constant vs smart at matched total injection"
    if title_suffix:
        title += f"\n{title_suffix}"
    return image_grid(
        cells,
        fig_dir / "figC_constant_vs_smart_grid.png",
        col_labels=col_labels,
        row_labels=["constant schedule", "smart (idea 2)"],
        title=title,
        panel_size=2.6,
    )


# ---------------------------------------------------------------------------
# Figure D — held-out seed comparison
# ---------------------------------------------------------------------------


def figD_heldout(
    *,
    fig_dir: Path,
    cells_by_seed: dict[int, dict[str, Path]],
    title_suffix: str = "",
) -> Path:
    seeds = sorted(cells_by_seed.keys())
    columns = ["poe", "mono", "constant_match", "smart"]
    rows: list[list[Path]] = []
    for seed in seeds:
        rows.append([cells_by_seed[seed][c] for c in columns])
    title = "Figure D — held-out: PoE / Mono / constant-match / smart"
    if title_suffix:
        title += f"\n{title_suffix}"
    return image_grid(
        rows,
        fig_dir / "figD_heldout.png",
        col_labels=["PoE", "Mono", "constant @ matched budget", "smart (idea 2)"],
        row_labels=[f"seed {s}" for s in seeds],
        title=title,
        panel_size=2.6,
    )
