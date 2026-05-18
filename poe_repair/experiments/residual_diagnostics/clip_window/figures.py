"""Stage 2 — render filmstrip + CLIP-score trajectory figures.

Three figures:

  - **fig01_filmstrip** — 3 rows × 11 columns image grid. Rows are PoE,
    λ=0.6, Mono. Columns are sparse step snapshots (Tweedie x̂_0 decoded
    via VAE). Plus an extra column showing the actual final x_0.
  - **fig02_clip_score_trajectories** — 3 stacked panels, one per λ. Each
    panel plots three lines: CLIP cosine vs each text target across
    snapshot steps. Tells you when each text target becomes a winner.
  - **fig03_cross_trajectory_comparison** — single plot. The "a cat and a
    dog" line for all three trajectories overlaid. Headline figure: when
    does CLIP first separate Mono from PoE on the joint target?
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from poe_repair.experiments.residual_diagnostics.clip_window.analyse import TrajectorySnapshots
from poe_repair.figures._common import image_grid, save_fig


# ---------------------------------------------------------------------------
# Figure 1 — filmstrip
# ---------------------------------------------------------------------------


def fig01_filmstrip(
    *,
    fig_dir: Path,
    snapshots_by_label: dict[str, TrajectorySnapshots],
    title_suffix: str = "",
    include_final: bool = True,
) -> Path:
    labels = list(snapshots_by_label.keys())
    if not labels:
        raise ValueError("snapshots_by_label is empty")
    first = snapshots_by_label[labels[0]]
    step_indices = list(first.step_indices)

    rows: list[list] = []
    for label in labels:
        snap = snapshots_by_label[label]
        row = list(snap.decoded_paths)
        if include_final and snap.final_image_path is not None:
            row.append(snap.final_image_path)
        rows.append(row)

    col_labels = [f"step {i:d}" for i in step_indices]
    if include_final:
        col_labels.append("final x_0")
    row_labels = [snapshots_by_label[lab].label for lab in labels]

    title = (
        "Figure 1 — Tweedie x̂_0 filmstrip across denoising "
        "(rows = trajectories, cols = step snapshots)"
    )
    if title_suffix:
        title += f"\n{title_suffix}"
    return image_grid(
        rows,
        fig_dir / "fig01_filmstrip.png",
        col_labels=col_labels,
        row_labels=row_labels,
        title=title,
        panel_size=1.6,
    )


# ---------------------------------------------------------------------------
# Figure 2 — CLIP-score trajectories per λ (one panel per trajectory)
# ---------------------------------------------------------------------------


def fig02_clip_score_trajectories(
    *,
    fig_dir: Path,
    snapshots_by_label: dict[str, TrajectorySnapshots],
    text_targets: Sequence[str],
) -> Path:
    n = len(snapshots_by_label)
    fig, axes = plt.subplots(n, 1, figsize=(9.0, 3.5 * n), sharex=True, squeeze=False)
    cmap = plt.get_cmap("tab10")
    for row, (label, snap) in enumerate(snapshots_by_label.items()):
        steps = snap.step_indices
        scores = np.asarray(snap.clip_scores)        # (snapshots, targets)
        ax = axes[row][0]
        for t_idx, target in enumerate(text_targets):
            ax.plot(
                steps,
                scores[:, t_idx],
                marker="o",
                color=cmap(t_idx),
                label=f'"{target}"',
            )
        ax.set_ylabel("CLIP cosine")
        ax.set_title(f"{snap.label}  (λ={snap.lam:g})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[-1][0].set_xlabel("denoising step index t  (0 = noisiest)")
    fig.suptitle(
        "Figure 2 — CLIP score trajectory per text target, per trajectory",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_fig(fig, fig_dir / "fig02_clip_score_trajectories.png")


# ---------------------------------------------------------------------------
# Figure 3 — cross-trajectory comparison on the joint target
# ---------------------------------------------------------------------------


def fig03_cross_trajectory_comparison(
    *,
    fig_dir: Path,
    snapshots_by_label: dict[str, TrajectorySnapshots],
    text_targets: Sequence[str],
    target: str = "a cat and a dog",
) -> Path:
    if target not in text_targets:
        raise ValueError(f"target {target!r} not in text_targets {list(text_targets)}")
    target_idx = list(text_targets).index(target)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    cmap = plt.get_cmap("viridis")
    n = max(1, len(snapshots_by_label) - 1)
    for i, (label, snap) in enumerate(snapshots_by_label.items()):
        steps = snap.step_indices
        scores = np.asarray(snap.clip_scores)[:, target_idx]
        ax.plot(
            steps, scores, marker="o",
            color=cmap(i / n),
            label=f"{snap.label}  (λ={snap.lam:g})",
        )
    ax.set_xlabel("denoising step index t  (0 = noisiest)")
    ax.set_ylabel(f'CLIP cosine vs "{target}"')
    ax.set_title(
        "Figure 3 — when does CLIP first separate Mono from PoE?  "
        f'target = "{target}"'
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save_fig(fig, fig_dir / "fig03_cross_trajectory_comparison.png")
