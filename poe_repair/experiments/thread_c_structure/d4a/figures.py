"""D4-A bar chart and D4-A-t small-multiples.

Both renderers consume ``D4aResult`` directly so the only IO is matplotlib.

D4-A (single panel):
    x-axis = seed group, four bars per group (oracle / shared-mean /
    shuffle / zero), VQA-min on y-axis, filled disc strip below encoding
    the detection regime (filled = both_distinct, hollow = anything else).
    Per-seed threshold line at zero + 0.7·(oracle − zero).

D4-A-t (four panels):
    Same bar set, repeated for each window, on a shared y-axis. Per-panel
    threshold line at panel-zero + 0.7·(panel-oracle − panel-zero).

Both renderers also write the VQA-mean numeric on top of each bar as a
sanity check, per the plan §7b D4-A "print the mean alongside min".
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from poe_repair.experiments.thread_c_structure.d4a.runner import (
    D4aResult, D4aSeedRow,
)
from poe_repair.experiments.thread_c_structure.d4a.overrides import Condition
from poe_repair.figures._common import save_fig


CONDITION_ORDER = (
    Condition.ORACLE, Condition.SHARED_MEAN,
    Condition.SHUFFLE, Condition.ZERO,
)

CONDITION_COLOUR = {
    Condition.ORACLE:      "#2E8B57",
    Condition.SHARED_MEAN: "#274C77",
    Condition.SHUFFLE:     "#9D4EDD",
    Condition.ZERO:        "#888888",
}

CONDITION_LABEL = {
    Condition.ORACLE:      "oracle",
    Condition.SHARED_MEAN: "shared-mean",
    Condition.SHUFFLE:     "shuffle",
    Condition.ZERO:        "zero (PoE)",
}


def _find_row(
    rows: list[D4aSeedRow], seed: int, condition: Condition,
    window_label: str,
) -> D4aSeedRow | None:
    for r in rows:
        if (
            r.seed == seed
            and r.condition is condition
            and r.window_label == window_label
        ):
            return r
    return None


def _vqa_min(row: D4aSeedRow | None) -> float:
    if row is None:
        return float("nan")
    return float(row.grade.vqa_min)


def _vqa_mean(row: D4aSeedRow | None) -> float:
    if row is None:
        return float("nan")
    return float(row.grade.vqa_mean)


def _is_both_distinct(row: D4aSeedRow | None) -> bool:
    return row is not None and row.grade.detection_regime == "both_distinct"


def _draw_panel(
    ax,
    result: D4aResult,
    *,
    window_label: str,
    threshold_strategy: str = "per_seed",   # "per_seed" or "per_panel"
    title: str | None = None,
    ymax: float = 1.0,
) -> dict:
    """Draw a single D4-A-style panel. Returns a dict of summary stats."""
    seeds = result.seeds
    cond_order = CONDITION_ORDER
    n_seeds = len(seeds)
    n_cond = len(cond_order)
    group_centres = np.arange(n_seeds, dtype=float)
    bar_w = 0.8 / max(n_cond, 1)

    per_seed_oracle: list[float] = []
    per_seed_zero: list[float] = []
    panel_pass: list[bool] = []
    panel_filled: list[bool] = []
    for i, seed in enumerate(seeds):
        oracle_row = _find_row(result.rows, seed, Condition.ORACLE, window_label)
        zero_row = _find_row(result.rows, seed, Condition.ZERO, window_label)
        shared_row = _find_row(result.rows, seed, Condition.SHARED_MEAN, window_label)
        per_seed_oracle.append(_vqa_min(oracle_row))
        per_seed_zero.append(_vqa_min(zero_row))
        for j, cond in enumerate(cond_order):
            row = _find_row(result.rows, seed, cond, window_label)
            v = _vqa_min(row)
            mean_v = _vqa_mean(row)
            x = float(group_centres[i] - 0.4 + j * bar_w + bar_w / 2.0)
            ax.bar(
                x, v if not np.isnan(v) else 0.0, width=bar_w * 0.95,
                color=CONDITION_COLOUR[cond],
                edgecolor="black", linewidth=0.5,
                label=CONDITION_LABEL[cond] if i == 0 else None,
            )
            if not np.isnan(v):
                ax.text(
                    x, min(ymax, v) + 0.02, f"μ={mean_v:.2f}",
                    ha="center", va="bottom", fontsize=6, color="#444444",
                )

        # Detection-regime strip (one disc per bar) just below y=0.
        for j, cond in enumerate(cond_order):
            row = _find_row(result.rows, seed, cond, window_label)
            x = float(group_centres[i] - 0.4 + j * bar_w + bar_w / 2.0)
            filled = _is_both_distinct(row)
            ax.scatter(
                x, -0.04 * ymax,
                marker="o", s=40,
                facecolor=CONDITION_COLOUR[cond] if filled else "white",
                edgecolor=CONDITION_COLOUR[cond], linewidths=1.2,
            )
            if cond is Condition.SHARED_MEAN:
                panel_filled.append(filled)

        # Per-seed threshold line.
        if threshold_strategy == "per_seed":
            o = _vqa_min(oracle_row)
            z = _vqa_min(zero_row)
            if not (np.isnan(o) or np.isnan(z)):
                thr = z + 0.7 * (o - z)
                xl = group_centres[i] - 0.4
                xr = group_centres[i] + 0.4
                ax.hlines(thr, xl, xr,
                          colors="#B23A48", linestyles="--", linewidth=1.0)
                sm = _vqa_min(shared_row)
                panel_pass.append(not np.isnan(sm) and sm >= thr)

    if threshold_strategy == "per_panel":
        # Per-panel threshold uses panel-mean of oracle and zero.
        o_panel = np.nanmean(per_seed_oracle)
        z_panel = np.nanmean(per_seed_zero)
        thr = z_panel + 0.7 * (o_panel - z_panel)
        ax.axhline(thr, color="#B23A48", ls="--", lw=1.0,
                   label=f"panel threshold = {thr:.2f}")
        for i, seed in enumerate(seeds):
            shared_row = _find_row(result.rows, seed, Condition.SHARED_MEAN, window_label)
            sm = _vqa_min(shared_row)
            panel_pass.append(not np.isnan(sm) and sm >= thr)

    ax.set_xticks(group_centres)
    ax.set_xticklabels([f"seed {s}" for s in seeds], fontsize=8)
    ax.set_ylim(-0.10 * ymax, ymax + 0.05)
    ax.set_ylabel("VQAScore min")
    if title is not None:
        ax.set_title(title, fontsize=9)
    return {
        "shared_mean_pass_count": int(sum(panel_pass)),
        "shared_mean_pass_total": int(len(panel_pass)),
        "shared_mean_both_distinct_count": int(sum(panel_filled)),
    }


def render_d4a(
    result: D4aResult,
    fig_path: Path,
    *,
    pair_slug: str,
    window_label: str = "all",
) -> Path:
    """Single-panel D4-A bar chart: per-seed threshold strategy."""
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    summary = _draw_panel(
        ax, result, window_label=window_label,
        threshold_strategy="per_seed",
        title=None,
    )
    pairing = result.shuffle_pairing
    pairing_str = ", ".join(
        f"{s}→{pairing[s]}" for s in result.seeds if s in pairing
    )
    n_seeds = summary["shared_mean_pass_total"] or len(result.seeds)
    n_pass = summary["shared_mean_pass_count"]
    n_filled = summary["shared_mean_both_distinct_count"]
    pass_vqa = (n_pass >= 2 and n_seeds >= 3) or (n_pass == n_seeds and n_seeds == 2)
    pass_det = (n_filled >= 2 and n_seeds >= 3) or (n_filled == n_seeds and n_seeds == 2)
    overall = pass_vqa and pass_det
    color = "#2E8B57" if overall else "#B23A48"
    ax.set_title(
        f"D4-A — substitution test  |  {pair_slug}  |  "
        f"window={window_label}  |  "
        f"shared-mean ≥ 0.7·(oracle − zero) on {n_pass}/{n_seeds} seeds  |  "
        f"both_distinct on {n_filled}/{n_seeds}  |  "
        f"[{'PASS' if overall else 'FAIL'}]\n"
        f"shuffle pairing: {pairing_str}",
        fontsize=9, color=color,
    )
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    return save_fig(fig, fig_path)


def render_d4a_t(
    result: D4aResult,
    fig_path: Path,
    *,
    pair_slug: str,
    panel_labels: tuple[str, ...] = ("pre_commit", "commit", "post_commit", "all"),
) -> Path:
    """Four-panel small-multiples D4-A-t: per-panel threshold strategy."""
    panels = [lab for lab in panel_labels if any(r.window_label == lab for r in result.rows)]
    n = max(len(panels), 1)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.6), squeeze=False, sharey=True)
    summaries: list[dict] = []
    for ax, lab in zip(axes[0], panels):
        window = next(w for r in result.rows for w in [r.window] if r.window_label == lab)
        title = f"{lab}  t ∈ [{window[0]}, {window[1]})"
        summaries.append(
            _draw_panel(
                ax, result, window_label=lab,
                threshold_strategy="per_panel", title=title,
            )
        )
    # Single legend on the leftmost axis.
    axes[0][0].legend(loc="upper right", fontsize=7, frameon=False)

    n_seeds = len(result.seeds)
    commit_summary = next(
        (s for s, lab in zip(summaries, panels) if lab == "commit"), None,
    )
    commit_pass = (
        commit_summary is not None
        and commit_summary["shared_mean_pass_count"] >= max(2, n_seeds - 1)
    )
    color = "#2E8B57" if commit_pass else "#B23A48"
    fig.suptitle(
        f"D4-A-t — windowed substitution  |  {pair_slug}  "
        f"|  commit-window pass on "
        f"{commit_summary['shared_mean_pass_count'] if commit_summary else 0}/{n_seeds} seeds  "
        f"[{'PASS' if commit_pass else 'FAIL'}]",
        fontsize=10, color=color,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return save_fig(fig, fig_path)
