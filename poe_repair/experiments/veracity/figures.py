"""Stage 3 — render the Phase 0 Veracity figure set (v3, post-audit).

Figure set per veracity-figure-plan.md:

    Fig 1     — fig1_existence_pmi              (Existence preflight)
    Fig 2     — fig2_reachability_anticorroboration  (Reachability supporting)
    Fig 4     — fig4_sufficiency                (Sufficiency HEADLINE)
    App A     — app_a_trajectory_independence   (PoE-anchor vs Mono-anchor)
    App B'    — app_b_detection_failure_modes   (chimera vs dominance via detection)
    App C     — app_c_cfg_timestep_grid         (CFG × timestep decoded grid)
    App E     — app_e_window_injection_sweep    (temporal locus; box/VQA overlay tbd)

v3 cuts: legacy fig01–fig09; cross-attention App-B; image-space residual App-D;
Reachability structure Fig 3; CFG line-plot App-C.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from poe_repair.experiments.veracity import metrics as M
from poe_repair.figures._common import overlay_boxes, save_fig
from poe_repair.figures._veracity_style import (
    ACCENT,
    ANCHOR_FRAME,
    CONTESTED_STYLE,
    CONTROL_GRAY,
    DISJOINT_STYLE,
    FP16_PMI_TOLERANCE,
    LAMBDA_CMAP,
    MONO_ANCHOR_STYLE,
    PEAK_BAND,
    POE_ANCHOR_STYLE,
    SELF_PAIR_STYLE,
    apply_veracity_style,
    lambda_color,
)


# Detection palette (cat=ACCENT, dog=desaturated red, butterfly=ACCENT,
# meadow=desaturated green). Matches plan v3 §"Box overlay palette".
DETECTION_PALETTE: dict[str, str] = {
    "a cat": ACCENT,
    "cat": ACCENT,
    "a dog": "#B36464",
    "dog": "#B36464",
    "a butterfly": ACCENT,
    "butterfly": ACCENT,
    "a flower meadow": "#5A8C5A",
    "flower meadow": "#5A8C5A",
    "meadow": "#5A8C5A",
}
DETECTION_BOX_THRESHOLD = 0.25
DETECTION_TEXT_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# Helpers (used across figures)
# ---------------------------------------------------------------------------


def _tweedie_x0(x_t: torch.Tensor, eps: torch.Tensor, alpha_bar: float) -> torch.Tensor:
    sa = float(alpha_bar) ** 0.5
    so = float(1.0 - alpha_bar) ** 0.5
    return (x_t - so * eps) / sa


def _decode_payload_x0(payload: dict, eps_key: str, ctx) -> torch.Tensor:
    """Decode Tweedie x̂_0 from a residual payload using ε given by ``eps_key``."""
    from poe_repair.runtime import decode_latents

    timestep = int(payload["timestep"])
    alpha_bar_t = float(
        ctx.scheduler.alphas_cumprod[timestep].to(dtype=torch.float64).item()
    )
    x_t = payload["x_t"].float().to(ctx.device, ctx.dtype)
    eps = payload[eps_key].float().to(ctx.device, ctx.dtype)
    x0 = _tweedie_x0(x_t, eps, alpha_bar_t)
    return decode_latents(ctx.models, x0).cpu()


def _frame_anchor(ax, color: str = ANCHOR_FRAME, lw: float = 2.0) -> None:
    """Add an accent-colored frame to an axis (used for λ=0/λ=1 panels)."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(lw)


# ---------------------------------------------------------------------------
# Fig 1 — Existence preflight: PMI / definition self-consistency
# ---------------------------------------------------------------------------


def fig1_existence_pmi(
    *,
    fig_dir: Path,
    pmi: dict,
    title_suffix: str = "",
) -> Path:
    """Per-step relative residual of the four-eps identity, log-y, with the
    fp16 tolerance line surfaced. Preflight check, not a scientific claim.
    """
    apply_veracity_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    cmap = plt.get_cmap(LAMBDA_CMAP)
    lambdas = list(pmi["per_lambda_curve"].keys())
    for i, lam_str in enumerate(lambdas):
        curve = pmi["per_lambda_curve"][lam_str]
        steps = list(range(len(curve)))
        ax.plot(
            steps, curve,
            color=cmap(i / max(1, len(lambdas) - 1)),
            linewidth=1.0, alpha=0.7, label=f"λ={lam_str}",
        )
    ax.axhline(
        FP16_PMI_TOLERANCE, color=CONTROL_GRAY, linestyle="--", linewidth=1.0,
        label=f"fp16 tolerance ({FP16_PMI_TOLERANCE:.0e})",
    )
    ax.set_yscale("log")
    ax.set_xlabel("denoising step index t  (0 = noisiest)")
    ax.set_ylabel(
        r"$\|\Delta - w(\varepsilon_J + \varepsilon_\emptyset"
        r" - \varepsilon_A - \varepsilon_B)\|\,/\,\|\Delta\|$"
    )
    suffix = f"\n{title_suffix}" if title_suffix else ""
    ax.set_title(
        f"Fig 1 — Existence preflight: PMI self-consistency  "
        f"(max={pmi['max_relative_error']:.2e}, "
        f"mean={pmi['mean_relative_error']:.2e}){suffix}"
    )
    ax.legend(fontsize=7, ncol=2, loc="best")
    return save_fig(fig, fig_dir / "fig1_existence_pmi.png")


# ---------------------------------------------------------------------------
# Fig 2 — Reachability supporting: ‖Δ_t‖ + anti-corroboration + basin-commit
# ---------------------------------------------------------------------------


def fig2_reachability_anticorroboration(
    *,
    fig_dir: Path,
    controls: dict,
    x0_stability: dict,
    basin_window: tuple[int, int],
    decoded_thumbs: dict[int, torch.Tensor] | None = None,
    title_suffix: str = "",
) -> Path:
    """Two-panel composite, shared x-axis (diffusion step t).

    Top: ‖Δ_t‖ vs t — three lines (contested, self-pair, disjoint).
    Bottom: x̂_0-prediction stability for the contested pair.
    Faint accent-tint background on both panels marks the basin-commit window.
    """
    apply_veracity_style()
    fig = plt.figure(figsize=(10.0, 6.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.4, 1.0], hspace=0.18)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)

    contested = controls["contested"]["mean"]
    n_steps = len(contested)
    steps = list(range(n_steps))

    cs, ce = basin_window
    if 0 <= cs <= ce < n_steps:
        ax_top.axvspan(cs, ce, color=PEAK_BAND, zorder=0)
        ax_bot.axvspan(cs, ce, color=PEAK_BAND, zorder=0)

    ax_top.plot(steps, contested, **CONTESTED_STYLE)

    sp = controls["self_pair"]
    if sp["mean"]:
        sp_n = min(len(sp["mean"]), n_steps)
        ax_top.plot(
            list(range(sp_n)), sp["mean"][:sp_n],
            **{**SELF_PAIR_STYLE, "label": f"self-pair (n={sp['n_seeds']})"},
        )
        if sp.get("std"):
            std = sp["std"][:sp_n]
            mean = sp["mean"][:sp_n]
            ax_top.fill_between(
                list(range(sp_n)),
                [m - s for m, s in zip(mean, std)],
                [m + s for m, s in zip(mean, std)],
                color=CONTROL_GRAY, alpha=0.15, linewidth=0,
            )

    dj = controls["disjoint"]
    if dj["mean"]:
        dj_n = min(len(dj["mean"]), n_steps)
        ax_top.plot(
            list(range(dj_n)), dj["mean"][:dj_n],
            **{**DISJOINT_STYLE, "label": f"disjoint (n={dj['n_seeds']})"},
        )
        if dj.get("std"):
            std = dj["std"][:dj_n]
            mean = dj["mean"][:dj_n]
            ax_top.fill_between(
                list(range(dj_n)),
                [m - s for m, s in zip(mean, std)],
                [m + s for m, s in zip(mean, std)],
                color=CONTROL_GRAY, alpha=0.10, linewidth=0,
            )

    ax_top.set_ylim(bottom=0.0)
    ax_top.set_ylabel(r"$\|\Delta_t\|$  (Frobenius)")
    ax_top.set_title(
        f"Fig 2 — Reachability: $\\|\\Delta_t\\|$ is composition-specific "
        f"and timed to basin commit{(' — ' + title_suffix) if title_suffix else ''}"
    )
    ax_top.legend(loc="upper right")

    if decoded_thumbs:
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage

        from poe_repair.figures._common import _to_imshow_array
        cur_top = ax_top.get_ylim()[1]
        ax_top.set_ylim(0.0, cur_top * 1.55)
        cur_top = ax_top.get_ylim()[1]
        thumb_y_data = cur_top * 0.85
        for step_idx, thumb in decoded_thumbs.items():
            if step_idx < 0 or step_idx >= n_steps:
                continue
            y_at = contested[step_idx]
            ax_top.plot(
                [step_idx, step_idx], [y_at, thumb_y_data * 0.88],
                color=ACCENT, linewidth=0.6, alpha=0.5, zorder=1,
            )
            arr = _to_imshow_array(thumb)
            imagebox = OffsetImage(arr, zoom=0.06)
            ab = AnnotationBbox(
                imagebox, (step_idx, thumb_y_data),
                frameon=True, pad=0.15,
                bboxprops=dict(edgecolor=ACCENT, linewidth=0.6),
                box_alignment=(0.5, 0.5),
            )
            ax_top.add_artist(ab)

    sb_steps = x0_stability.get("step_indices", [])
    sb_vals = x0_stability.get("stability", [])
    if sb_steps and sb_vals:
        ax_bot.plot(
            sb_steps, sb_vals,
            color=ACCENT, linewidth=1.6,
            label=r"$1 - \mathrm{corr}(\hat x_0(t), \hat x_0(T))$",
        )
        ax_bot.axhline(
            0.05, color=CONTROL_GRAY, linestyle=":", linewidth=0.9,
            label="basin-commit threshold (0.05)",
        )
    ax_bot.set_xlabel("denoising step index t  (0 = noisiest)")
    ax_bot.set_ylabel(r"$1 - \mathrm{corr}(\hat x_0(t), \hat x_0(T))$")
    ax_bot.legend(loc="upper right")

    return save_fig(fig, fig_dir / "fig2_reachability_anticorroboration.png")


# ---------------------------------------------------------------------------
# Fig 4 — Sufficiency HEADLINE: λ-sweep strip + curves with detection overlay
# ---------------------------------------------------------------------------


def fig4_sufficiency(
    *,
    fig_dir: Path,
    paths_by_lambda: dict[float, Path],
    distances: dict,
    title_suffix: str = "",
    target_lambdas: tuple[float, ...] = (0.0, 0.2, 0.5, 0.8, 1.0),
    detection_queries: tuple[str, ...] | None = None,
    per_concept_texts: tuple[str, ...] | None = None,
) -> Path:
    """Headline figure of the v3 set.

    Two-row composite:
      Top: 5-panel decoded x̂_0 strip at λ ∈ ``target_lambdas``,
           with GroundingDINO bounding boxes overlaid for ``detection_queries``.
      Bottom: 3 panels —
        (left)   latent-L2 distance vs λ
        (middle) CLIP image-cosine distance vs λ
        (right)  load-bearing detection panel — GroundingDINO max-confidence
                 per query vs λ, with a horizontal at the 0.35 detection
                 threshold; per-concept CLIP-Score curves overlaid faintly
                 as proxies (clearly labelled).

    Detection is best-effort: if GroundingDINO is unavailable, the top-row
    boxes are silently omitted and the bottom-right panel falls back to
    the per-concept CLIP-Score curves only (with a caption note in the
    panel title).
    """
    apply_veracity_style()

    available_lambdas = sorted(paths_by_lambda.keys())
    snapped: list[tuple[float, float, Path]] = []
    for tlam in target_lambdas:
        if not available_lambdas:
            break
        closest = min(available_lambdas, key=lambda l: abs(l - tlam))
        snapped.append((tlam, closest, paths_by_lambda[closest]))

    n_strip = len(snapped)
    n_bottom_panels = 3
    fig = plt.figure(figsize=(max(11.0, 2.4 * n_strip), 8.0))
    gs = fig.add_gridspec(
        2, n_bottom_panels, height_ratios=[1.4, 1.0], hspace=0.34, wspace=0.28,
    )
    strip_grid = gs[0, :].subgridspec(1, n_strip, wspace=0.05)

    from poe_repair.figures._common import _to_imshow_array

    # Pre-detect on each strip image (cheap to do here so we have one consistent
    # call site that handles graceful-fallback).
    detections_by_path: dict[Path, list[dict]] = {}
    detection_failed = False
    if detection_queries:
        try:
            for _, _, ppath in snapped:
                detections_by_path[ppath] = M.detect_boxes(
                    ppath,
                    list(detection_queries),
                    box_threshold=DETECTION_BOX_THRESHOLD,
                    text_threshold=DETECTION_TEXT_THRESHOLD,
                )
        except Exception as exc:  # noqa: BLE001 — surface but don't crash render
            import traceback
            print(f"[fig4] GroundingDINO detection skipped: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            detections_by_path = {}
            detection_failed = True

    for i, (tlam, slam, ppath) in enumerate(snapped):
        ax = fig.add_subplot(strip_grid[0, i])
        img = _to_imshow_array(ppath)
        ax.imshow(img)
        if ppath in detections_by_path:
            overlay_boxes(
                ax,
                detections=detections_by_path[ppath],
                palette=DETECTION_PALETTE,
                missing_threshold=DETECTION_BOX_THRESHOLD,
            )
        is_anchor = abs(slam - 0.0) < 1e-6 or abs(slam - 1.0) < 1e-6
        if is_anchor:
            _frame_anchor(ax)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            ax.axis("off")
        match_note = (
            f"λ={slam:.2f}" if abs(slam - tlam) < 1e-6
            else f"λ={slam:.2f}≈{tlam:.2f}"
        )
        ax.set_title(match_note, fontsize=9)

    lambdas = list(distances["lambdas"])

    ax_l2 = fig.add_subplot(gs[1, 0])
    ax_l2.plot(
        lambdas, distances["latent_l2"]["d_poe"],
        marker="o", color=CONTROL_GRAY, label=r"$d_{\mathrm{PoE}}$",
    )
    ax_l2.plot(
        lambdas, distances["latent_l2"]["d_mono"],
        marker="s", color=ACCENT, label=r"$d_{\mathrm{Mono}}$",
    )
    ax_l2.set_xlabel("λ")
    ax_l2.set_ylabel("latent-L2 distance")
    ax_l2.set_title("latent-L2")
    ax_l2.legend()

    ax_cl = fig.add_subplot(gs[1, 1])
    ax_cl.plot(
        lambdas, distances["clip_image_cosine"]["d_poe"],
        marker="o", color=CONTROL_GRAY, label=r"$d_{\mathrm{PoE}}$",
    )
    ax_cl.plot(
        lambdas, distances["clip_image_cosine"]["d_mono"],
        marker="s", color=ACCENT, label=r"$d_{\mathrm{Mono}}$",
    )
    ax_cl.set_xlabel("λ")
    ax_cl.set_ylabel("CLIP cosine distance")
    ax_cl.set_title("CLIP image cosine")
    ax_cl.legend()

    # Bottom-right load-bearing panel: GroundingDINO max-confidence per query
    # vs λ. Per-concept CLIP-Score overlaid as faint proxy curves.
    ax_pc = fig.add_subplot(gs[1, 2])
    full_lambdas = sorted(paths_by_lambda.keys())
    full_paths = [paths_by_lambda[lam] for lam in full_lambdas]

    detection_curves_drawn = False
    if detection_queries and not detection_failed:
        try:
            per_query_conf: dict[str, list[float]] = {q: [] for q in detection_queries}
            for ppath in full_paths:
                dets = M.detect_boxes(
                    ppath, list(detection_queries),
                    box_threshold=DETECTION_BOX_THRESHOLD,
                    text_threshold=DETECTION_TEXT_THRESHOLD,
                )
                # max confidence per query among detections that match the query
                for q in detection_queries:
                    matches = [
                        d["confidence"] for d in dets
                        if d.get("label", "").strip().lower()
                        == q.strip().lower()
                    ]
                    per_query_conf[q].append(max(matches) if matches else 0.0)
            for i, q in enumerate(detection_queries):
                col = DETECTION_PALETTE.get(q.strip().lower(), ACCENT)
                ax_pc.plot(
                    full_lambdas, per_query_conf[q],
                    marker="o", color=col, linewidth=1.8,
                    label=f'detect "{q}"  (max conf)',
                )
            ax_pc.axhline(
                DETECTION_BOX_THRESHOLD, color=CONTROL_GRAY, linestyle=":",
                linewidth=0.9,
                label=f"detection threshold ({DETECTION_BOX_THRESHOLD:.2f})",
            )
            detection_curves_drawn = True
        except Exception as exc:  # noqa: BLE001
            print(f"[fig4] detection curves skipped: {exc}")

    if per_concept_texts:
        try:
            sims = M.clip_image_text_similarities(
                full_paths, list(per_concept_texts),
            )
            for i, text in enumerate(per_concept_texts):
                col = lambda_color(i, max(2, len(per_concept_texts)))
                ax_pc.plot(
                    full_lambdas, sims[text],
                    linestyle="--", color=col, linewidth=1.0, alpha=0.7,
                    label=f'CLIP-Score "{text}"  (proxy)',
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[fig4] per-concept CLIP overlay skipped: {exc}")

    ax_pc.set_xlabel("λ")
    if detection_curves_drawn:
        ax_pc.set_ylabel("GroundingDINO confidence  (logit, not calibrated)")
        ax_pc.set_title("per-concept detection (load-bearing)")
    else:
        ax_pc.set_ylabel("CLIP image-text similarity")
        ax_pc.set_title("per-concept CLIP score (detection unavailable)")
    ax_pc.legend(fontsize=7, loc="best")

    suffix = f"  |  {title_suffix}" if title_suffix else ""
    fig.suptitle(
        f"Fig 4 — Sufficiency (HEADLINE): walking along Δ_t flips chimera → "
        f"co-occurrence{suffix}",
        fontsize=12, y=0.99,
    )
    return save_fig(fig, fig_dir / "fig4_sufficiency.png")


# ---------------------------------------------------------------------------
# App-A — Trajectory dependence (PoE-anchor vs Mono-anchor)
# ---------------------------------------------------------------------------


def app_a_trajectory_independence(
    *,
    fig_dir: Path,
    cells: list[dict] | None = None,
    poe_anchor_run_dir: Path | None = None,
    mono_anchor_run_dir: Path | None = None,
    scheduler=None,
    basin_threshold: float = 0.05,
    title_suffix: str = "",
) -> Path:
    """``‖Δ_t‖`` along PoE-anchor (solid) vs Mono-anchor (dashed) trajectories.

    2-row × N-column composite. Top row: absolute Frobenius norms (shared y).
    Bottom row: each curve self-normalised to its own peak (shape claim).
    Basin-commit window overlaid as a faint accent band on every panel.

    Modes:
      - Multi-cell: ``cells = [{"label", "regime", "poe_run_dir(s)",
        "mono_run_dir(s)"}, ...]``.
      - Single-cell (legacy): pass ``poe_anchor_run_dir`` and
        ``mono_anchor_run_dir``.
    """
    apply_veracity_style()

    if cells is None:
        if poe_anchor_run_dir is None or mono_anchor_run_dir is None:
            raise ValueError(
                "app_a_trajectory_independence requires either `cells` or "
                "both `poe_anchor_run_dir` and `mono_anchor_run_dir`."
            )
        cells = [{
            "label": "",
            "regime": "",
            "poe_run_dir": poe_anchor_run_dir,
            "mono_run_dir": mono_anchor_run_dir,
        }]

    def _load_seed_curves(c: dict, key_single: str, key_list: str) -> list[list[float]]:
        if key_list in c and c[key_list]:
            dirs = c[key_list]
        elif key_single in c and c[key_single] is not None:
            dirs = [c[key_single]]
        else:
            return []
        out: list[list[float]] = []
        for d in dirs:
            try:
                norms = M.per_step_residual_norms_from_residuals(d)
                if norms:
                    out.append(norms)
            except FileNotFoundError:
                continue
        return out

    def _aggregate(per_seed: list[list[float]]) -> tuple[list[float], list[float], int]:
        if not per_seed:
            return ([], [], 0)
        n_steps = min(len(s) for s in per_seed)
        truncated = [s[:n_steps] for s in per_seed]
        arr = np.asarray(truncated, dtype=np.float64)
        mean = arr.mean(axis=0).tolist()
        std = (arr.std(axis=0, ddof=0)).tolist() if arr.shape[0] > 1 else [0.0] * n_steps
        return (mean, std, arr.shape[0])

    loaded: list = []
    y_max = 0.0
    for c in cells:
        poe_seeds = _load_seed_curves(c, "poe_run_dir", "poe_run_dirs")
        mono_seeds = _load_seed_curves(c, "mono_run_dir", "mono_run_dirs")
        poe_mean, poe_std, poe_n = _aggregate(poe_seeds)
        mono_mean, mono_std, mono_n = _aggregate(mono_seeds)
        if poe_mean:
            y_max = max(y_max, max(poe_mean) + (max(poe_std) if poe_std else 0.0))
        if mono_mean:
            y_max = max(y_max, max(mono_mean) + (max(mono_std) if mono_std else 0.0))
        basin: tuple[int, int] | None = None
        if scheduler is not None and mono_seeds:
            mono_dir = c.get("mono_run_dirs", [c.get("mono_run_dir")])[0]
            try:
                x0_stab = M.compute_x0_stability(
                    mono_dir, eps_key="eps_j", scheduler=scheduler,
                )
                basin = M.basin_commit_window(
                    x0_stab["stability"], threshold=basin_threshold,
                )
            except (KeyError, FileNotFoundError, ValueError):
                basin = None
        loaded.append(
            (c, (poe_mean, poe_std, poe_n), (mono_mean, mono_std, mono_n), basin)
        )

    n = len(loaded)
    fig, axes = plt.subplots(
        2, n, figsize=(6.5 * n, 7.6),
        sharex="col", squeeze=False,
    )

    for col, (c, poe_agg, mono_agg, basin) in enumerate(loaded):
        ax_abs = axes[0, col]
        ax_norm = axes[1, col]
        poe_mean, poe_std, poe_n = poe_agg
        mono_mean, mono_std, mono_n = mono_agg

        if basin is not None:
            cs, ce = basin
            ax_abs.axvspan(cs, ce, color=PEAK_BAND, zorder=0)
            ax_norm.axvspan(cs, ce, color=PEAK_BAND, zorder=0)

        def _plot_band(ax, mean, std, style, n):
            if not mean:
                return
            xs = list(range(len(mean)))
            label = style.get("label", "")
            if n > 1:
                label = f"{label}  (n={n})"
            kw = {**style, "label": label}
            ax.plot(xs, mean, **kw)
            if std and any(s > 0 for s in std) and n > 1:
                lower = [m - s for m, s in zip(mean, std)]
                upper = [m + s for m, s in zip(mean, std)]
                ax.fill_between(
                    xs, lower, upper, color=style["color"], alpha=0.18, linewidth=0,
                )

        _plot_band(ax_abs, poe_mean, poe_std, POE_ANCHOR_STYLE, poe_n)
        _plot_band(ax_abs, mono_mean, mono_std, MONO_ANCHOR_STYLE, mono_n)
        ax_abs.set_ylim(0.0, y_max * 1.05 if y_max > 0 else 1.0)
        regime = c.get("regime", "")
        label = c.get("label", "")
        title_bits = [b for b in [label, regime] if b]
        ax_abs.set_title("  |  ".join(title_bits) if title_bits else "")
        ax_abs.legend(loc="upper right")

        poe_peak = max(poe_mean) if poe_mean else 1.0
        mono_peak = max(mono_mean) if mono_mean else 1.0
        if poe_mean and poe_peak > 0:
            norm_mean = [v / poe_peak for v in poe_mean]
            norm_std = [v / poe_peak for v in poe_std] if poe_std else []
            _plot_band(ax_norm, norm_mean, norm_std, POE_ANCHOR_STYLE, poe_n)
        if mono_mean and mono_peak > 0:
            norm_mean = [v / mono_peak for v in mono_mean]
            norm_std = [v / mono_peak for v in mono_std] if mono_std else []
            _plot_band(ax_norm, norm_mean, norm_std, MONO_ANCHOR_STYLE, mono_n)
        ax_norm.set_ylim(0.0, 1.15)
        ax_norm.set_xlabel("denoising step index t")
        ax_norm.legend(loc="upper right")

    axes[0, 0].set_ylabel(r"$\|\Delta_t\|$  (Frobenius)")
    axes[1, 0].set_ylabel(r"$\|\Delta_t\| \,/\, \mathrm{peak}$  (self-normalised)")

    fig.text(
        0.005, 0.74, "absolute",
        rotation=90, va="center", ha="left", fontsize=9, alpha=0.75,
    )
    fig.text(
        0.005, 0.30, "shape only",
        rotation=90, va="center", ha="left", fontsize=9, alpha=0.75,
    )
    if any(b is not None for *_, b in loaded):
        fig.text(
            0.5, 0.005,
            "shaded band = basin-commit window  "
            "(measured from x̂_0-stability on the Mono trajectory)",
            ha="center", fontsize=8, alpha=0.7,
        )

    suffix = f"  |  {title_suffix}" if title_suffix else ""
    fig.suptitle(
        f"App-A — Trajectory dependence of ‖Δ_t‖: absolute and shape-only{suffix}",
        fontsize=12, y=1.00,
    )
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.97))
    return save_fig(fig, fig_dir / "app_a_trajectory_independence.png")


# ---------------------------------------------------------------------------
# App-B' — Detection-based failure-mode classification (NEW v3)
# ---------------------------------------------------------------------------


def app_b_detection_failure_modes(
    *,
    fig_dir: Path,
    image_paths_by_seed: dict[int, Path],
    detection_queries: tuple[str, str] = ("a cat", "a dog"),
    chimera_seed: int | None = None,
    title_suffix: str = "",
) -> Path:
    """1×N composite of decoded PoE images per seed at λ=0, with GroundingDINO
    boxes overlaid for ``detection_queries``. Below the strip: a small
    table-row per seed reading {confidence_q1, confidence_q2, box_IoU,
    regime_classification}. ``chimera_seed`` (optional) labels which seed
    is the chimera; its row's IoU is *not* bolded by default — the visual
    already does the work.
    """
    apply_veracity_style()

    seeds = sorted(image_paths_by_seed.keys())
    n = len(seeds)
    if n == 0:
        raise ValueError("app_b_detection_failure_modes: no seeds provided")

    # Run detection per seed (best-effort; if GroundingDINO is unavailable,
    # we fall back to a placeholder strip with an explanatory annotation).
    detections_by_seed: dict[int, list[dict]] = {}
    detection_error: str | None = None
    try:
        for seed, ppath in image_paths_by_seed.items():
            detections_by_seed[seed] = M.detect_boxes(
                ppath, list(detection_queries),
                box_threshold=DETECTION_BOX_THRESHOLD,
                text_threshold=DETECTION_TEXT_THRESHOLD,
            )
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        detection_error = f"{type(exc).__name__}: {exc}"

    # Compose figure: top = image strip, bottom = small table.
    fig = plt.figure(figsize=(max(9.0, 3.0 * n), 5.0))
    gs = fig.add_gridspec(2, n, height_ratios=[3.2, 1.0], hspace=0.28, wspace=0.05)

    from poe_repair.figures._common import _to_imshow_array

    q1, q2 = detection_queries[0], detection_queries[1]

    for col, seed in enumerate(seeds):
        ax = fig.add_subplot(gs[0, col])
        ppath = image_paths_by_seed[seed]
        ax.imshow(_to_imshow_array(ppath))
        if seed in detections_by_seed:
            overlay_boxes(
                ax,
                detections=detections_by_seed[seed],
                palette=DETECTION_PALETTE,
                missing_threshold=DETECTION_BOX_THRESHOLD,
            )
        is_chimera = (chimera_seed is not None and seed == chimera_seed)
        title = f"seed {seed}" + ("  (chimera)" if is_chimera else "")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Bottom table-row per seed with confidence/IoU/regime.
    for col, seed in enumerate(seeds):
        ax = fig.add_subplot(gs[1, col])
        ax.axis("off")
        if seed in detections_by_seed:
            dets = detections_by_seed[seed]

            def _max_conf(label: str) -> float:
                vals = [
                    d["confidence"] for d in dets
                    if d.get("label", "").strip().lower() == label.strip().lower()
                ]
                return max(vals) if vals else 0.0

            def _best_box(label: str):
                cands = [
                    d for d in dets
                    if d.get("label", "").strip().lower() == label.strip().lower()
                ]
                if not cands:
                    return None
                return max(cands, key=lambda d: d["confidence"])["box"]

            c1 = _max_conf(q1)
            c2 = _max_conf(q2)
            b1 = _best_box(q1)
            b2 = _best_box(q2)
            if b1 is not None and b2 is not None:
                iou = M.box_iou(b1, b2)
                iou_str = f"{iou:.2f}"
            else:
                iou = None
                iou_str = "—"
            regime = M.classify_detection_regime(
                dets, queries=detection_queries,
                threshold=DETECTION_BOX_THRESHOLD,
            )
            rows = [
                f'conf "{q1}":  {c1:.2f}' if c1 > 0 else f'conf "{q1}":  miss',
                f'conf "{q2}":  {c2:.2f}' if c2 > 0 else f'conf "{q2}":  miss',
                f"box IoU:  {iou_str}",
                f"regime:  {regime}",
            ]
            ax.text(
                0.02, 0.95, "\n".join(rows),
                ha="left", va="top", transform=ax.transAxes,
                fontsize=8, family="monospace",
            )
        else:
            msg = f"detection unavailable\n({detection_error or 'no result'})"
            ax.text(
                0.5, 0.5, msg, ha="center", va="center",
                transform=ax.transAxes, fontsize=8, alpha=0.7,
            )

    suffix = f"  |  {title_suffix}" if title_suffix else ""
    fig.suptitle(
        f"App-B′ — Detection-based failure modes (GroundingDINO @ "
        f"box≥{DETECTION_BOX_THRESHOLD:.2f}, text≥{DETECTION_TEXT_THRESHOLD:.2f})"
        f"{suffix}",
        fontsize=11, y=1.00,
    )
    return save_fig(fig, fig_dir / "app_b_detection_failure_modes.png")


# ---------------------------------------------------------------------------
# App-C — CFG × timestep qualitative grid (NEW v3, replaces line plot)
# ---------------------------------------------------------------------------


def app_c_cfg_timestep_grid(
    *,
    fig_dir: Path,
    cfg_run_dirs: dict[float, Path],
    ctx,
    timesteps: tuple[int, ...] = (0, 10, 20, 30, 40, 49),
    commit_threshold: float = 0.5,
    title_suffix: str = "",
) -> Path:
    """5-row × N-column qualitative grid.

    Rows = CFG values (sorted ascending).  Columns = denoising step indices.
    Each cell = decoded ``x̂_0 = tweedie(x_t, ε_PoE)`` from the cached
    residual at that step. Optional thin vertical highlight on the column
    closest to the basin-commit threshold (x̂_0-stability ≤ ``commit_threshold``).

    ``cfg_run_dirs[cfg_value] -> run_dir`` (the λ=0 / PoE-anchor run dir for
    each CFG value). Decoding requires ``ctx`` for VAE access.
    """
    apply_veracity_style()

    if not cfg_run_dirs:
        raise ValueError("app_c_cfg_timestep_grid: no CFG run dirs supplied")
    cfg_values = sorted(cfg_run_dirs.keys())
    n_rows = len(cfg_values)
    n_cols = len(timesteps)

    # Precompute commit column per CFG row (closest column index where
    # x̂_0-stability ≤ commit_threshold). One vertical highlight per row.
    commit_col_by_cfg: dict[float, int | None] = {}
    if ctx is not None:
        for cfg_val, run_dir in cfg_run_dirs.items():
            try:
                x0_stab = M.compute_x0_stability(
                    run_dir, eps_key="eps_poe", scheduler=ctx.scheduler,
                )
                stab = x0_stab.get("stability", [])
                steps = x0_stab.get("step_indices", [])
                if not stab:
                    commit_col_by_cfg[cfg_val] = None
                    continue
                # First step where stability ≤ commit_threshold.
                commit_step: int | None = None
                for s, v in zip(steps, stab):
                    if v <= commit_threshold:
                        commit_step = s
                        break
                if commit_step is None:
                    commit_col_by_cfg[cfg_val] = None
                else:
                    # Column = nearest of ``timesteps`` to commit_step.
                    commit_col_by_cfg[cfg_val] = min(
                        range(n_cols),
                        key=lambda i: abs(timesteps[i] - commit_step),
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"[app-c] commit column for CFG={cfg_val} unavailable: {exc}")
                commit_col_by_cfg[cfg_val] = None

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.6 * n_cols + 0.4, 1.6 * n_rows + 0.6),
        squeeze=False,
    )

    from poe_repair.figures._common import _to_imshow_array

    for r, cfg_val in enumerate(cfg_values):
        run_dir = cfg_run_dirs[cfg_val]
        commit_col = commit_col_by_cfg.get(cfg_val)
        for c, t in enumerate(timesteps):
            ax = axes[r, c]
            ax.axis("off")
            f = M.residuals_dir_for(run_dir) / f"step_{t:03d}.pt"
            if not f.exists() or ctx is None:
                ax.text(
                    0.5, 0.5,
                    "(missing)" if not f.exists() else "(no ctx)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=7, alpha=0.6,
                )
            else:
                try:
                    payload = torch.load(f, map_location="cpu")
                    img = _decode_payload_x0(payload, "eps_poe", ctx)
                    ax.imshow(_to_imshow_array(img))
                except Exception as exc:  # noqa: BLE001
                    ax.text(
                        0.5, 0.5, f"err\n{type(exc).__name__}",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=7, alpha=0.6,
                    )

            if commit_col is not None and c == commit_col:
                # Thin accent-colored frame to mark the commit column.
                _frame_anchor(ax, color=ACCENT, lw=1.5)

            if r == 0:
                ax.set_title(f"t={t}", fontsize=9)
            if c == 0:
                ax.text(
                    -0.18, 0.5, f"CFG={cfg_val:.1f}",
                    ha="right", va="center", transform=ax.transAxes,
                    fontsize=9,
                )

    suffix = f"  |  {title_suffix}" if title_suffix else ""
    fig.suptitle(
        f"App-C — CFG × timestep decoded grid: basin-commit column is "
        f"composition-anchored (highlight = x̂₀-stability ≤ {commit_threshold:.2f})"
        f"{suffix}",
        fontsize=11, y=1.00,
    )
    return save_fig(fig, fig_dir / "app_c_cfg_timestep_grid.png")


# ---------------------------------------------------------------------------
# App-E — Window-localised injection (placeholder; box/VQA overlay tbd)
# ---------------------------------------------------------------------------


def app_e_window_injection_sweep(
    *,
    fig_dir: Path,
    cell,
    ctx,
    exp_name: str = "veracity_window_injection",
    window_starts: tuple[int, ...] = (0, 5, 10, 15, 20, 25, 30, 35, 40, 45),
    window_width: int = 5,
    poe_baseline_path: Path | None = None,
    mono_baseline_path: Path | None = None,
    title_suffix: str = "",
) -> Path:
    """Temporal-locus figure: VQAScore (load-bearing) + CLIP image-text
    similarity (proxy) for the joint prompt vs window-start step. Top
    strip of decoded window-injection images with GroundingDINO boxes
    overlaid (one per window).

    Render is best-effort: if window-injection runs aren't on disk, a
    placeholder is written instead (so the orchestrator doesn't crash).
    """
    apply_veracity_style()

    cfg = __import__(
        "poe_repair.config", fromlist=["RunConfig"]
    ).RunConfig()

    joint_text = (
        ctx.joint_template.format(a=cell.prompt_a, b=cell.prompt_b)
        if ctx is not None else f"{cell.prompt_a} and {cell.prompt_b}"
    )

    base = (
        cfg.paths.output_root / exp_name / "pairs" / cell.pair_slug
        / f"seed_{cell.seed}"
    )
    pairs_xy: list[tuple[int, Path]] = []
    for start in window_starts:
        end = start + window_width
        method = f"teacher_residual_const_lam100_w{start}-{end}"
        img = base / method / f"{method}.png"
        if img.exists():
            pairs_xy.append((start, img))

    if not pairs_xy:
        fig, ax = plt.subplots(figsize=(8.0, 3.5))
        ax.text(
            0.5, 0.5,
            f"App-E pending — no window-injection runs found under\n"
            f"{base}\n(blocked on Round B GPU sweeps; see plan-figures plan.)",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=9, alpha=0.7,
        )
        ax.axis("off")
        return save_fig(fig, fig_dir / "app_e_window_injection.png")

    image_paths = [p for _, p in pairs_xy]
    starts_x = [s for s, _ in pairs_xy]

    # Load-bearing: VQAScore (min over three grounded questions).
    vqa_questions = (
        f"Is there a {cell.prompt_a.removeprefix('a ').strip()} in this image?",
        f"Is there a {cell.prompt_b.removeprefix('a ').strip()} in this image?",
        f"Is the {cell.prompt_a.removeprefix('a ').strip()} clearly separate "
        f"from the {cell.prompt_b.removeprefix('a ').strip()}?",
    )
    vqa_min: list[float] = []
    vqa_per_q: list[list[float]] = [[] for _ in vqa_questions]
    vqa_failed = False
    try:
        for ppath in image_paths:
            probs = M.vqascore_yesno(ppath, list(vqa_questions))
            vqa_min.append(min(probs))
            for i, p in enumerate(probs):
                vqa_per_q[i].append(p)
    except Exception as exc:  # noqa: BLE001
        print(f"[app-e] VQAScore unavailable: {exc}")
        vqa_failed = True
        vqa_min = []

    # Proxy: CLIP image-text similarity for the joint prompt.
    sims = M.clip_image_text_similarities(image_paths, [joint_text])
    clip_ys = sims[joint_text]

    # Baselines
    poe_y = None
    mono_y = None
    if poe_baseline_path is not None and poe_baseline_path.exists():
        poe_y = M.clip_image_text_similarities(
            [poe_baseline_path], [joint_text],
        )[joint_text][0]
    if mono_baseline_path is not None and mono_baseline_path.exists():
        mono_y = M.clip_image_text_similarities(
            [mono_baseline_path], [joint_text],
        )[joint_text][0]

    # Detection on each window image (best-effort)
    detection_queries = (f"a {cell.prompt_a.removeprefix('a ').strip()}",
                         f"a {cell.prompt_b.removeprefix('a ').strip()}")
    detections_by_path: dict[Path, list[dict]] = {}
    try:
        for ppath in image_paths:
            detections_by_path[ppath] = M.detect_boxes(
                ppath, list(detection_queries),
                box_threshold=DETECTION_BOX_THRESHOLD,
                text_threshold=DETECTION_TEXT_THRESHOLD,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[app-e] detection skipped: {exc}")
        detections_by_path = {}

    # Layout: top = strip of windows w/ boxes; bottom = curves.
    n_strip = len(image_paths)
    fig = plt.figure(figsize=(max(11.0, 1.6 * n_strip), 7.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.4], hspace=0.30)
    strip_grid = gs[0, 0].subgridspec(1, n_strip, wspace=0.05)

    from poe_repair.figures._common import _to_imshow_array

    for i, (start, ppath) in enumerate(pairs_xy):
        ax = fig.add_subplot(strip_grid[0, i])
        ax.imshow(_to_imshow_array(ppath))
        if ppath in detections_by_path:
            overlay_boxes(
                ax, detections=detections_by_path[ppath],
                palette=DETECTION_PALETTE,
                missing_threshold=DETECTION_BOX_THRESHOLD,
            )
        ax.set_title(f"[{start},{start+window_width}]", fontsize=8)
        ax.axis("off")

    ax = fig.add_subplot(gs[1, 0])

    if poe_y is not None:
        ax.axhline(
            poe_y, color=CONTROL_GRAY, linestyle=":", linewidth=1.0,
            label=f"PoE baseline (no injection) = {poe_y:.3f}",
        )
    if mono_y is not None:
        ax.axhline(
            mono_y, color=ACCENT, linestyle=":", linewidth=1.0,
            label=f"Mono baseline (full injection) = {mono_y:.3f}",
        )

    if vqa_min:
        ax.plot(
            starts_x, vqa_min,
            marker="o", color=ACCENT, linewidth=2.0,
            label="VQAScore  min(p_yes)  (load-bearing)",
        )
        # Per-question faint envelope
        for i, q in enumerate(vqa_questions):
            ax.plot(
                starts_x, vqa_per_q[i],
                color=ACCENT, linestyle=":", linewidth=0.8, alpha=0.4,
            )
    elif not vqa_failed:
        # vqa empty for some other reason (shouldn't happen)
        pass

    ax.plot(
        starts_x, clip_ys,
        marker="s", color=CONTROL_GRAY, linestyle="--", linewidth=1.4,
        label="CLIP image-text  (proxy)",
    )

    ax.set_xlabel("window start step  (window = [start, start+%d])" % window_width)
    ax.set_ylabel("score")
    suffix = f"  |  {title_suffix}" if title_suffix else ""
    title = "App-E — Temporal locus of effectiveness"
    if vqa_failed:
        title += "  (VQAScore unavailable; CLIP proxy only)"
    ax.set_title(title + suffix)
    ax.legend(loc="best", fontsize=8)
    return save_fig(fig, fig_dir / "app_e_window_injection.png")
