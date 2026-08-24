"""Render the surviving Veracity diagnostic figure set.

Figure set:

    Fig 1   — fig1_existence_pmi              (Existence preflight)
    Fig 4   — fig4_sufficiency                (Sufficiency HEADLINE)
    App A   — app_a_trajectory_independence   (PoE-anchor vs Mono-anchor)
    App B'  — app_b_detection_failure_modes   (chimera vs dominance via detection)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from poe_repair.experiments.residual_between_mono_and_poe import metrics as M
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


