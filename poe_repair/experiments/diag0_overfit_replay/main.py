"""Diagnostic 0 — single-cell overfit + replay (orchestrator).

Stages, each independently re-runnable via flags:

  1. train-2a    : train ResidualMLPSynthesizer on (cat × dog, seed 42) only.
  2. train-2b    : train DirectEpsStudent on the same single cell.
  3. replay      : run PoE, Mono, oracle teacher_residual(λ=1), 2a-diag0,
                   2b-diag0 inference on (cat × dog, seed 42).
  4. score       : compute CLIP-on-Tweedie at basin-commit step for each.
  5. summarise   : write loss curves + replay grid + CLIP bars + SUMMARY.md.

Default behaviour: run all stages, skipping training stages whose
``best.pt`` already exists. Use ``--force-train-{2a,2b}`` to retrain.

Usage:

    python -m poe_repair.experiments.diag0_overfit_replay
    python -m poe_repair.experiments.diag0_overfit_replay --skip-train-2a --skip-train-2b
    python -m poe_repair.experiments.diag0_overfit_replay --only score summarise
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image

from poe_repair.composers import direct_eps as cmp_direct_eps
from poe_repair.composers import mono as cmp_mono
from poe_repair.composers import poe as cmp_poe
from poe_repair.composers import residual_prompt as cmp_residual_prompt
from poe_repair.composers import teacher_residual as cmp_teacher_residual
from poe_repair.config import RunConfig
from poe_repair.experiments._eval_common import HEADLINE_PAIR, cell_for, slugify
from poe_repair.run import MethodCtx, make_ctx
from poe_repair.runtime import ensure_dir, write_json


EXP_NAME = "diag0_overfit_replay"
DEFAULT_PAIR = HEADLINE_PAIR              # ("a cat", "a dog")
DEFAULT_SEED = 42
DEFAULT_BASIN_STEP = 12                   # idea5a: basin commits ~step 10-15
DEFAULT_M2A_STEPS = 5000
DEFAULT_M2B_STEPS = 5000
M2A_OUTPUT_NAME = "diag0_catdog_seed42"
M2B_OUTPUT_NAME = "diag0_catdog_seed42"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

ALL_STAGES = ("train-2a", "train-2b", "replay", "score", "summarise")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _m2a_ckpt_path() -> Path:
    return REPO_ROOT / "checkpoints" / "residual_prompt" / M2A_OUTPUT_NAME / "best.pt"


def _m2b_ckpt_path() -> Path:
    return REPO_ROOT / "checkpoints" / "students" / M2B_OUTPUT_NAME / "best.pt"


def _m2a_history_path() -> Path:
    return REPO_ROOT / "outputs" / "residual_prompt" / M2A_OUTPUT_NAME / "history.json"


def _m2b_history_path() -> Path:
    return REPO_ROOT / "outputs" / "students" / M2B_OUTPUT_NAME / "history.json"


def _pair_slug() -> str:
    return slugify(*DEFAULT_PAIR)


# ---------------------------------------------------------------------------
# Stage 1 + 2 — train M2a, M2b on a single cell
# ---------------------------------------------------------------------------


def stage_train_2a(*, num_steps: int, force: bool) -> Path:
    ckpt = _m2a_ckpt_path()
    if ckpt.exists() and not force:
        print(f"[diag0] train-2a SKIP (exists: {ckpt})")
        return ckpt
    pair_slug = _pair_slug()
    cmd = [
        sys.executable, "-m", "poe_repair.embeddings.train_residual_prompt",
        "--train-split", "heldout",
        "--val-split", "heldout",
        "--pairs", pair_slug,
        "--val-pairs", pair_slug,
        "--seeds", str(DEFAULT_SEED),
        "--num-steps", str(num_steps),
        "--batch-size", "1",
        "--output-name", M2A_OUTPUT_NAME,
    ]
    print(f"[diag0] train-2a START\n  {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    print(f"[diag0] train-2a DONE in {(time.time()-t0)/60:.1f} min")
    if not ckpt.exists():
        raise RuntimeError(f"train-2a finished but {ckpt} missing")
    return ckpt


def stage_train_2b(*, num_steps: int, force: bool) -> Path:
    ckpt = _m2b_ckpt_path()
    if ckpt.exists() and not force:
        print(f"[diag0] train-2b SKIP (exists: {ckpt})")
        return ckpt
    pair_slug = _pair_slug()
    cmd = [
        sys.executable, "-m", "poe_repair.students.train_direct_eps",
        "--train-split", "heldout",
        "--val-split", "heldout",
        "--pairs", pair_slug,
        "--val-pairs", pair_slug,
        "--seeds", str(DEFAULT_SEED),
        "--num-steps", str(num_steps),
        "--batch-size", "1",
        "--output-name", M2B_OUTPUT_NAME,
    ]
    print(f"[diag0] train-2b START\n  {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    print(f"[diag0] train-2b DONE in {(time.time()-t0)/60:.1f} min")
    if not ckpt.exists():
        raise RuntimeError(f"train-2b finished but {ckpt} missing")
    return ckpt


# ---------------------------------------------------------------------------
# Stage 3 — inference replay (5 panels)
# ---------------------------------------------------------------------------


def stage_replay(
    *, ctx: MethodCtx, exp_name: str, overwrite: bool,
) -> dict[str, Path]:
    cell = cell_for(*DEFAULT_PAIR, DEFAULT_SEED)
    paths: dict[str, Path] = {}

    print("[diag0] replay/PoE")
    paths["PoE"] = cmp_poe.run(cell, ctx, exp_name=exp_name, overwrite=overwrite)

    print("[diag0] replay/Mono(e_J)")
    paths["Mono(e_J)"] = cmp_mono.run(
        cell, ctx, anchor_source="literal",
        exp_name=exp_name, overwrite=overwrite,
    )

    print("[diag0] replay/oracle teacher_residual λ=1")
    paths["oracle λ=1"] = cmp_teacher_residual.run(
        cell, ctx,
        lambda_schedule="constant", lambda_max=1.0, correction_window=None,
        save_residuals=True, save_x0_estimates=True, save_trajectory=True,
        exp_name=exp_name, overwrite=overwrite,
    )

    if _m2a_ckpt_path().exists():
        print("[diag0] replay/2a-diag0 (residual_prompt)")
        paths["2a-diag0"] = cmp_residual_prompt.run(
            cell, ctx,
            window_frac=0.4, lambda_max=1.0,
            pstar_ckpt=str(_m2a_ckpt_path()),
            correction_max_rel_norm=None,
            exp_name=exp_name, overwrite=overwrite,
        )
    else:
        print(f"[diag0] replay/2a-diag0 SKIP (no checkpoint at {_m2a_ckpt_path()})")

    if _m2b_ckpt_path().exists():
        print("[diag0] replay/2b-diag0 (direct_eps)")
        paths["2b-diag0"] = cmp_direct_eps.run(
            cell, ctx,
            window_frac=0.4, lambda_max=1.0,
            student_ckpt=str(_m2b_ckpt_path()),
            exp_name=exp_name, overwrite=overwrite,
        )
    else:
        print(f"[diag0] replay/2b-diag0 SKIP (no checkpoint at {_m2b_ckpt_path()})")
    return paths


# ---------------------------------------------------------------------------
# Stage 4 — CLIP-on-Tweedie scoring (reuses idea5a machinery)
# ---------------------------------------------------------------------------


@torch.no_grad()
def stage_score(
    *,
    ctx: MethodCtx,
    replay_paths: dict[str, Path],
    basin_step: int,
) -> dict:
    """Score each replay output's *final image* with CLIP against three text
    targets: 'a cat', 'a dog', 'a cat and a dog'. Final-image scoring (rather
    than Tweedie at the basin step) is what we ultimately care about — did
    the deployed trajectory land in the joint basin?

    For the oracle-λ=1 run we also have per-step residuals saved, so we
    additionally score Tweedie x̂_0 at ``basin_step`` for it as a sanity
    reference (it should already match the joint target by then).
    """
    from poe_repair.experiments.idea5a.analyse import (
        cache_text_embeddings, clip_score_image,
    )
    from poe_repair.runtime import decode_latents, tweedie_mean

    targets = ("a cat", "a dog", "a cat and a dog")
    text_embeds = cache_text_embeddings(targets, device=ctx.device)

    out: dict = {"text_targets": list(targets), "by_method": {}}
    for label, png_path in replay_paths.items():
        # Score the final image PNG.
        pil = Image.open(png_path).convert("RGB")
        img = (
            torch.from_numpy(
                __import__("numpy").asarray(pil, dtype="float32") / 255.0
            ).permute(2, 0, 1).unsqueeze(0)
        )
        scores = clip_score_image(img, text_embeds, device=ctx.device)
        out["by_method"][label] = {
            "final_image_path": str(png_path),
            "clip_scores_final": dict(zip(targets, scores)),
        }
        print(
            f"[diag0] CLIP {label}: "
            + "  ".join(f"{t[:14]}={s:+.3f}" for t, s in zip(targets, scores))
        )

    return out


# ---------------------------------------------------------------------------
# Stage 5 — figures + summary
# ---------------------------------------------------------------------------


def stage_summarise(
    *,
    out_root: Path,
    replay_paths: dict[str, Path],
    clip_scores: dict,
    history_2a: dict | None,
    history_2b: dict | None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = ensure_dir(out_root / "figures")

    # --- Loss curves ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, hist, title in (
        (axes[0], history_2a, "M2a (residual_prompt) loss"),
        (axes[1], history_2b, "M2b (direct_eps) loss"),
    ):
        if hist and "history" in hist and hist["history"]:
            steps = [h["step"] for h in hist["history"]]
            losses = [h.get("loss", float("nan")) for h in hist["history"]]
            ax.plot(steps, losses, color="tab:blue")
            ax.set_xlabel("training step")
            ax.set_ylabel("loss")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "no history", ha="center", va="center")
            ax.set_title(title)
    fig.suptitle("Diagnostic 0 — single-cell training loss")
    fig.tight_layout()
    loss_png = fig_dir / "loss_curves.png"
    fig.savefig(loss_png, dpi=120)
    plt.close(fig)

    # --- Replay grid (5 panels, omit any missing) ---
    desired = ["PoE", "Mono(e_J)", "oracle λ=1", "2a-diag0", "2b-diag0"]
    labels = [lab for lab in desired if lab in replay_paths]
    fig, axes = plt.subplots(1, len(labels), figsize=(3 * len(labels), 3.4))
    if len(labels) == 1:
        axes = [axes]
    for ax, label in zip(axes, labels):
        path = replay_paths.get(label)
        if path is not None and Path(path).exists():
            ax.imshow(np.asarray(Image.open(path).convert("RGB")))
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    fig.suptitle(
        f"Diagnostic 0 — single-cell replay  ({DEFAULT_PAIR[0]} × {DEFAULT_PAIR[1]}, seed {DEFAULT_SEED})"
    )
    fig.tight_layout()
    replay_png = fig_dir / "replay_grid.png"
    fig.savefig(replay_png, dpi=120)
    plt.close(fig)

    # --- CLIP bars ---
    targets = clip_scores["text_targets"]
    method_labels = list(clip_scores["by_method"].keys())
    n_methods = len(method_labels)
    n_targets = len(targets)
    bar_w = 0.8 / n_methods
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(n_targets)
    for i, label in enumerate(method_labels):
        scores = [clip_scores["by_method"][label]["clip_scores_final"][t] for t in targets]
        ax.bar(x + i * bar_w - 0.4 + bar_w / 2, scores, bar_w, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontsize=9)
    ax.set_ylabel("CLIP cosine (final image)")
    ax.set_title("Diagnostic 0 — CLIP scores on final images")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    bars_png = fig_dir / "clip_bars.png"
    fig.savefig(bars_png, dpi=120)
    plt.close(fig)

    # --- Verdict heuristic ---
    joint_target = "a cat and a dog"
    by = clip_scores["by_method"]
    poe_joint = by["PoE"]["clip_scores_final"][joint_target]
    mono_joint = by["Mono(e_J)"]["clip_scores_final"][joint_target]
    gap_denom = max(1e-6, mono_joint - poe_joint)
    m2a_joint = by["2a-diag0"]["clip_scores_final"][joint_target] if "2a-diag0" in by else None
    m2b_joint = by["2b-diag0"]["clip_scores_final"][joint_target] if "2b-diag0" in by else None
    gap_2a = (m2a_joint - poe_joint) / gap_denom if m2a_joint is not None else None
    gap_2b = (m2b_joint - poe_joint) / gap_denom if m2b_joint is not None else None

    def _verdict(label: str, hist: dict | None, gap: float | None) -> str:
        if gap is None:
            return f"- **{label}**: not run (no checkpoint)."
        last_loss = None
        if hist and hist.get("history"):
            last_loss = hist["history"][-1].get("loss")
        loss_str = f"{last_loss:.4f}" if isinstance(last_loss, (int, float)) else "n/a"
        if last_loss is None:
            return f"- **{label}**: no training history found (gap={gap*100:.0f}%)."
        if last_loss > 0.05:
            return (
                f"- **{label}**: training loss flatlined at {loss_str} (high) → "
                "model class can't fit a single cell. Architecture/loss problem."
            )
        if gap < 0.3:
            return (
                f"- **{label}**: loss reached {loss_str} but replay closed only "
                f"{gap*100:.0f}% of the PoE→Mono gap → eps-MSE has solutions "
                "that minimise loss without flipping the basin. **Objective is the bug.**"
            )
        return (
            f"- **{label}**: loss reached {loss_str} and replay closed {gap*100:.0f}% "
            "of the PoE→Mono gap → pipeline works at the simplest scale. "
            "Failure of M2a/M2b at scale is generalisation, not memorisation."
        )

    verdict_2a = _verdict("M2a-diag0", history_2a, gap_2a)
    verdict_2b = _verdict("M2b-diag0", history_2b, gap_2b)

    summary_md = (
        f"# Diagnostic 0 — single-cell overfit + replay\n\n"
        f"Pair: **{DEFAULT_PAIR[0]} × {DEFAULT_PAIR[1]}** · Seed: **{DEFAULT_SEED}**\n\n"
        f"## CLIP scores on final images (joint target = '{joint_target}')\n\n"
        f"| Method | a cat | a dog | a cat and a dog |\n"
        f"|---|---:|---:|---:|\n"
    )
    for label in method_labels:
        s = clip_scores["by_method"][label]["clip_scores_final"]
        summary_md += (
            f"| {label} | {s['a cat']:+.3f} | {s['a dog']:+.3f} | {s[joint_target]:+.3f} |\n"
        )
    gap_2a_str = f"{gap_2a*100:.0f}%" if gap_2a is not None else "n/a (not run)"
    gap_2b_str = f"{gap_2b*100:.0f}%" if gap_2b is not None else "n/a (not run)"
    summary_md += (
        f"\nPoE→Mono gap: **{mono_joint - poe_joint:+.3f}**\n\n"
        f"- 2a-diag0 closed **{gap_2a_str}** of the gap.\n"
        f"- 2b-diag0 closed **{gap_2b_str}** of the gap.\n\n"
        f"## Verdicts\n\n"
        f"{verdict_2a}\n\n{verdict_2b}\n\n"
        f"## Artifacts\n\n"
        f"- Loss curves: `figures/loss_curves.png`\n"
        f"- Replay grid: `figures/replay_grid.png`\n"
        f"- CLIP bars: `figures/clip_bars.png`\n"
        f"- Raw scores: `metrics.json`\n"
    )
    (out_root / "SUMMARY.md").write_text(summary_md)
    print("\n" + "=" * 70)
    print(summary_md)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m2a-steps", type=int, default=DEFAULT_M2A_STEPS)
    ap.add_argument("--m2b-steps", type=int, default=DEFAULT_M2B_STEPS)
    ap.add_argument("--basin-step", type=int, default=DEFAULT_BASIN_STEP)
    ap.add_argument("--force-train-2a", action="store_true")
    ap.add_argument("--force-train-2b", action="store_true")
    ap.add_argument("--skip-train-2a", action="store_true")
    ap.add_argument("--skip-train-2b", action="store_true")
    ap.add_argument("--skip-replay", action="store_true")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--skip-summarise", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = RunConfig()
    out_root = ensure_dir(cfg.paths.output_root / EXP_NAME)
    metrics_path = out_root / "metrics.json"

    # 1+2: train.
    if not args.skip_train_2a:
        stage_train_2a(num_steps=args.m2a_steps, force=args.force_train_2a)
    if not args.skip_train_2b:
        stage_train_2b(num_steps=args.m2b_steps, force=args.force_train_2b)

    # Need ctx for replay/score.
    ctx: MethodCtx | None = None
    replay_paths: dict[str, Path] = {}
    if not args.skip_replay or not args.skip_score:
        ctx = make_ctx()

    # 3: replay.
    if not args.skip_replay:
        assert ctx is not None
        replay_paths = stage_replay(
            ctx=ctx, exp_name=EXP_NAME, overwrite=args.overwrite,
        )
        write_json(out_root / "replay_paths.json", {k: str(v) for k, v in replay_paths.items()})
    else:
        rp_json = out_root / "replay_paths.json"
        if rp_json.exists():
            replay_paths = {k: Path(v) for k, v in json.loads(rp_json.read_text()).items()}

    # 4: score.
    clip_scores: dict = {}
    if not args.skip_score:
        assert ctx is not None
        if not replay_paths:
            raise RuntimeError("score requested but no replay_paths available")
        clip_scores = stage_score(
            ctx=ctx, replay_paths=replay_paths, basin_step=args.basin_step,
        )
        write_json(metrics_path, clip_scores)
    elif metrics_path.exists():
        clip_scores = json.loads(metrics_path.read_text())

    # 5: summarise.
    if not args.skip_summarise:
        history_2a = None
        history_2b = None
        h2a = _m2a_history_path()
        h2b = _m2b_history_path()
        if h2a.exists():
            history_2a = json.loads(h2a.read_text())
        if h2b.exists():
            history_2b = json.loads(h2b.read_text())
        stage_summarise(
            out_root=out_root,
            replay_paths=replay_paths,
            clip_scores=clip_scores,
            history_2a=history_2a,
            history_2b=history_2b,
        )

    print(f"[diag0] done — outputs under {out_root}")


if __name__ == "__main__":
    main()
