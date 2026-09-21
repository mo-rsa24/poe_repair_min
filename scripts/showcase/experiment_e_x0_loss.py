#!/usr/bin/env python
"""Experiment E readout (scope 01 plan 16): does training the rank-32 adapter on the
clean-estimate residual, instead of the noise residual, give a crisper held-out fix?

The bars live here, in source, so they cannot move after the answer is seen. They were written
on 2026-09-06 before the run started, from the rank-32 baseline numbers in
report/does-training-longer-help-the-pooled-lora/ and from the contrast read of the baseline's
renders (grey-level standard deviation, 8 seeds at lambda 1.0: 37.5 against the joint prompt's
46.4 and plain PoE's 40.6 on the 1024 px renders).

Stages (each is idempotent and reads what the earlier ones wrote):
  --frames    per-step running estimates of the new adapter at lambda 1.0 (GPU; the baseline's
              frames already exist under where_each_condition_lands/frames/)
  --figures   training curves (W&B), contrast over steps, where each condition lands in
              DINOv2 space, both-ness over steps, checkpoint bars
  --strips    Mono | plain PoE | baseline adapter | new adapter, one strip per seed and one sheet
  --verdict   the bars against results.json; verdict.json and cell-table.md
  --wandb     everything above into one W&B run

Every output lands under OUT_ROOT on /datasets and is copied into RESULTS in the repo.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SHOWCASE = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase")
TRAINING_CACHE = Path("/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache")
NEW_RUN_ID = "phase1_r32_x0loss_40k"
NEW_RUN_DIR = SHOWCASE / NEW_RUN_ID
BASELINE_RUN_WANDB = "prime_lab/poe-repair-animals-compose/6xc2l8ix"
WANDB_PROJECT = "prime_lab/poe-repair-animals-compose"
BASELINE_PROBE = SHOWCASE / "figure_r32_030050"           # rank 32, noise-space loss, step 30,050
NEW_PROBE = {30000: SHOWCASE / "figure_r32_x0loss_030000", 40000: SHOWCASE / "figure_r32_x0loss_040000"}
NEW_CKPT_30K = NEW_RUN_DIR / "checkpoints" / "lora_step_030000.pt"
LANDS = SHOWCASE / "where_each_condition_lands"
FRAMES_BASE = LANDS / "frames"                            # solo_a, solo_b, joint, poe, lora_1.0 (baseline)
FRAMES_NEW = LANDS / "frames_x0loss"                      # lora_1.0 (new adapter)
OUT_ROOT = SHOWCASE / "experiment_e_x0_loss"
RESULTS = REPO_ROOT / "artifacts/results/does-training-on-the-clean-estimate-residual-sharpen-the-fix"
SEEDS = (9, 10, 11, 12, 13, 14, 15, 16)
FRAME_STEPS = (0, 2, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 49, 50)
LAMBDA = "1.0"

# --- bars, fixed before the run ------------------------------------------------------------
# Baseline rank 32 at 30,050, lambda 1.0, all 50 steps: 8-seed DINOv2 drift -0.091, 7 of 8 composing.
BASELINE_DRIFT_30K = -0.091
BASELINE_COMPOSE_30K = 7
# The smallest drift gap the baseline grids told apart (30k against 60k) is 0.03.
DRIFT_MARGIN = 0.03
# Question 1 (moves the plan), read at 30k: support if drift is at or below the baseline minus the
# margin; null if at or above the baseline plus the margin; inconclusive between.
DRIFT_SUPPORT = BASELINE_DRIFT_30K - DRIFT_MARGIN     # -0.121
DRIFT_NULL = BASELINE_DRIFT_30K + DRIFT_MARGIN        # -0.061
# The read is only valid if the new adapter still composes.
MIN_COMPOSE_COUNT = 6
# Question 2, the contrast read on the 1024 px renders: closure of the gap between the baseline
# adapter's mean grey-level standard deviation and the joint prompt's, both computed here from
# the files. Support at half the gap closed, null at a tenth or less.
CONTRAST_SUPPORT_CLOSURE = 0.5
CONTRAST_NULL_CLOSURE = 0.1

CONDITION_LABELS = {
    "solo_a": "a cat alone", "solo_b": "a dog alone", "joint": "\"a cat and a dog\"",
    "poe": "PoE, no correction", "lora_base": "PoE + baseline adapter (noise loss, 30,050)",
    "lora_new": "PoE + new adapter (clean-estimate loss, 30,000)",
}
COLORS = {"solo_a": "#d95f02", "solo_b": "#1b9e77", "joint": "#7570b3", "poe": "#111111",
          "lora_base": "#888888", "lora_new": "#d7191c"}

# Smoke mode: EXPERIMENT_E_SMOKE=<scratch dir> reads the baseline's probe folder and frames as a
# stand-in for the new adapter's, so every stage runs end to end before the real run exists.
import os as _os  # noqa: E402
if _os.environ.get("EXPERIMENT_E_SMOKE"):
    NEW_PROBE = {30000: BASELINE_PROBE, 40000: BASELINE_PROBE}
    FRAMES_NEW = FRAMES_BASE
    OUT_ROOT = Path(_os.environ["EXPERIMENT_E_SMOKE"])
    RESULTS = OUT_ROOT / "results_copy"


def _log(msg: str) -> None:
    print(f"[experiment_e] {msg}", flush=True)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _grey_std(path: Path) -> float:
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return float(g.std())


def _copy_to_results(*paths: Path) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for p in paths:
        if p.exists():
            shutil.copy2(p, RESULTS / p.name)


# ------------------------------------------------------------------ paths of the four conditions
def final_render(cond: str, seed: int) -> Path:
    if cond in ("solo_a", "solo_b", "joint"):
        return LANDS / cond / f"seed_{seed}.png"
    if cond == "mono":
        return TRAINING_CACHE / "heldout" / "a_cat__x__a_dog" / f"seed_{seed}" / "mono.png"
    if cond == "poe":
        return BASELINE_PROBE / "renders" / "full" / f"seed_{seed}_lambda_0.0.png"
    if cond == "lora_base":
        return BASELINE_PROBE / "renders" / "full" / f"seed_{seed}_lambda_{LAMBDA}.png"
    if cond == "lora_new":
        return NEW_PROBE[30000] / "renders" / "full" / f"seed_{seed}_lambda_{LAMBDA}.png"
    raise KeyError(cond)


def frame(cond: str, seed: int, step: int) -> Path:
    root = FRAMES_NEW if cond == "lora_new" else FRAMES_BASE
    name = {"lora_base": "lora_1.0", "lora_new": "lora_1.0"}.get(cond, cond)
    return root / name / f"seed_{seed}" / f"step_{step:03d}.png"


def _rows(results_json: Path) -> dict[tuple[int, str], dict]:
    d = json.loads(results_json.read_text())
    return {(int(r["seed"]), str(r["lambda"])): r for r in d["rows"] if r["window"] == "full"}


# ------------------------------------------------------------------------------------ --frames
def stage_frames() -> None:
    """Per-step running estimates of the new adapter at lambda 1.0, same renderer and frame steps
    as the baseline's frames, so the two are read on one axis."""
    import where_each_condition_lands_trajectories as W
    if not NEW_CKPT_30K.exists():
        raise SystemExit(f"no checkpoint at {NEW_CKPT_30K}")
    W.CHECKPOINT = NEW_CKPT_30K
    W.LORA_RANK = 32
    FRAMES_NEW.mkdir(parents=True, exist_ok=True)
    W.main(["--out-root", str(FRAMES_NEW), "--conditions", "lora_1.0",
            "--seeds", ",".join(str(s) for s in SEEDS)])
    _log(f"frames of the new adapter under {FRAMES_NEW}")


# ----------------------------------------------------------------------------------- embedding
def _embedder():
    from scripts.build_lora_inspector_mds_semantic import DinoEmbedder
    return DinoEmbedder(device=_device())


def _embed(embedder, paths: list[Path]) -> np.ndarray:
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        ims.append(torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1))
    feats = []
    for i in range(0, len(ims), 8):
        feats.append(embedder.embed_decoded_batch(torch.stack(ims[i:i + 8])))
    return np.concatenate(feats)


def _axes(feats: dict[str, np.ndarray]) -> dict:
    """x: unit direction from the cat-alone centroid to the dog-alone centroid; y: unit direction
    from the midpoint of those two toward the joint-prompt centroid, orthogonalised against x.
    Both-ness is the y coordinate. Same construction as the landing finding's second view."""
    ca, cb, cj = feats["solo_a"].mean(0), feats["solo_b"].mean(0), feats["joint"].mean(0)
    origin = 0.5 * (ca + cb)
    x = cb - ca; x = x / np.linalg.norm(x)
    y = cj - origin; y = y - (y @ x) * x; y = y / np.linalg.norm(y)
    return {"origin": origin, "x": x, "y": y, "joint_both_ness": float((cj - origin) @ y)}


def _project(ax: dict, f: np.ndarray) -> np.ndarray:
    d = f - ax["origin"]
    return np.stack([d @ ax["x"], d @ ax["y"]], axis=-1)


# ----------------------------------------------------------------------------------- --figures
def fig_training_curves() -> Path | None:
    """train/loss_eps (the unweighted noise-space MSE, comparable across the two runs) and the
    held-out tracking drift against optimizer step, baseline and new run."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import wandb
        api = wandb.Api()
        base = api.run(BASELINE_RUN_WANDB)
        new = None
        for r in api.runs(WANDB_PROJECT, filters={"display_name": NEW_RUN_ID}):
            new = r
        if new is None:
            for r in api.runs(WANDB_PROJECT, filters={"config.run_id": NEW_RUN_ID}):
                new = r
        if new is None:
            _log("training curves: the new run is not on W&B yet; skipping this figure")
            return None
    except Exception as exc:  # noqa: BLE001
        _log(f"training curves: W&B unavailable ({type(exc).__name__}: {exc}); skipping this figure")
        return None

    def hist(run, keys):
        df = run.history(keys=["train/optimizer_step", *keys], samples=20000, pandas=True)
        return df

    def ema(x, a=0.02):
        out, m = [], None
        for v in x:
            m = v if m is None else (1 - a) * m + a * v
            out.append(m)
        return np.array(out)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    for run, label, color in ((base, "baseline: noise-space loss (6xc2l8ix)", COLORS["lora_base"]),
                              (new, f"new: clean-estimate loss ({new.id})", COLORS["lora_new"])):
        key = "train/loss_eps" if "train/loss_eps" in run.summary.keys() or run is new else "train/loss"
        try:
            df = hist(run, [key]).dropna()
            axs[0].plot(df["train/optimizer_step"], ema(df[key].values), color=color, label=f"{label}, {key}")
        except Exception as exc:  # noqa: BLE001
            _log(f"training curves: {label} loss history failed ({exc})")
        try:
            df = hist(run, []).copy()
            cols = [c for c in run.history(samples=20000, pandas=True).columns
                    if c.startswith("eval/tracking/embedding_drift/dino/out_out")]
            if cols:
                d2 = run.history(keys=["train/optimizer_step", *cols], samples=20000, pandas=True).dropna()
                axs[1].plot(d2["train/optimizer_step"], d2[cols].mean(axis=1), "o-", color=color, label=label)
        except Exception as exc:  # noqa: BLE001
            _log(f"training curves: {label} tracking history failed ({exc})")
    axs[0].set_xlabel("optimizer step"); axs[0].set_ylabel("noise-space MSE, EMA 0.02")
    axs[0].set_title("training loss measured the same way on both runs"); axs[0].legend(fontsize=8)
    axs[0].set_xlim(0, 45000)
    axs[1].axhline(0, color="grey", lw=0.6); axs[1].set_xlim(0, 45000)
    axs[1].set_xlabel("optimizer step"); axs[1].set_ylabel("DINOv2 drift of the 2 held-out tracking cells\n(negative = nearer the joint-prompt image)")
    axs[1].set_title("held-out drift while training (cat × dog seeds 9, 10)"); axs[1].legend(fontsize=8)
    fig.tight_layout()
    out = OUT_ROOT / "figures" / "training-curves.png"
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def fig_contrast_over_steps() -> tuple[Path, dict]:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = ["joint", "poe", "lora_base", "lora_new"]
    table = {c: {s: [] for s in SEEDS} for c in conds}
    for c in conds:
        for s in SEEDS:
            for k in FRAME_STEPS:
                p = frame(c, s, k)
                table[c][s].append(_grey_std(p) if p.exists() else float("nan"))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c in conds:
        arr = np.array([table[c][s] for s in SEEDS])
        for row in arr:
            ax.plot(FRAME_STEPS, row, color=COLORS[c], lw=0.6, alpha=0.35)
        ax.plot(FRAME_STEPS, np.nanmean(arr, 0), color=COLORS[c], lw=2.2, label=CONDITION_LABELS[c])
    ax.set_xlabel("DDIM step (50-step run)"); ax.set_ylabel("grey-level standard deviation of the 256 px running estimate")
    ax.set_title("contrast of the running estimate over the run, thin per seed, thick mean of 8")
    ax.legend(fontsize=8); fig.tight_layout()
    out = OUT_ROOT / "figures" / "contrast-over-steps.png"
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=140); plt.close(fig)
    summary = {c: {str(k): round(float(v), 2) for k, v in zip(FRAME_STEPS, np.nanmean(np.array([table[c][s] for s in SEEDS]), 0))}
               for c in conds}
    (OUT_ROOT / "figures" / "contrast-over-steps.json").write_text(json.dumps({"mean_over_seeds": summary, "per_seed": {c: {str(s): table[c][s] for s in SEEDS} for c in conds}}, indent=1))
    return out, summary


def fig_where_it_lands(embedder) -> tuple[Path, dict]:
    """Final renders of every condition embedded with the scorer's DINOv2 encoder and projected
    onto the cat-to-dog axis (x) and the both-ness axis (y) fitted on the three reference clouds."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    conds = ["solo_a", "solo_b", "joint", "poe", "lora_base", "lora_new"]
    feats = {c: _embed(embedder, [final_render(c, s) for s in SEEDS]) for c in conds}
    ax_ = _axes(feats)
    xy = {c: _project(ax_, feats[c]) for c in conds}
    fig, ax = plt.subplots(figsize=(9.5, 8))
    for c in ("solo_a", "solo_b", "joint"):
        pts = xy[c]; hull = ConvexHull(pts); poly = pts[hull.vertices]
        ax.fill(poly[:, 0], poly[:, 1], color=COLORS[c], alpha=0.12, lw=0)
        ax.scatter(pts[:, 0], pts[:, 1], s=40, color=COLORS[c], label=CONDITION_LABELS[c], zorder=3)
        ax.annotate(CONDITION_LABELS[c], pts.mean(0), color=COLORS[c], fontsize=10, ha="center", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85), zorder=4)
    for i, s in enumerate(SEEDS):
        p, b, n = xy["poe"][i], xy["lora_base"][i], xy["lora_new"][i]
        ax.annotate("", xy=b, xytext=p, arrowprops=dict(arrowstyle="-|>", color=COLORS["lora_base"], lw=1.0), zorder=5)
        ax.annotate("", xy=n, xytext=p, arrowprops=dict(arrowstyle="-|>", color=COLORS["lora_new"], lw=1.4), zorder=5)
        ax.annotate(f"s{s}", p, fontsize=8, color=COLORS["poe"], xytext=(4, -9), textcoords="offset points")
    ax.scatter(xy["poe"][:, 0], xy["poe"][:, 1], s=60, color=COLORS["poe"], marker="s", label=CONDITION_LABELS["poe"], zorder=6)
    ax.scatter(xy["lora_base"][:, 0], xy["lora_base"][:, 1], s=60, color=COLORS["lora_base"], marker="o", label=CONDITION_LABELS["lora_base"], zorder=6)
    ax.scatter(xy["lora_new"][:, 0], xy["lora_new"][:, 1], s=70, color=COLORS["lora_new"], marker="D", label=CONDITION_LABELS["lora_new"], zorder=7)
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("cat-alone centroid  ←  x (cosine units)  →  dog-alone centroid")
    ax.set_ylabel("both-ness: 0 at the solo midpoint, up toward the \"a cat and a dog\" centroid")
    ax.set_title("cat × dog, held-out seeds 9 to 16: where each final render lands in DINOv2 space\n"
                 "axes fitted on the three reference clouds; arrows: plain PoE → adapter at λ 1.0", fontsize=10)
    ax.legend(loc="best", fontsize=8, framealpha=0.9); fig.tight_layout()
    out = OUT_ROOT / "figures" / "where-each-condition-lands.png"
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=150); plt.close(fig)
    summary = {c: {"both_ness_mean": round(float(xy[c][:, 1].mean()), 4), "x_mean": round(float(xy[c][:, 0].mean()), 4),
                   "per_seed_both_ness": {str(s): round(float(xy[c][i, 1]), 4) for i, s in enumerate(SEEDS)}} for c in conds}
    summary["joint_centroid_both_ness"] = round(ax_["joint_both_ness"], 4)
    (OUT_ROOT / "figures" / "where-each-condition-lands.json").write_text(json.dumps(summary, indent=1))
    np.savez(OUT_ROOT / "figures" / "final-render-dino-feats.npz", **feats)
    return out, summary


def fig_both_ness_over_steps(embedder) -> tuple[Path, dict]:
    """The per-step running estimates projected onto the same two axes, fitted here on the step-50
    frames of the three references (256 px), so every point is motion in one fixed plane."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    refs = {c: _embed(embedder, [frame(c, s, 50) for s in SEEDS]) for c in ("solo_a", "solo_b", "joint")}
    ax_ = _axes(refs)
    conds = ["joint", "poe", "lora_base", "lora_new"]
    both = {c: np.full((len(SEEDS), len(FRAME_STEPS)), np.nan) for c in conds}
    for c in conds:
        paths, where = [], []
        for i, s in enumerate(SEEDS):
            for j, k in enumerate(FRAME_STEPS):
                p = frame(c, s, k)
                if p.exists():
                    paths.append(p); where.append((i, j))
        if paths:
            f = _project(ax_, _embed(embedder, paths))
            for (i, j), v in zip(where, f):
                both[c][i, j] = v[1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c in conds:
        for row in both[c]:
            ax.plot(FRAME_STEPS, row, color=COLORS[c], lw=0.6, alpha=0.35)
        ax.plot(FRAME_STEPS, np.nanmean(both[c], 0), color=COLORS[c], lw=2.2, label=CONDITION_LABELS[c])
    ax.axhline(ax_["joint_both_ness"], color=COLORS["joint"], ls="--", lw=0.8)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("DDIM step (50-step run)"); ax.set_ylabel("both-ness of the running estimate\n(0 solo midpoint, dashed: joint-prompt centroid)")
    ax.set_title("how each run moves toward \"a cat and a dog\" over the 50 steps, thin per seed, thick mean of 8")
    ax.legend(fontsize=8); fig.tight_layout()
    out = OUT_ROOT / "figures" / "both-ness-over-steps.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    summary = {c: {str(k): round(float(v), 4) for k, v in zip(FRAME_STEPS, np.nanmean(both[c], 0))} for c in conds}
    (OUT_ROOT / "figures" / "both-ness-over-steps.json").write_text(json.dumps({"mean_over_seeds": summary, "joint_centroid_both_ness": round(ax_["joint_both_ness"], 4)}, indent=1))
    return out, summary


def fig_checkpoint_bars() -> tuple[Path, dict]:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = [("baseline 30,050\nnoise loss", BASELINE_PROBE)] + [(f"new {k:,}\nclean-estimate loss", v) for k, v in NEW_PROBE.items()]
    stats = []
    for label, root in cells:
        rj = root / "results.json"
        if not rj.exists():
            stats.append({"label": label, "missing": True}); continue
        d = json.loads(rj.read_text())["summary"]["full"][LAMBDA]
        rows = _rows(rj)
        contrast = float(np.mean([_grey_std(Path(rows[(s, LAMBDA)]["image_path"])) for s in SEEDS]))
        stats.append({"label": label, "compose_count": int(round(d["compose_rate"] * d["n"])), "n": d["n"],
                      "mean_dino_drift": round(d["mean_dino_drift"], 4), "contrast_mean": round(contrast, 2)})
    joint_contrast = float(np.mean([_grey_std(final_render("joint", s)) for s in SEEDS]))
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    labels = [s["label"] for s in stats]; xs = np.arange(len(stats))
    colors = [COLORS["lora_base"]] + [COLORS["lora_new"]] * (len(stats) - 1)
    axs[0].bar(xs, [s.get("compose_count", 0) for s in stats], color=colors); axs[0].set_ylim(0, 8)
    axs[0].axhline(MIN_COMPOSE_COUNT, color="k", ls="--", lw=0.8, label=f"read valid at ≥ {MIN_COMPOSE_COUNT}")
    axs[0].set_title("seeds composing, of 8 (detector count ≥ 2)"); axs[0].legend(fontsize=8)
    axs[1].bar(xs, [s.get("mean_dino_drift", 0) for s in stats], color=colors)
    axs[1].axhline(DRIFT_SUPPORT, color="green", ls="--", lw=0.8, label=f"support ≤ {DRIFT_SUPPORT:.3f}")
    axs[1].axhline(DRIFT_NULL, color="red", ls="--", lw=0.8, label=f"null ≥ {DRIFT_NULL:.3f}")
    axs[1].set_title("8-seed DINOv2 drift at λ 1.0\n(negative = nearer the joint-prompt image)"); axs[1].legend(fontsize=8)
    axs[2].bar(xs, [s.get("contrast_mean", 0) for s in stats], color=colors)
    axs[2].axhline(joint_contrast, color=COLORS["joint"], ls="--", lw=0.8, label=f"joint prompt {joint_contrast:.1f}")
    axs[2].set_title("mean grey-level std of the 1024 px renders at λ 1.0"); axs[2].legend(fontsize=8)
    for a in axs:
        a.set_xticks(xs); a.set_xticklabels(labels, fontsize=8)
    fig.tight_layout()
    out = OUT_ROOT / "figures" / "checkpoint-bars.png"
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=140); plt.close(fig)
    summary = {"cells": stats, "joint_contrast_mean": round(joint_contrast, 2)}
    (OUT_ROOT / "figures" / "checkpoint-bars.json").write_text(json.dumps(summary, indent=1))
    return out, summary


def stage_figures() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    made = []
    p = fig_training_curves()
    if p: made.append(p)
    p, _ = fig_contrast_over_steps(); made.append(p)
    embedder = _embedder()
    p, _ = fig_where_it_lands(embedder); made.append(p)
    p, _ = fig_both_ness_over_steps(embedder); made.append(p)
    p, _ = fig_checkpoint_bars(); made.append(p)
    _copy_to_results(*made, *[m.with_suffix(".json") for m in made if m.with_suffix(".json").exists()])
    _log("figures: " + ", ".join(str(m) for m in made))


# ------------------------------------------------------------------------------------ --strips
def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


STRIP_COLS = (("mono", "Mono: \"a cat and a dog\""), ("poe", "plain PoE (λ 0)"),
              ("lora_base", "baseline adapter, noise loss, 30,050, λ 1"),
              ("lora_new", "new adapter, clean-estimate loss, 30,000, λ 1"))


def stage_strips() -> None:
    base_rows, new_rows = _rows(BASELINE_PROBE / "results.json"), _rows(NEW_PROBE[30000] / "results.json")
    tile, pad, cap_h = 512, 12, 78
    font, small = _font(18), _font(15)
    strips_dir = OUT_ROOT / "strips"; strips_dir.mkdir(parents=True, exist_ok=True)
    sheet_rows = []
    for s in SEEDS:
        w = len(STRIP_COLS) * (tile + pad) + pad
        strip = Image.new("RGB", (w, tile + cap_h + 2 * pad), "white")
        d = ImageDraw.Draw(strip)
        for j, (cond, label) in enumerate(STRIP_COLS):
            p = final_render(cond, s)
            im = Image.open(p).convert("RGB").resize((tile, tile), Image.LANCZOS)
            x0 = pad + j * (tile + pad)
            strip.paste(im, (x0, pad))
            line2 = f"contrast {_grey_std(p):.1f}"
            if cond == "poe":
                r = base_rows[(s, "0.0")]; line2 += f"  |  animals {r['n_instances']}"
            elif cond == "lora_base":
                r = base_rows[(s, LAMBDA)]; line2 += f"  |  animals {r['n_instances']}  |  drift {r['drift']['dino']['drift']:+.3f}"
            elif cond == "lora_new":
                r = new_rows[(s, LAMBDA)]; line2 += f"  |  animals {r['n_instances']}  |  drift {r['drift']['dino']['drift']:+.3f}"
            d.text((x0, tile + pad + 6), label, fill="black", font=font)
            d.text((x0, tile + pad + 34), line2, fill="#333333", font=small)
        d.text((pad, tile + pad + 56), f"seed {s}", fill="#666666", font=small)
        out = strips_dir / f"strip-seed_{s}.png"; strip.save(out); sheet_rows.append(strip)
    sheet = Image.new("RGB", (sheet_rows[0].width, sum(r.height for r in sheet_rows)), "white")
    y = 0
    for r in sheet_rows:
        sheet.paste(r, (0, y)); y += r.height
    sheet.save(OUT_ROOT / "strips" / "sheet-all-seeds.png")
    small_sheet = sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS)
    small_sheet.save(OUT_ROOT / "strips" / "sheet-all-seeds-half.png")
    RESULTS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT_ROOT / "strips" / "sheet-all-seeds-half.png", RESULTS / "mono-vs-poe-vs-adapters-all-seeds.png")
    _log(f"strips under {strips_dir}")


# ----------------------------------------------------------------------------------- --verdict
def stage_verdict() -> dict:
    base = json.loads((BASELINE_PROBE / "results.json").read_text())["summary"]["full"][LAMBDA]
    out = {"bars": {"DRIFT_SUPPORT": DRIFT_SUPPORT, "DRIFT_NULL": DRIFT_NULL, "MIN_COMPOSE_COUNT": MIN_COMPOSE_COUNT,
                    "CONTRAST_SUPPORT_CLOSURE": CONTRAST_SUPPORT_CLOSURE, "CONTRAST_NULL_CLOSURE": CONTRAST_NULL_CLOSURE,
                    "source": "scripts/showcase/experiment_e_x0_loss.py"},
           "baseline_30050": {"compose_count": int(round(base["compose_rate"] * base["n"])), "mean_dino_drift": round(base["mean_dino_drift"], 4)}}
    joint_c = float(np.mean([_grey_std(final_render("joint", s)) for s in SEEDS]))
    base_c = float(np.mean([_grey_std(final_render("lora_base", s)) for s in SEEDS]))
    poe_c = float(np.mean([_grey_std(final_render("poe", s)) for s in SEEDS]))
    out["contrast"] = {"joint_mean": round(joint_c, 2), "baseline_adapter_mean": round(base_c, 2), "plain_poe_mean": round(poe_c, 2)}
    lines = ["| checkpoint | seeds composing of 8 | 8-seed DINOv2 drift at λ 1 | mean contrast at λ 1 | gap closed | verdict |", "|---|---|---|---|---|---|",
             f"| baseline rank 32, noise loss, 30,050 | {out['baseline_30050']['compose_count']} | {out['baseline_30050']['mean_dino_drift']:+.3f} | {base_c:.1f} | 0 | (reference) |"]
    for step, root in NEW_PROBE.items():
        rj = root / "results.json"
        if not rj.exists():
            out[f"new_{step}"] = {"missing": str(rj)}; continue
        d = json.loads(rj.read_text())["summary"]["full"][LAMBDA]
        rows = _rows(rj)
        n = int(round(d["compose_rate"] * d["n"])); drift = float(d["mean_dino_drift"])
        c = float(np.mean([_grey_std(Path(rows[(s, LAMBDA)]["image_path"])) for s in SEEDS]))
        closure = (c - base_c) / (joint_c - base_c) if joint_c != base_c else float("nan")
        if n < MIN_COMPOSE_COUNT:
            q1 = "invalid: composes fewer than the minimum, the fidelity read does not apply"
        elif drift <= DRIFT_SUPPORT:
            q1 = "support"
        elif drift >= DRIFT_NULL:
            q1 = "null"
        else:
            q1 = "inconclusive"
        q2 = "support" if closure >= CONTRAST_SUPPORT_CLOSURE else ("null" if closure <= CONTRAST_NULL_CLOSURE else "inconclusive")
        out[f"new_{step}"] = {"compose_count": n, "mean_dino_drift": round(drift, 4), "contrast_mean": round(c, 2),
                              "contrast_gap_closed": round(closure, 3), "q1_drift": q1, "q2_contrast": q2}
        lines.append(f"| new rank 32, clean-estimate loss, {step:,} | {n} | {drift:+.3f} | {c:.1f} | {closure:.2f} | drift {q1}; contrast {q2} |")
    out["verdict_30000"] = out.get("new_30000", {}).get("q1_drift", "not run")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "verdict.json").write_text(json.dumps(out, indent=1))
    (OUT_ROOT / "cell-table.md").write_text("\n".join(lines) + "\n")
    _copy_to_results(OUT_ROOT / "verdict.json", OUT_ROOT / "cell-table.md")
    _log("verdict: " + json.dumps({k: v for k, v in out.items() if k.startswith("new_") or k == "verdict_30000"}))
    return out


# ------------------------------------------------------------------------------------- --wandb
def stage_wandb() -> None:
    import wandb
    entity, project = WANDB_PROJECT.split("/")
    idf = OUT_ROOT / "wandb_run_id.txt"
    run_id = idf.read_text().strip() if idf.exists() else None
    run = wandb.init(entity=entity, project=project, name="experiment_e_x0_loss_readout", id=run_id, resume="allow",
                     tags=["plan-16", "scope-01", "experiment-e", "clean-estimate-loss"],
                     config={"new_run_id": NEW_RUN_ID, "baseline": BASELINE_RUN_WANDB, "bars": {
                         "DRIFT_SUPPORT": DRIFT_SUPPORT, "DRIFT_NULL": DRIFT_NULL, "MIN_COMPOSE_COUNT": MIN_COMPOSE_COUNT,
                         "CONTRAST_SUPPORT_CLOSURE": CONTRAST_SUPPORT_CLOSURE, "CONTRAST_NULL_CLOSURE": CONTRAST_NULL_CLOSURE}})
    idf.write_text(run.id)
    log = {}
    for name in ("training-curves", "contrast-over-steps", "where-each-condition-lands", "both-ness-over-steps", "checkpoint-bars"):
        p = OUT_ROOT / "figures" / f"{name}.png"
        if p.exists():
            log[f"figures/{name}"] = wandb.Image(str(p))
    for s in SEEDS:
        p = OUT_ROOT / "strips" / f"strip-seed_{s}.png"
        if p.exists():
            log[f"strips/seed_{s}"] = wandb.Image(str(p), caption=" | ".join(lab for _, lab in STRIP_COLS))
    sheet = OUT_ROOT / "strips" / "sheet-all-seeds-half.png"
    if sheet.exists():
        log["strips/sheet_all_seeds"] = wandb.Image(str(sheet))
    vj = OUT_ROOT / "verdict.json"
    if vj.exists():
        v = json.loads(vj.read_text())
        tab = wandb.Table(columns=["checkpoint", "compose_count", "mean_dino_drift", "contrast_mean", "contrast_gap_closed", "q1_drift", "q2_contrast"])
        tab.add_data("baseline 30050", v["baseline_30050"]["compose_count"], v["baseline_30050"]["mean_dino_drift"], v["contrast"]["baseline_adapter_mean"], 0.0, "reference", "reference")
        for step in NEW_PROBE:
            c = v.get(f"new_{step}", {})
            if "compose_count" in c:
                tab.add_data(f"new {step}", c["compose_count"], c["mean_dino_drift"], c["contrast_mean"], c["contrast_gap_closed"], c["q1_drift"], c["q2_contrast"])
        log["verdict_table"] = tab
        run.summary["verdict_30000"] = v.get("verdict_30000")
    # the per-seed strip table: one row per seed, one tile per column
    base_rows, new_rows = _rows(BASELINE_PROBE / "results.json"), _rows(NEW_PROBE[30000] / "results.json")
    cols = ["seed"] + [c for c, _ in STRIP_COLS] + ["animals_base", "animals_new", "drift_base", "drift_new", "contrast_base", "contrast_new"]
    tab = wandb.Table(columns=cols)
    for s in SEEDS:
        tiles = [wandb.Image(Image.open(final_render(c, s)).convert("RGB").resize((384, 384), Image.LANCZOS), caption=lab) for c, lab in STRIP_COLS]
        rb, rn = base_rows[(s, LAMBDA)], new_rows[(s, LAMBDA)]
        tab.add_data(s, *tiles, rb["n_instances"], rn["n_instances"], round(rb["drift"]["dino"]["drift"], 3), round(rn["drift"]["dino"]["drift"], 3),
                     round(_grey_std(final_render("lora_base", s)), 1), round(_grey_std(final_render("lora_new", s)), 1))
    log["strips_by_seed"] = tab
    run.log(log)
    _log(f"wandb readout logged to run {run.id} url {run.url}")
    run.finish()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    for flag in ("frames", "figures", "strips", "verdict", "wandb"):
        ap.add_argument(f"--{flag}", action="store_true")
    args = ap.parse_args(argv)
    if args.frames:
        stage_frames()
    if args.figures:
        stage_figures()
    if args.strips:
        stage_strips()
    if args.verdict:
        stage_verdict()
    if args.wandb:
        stage_wandb()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
