#!/usr/bin/env python
"""Readout of plan 22, scope 01: does charging the adapter for its energy sharpen the fix?

Three arms resumed from the shipped rank-32 checkpoint (step 30,050) for 10,000 steps with the
energy penalty beta in {0, 0.01, 0.05}, read at 40,050 against each other and against the
baseline lineage's own 30,050 and 40,050. The bars live here, in source, written on 2026-09-06
before any arm had trained.

Stages (each idempotent; each reads what the earlier ones wrote):
  --baseline  the baseline's own 40,050 8-seed grid, and the on-policy energy read of the
              baseline at 30,050 and 40,050 (GPU); skipped where the outputs exist
  --figures   training curves (W&B), mismatch against energy, where each condition lands in the
              DINOv2 plane, the tracks in that plane, both-ness over steps, checkpoint bars
  --strips    Mono | plain PoE | shipped 30,050 | control 40,050 | beta 0.01 | beta 0.05, one
              strip per seed and one sheet
  --verdict   the bars against the probe folders and the on-policy sidecars; verdict.json,
              cell-table.md
  --wandb     everything above into one W&B run
  --all       the four non-GPU stages after --baseline

Every output lands under OUT_ROOT on /datasets and is copied into RESULTS in the repo.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The embedding, axis and font helpers of experiment E's readout; importing the module has no
# side effect beyond reading an env var it owns.
from experiment_e_x0_loss import _axes, _embed, _embedder, _font, _grey_std, _project  # noqa: E402

PY = "/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python"
SHOWCASE = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase")
TRAINING_CACHE = Path("/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache")
WANDB_PROJECT = "prime_lab/poe-repair-animals-compose"
BASELINE_RUN_WANDB = "prime_lab/poe-repair-animals-compose/6xc2l8ix"
BASELINE_CKPTS = SHOWCASE / "phase1_r32_100k" / "checkpoints"
LANDS = SHOWCASE / "where_each_condition_lands"
OUT_ROOT = SHOWCASE / "energy_penalty"
RESULTS = REPO_ROOT / "artifacts/results/does-charging-the-adapter-for-its-energy-sharpen-the-fix"
SEEDS = (9, 10, 11, 12, 13, 14, 15, 16)
FRAME_STEPS = (0, 2, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 49, 50)
LAMBDA = "1.0"
BETAS = ("0", "0.01", "0.05")

# --- bars, fixed before the arms ran ------------------------------------------------------------
MIN_COMPOSE_COUNT = 6            # a read is only valid if the arm still composes
DRIFT_MARGIN = 0.03              # the smallest drift gap the baseline grids told apart
CONTROL_VALIDITY_DRIFT_TOL = 0.03  # control arm against the baseline's own 40,050
ENERGY_SUPPORT_RATIO = 0.7       # energy_hat(beta 0.05) / energy_hat(beta 0) at or below: the price bit
ENERGY_NULL_RATIO = 0.9          # above: it did not
MISMATCH_TOLERANCE = 1.10        # mismatch(beta 0.05) / mismatch(beta 0) at or below: the fit was kept
CONTRAST_SUPPORT_CLOSURE = 0.5   # of the gap between the control's contrast and the joint prompt's
CONTRAST_NULL_CLOSURE = 0.1
PRIMARY_BETA = "0.05"

# --- the conditions -------------------------------------------------------------------------------
# key -> (label, probe folder, frames folder, on-policy folder)
CONDS = {
    "base30": ("shipped 30,050", SHOWCASE / "figure_r32_030050", LANDS / "frames", OUT_ROOT / "on_policy" / "baseline_030050"),
    "base40": ("baseline lineage 40,050", SHOWCASE / "figure_r32_040050", None, OUT_ROOT / "on_policy" / "baseline_040050"),
    "b0": ("control β 0, 40,050", SHOWCASE / "figure_r32_energy0_040050", LANDS / "frames_energy0", OUT_ROOT / "on_policy" / "energy0_040050"),
    "b0.01": ("β 0.01, 40,050", SHOWCASE / "figure_r32_energy0.01_040050", LANDS / "frames_energy0.01", OUT_ROOT / "on_policy" / "energy0.01_040050"),
    "b0.05": ("β 0.05, 40,050", SHOWCASE / "figure_r32_energy0.05_040050", LANDS / "frames_energy0.05", OUT_ROOT / "on_policy" / "energy0.05_040050"),
}
ARM_RUNS = {b: f"phase1_r32_energy{b}_from30050_40k" for b in BETAS}
COLORS = {"solo_a": "#d95f02", "solo_b": "#1b9e77", "joint": "#7570b3", "poe": "#111111",
          "base30": "#888888", "base40": "#bbbbbb", "b0": "#2c7bb6", "b0.01": "#fdae61", "b0.05": "#d7191c"}
MARKERS = {"poe": "s", "base30": "o", "base40": "o", "b0": "^", "b0.01": "P", "b0.05": "D"}


def _log(msg: str) -> None:
    print(f"[readout] {msg}", flush=True)


def _copy(*paths: Path) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for p in paths:
        if p and p.exists():
            shutil.copy2(p, RESULTS / p.name)


def _fig_dir() -> Path:
    d = OUT_ROOT / "figures"; d.mkdir(parents=True, exist_ok=True); return d


def final_render(cond: str, seed: int) -> Path:
    if cond in ("solo_a", "solo_b", "joint"):
        return LANDS / cond / f"seed_{seed}.png"
    if cond == "mono":
        return TRAINING_CACHE / "heldout" / "a_cat__x__a_dog" / f"seed_{seed}" / "mono.png"
    if cond == "poe":
        return CONDS["base30"][1] / "renders" / "full" / f"seed_{seed}_lambda_0.0.png"
    return CONDS[cond][1] / "renders" / "full" / f"seed_{seed}_lambda_{LAMBDA}.png"


def frame(cond: str, seed: int, step: int) -> Path | None:
    if cond in ("solo_a", "solo_b", "joint", "poe"):
        return LANDS / "frames" / cond / f"seed_{seed}" / f"step_{step:03d}.png"
    if cond == "base30":
        return LANDS / "frames" / "lora_1.0" / f"seed_{seed}" / f"step_{step:03d}.png"
    root = CONDS[cond][2]
    return None if root is None else root / "lora_1.0" / f"seed_{seed}" / f"step_{step:03d}.png"


def _probe(cond: str) -> dict | None:
    p = CONDS[cond][1] / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def _rows(cond: str) -> dict[int, dict]:
    res = _probe(cond)
    if res is None:
        return {}
    return {int(r["seed"]): r for r in res["rows"] if r["window"] == "full" and str(r["lambda"]) == LAMBDA}


def _onpolicy(cond: str) -> dict | None:
    p = CONDS[cond][3] / "on_policy_energy.json"
    return json.loads(p.read_text()) if p.exists() else None


def _contrast_mean(cond: str) -> float | None:
    paths = [final_render(cond, s) for s in SEEDS]
    if not all(p.exists() for p in paths):
        return None
    return float(np.mean([_grey_std(p) for p in paths]))


def _summary(cond: str) -> dict:
    res = _probe(cond); op = _onpolicy(cond)
    d = {"label": CONDS[cond][0], "probe_exists": res is not None, "onpolicy_exists": op is not None}
    if res is not None:
        s = res["summary"]["full"][LAMBDA]
        d.update(compose_n=int(round(s["compose_rate"] * s["n"])), n=int(s["n"]),
                 drift=float(s["mean_dino_drift"]), contrast=_contrast_mean(cond))
    if op is not None:
        sm = op["summary_mean_over_seeds"]
        d.update(energy_hat=sm["energy_hat_nats"], energy_true=sm["energy_true_nats"],
                 mismatch=sm["mismatch_nats"], energy_hat_late=sm["energy_hat_late_nats"], cos=sm["cos_mean"])
    return d


# ------------------------------------------------------------------------------------- --baseline
def stage_baseline() -> None:
    """The baseline lineage's own 40,050 grid and the two baseline on-policy reads. GPU."""
    env = dict(os.environ, POE_REPAIR_TRAINING_CACHE=str(TRAINING_CACHE))
    probe = CONDS["base40"][1]
    if not (probe / "results.json").exists():
        _log(f"probe: baseline 40,050 -> {probe}")
        subprocess.run([PY, "scripts/showcase/lambda_boundary_probe.py", "--rank", "32",
                        "--checkpoint", str(BASELINE_CKPTS / "lora_step_040050.pt"),
                        "--out-root", str(probe), "--windows", "full", "--lambdas", LAMBDA],
                       cwd=REPO_ROOT, env=env, check=False)
    for cond, step in (("base30", "030050"), ("base40", "040050")):
        out = CONDS[cond][3]
        if not (out / "on_policy_energy.json").exists():
            _log(f"on-policy energy: baseline {step} -> {out}")
            subprocess.run([PY, "scripts/showcase/on_policy_energy.py", "--rank", "32",
                            "--checkpoint", str(BASELINE_CKPTS / f"lora_step_{step}.pt"),
                            "--out", str(out), "--tag", f"baseline lineage, step {step}"],
                           cwd=REPO_ROOT, env=env, check=False)


# -------------------------------------------------------------------------------------- --figures
def fig_training_curves() -> Path | None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import wandb
        api = wandb.Api()
        runs = {}
        for b, rid in ARM_RUNS.items():
            found = api.runs(WANDB_PROJECT, filters={"display_name": rid})
            for r in found:
                runs[f"b{b}"] = r; break
        base = api.run(BASELINE_RUN_WANDB)
    except Exception as exc:  # noqa: BLE001
        _log(f"training curves skipped: {exc}"); return None
    keys = ["train/loss", "train/loss_fit", "train/loss_energy", "train/lora_weight_norm", "train/optimizer_step"]

    def hist(run, keys):
        try:
            df = run.history(keys=keys, samples=20000, pandas=True)
        except Exception:  # noqa: BLE001
            return None
        return df

    def ema(x, a=0.02):
        out, m = [], None
        for v in x:
            m = v if m is None else (1 - a) * m + a * v
            out.append(m)
        return np.array(out)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    bd = hist(base, ["train/loss", "train/lora_weight_norm", "train/optimizer_step"])
    if bd is not None and "train/loss" in bd:
        bd = bd.dropna(subset=["train/optimizer_step"]).sort_values("train/optimizer_step")
        m = (bd["train/optimizer_step"] >= 25000) & (bd["train/optimizer_step"] <= 45000)
        sub = bd[m].dropna(subset=["train/loss"])
        axes[0].plot(sub["train/optimizer_step"], ema(sub["train/loss"].values), color="#888888", lw=1.4, label="baseline lineage train/loss")
    for key, run in runs.items():
        df = hist(run, keys)
        if df is None:
            continue
        df = df.dropna(subset=["train/optimizer_step"]).sort_values("train/optimizer_step")
        for ax, col in ((axes[0], "train/loss_fit"), (axes[1], "train/loss_energy"), (axes[2], "train/lora_weight_norm")):
            if col in df:
                sub = df.dropna(subset=[col])
                y = ema(sub[col].values) if col != "train/lora_weight_norm" else sub[col].values
                ax.plot(sub["train/optimizer_step"], y, color=COLORS[key], lw=1.6, label=CONDS[key][0])
    axes[0].set_title("fit loss: noise-space MSE to the cached joint prediction (EMA 0.02)")
    axes[1].set_title("energy term: mean_k w_k · mean(r̂_k²) (EMA 0.02), before β")
    axes[2].set_title("Frobenius norm of the LoRA weights")
    for ax in axes:
        ax.set_xlabel("optimizer step"); ax.legend(fontsize=8); ax.axvline(30050, color="grey", lw=0.6, ls=":")
    fig.suptitle("the three arms resumed from 30,050, one axis moved: the price β on the correction's energy", fontsize=10)
    fig.tight_layout()
    out = _fig_dir() / "training-curves.png"; fig.savefig(out, dpi=140); plt.close(fig)
    return out


def fig_mismatch_vs_energy() -> tuple[Path | None, dict]:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = {c: _onpolicy(c) for c in CONDS}
    data = {c: d for c, d in data.items() if d is not None}
    if not data:
        return None, {}
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    summary = {}
    for c, d in data.items():
        rows = d["rows"]; n = max(r["step"] for r in rows) + 1
        e = np.zeros((len(SEEDS), n)); m = np.zeros((len(SEEDS), n)); t = np.zeros((len(SEEDS), n))
        for r in rows:
            i = SEEDS.index(r["seed"]); e[i, r["step"]] = r["energy_hat_nats"]; m[i, r["step"]] = r["mismatch_nats"]; t[i, r["step"]] = r["energy_true_nats"]
        axes[0].plot(range(n), e.mean(0), color=COLORS[c], lw=2.0, label=CONDS[c][0])
        axes[1].plot(range(n), m.mean(0), color=COLORS[c], lw=2.0, label=CONDS[c][0])
        if c == "base30":
            axes[0].plot(range(n), t.mean(0), color="#7570b3", lw=1.2, ls="--", label="true correction at the visited states (shipped run)")
        summary[c] = {"energy_hat_nats": round(float(e.sum(1).mean()), 1), "mismatch_nats": round(float(m.sum(1).mean()), 1),
                      "energy_true_nats": round(float(t.sum(1).mean()), 1),
                      "energy_hat_late_share": round(float(e[:, 20:].sum() / max(e.sum(), 1e-9)), 3)}
    axes[0].set_ylabel("adapter's spend per step\n½ γ_k ‖r̂_k‖² (nats)"); axes[0].legend(fontsize=8)
    axes[1].set_ylabel("mismatch per step\n½ γ_k ‖r̂_k − r_true,k‖² (nats)"); axes[1].set_xlabel("DDIM step (0 = pure noise)")
    axes[0].set_title("on-policy, cat × dog seeds 9 to 16, mean of 8: what each checkpoint spends, and how far it is from the true correction", fontsize=10)
    fig.tight_layout()
    out = _fig_dir() / "mismatch-vs-energy.png"; fig.savefig(out, dpi=140); plt.close(fig)
    (_fig_dir() / "mismatch-vs-energy.json").write_text(json.dumps(summary, indent=1))
    return out, summary


def fig_where_it_lands(embedder) -> tuple[Path | None, dict]:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    conds = ["solo_a", "solo_b", "joint", "poe"] + [c for c in ("base30", "b0", "b0.01", "b0.05") if _probe(c) is not None]
    conds = [c for c in conds if all(final_render(c, s).exists() for s in SEEDS)]
    feats = {c: _embed(embedder, [final_render(c, s) for s in SEEDS]) for c in conds}
    ax_ = _axes(feats); xy = {c: _project(ax_, feats[c]) for c in conds}
    fig, ax = plt.subplots(figsize=(10, 8.5))
    for c in ("solo_a", "solo_b", "joint"):
        pts = xy[c]; hull = ConvexHull(pts); poly = pts[hull.vertices]
        ax.fill(poly[:, 0], poly[:, 1], color=COLORS[c], alpha=0.12, lw=0)
        ax.scatter(pts[:, 0], pts[:, 1], s=36, color=COLORS[c], zorder=3)
        ax.annotate({"solo_a": "a cat alone", "solo_b": "a dog alone", "joint": "\"a cat and a dog\""}[c], pts.mean(0),
                    color=COLORS[c], fontsize=10, ha="center", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85), zorder=4)
    for c in conds:
        if c in ("solo_a", "solo_b", "joint"):
            continue
        ax.scatter(xy[c][:, 0], xy[c][:, 1], s=70, color=COLORS[c], marker=MARKERS[c], zorder=6,
                   label={"poe": "plain PoE, λ 0"}.get(c, CONDS.get(c, ("",))[0] + ", λ 1"), edgecolor="black", lw=0.4)
        if c != "poe":
            for i in range(len(SEEDS)):
                ax.annotate("", xy=xy[c][i], xytext=xy["poe"][i], arrowprops=dict(arrowstyle="-|>", color=COLORS[c], lw=0.9, alpha=0.7), zorder=5)
    for i, s in enumerate(SEEDS):
        ax.annotate(f"s{s}", xy["poe"][i], fontsize=8, color="#111111", xytext=(4, -9), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("cat-alone centroid  ←  x (DINOv2 cosine units)  →  dog-alone centroid")
    ax.set_ylabel("both-ness: 0 at the solo midpoint, up toward the \"a cat and a dog\" centroid")
    ax.set_title("cat × dog, held-out seeds 9 to 16: where each final render lands in DINOv2 space\n"
                 "axes fitted on the three reference clouds; arrows from plain PoE to each adapter at λ 1.0", fontsize=10)
    ax.legend(loc="best", fontsize=8, framealpha=0.9); fig.tight_layout()
    out = _fig_dir() / "where-each-condition-lands.png"; fig.savefig(out, dpi=150); plt.close(fig)
    summary = {c: {"both_ness_mean": round(float(xy[c][:, 1].mean()), 4), "x_mean": round(float(xy[c][:, 0].mean()), 4),
                   "per_seed_both_ness": {str(s): round(float(xy[c][i, 1]), 4) for i, s in enumerate(SEEDS)}} for c in conds}
    summary["joint_centroid_both_ness"] = round(ax_["joint_both_ness"], 4)
    (_fig_dir() / "where-each-condition-lands.json").write_text(json.dumps(summary, indent=1))
    np.savez(_fig_dir() / "final-render-dino-feats.npz", **feats)
    return out, summary


def _track_data(embedder):
    """Both-ness and which-animal of the running estimate at the 14 saved steps, on axes fitted on
    the step-50 frames of the three references (256 px)."""
    refs = {c: _embed(embedder, [frame(c, s, 50) for s in SEEDS]) for c in ("solo_a", "solo_b", "joint")}
    ax_ = _axes(refs)
    conds = ["joint", "poe", "base30", "b0", "b0.01", "b0.05"]
    xy = {}
    for c in conds:
        arr = np.full((len(SEEDS), len(FRAME_STEPS), 2), np.nan)
        paths, where = [], []
        for i, s in enumerate(SEEDS):
            for j, k in enumerate(FRAME_STEPS):
                p = frame(c, s, k)
                if p is not None and p.exists():
                    paths.append(p); where.append((i, j))
        if paths:
            f = _project(ax_, _embed(embedder, paths))
            for (i, j), v in zip(where, f):
                arr[i, j] = v
            xy[c] = arr
    return ax_, xy


def fig_tracks(embedder) -> tuple[Path | None, Path | None, dict]:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ax_, xy = _track_data(embedder)
    if not xy:
        return None, None, {}
    labels = {"joint": "\"a cat and a dog\"", "poe": "plain PoE", **{c: CONDS[c][0] for c in CONDS}}
    # the plane, one path per run; seed 15 bold, others faint
    fig, ax = plt.subplots(figsize=(10, 8.5))
    big = SEEDS.index(15)
    for c, arr in xy.items():
        for i in range(len(SEEDS)):
            pts = arr[i]; ok = ~np.isnan(pts[:, 0])
            ax.plot(pts[ok, 0], pts[ok, 1], color=COLORS.get(c, "#333333"), lw=2.2 if i == big else 0.6,
                    alpha=1.0 if i == big else 0.25, marker="o" if i == big else None, ms=3,
                    label=labels.get(c, c) if i == big else None)
        pts = arr[big]
        if not np.isnan(pts[0, 0]):
            ax.annotate("step 0", pts[0], fontsize=7, color=COLORS.get(c, "#333333"), xytext=(3, 3), textcoords="offset points")
    ax.axhline(ax_["joint_both_ness"], color=COLORS["joint"], ls="--", lw=0.8)
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("which animal: cat-alone centroid  ←  x  →  dog-alone centroid (DINOv2 cosine units)")
    ax.set_ylabel("both-ness of the running estimate (dashed: joint-prompt centroid)")
    ax.set_title("the path each run takes through the DINOv2 plane over the 50 steps\n"
                 "running estimate decoded at 14 steps; seed 15 bold, seeds 9 to 16 faint", fontsize=10)
    ax.legend(fontsize=8); fig.tight_layout()
    out1 = _fig_dir() / "tracks-in-the-dino-plane.png"; fig.savefig(out1, dpi=150); plt.close(fig)
    # both-ness against step
    fig, ax = plt.subplots(figsize=(9, 4.8))
    summary = {}
    for c, arr in xy.items():
        y = arr[:, :, 1]
        for row in y:
            ax.plot(FRAME_STEPS, row, color=COLORS.get(c, "#333333"), lw=0.5, alpha=0.3)
        ax.plot(FRAME_STEPS, np.nanmean(y, 0), color=COLORS.get(c, "#333333"), lw=2.2, label=labels.get(c, c))
        summary[c] = {str(k): round(float(v), 4) for k, v in zip(FRAME_STEPS, np.nanmean(y, 0))}
    ax.axhline(ax_["joint_both_ness"], color=COLORS["joint"], ls="--", lw=0.8); ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("DDIM step (50-step run)"); ax.set_ylabel("both-ness of the running estimate")
    ax.set_title("how each run moves toward \"a cat and a dog\": thin per seed, thick mean of 8", fontsize=10)
    ax.legend(fontsize=8); fig.tight_layout()
    out2 = _fig_dir() / "both-ness-over-steps.png"; fig.savefig(out2, dpi=140); plt.close(fig)
    payload = {"mean_over_seeds": summary, "joint_centroid_both_ness": round(ax_["joint_both_ness"], 4),
               "tracks": {c: np.nan_to_num(arr, nan=-99.0).round(4).tolist() for c, arr in xy.items()}, "frame_steps": list(FRAME_STEPS)}
    (_fig_dir() / "tracks-in-the-dino-plane.json").write_text(json.dumps(payload, indent=1))
    (_fig_dir() / "both-ness-over-steps.json").write_text(json.dumps({"mean_over_seeds": summary, "joint_centroid_both_ness": payload["joint_centroid_both_ness"]}, indent=1))
    return out1, out2, summary


def fig_checkpoint_bars() -> tuple[Path | None, dict]:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stats = {c: _summary(c) for c in CONDS}
    have = [c for c in CONDS if stats[c]["probe_exists"]]
    if not have:
        return None, stats
    joint_contrast = float(np.mean([_grey_std(final_render("mono", s)) for s in SEEDS]))
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    x = np.arange(len(have)); labels = [CONDS[c][0].replace(", ", "\n") for c in have]; cols = [COLORS[c] for c in have]
    axes[0].bar(x, [stats[c]["compose_n"] for c in have], color=cols); axes[0].axhline(MIN_COMPOSE_COUNT, color="k", ls="--", lw=0.8)
    axes[0].set_ylim(0, 8); axes[0].set_title("seeds composing of 8\n(dashed: validity floor 6)")
    axes[1].bar(x, [stats[c]["drift"] for c in have], color=cols)
    if "b0" in stats and stats["b0"]["probe_exists"]:
        axes[1].axhline(stats["b0"]["drift"] - DRIFT_MARGIN, color="k", ls="--", lw=0.8, label="support: control − 0.03")
        axes[1].legend(fontsize=7)
    axes[1].set_title("DINOv2 drift at λ 1, mean of 8\n(negative: nearer the joint-prompt image)")
    axes[2].bar(x, [stats[c]["contrast"] or 0 for c in have], color=cols); axes[2].axhline(joint_contrast, color=COLORS["joint"], ls="--", lw=0.8, label="joint prompt")
    axes[2].set_title("contrast: grey-level std of the 1024 px render"); axes[2].legend(fontsize=7)
    eh = [stats[c].get("energy_hat", 0) for c in have]; el = [stats[c].get("energy_hat_late", 0) for c in have]
    axes[3].bar(x, eh, color=cols); axes[3].bar(x, el, color="none", edgecolor="k", hatch="///", lw=0.5, label="of which steps 25 to 49")
    axes[3].set_title("on-policy control energy of r̂ (nats)"); axes[3].legend(fontsize=7)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    fig.suptitle("cat × dog, held-out seeds 9 to 16, every adapter at λ 1.0 on all 50 steps", fontsize=10)
    fig.tight_layout()
    out = _fig_dir() / "checkpoint-bars.png"; fig.savefig(out, dpi=140); plt.close(fig)
    (_fig_dir() / "checkpoint-bars.json").write_text(json.dumps({**stats, "joint_contrast": joint_contrast}, indent=1, default=str))
    return out, stats


def stage_figures() -> None:
    outs = [fig_training_curves()]
    o, _ = fig_mismatch_vs_energy(); outs.append(o)
    emb = _embedder()
    o, _ = fig_where_it_lands(emb); outs.append(o)
    o1, o2, _ = fig_tracks(emb); outs += [o1, o2]
    o, _ = fig_checkpoint_bars(); outs.append(o)
    _copy(*[p for p in outs if p], *sorted(_fig_dir().glob("*.json")), _fig_dir() / "final-render-dino-feats.npz")
    _log("figures: " + ", ".join(p.name for p in outs if p))


# --------------------------------------------------------------------------------------- --strips
STRIP_COLS = [("mono", "Mono: \"a cat and a dog\""), ("poe", "plain PoE (λ 0)"), ("base30", "shipped 30,050, λ 1"),
              ("b0", "control β 0, 40,050, λ 1"), ("b0.01", "β 0.01, 40,050, λ 1"), ("b0.05", "β 0.05, 40,050, λ 1")]


def stage_strips() -> None:
    cols = [(c, l) for c, l in STRIP_COLS if c in ("mono", "poe") or _probe(c) is not None]
    rows_by = {c: _rows(c) for c, _ in cols if c not in ("mono", "poe")}
    poe_rows = _rows("base30")
    tile, pad, cap_h = 448, 12, 78
    font, small = _font(17), _font(14)
    sdir = OUT_ROOT / "strips"; sdir.mkdir(parents=True, exist_ok=True)
    strips = []
    for s in SEEDS:
        W = len(cols) * (tile + pad) + pad; H = tile + cap_h + 2 * pad
        im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im)
        for j, (c, label) in enumerate(cols):
            p = final_render(c, s)
            if not p.exists():
                continue
            t = Image.open(p).convert("RGB").resize((tile, tile), Image.LANCZOS)
            x0 = pad + j * (tile + pad); im.paste(t, (x0, pad))
            d.text((x0, pad + tile + 4), label, fill="black", font=font)
            line = ""
            if c == "poe":
                r = poe_rows.get(s)
                r0 = None
                res = _probe("base30")
                if res:
                    r0 = next((x for x in res["rows"] if x["seed"] == s and x["window"] == "full" and float(x["lambda"]) == 0.0), None)
                if r0:
                    line = f"count {r0.get('n_instances', '?')}  contrast {_grey_std(p):.1f}"
            elif c in rows_by and s in rows_by[c]:
                r = rows_by[c][s]
                line = f"count {r.get('n_instances', '?')}  drift {r['drift']['dino']['drift']:+.3f}  contrast {_grey_std(p):.1f}"
            elif c == "mono":
                line = f"contrast {_grey_std(p):.1f}"
            d.text((x0, pad + tile + 30), line, fill="#333333", font=small)
        out = sdir / f"strip-seed_{s}.png"; im.save(out); strips.append(out)
    sheet = Image.new("RGB", (Image.open(strips[0]).width, sum(Image.open(p).height for p in strips)), "white")
    y = 0
    for p in strips:
        t = Image.open(p); sheet.paste(t, (0, y)); y += t.height
    sheet_path = OUT_ROOT / "sheet-all-seeds.png"; sheet.save(sheet_path)
    _copy(sheet_path); (RESULTS / "strips").mkdir(parents=True, exist_ok=True)
    for p in strips:
        shutil.copy2(p, RESULTS / "strips" / p.name)
    _log(f"strips: {len(strips)} and the sheet")


# -------------------------------------------------------------------------------------- --verdict
def stage_verdict() -> dict:
    stats = {c: _summary(c) for c in CONDS}
    joint_contrast = float(np.mean([_grey_std(final_render("mono", s)) for s in SEEDS]))
    v: dict = {"bars": {k: globals()[k] for k in ("MIN_COMPOSE_COUNT", "DRIFT_MARGIN", "CONTROL_VALIDITY_DRIFT_TOL",
                                                    "ENERGY_SUPPORT_RATIO", "ENERGY_NULL_RATIO", "MISMATCH_TOLERANCE",
                                                    "CONTRAST_SUPPORT_CLOSURE", "CONTRAST_NULL_CLOSURE")},
               "cells": stats, "joint_contrast": joint_contrast}
    ctl, arm = stats["b0"], stats[f"b{PRIMARY_BETA}"]
    # validity of the control against the shipped lineage
    if ctl["probe_exists"] and stats["base40"]["probe_exists"]:
        ok = ctl["compose_n"] >= MIN_COMPOSE_COUNT and abs(ctl["drift"] - stats["base40"]["drift"]) <= CONTROL_VALIDITY_DRIFT_TOL
        v["control_valid_against_lineage"] = {"pass": bool(ok), "control_drift": ctl["drift"], "lineage_40050_drift": stats["base40"]["drift"],
                                              "control_compose_n": ctl["compose_n"]}
    else:
        v["control_valid_against_lineage"] = {"pass": None, "reason": "a probe is missing"}
    # question 1
    if ctl["probe_exists"] and arm["probe_exists"]:
        if arm["compose_n"] < MIN_COMPOSE_COUNT:
            q1, why = "null", f"composition lost: {arm['compose_n']} of 8 compose, floor {MIN_COMPOSE_COUNT}"
        elif arm["drift"] <= ctl["drift"] - DRIFT_MARGIN:
            q1, why = "support", f"drift {arm['drift']:+.3f} against control {ctl['drift']:+.3f}, margin {DRIFT_MARGIN}"
        elif arm["drift"] >= ctl["drift"] + DRIFT_MARGIN:
            q1, why = "null", f"the price hurt: drift {arm['drift']:+.3f} against control {ctl['drift']:+.3f}"
        else:
            q1, why = "null", f"no change the reader can see: drift {arm['drift']:+.3f} against control {ctl['drift']:+.3f}, inside ±{DRIFT_MARGIN}"
        gap = joint_contrast - (ctl["contrast"] or joint_contrast)
        closure = ((arm["contrast"] or 0) - (ctl["contrast"] or 0)) / gap if abs(gap) > 1e-6 else float("nan")
        q1c = "support" if closure >= CONTRAST_SUPPORT_CLOSURE else ("null" if closure <= CONTRAST_NULL_CLOSURE else "inconclusive")
        v["q1_fidelity"] = {"verdict": q1, "why": why, "contrast_closure": round(float(closure), 3), "contrast_verdict": q1c}
    else:
        v["q1_fidelity"] = {"verdict": "inconclusive", "why": "a probe is missing"}
    # question 2
    if ctl["onpolicy_exists"] and arm["onpolicy_exists"]:
        er = arm["energy_hat"] / max(ctl["energy_hat"], 1e-9); mr = arm["mismatch"] / max(ctl["mismatch"], 1e-9)
        if er <= ENERGY_SUPPORT_RATIO and mr <= MISMATCH_TOLERANCE:
            q2 = "support"
        elif er > ENERGY_NULL_RATIO:
            q2 = "null"
        elif er <= ENERGY_SUPPORT_RATIO and mr > MISMATCH_TOLERANCE:
            q2 = "fit lost"
        else:
            q2 = "inconclusive"
        v["q2_energy_without_fit"] = {"verdict": q2, "energy_ratio": round(er, 3), "mismatch_ratio": round(mr, 3),
                                      "late_share_control": None, "late_share_arm": None}
    else:
        v["q2_energy_without_fit"] = {"verdict": "inconclusive", "why": "an on-policy read is missing"}
    # question 3
    trio = [stats["b0"], stats["b0.01"], stats[f"b{PRIMARY_BETA}"]]
    if all(t["onpolicy_exists"] and t["probe_exists"] for t in trio):
        e = [t["energy_hat"] for t in trio]; d = [t["drift"] for t in trio]
        v["q3_monotone"] = {"energy_monotone": bool(e[0] >= e[1] >= e[2]), "drift_monotone": bool(d[0] >= d[1] >= d[2]),
                            "energy": e, "drift": d}
    else:
        v["q3_monotone"] = {"verdict": "inconclusive", "why": "an arm is missing"}
    (OUT_ROOT / "verdict.json").write_text(json.dumps(v, indent=1, default=str))
    lines = ["| checkpoint | compose of 8 | drift | contrast | on-policy energy (nats) | mismatch (nats) | cos |", "|---|---|---|---|---|---|---|"]
    for c, s in stats.items():
        if not s["probe_exists"]:
            lines.append(f"| {s['label']} | missing | | | | | |"); continue
        lines.append(f"| {s['label']} | {s['compose_n']} | {s['drift']:+.3f} | {s['contrast']:.1f} | "
                     f"{s.get('energy_hat', float('nan')):.0f} | {s.get('mismatch', float('nan')):.0f} | {s.get('cos', float('nan')):.3f} |")
    lines += ["", f"Q1 fidelity: **{v['q1_fidelity'].get('verdict')}** ({v['q1_fidelity'].get('why', '')}); contrast: {v['q1_fidelity'].get('contrast_verdict', '')}",
              f"Q2 energy without fit: **{v['q2_energy_without_fit'].get('verdict')}** {json.dumps({k: v['q2_energy_without_fit'].get(k) for k in ('energy_ratio', 'mismatch_ratio')})}",
              f"Q3 monotone: {json.dumps({k: v['q3_monotone'].get(k) for k in ('energy_monotone', 'drift_monotone')})}",
              f"control valid against the lineage: {v['control_valid_against_lineage'].get('pass')}"]
    (OUT_ROOT / "cell-table.md").write_text("\n".join(lines) + "\n")
    _copy(OUT_ROOT / "verdict.json", OUT_ROOT / "cell-table.md")
    print("\n".join(lines[-4:]), flush=True)
    return v


# ---------------------------------------------------------------------------------------- --wandb
def stage_wandb() -> None:
    import wandb
    verdict = json.loads((OUT_ROOT / "verdict.json").read_text()) if (OUT_ROOT / "verdict.json").exists() else {}
    run = wandb.init(project=WANDB_PROJECT.split("/")[1], entity=WANDB_PROJECT.split("/")[0], name="energy_penalty_readout",
                     config={"plan": "01-showcase-the-trained-lora/22-charge-the-adapter-for-its-energy", "arms": ARM_RUNS,
                             "bars": verdict.get("bars", {})}, reinit=True)
    for p in sorted(_fig_dir().glob("*.png")) + [OUT_ROOT / "control-energy-over-steps.png", OUT_ROOT / "sheet-all-seeds.png"]:
        if p.exists():
            run.log({f"figures/{p.stem}": wandb.Image(str(p))})
    for p in sorted((OUT_ROOT / "strips").glob("strip-seed_*.png")):
        run.log({f"strips/{p.stem}": wandb.Image(str(p))})
    cells = verdict.get("cells", {})
    if cells:
        tbl = wandb.Table(columns=["checkpoint", "compose_n", "drift", "contrast", "energy_hat_nats", "mismatch_nats", "cos"])
        for c, s in cells.items():
            if s.get("probe_exists"):
                tbl.add_data(s["label"], s.get("compose_n"), s.get("drift"), s.get("contrast"), s.get("energy_hat"), s.get("mismatch"), s.get("cos"))
        run.log({"cell_table": tbl})
    run.summary.update({"q1": verdict.get("q1_fidelity"), "q2": verdict.get("q2_energy_without_fit"), "q3": verdict.get("q3_monotone"),
                        "control_valid": verdict.get("control_valid_against_lineage")})
    art = wandb.Artifact("energy_penalty_readout", type="results")
    for p in list(_fig_dir().glob("*.json")) + [OUT_ROOT / "verdict.json", OUT_ROOT / "cell-table.md", OUT_ROOT / "control-energy-over-steps.json"]:
        if p.exists():
            art.add_file(str(p))
    run.log_artifact(art)
    _log(f"W&B run {run.id}: {run.url}")
    run.finish()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    for flag in ("baseline", "figures", "strips", "verdict", "wandb", "all"):
        ap.add_argument(f"--{flag}", action="store_true")
    args = ap.parse_args(argv)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.baseline:
        stage_baseline()
    if args.figures or args.all:
        stage_figures()
    if args.strips or args.all:
        stage_strips()
    if args.verdict or args.all:
        stage_verdict()
    if args.wandb or args.all:
        stage_wandb()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
