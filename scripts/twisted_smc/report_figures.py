#!/usr/bin/env python
"""Two report figures for the twisted-SMC baseline (scope 06, step 51), from files already on disk.

1. twist-training-curves.png: left, validation and training accuracy of the twist head against
   optimizer step with the 0.60 bar; right, training loss per timestep bucket on a log axis with
   the chance line. Read off the run's history.json.
2. twisted-smc-checkpoint-timeline.png: one row per render cell; columns Mono, PoE control, then
   the twisted-SMC pick at every checkpoint. Composed from the PNGs the run saved.

Writes both, plus a JSON sidecar with every number drawn, into
artifacts/results/is-the-gap-the-samplers-or-the-models/.

    python scripts/twisted_smc/report_figures.py [--run-dir DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts/results/is-the-gap-the-samplers-or-the-models"
DEFAULT_RUN = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/twisted_smc/"
                   "twist_w64_b16_lr1e-04_s100000_20260905-072331")

# dataviz reference palette, light mode
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = {"near_clean": "#2a78d6", "mid": "#eb6834", "late_noise": "#1baf7a"}
BUCKET_LABEL = {"near_clean": "timestep < 300 (near clean)", "mid": "300 to 700",
                "late_noise": "timestep >= 700 (high noise)"}
MIN_VAL_ACC, CHANCE = 0.60, 0.6931


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def curves(history: list[dict], out: Path) -> dict:
    val = [(r["_step"], r["val/acc"]) for r in history if "val/acc" in r]
    tr_acc = [(r["_step"], r["train/acc"]) for r in history if "train/acc" in r]
    buckets = {k: [(r["_step"], r[f"train/loss_bucket/{k}"]) for r in history
                   if f"train/loss_bucket/{k}" in r] for k in SERIES}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax1, ax2):
        _style(ax)

    ax1.plot(*zip(*tr_acc), color=MUTED, linewidth=1.2, label="training accuracy (batch)")
    ax1.plot(*zip(*val), color=SERIES["near_clean"], linewidth=2, label="validation accuracy (16 held-out cells)")
    ax1.axhline(MIN_VAL_ACC, color=INK2, linewidth=1, linestyle="--")
    ax1.text(val[-1][0], MIN_VAL_ACC + 0.01, "bar: 0.60", color=INK2, fontsize=9, ha="right")
    ax1.set_ylim(0.4, 1.02)
    ax1.set_xlabel("optimizer step", color=INK2)
    ax1.set_ylabel("accuracy: joint-prompt vs PoE latent", color=INK2)
    ax1.set_title("The twist head separates its training set and not held-out pairs", color=INK, fontsize=10, loc="left")
    ax1.legend(frameon=False, fontsize=8, loc="center right")
    ax1.text(val[-1][0], val[-1][1] - 0.03, f"{val[-1][1]:.2f}", color=SERIES["near_clean"], fontsize=9, ha="right")

    for k, pts in buckets.items():
        xs, ys = zip(*pts)
        ax2.plot(xs, [max(y, 1e-4) for y in ys], color=SERIES[k], linewidth=2, label=BUCKET_LABEL[k])
    ax2.axhline(CHANCE, color=INK2, linewidth=1, linestyle="--")
    ax2.text(xs[0], CHANCE * 1.15, "chance: 0.69", color=INK2, fontsize=9, ha="left")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-4, 1.5)
    ax2.set_xlabel("optimizer step", color=INK2)
    ax2.set_ylabel("training loss (binary cross-entropy, floor 1e-4)", color=INK2)
    ax2.set_title("Loss reaches zero even at high noise, where the classes coincide", color=INK, fontsize=10, loc="left")
    ax2.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)

    def near(pts, s):
        return min(pts, key=lambda p: abs(p[0] - s))
    return {
        "val_acc": {"first": val[0], "max": max(val, key=lambda p: p[1]), "min": min(val, key=lambda p: p[1]), "last": val[-1]},
        "train_acc_last": tr_acc[-1],
        "bucket_loss": {k: {str(s): near(pts, s) for s in (1, 10000, 30000, 100000)} for k, pts in buckets.items()},
        "chance_loss": CHANCE, "min_val_acc_bar": MIN_VAL_ACC,
    }


def timeline(run: Path, history: list[dict], out: Path, thumb: int = 150) -> dict:
    cells = json.loads((run / "cells.json").read_text())["render"]
    steps = sorted(int(p.name.split("_")[1]) for p in (run / "samples").glob("step_*"))
    cols = ["Mono", "PoE"] + [f"SMC @{s // 1000}k" if s else "SMC @0" for s in steps]
    pad, left, top = 6, 190, 44
    W = left + len(cols) * (thumb + pad) + pad
    H = top + len(cells) * (thumb + pad) + pad
    im = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
        small = ImageFont.truetype("DejaVuSans.ttf", 11)
    except Exception:
        font = small = ImageFont.load_default()
    d.text((pad, 8), "Twisted SMC across training checkpoints. Mono and PoE are fixed; every SMC column is the largest-weight of 4 particles.", fill=INK, font=font)
    for j, c in enumerate(cols):
        d.text((left + j * (thumb + pad), top - 16), c, fill=INK2, font=small)
    compose = {}
    for i, cell in enumerate(cells):
        split, pair, seed = cell.split("/")
        stem = f"{split}__{pair}__{seed.replace('seed_', 'seed_0')}"
        y = top + i * (thumb + pad)
        d.text((pad, y + thumb // 2 - 14), pair.replace("__x__", " x ").replace("_", " "), fill=INK, font=font)
        d.text((pad, y + thumb // 2 + 4), f"{split}, {seed.replace('_', ' ')}", fill=MUTED, font=small)
        paths = [run / "references" / f"{stem}__mono.png", run / "references" / f"{stem}__poe.png"]
        paths += [run / "samples" / f"step_{s:06d}" / f"{stem}__smc.png" for s in steps]
        for j, p in enumerate(paths):
            t = Image.open(p).convert("RGB"); t.thumbnail((thumb, thumb))
            im.paste(t, (left + j * (thumb + pad), y))
        key = f"{split}/{pair}/{seed.replace('seed_', 'seed_0')}"
        compose[key] = {str(s): next((r[f"eval/compose/smc/{key}"] for r in history
                                       if r["_step"] == s and f"eval/compose/smc/{key}" in r), None) for s in steps}
    im.save(out)
    return {"cells": cells, "checkpoints": steps, "compose_smc_shown_particle": compose}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    history = json.loads((a.run_dir / "history.json").read_text())
    side = {"run_dir": str(a.run_dir), "history": str(a.run_dir / "history.json")}
    side["curves"] = curves(history, OUT / "twist-training-curves.png")
    side["timeline"] = timeline(a.run_dir, history, OUT / "twisted-smc-checkpoint-timeline.png")
    (OUT / "twisted-smc-report-figures.json").write_text(json.dumps(side, indent=1))
    print(json.dumps(side["curves"], indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
