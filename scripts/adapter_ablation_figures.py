#!/usr/bin/env python
"""The two grids of the adapter-ablation appendix, drawn from the runs' own logged values.

One figure per axis. The first varies which layers carry the adapter at fixed rank; the second
varies rank at fixed layers. Both read `measures.json`, produced by `extract_measures.py` from each
run's `history.json` on the cluster, so every number here was logged by the run that claims it.

The measure is the DINOv2 embedding drift, defined in
`poe_repair/experiments/cross_pair_lora_pooling/_inline_sampling.py::embedding_drift`: the corrected
render's cosine distance to the joint-prompt render minus its cosine distance to the plain-product
render, in DINOv2 embedding space. It runs from -2 to +2 and is negative when the render sits nearer
the joint-prompt target than the product it repairs, so lower is better. The compose rate cannot
carry these figures: every configuration reads 1.0 on the tracking set.

    python scripts/adapter_ablation_figures.py <measures.json> <out_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MEASURE = "eval/tracking/embedding_drift/dino"
MEASURE_NAME = "DINOv2 drift: distance to the joint-prompt render minus distance to the plain product"

# Which layers each run adapts, and what else moved with it. Anything that moved beside the axis
# makes the comparison mixed, and the figure says so rather than leaving it to be worked out.
LAYERS_AXIS = [
    ("v57-08-x0-w25", "cross-attention", "trained on steps 0 to 25"),
    ("v57-09-x0-seeds", "cross-attention", "every cached seed"),
    ("v57-10-x0-contrast", "cross-attention", "contrast term at 0.01"),
    ("v57-11b-x0-self-decay", "cross + self-attention", "rate decaying, resumed at step 5,000"),
]
RANK_AXIS = [
    ("phase1_r8_200k", 8, "epsilon loss, 200k steps"),
    ("phase1_r16_100k", 16, "epsilon loss, 100k steps"),
    ("phase1_r32_100k", 32, "epsilon loss, 100k steps"),
]
CELL_ORDER = [
    ("in_in/a_lion__x__a_meerkat/seed_01", "lion and meerkat, seed 1 (trained)"),
    ("in_in/a_typewriter__x__a_cactus/seed_01", "typewriter and cactus, seed 1 (trained)"),
    ("out_out/a_cat__x__a_dog/seed_09", "cat and dog, seed 9 (held out)"),
    ("out_out/a_cat__x__a_dog/seed_10", "cat and dog, seed 10 (held out)"),
    ("out_out/an_elephant__x__a_penguin/seed_09", "elephant and penguin, seed 9 (held out)"),
    ("out_out/an_elephant__x__a_penguin/seed_10", "elephant and penguin, seed 10 (held out)"),
]
COMMON_CELLS = [("out_out/a_cat__x__a_dog/seed_09", "cat and dog, seed 9"),
                ("out_out/a_cat__x__a_dog/seed_10", "cat and dog, seed 10")]


def value(run: dict, cell: str) -> float | None:
    v = run["eval"].get(f"{MEASURE}/{cell}")
    return None if v is None else float(v[0])


def params(run: dict) -> str:
    t, n = run.get("trainable"), run.get("total_params")
    return "" if not t or not n else f"{t:,} of {n:,} ({100 * t / n:.2f}%)"


def layers_figure(m: dict, out: Path) -> dict:
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    n = len(LAYERS_AXIS)
    h = 0.8 / n
    colours = ["#4c72b0", "#6b93d6", "#9ab8e8", "#c44e52"]
    record = {"measure": MEASURE, "measure_meaning": MEASURE_NAME, "cells": {}, "runs": {}}
    for i, (name, layers, other) in enumerate(LAYERS_AXIS):
        run = m[name]
        ys, xs = [], []
        for j, (cell, _) in enumerate(CELL_ORDER):
            v = value(run, cell)
            if v is None:
                continue
            ys.append(j + (i - (n - 1) / 2) * h)
            xs.append(v)
            record["cells"].setdefault(cell, {})[name] = v
        t, tot = run.get("trainable"), run.get("total_params")
        share = f"{t:,} trainable, {100 * t / tot:.2f}% of the model" if t and tot else ""
        ax.barh(ys, xs, height=h * 0.92, color=colours[i],
                label=f"{layers}: {share}\n    {name}, {other}")
        record["runs"][name] = {"wandb": run["wandb"], "layers": layers, "rank": run["rank"],
                                "trainable_params": run.get("trainable"),
                                "total_params": run.get("total_params"),
                                "step": run.get("last_step"), "other_axes_moved": other}
    ax.axvline(0, color="black", lw=1)
    ax.annotate("worse than the plain product →", xy=(0.004, 0.02), fontsize=8,
                xycoords=("data", "axes fraction"), va="bottom")
    ax.annotate("← nearer the joint-prompt render", xy=(-0.004, 0.02), fontsize=8,
                xycoords=("data", "axes fraction"), va="bottom", ha="right")
    ax.set_yticks(range(len(CELL_ORDER)))
    ax.set_yticklabels([lab for _, lab in CELL_ORDER], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(MEASURE_NAME + "  (lower is better)", fontsize=9)
    ax.set_title("Which layers carry the adapter, at rank 32 and the same loss\n"
                 "every bar reads compose rate 1.0, so composition cannot separate them", fontsize=11)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              framealpha=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return record


def rank_figure(m: dict, out: Path) -> dict:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    record = {"measure": MEASURE, "measure_meaning": MEASURE_NAME, "cells": {}, "runs": {}}
    for cell, label in COMMON_CELLS:
        xs, ys = [], []
        for name, rank, other in RANK_AXIS:
            v = value(m[name], cell)
            if v is None:
                continue
            xs.append(rank)
            ys.append(v)
            record["cells"].setdefault(cell, {})[name] = v
        ax.plot(xs, ys, marker="o", label=label)
    for name, rank, other in RANK_AXIS:
        run = m[name]
        record["runs"][name] = {"wandb": run["wandb"], "rank": rank, "layers": "cross-attention",
                                "trainable_params": run.get("trainable"),
                                "total_params": run.get("total_params"),
                                "step": run.get("last_step"), "other_axes_moved": other}
    ax.axhline(0, color="black", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32])
    ax.set_xticklabels(["8", "16", "32"])
    ax.set_xlabel("adapter rank, cross-attention only (210 adapted projections)", fontsize=9)
    ax.set_ylabel("DINOv2 drift (lower is better)", fontsize=9)
    ax.set_title("Rank, at fixed layers, on the two cells common to both eras\n"
                 "mixed: these runs also differ in loss and in training length", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return record


if __name__ == "__main__":
    measures = json.loads(Path(sys.argv[1]).read_text())
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    f1 = out_dir / "which-layers-carry-the-adapter.png"
    rec1 = layers_figure(measures, f1)
    rec1["figure"] = f1.name
    rec1["axis"] = "which layers are adapted, at rank 32"
    rec1["clean"] = False
    rec1["why_not_clean"] = ("the cross+self run also decays its learning rate and resumed from a "
                             "step-5,000 checkpoint, so layers is not the only thing that moved")
    (f1.with_suffix(".json")).write_text(json.dumps(rec1, indent=1))

    f2 = out_dir / "how-much-rank-the-correction-needs.png"
    rec2 = rank_figure(measures, f2)
    rec2["figure"] = f2.name
    rec2["axis"] = "adapter rank, cross-attention only"
    rec2["clean"] = False
    rec2["why_not_clean"] = ("these runs differ in loss space and training length as well as rank, "
                             "and only cat and dog seeds 9 and 10 are common to both tracking sets")
    (f2.with_suffix(".json")).write_text(json.dumps(rec2, indent=1))

    for name in ("phase1_r8_200k", "phase1_r16_100k", "phase1_r32_100k",
                 "v57-09-x0-seeds", "v57-11b-x0-self-decay"):
        print(f"{name:24s} {params(measures[name])}")
    print("wrote", f1, "and", f2)
