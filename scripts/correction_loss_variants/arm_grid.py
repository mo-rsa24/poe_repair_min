"""Three-column comparison for each arm of the correction-loss variant set.

A copy of scripts/joint_poe_adapter_grid.py with RUN and STEP_DIR taken as
arguments instead of hard-coded, because a paper figure depends on the
original's fixed paths.

Columns per row: the joint-prompt render (the target the correction was
defined from), uncorrected product-of-experts, then the adapter-corrected
sample from one arm. Rows are held-out cells.

Only two held-out seeds are rendered by these runs (seed_pool.held_out[:2],
so seeds 9 and 10), and of those four cells only two carry a joint-prompt
reference that actually shows one of each animal, measured in
outputs/showcase/target_quality/target_quality.json:

    a_cat__x__a_dog            seed 9 sound, seed 10 shows three dogs
    an_elephant__x__a_penguin  seed 10 sound, seed 9 shows an elephant alone

The sound pair of cells is the default. --broken renders the other two with
the reference labelled as wrong, so the exclusion is visible rather than
silent. No score is averaged over cells whose reference is wrong.

    co3 python scripts/correction_loss_variants/arm_grid.py --all
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
CACHE = Path("/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache")
OUTROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/correction_loss_variants")
ARMS = ["plain", "anchor", "plurality", "connective"]

# cell -> whether the joint-prompt reference shows one of each animal
SOUND = [("a_cat__x__a_dog", 9), ("an_elephant__x__a_penguin", 10)]
BROKEN = [("a_cat__x__a_dog", 10, "reference draws three dogs"),
          ("an_elephant__x__a_penguin", 9, "reference draws an elephant alone")]

DEST = REPO / "artifacts/results/does-the-branch-conditioning-change-what-the-adapter-learns"


def run_dir(arm: str) -> Path:
    return OUTROOT / f"r16_s0_25_{arm}"


def latest_step_dir(arm: str) -> Path:
    d = run_dir(arm) / "samples/per_epoch"
    cands = [p for p in d.iterdir() if p.is_dir()] if d.exists() else []
    if not cands:
        raise SystemExit(f"no per_epoch samples under {d}")
    def step_of(p: Path) -> int:
        m = re.search(r"step_(\d+)", p.name)
        return int(m.group(1)) if m else -1
    return max(cands, key=step_of)


def pretty(slug: str) -> str:
    a, b = slug.split("__x__")
    strip = lambda s: s.replace("a_", "", 1).replace("an_", "", 1).replace("_", " ")
    return f"{strip(a)} x {strip(b)}"


def cell_paths(slug: str, seed: int) -> dict[str, Path]:
    cell = CACHE / "heldout" / slug / f"seed_{seed}"
    return {"joint": cell / "mono.png", "poe": cell / "poe.png"}


def adapter_png(arm: str, slug: str, seed: int, step_dir: Path) -> Path:
    return step_dir / f"out_out__{slug}__seed{seed:02d}.png"


def show(ax, path: Path):
    if path.exists():
        ax.imshow(Image.open(path).convert("RGB"))
    else:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def sidecar(name: str, rows: list[dict], steps: dict[str, int], extra: dict) -> None:
    cfgs = {}
    for arm in steps:
        p = run_dir(arm) / "config.json"
        if p.exists():
            cfgs[arm] = json.loads(p.read_text()).get("sampler")
    (DEST / f"{name}.json").write_text(json.dumps({
        "figure": name,
        "checkpoint_step_per_arm": steps,
        "runs": {a: str(run_dir(a)) for a in steps},
        "wandb": {"plain": "tqs7qf95", "anchor": "72as6yk3",
                  "plurality": "mw1cqrpk", "connective": "pckb2za7"},
        "sampler_per_arm": cfgs,
        "lambda": 1.0,
        "reference_validity_source": (
            "/datasets/mmolefe/poe_repair_min/outputs/showcase/"
            "target_quality/target_quality.json"),
        "note": ("joint and poe columns are the cache cell's own renders (mono.png, poe.png); "
                 "arm columns are that arm's per-epoch sample at the step named above, "
                 "lambda 1. Every column of a row shares the cell's starting seed. These "
                 "runs render held-out seeds 9 and 10 only; seeds 11 to 16 were never "
                 "sampled, so they are absent rather than excluded."),
        **extra,
        "rows": rows,
    }, indent=2))


def combined(cells, name: str, title_note: str, labels: dict[tuple[str, int], str] | None) -> None:
    steps = {a: int(re.search(r"step_(\d+)", latest_step_dir(a).name).group(1)) for a in ARMS}
    sdirs = {a: latest_step_dir(a) for a in ARMS}
    cols = ["joint prompt\n(the target)", "plain PoE"] + ARMS
    fig, axes = plt.subplots(len(cells), len(cols),
                             figsize=(1.55 * len(cols), 1.75 * len(cells) + 0.55),
                             squeeze=False)
    rows_meta = []
    for r, (slug, seed) in enumerate(cells):
        p = cell_paths(slug, seed)
        show(axes[r][0], p["joint"]); show(axes[r][1], p["poe"])
        for c, arm in enumerate(ARMS, start=2):
            show(axes[r][c], adapter_png(arm, slug, seed, sdirs[arm]))
        lab = pretty(slug) + f"\nseed {seed}"
        if labels and (slug, seed) in labels:
            lab += f"\n({labels[(slug, seed)]})"
        axes[r][0].set_ylabel(lab, fontsize=8, rotation=0, ha="right", va="center", labelpad=8)
        rows_meta.append({"pair": slug, "seed": seed,
                          "reference_sound": not (labels and (slug, seed) in labels),
                          "joint": str(p["joint"]), "poe": str(p["poe"]),
                          **{arm: str(adapter_png(arm, slug, seed, sdirs[arm])) for arm in ARMS}})
    for c, t in enumerate(cols):
        axes[0][c].set_title(t, fontsize=8)
    fig.subplots_adjust(left=0.17, right=0.995, top=0.88, bottom=0.01, wspace=0.02, hspace=0.06)
    DEST.mkdir(parents=True, exist_ok=True)
    fig.savefig(DEST / f"{name}.png", dpi=130); plt.close(fig)
    sidecar(name, rows_meta, steps, {"reference_note": title_note})
    print("wrote", DEST / f"{name}.png")


def per_arm(arm: str) -> None:
    sdir = latest_step_dir(arm)
    step = int(re.search(r"step_(\d+)", sdir.name).group(1))
    fig, axes = plt.subplots(len(SOUND), 3, figsize=(5.0, 1.75 * len(SOUND) + 0.55),
                             squeeze=False)
    rows_meta = []
    for r, (slug, seed) in enumerate(SOUND):
        p = cell_paths(slug, seed)
        show(axes[r][0], p["joint"]); show(axes[r][1], p["poe"])
        show(axes[r][2], adapter_png(arm, slug, seed, sdir))
        axes[r][0].set_ylabel(f"{pretty(slug)}\nseed {seed}", fontsize=8, rotation=0,
                              ha="right", va="center", labelpad=8)
        rows_meta.append({"pair": slug, "seed": seed, "reference_sound": True,
                          "joint": str(p["joint"]), "poe": str(p["poe"]),
                          "adapter": str(adapter_png(arm, slug, seed, sdir))})
    for c, t in enumerate(["joint prompt\n(the target)", "plain PoE", f"{arm}\n(adapter)"]):
        axes[0][c].set_title(t, fontsize=8)
    fig.subplots_adjust(left=0.26, right=0.995, top=0.86, bottom=0.01, wspace=0.02, hspace=0.06)
    DEST.mkdir(parents=True, exist_ok=True)
    name = f"three-column-comparison-{arm}"
    fig.savefig(DEST / f"{name}.png", dpi=130); plt.close(fig)
    sidecar(name, rows_meta, {arm: step},
            {"reference_note": "both rows have a joint-prompt reference showing one of each animal"})
    print("wrote", DEST / f"{name}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--broken", action="store_true")
    a = ap.parse_args()
    if a.arm:
        per_arm(a.arm); return
    if a.all:
        for arm in ARMS:
            per_arm(arm)
        combined(SOUND, "all-four-arms-on-the-sound-reference-cells",
                 "both rows have a joint-prompt reference showing one of each animal", None)
        combined([(s, sd) for s, sd, _ in BROKEN],
                 "all-four-arms-on-the-broken-reference-cells",
                 "the joint column is wrong on both rows; shown for completeness, never scored",
                 {(s, sd): why for s, sd, why in BROKEN})
        return
    ap.error("pass --arm <arm> or --all")


if __name__ == "__main__":
    main()
