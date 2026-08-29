"""F9: joint-prompt, plain-PoE and adapter samples for six pairs.

Three columns per row, all three from the same pair and starting seed:
the joint-prompt render (the target the correction was defined from),
uncorrected product-of-experts, and the adapter-corrected sample. Four
held-out pairs only; the trained-on rows were cropped by decision so the
figure carries the transfer claim alone. The sidecar records checkpoint
step, seeds and sampler settings, which the captured JPEG this replaces
never did.

    co3 python scripts/joint_poe_adapter_grid.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
CACHE = Path("/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache")
RUN = REPO / "artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k"
STEP_DIR = RUN / "samples/per_epoch/epoch_2000_step_100000"
OUT = REPO / "paper/iclr/figures/joint-prompt-poe-and-adapter-samples-for-four-held-out-pairs"

HELD = [("a_cat__x__a_dog", 9), ("an_eagle__x__a_hawk", 9),
        ("a_frog__x__a_toad", 9), ("a_goose__x__a_swan", 9)]
TRAIN = [("a_wolf__x__a_husky", 1), ("a_lion__x__a_tiger", 1)]


def pretty(slug: str) -> str:
    a, b = slug.split("__x__")
    strip = lambda s: s.replace("a_", "", 1).replace("an_", "", 1).replace("_", " ")
    return f"{strip(a)} × {strip(b)}"


def row_paths(slug: str, seed: int, split: str) -> dict[str, Path]:
    cell = CACHE / split / slug / f"seed_{seed}"
    prefix = "out_out" if split == "heldout" else "in_in"
    return {
        "joint": cell / "mono.png",
        "poe": cell / "poe.png",
        "adapter": STEP_DIR / f"{prefix}__{slug}__seed{seed:02d}.png",
    }


def main() -> None:
    rows = [(s, seed, "heldout") for s, seed in HELD]
    fig, axes = plt.subplots(len(rows), 3, figsize=(6.6, 9.1))
    col_titles = ["joint prompt\n(the target)", "plain PoE", "adapter-corrected"]
    meta_rows = []
    for r, (slug, seed, split) in enumerate(rows):
        paths = row_paths(slug, seed, split)
        for c, key in enumerate(["joint", "poe", "adapter"]):
            ax = axes[r, c]
            ax.imshow(Image.open(paths[key]).convert("RGB"))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=9)
        axes[r, 0].set_ylabel(pretty(slug), fontsize=9, rotation=0,
                              ha="right", va="center", labelpad=8)
        meta_rows.append({"pair": slug, "seed": seed, "split": split,
                          **{k: str(v) for k, v in paths.items()}})
    fig.subplots_adjust(left=0.145, right=0.998, top=0.96, bottom=0.005,
                        wspace=0.02, hspace=0.05)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=110)
    cfg = json.loads((RUN / "config.json").read_text())
    sidecar = {
        "figure": "F9",
        "checkpoint_step": 100000,
        "adapter_run": str(RUN),
        "sampler": cfg["sampler"],
        "lambda": 1.0,
        "note": ("joint and poe columns are the cache cell's own renders (mono.png, poe.png); "
                 "adapter column is the pooled run's per-epoch sample at step 100000, lambda 1. "
                 "All three columns of a row share the pair's starting seed."),
        "rows": meta_rows,
    }
    OUT.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    print("wrote", OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
