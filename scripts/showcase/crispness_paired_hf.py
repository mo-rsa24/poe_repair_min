"""Task 8.1: a paired, same-seed crispness read against the joint-prompt reference.

Every held-out seed already has a reference for what a crisp render of this pair looks like on
this noise: the joint-prompt render `mono.png` in the training cache. The read is the share of
image energy above a radial frequency cutoff, taken on the render and on its reference, and
reported as a ratio. 1.0 means as much fine detail as the joint prompt drew; below 1.0 means
softer. Being a ratio per seed sidesteps the per-seed spread that broke Laplacian variance (the
drawn-texture seeds 11, 14, 15 dominate any band built across seeds).

Conditions read, all at lambda 1.2, rank 32 @ 30050, cat x dog seeds 9 to 16, from
`/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/`:
  adapter-alone   tail/<pair>/seed_N/lambda_1.2_k000_w35-50.png
  hand-off @20    clean_tail/<pair>/seed_N/lambda_1.2_on0-20_k000_frozen_w35-50.png
  hand-off @30    clean_tail/<pair>/seed_N/lambda_1.2_on0-30_k000_frozen_w35-50.png
  joint prompt    training_cache/heldout/<pair>/seed_N/mono.png  (the reference, ratio 1 by definition)
  plain PoE       training_cache/heldout/<pair>/seed_N/poe.png

What would validate it: adapter-alone below hand-off below joint, matching the DINOv2 ordering the
clean-tail finding already recorded. What would surprise: hand-off below adapter-alone here while
DINOv2 says it is nearer, which would mean nearer and crisper are different things.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np
from PIL import Image

CORR = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector")
CACHE = Path("/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache/heldout")
OUT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/crispness")
PAIR = "a_cat__x__a_dog"
SEEDS = (9, 10, 11, 12, 13, 14, 15, 16)
# Fraction of the Nyquist radius above which energy counts as fine detail. 0.25 on a 1024 image
# is wavelengths shorter than ~8 px: fur texture, edges, whisker-scale structure.
HF_CUTOFF = 0.25


def hf_share(path: Path, cutoff: float = HF_CUTOFF) -> float:
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    g = g - g.mean()
    F = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    h, w = g.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - h / 2, xx - w / 2) / (min(h, w) / 2)
    return float(F[r >= cutoff].sum() / F.sum())


def conditions(seed: int, extra_cutoffs: tuple[int, ...] = ()) -> dict[str, Path]:
    c = {
        "plain PoE": CACHE / PAIR / f"seed_{seed}" / "poe.png",
        "adapter alone": CORR / "tail" / PAIR / f"seed_{seed}" / "lambda_1.2_k000_w35-50.png",
        "hand-off @20": CORR / "clean_tail" / PAIR / f"seed_{seed}" / "lambda_1.2_on0-20_k000_frozen_w35-50.png",
        "hand-off @30": CORR / "clean_tail" / PAIR / f"seed_{seed}" / "lambda_1.2_on0-30_k000_frozen_w35-50.png",
    }
    for k in extra_cutoffs:
        c[f"hand-off @{k}"] = CORR / "clean_tail" / PAIR / f"seed_{seed}" / f"lambda_1.2_on0-{k}_k000_frozen_w35-50.png"
    c["joint prompt"] = CACHE / PAIR / f"seed_{seed}" / "mono.png"
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-cutoffs", type=int, nargs="*", default=[])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    table: dict[str, dict[int, float]] = {}
    for seed in SEEDS:
        conds = conditions(seed, tuple(args.extra_cutoffs))
        ref = hf_share(conds["joint prompt"])
        for name, p in conds.items():
            if not p.exists():
                continue
            table.setdefault(name, {})[seed] = round(hf_share(p) / ref, 3)
    names = list(table)
    print(f"{'condition':<16}" + "".join(f"{s:>7}" for s in SEEDS) + f"{'median':>9}")
    for n in names:
        vals = [table[n].get(s, float('nan')) for s in SEEDS]
        med = st.median(v for v in vals if v == v)
        print(f"{n:<16}" + "".join(f"{v:>7.3f}" for v in vals) + f"{med:>9.3f}")
    (OUT / "paired_hf_ratio.json").write_text(json.dumps(
        {"pair": PAIR, "hf_cutoff": HF_CUTOFF, "lambda": 1.2, "checkpoint": "r32 @ 30050",
         "ratio_to_joint_by_condition": table}, indent=1))
    print(f"\nwrote {OUT / 'paired_hf_ratio.json'}")


if __name__ == "__main__":
    main()
