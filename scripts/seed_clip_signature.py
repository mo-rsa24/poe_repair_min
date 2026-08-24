#!/usr/bin/env python
"""Hypothesis A, single-pair diagnostic: does CLIP image space separate
composed from blended seeds where the correction's own curve-shape already
failed?

Same 12 seeds, same labels, same leave-one-out-plus-shuffled-null test as
scripts/seed_curve_signature.py, so this is a direct, apples-to-apples
rematch, not a fresh question. Only the feature changes: CLIP's image
embedding of the actual rendered picture, in place of the correction's own
size/turn curves.

Label source, unchanged from the existing script: whether that seed composed
with the correction applied in the earliest window (0-10), from the 12-seed
sweep, outputs/interaction_term/window_seeds/seed_breadth.json. Same images
that labeling already scored are the ones embedded here.

The honest guard, copied from the existing script rather than re-invented:
12 points is too few to trust an in-sample fit, so nothing is fitted. The
real leave-one-out accuracy is read against a shuffled-label null built from
many random relabelings of the same 12 points, not against 50%.

    python scripts/seed_clip_signature.py
    python scripts/seed_clip_signature.py --pair a_cat__x__a_dog
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from poe_repair import paths

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LABELS = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/"
              "window_seeds/seed_breadth.json")
IMAGE_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/"
                  "window_seeds/pairs")
IMAGE_TAG = "teacher_residual_const_lam100_w0-10"
OUT_DIR = paths.resolve(paths.CACHE_ANALYSES)
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"   # same checkpoint used elsewhere in this project
N_SHUFFLES = 2000


def loo_nearest_neighbour(X: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out 1-NN accuracy. Nothing is fitted: the held-out point is
    classified by the nearest of the others, so a perfect fit on 12 points
    cannot inflate it. Copied verbatim from seed_curve_signature.py so the
    two results are comparable on more than just the label."""
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    ok = 0
    for i in range(len(y)):
        d = np.linalg.norm(Z - Z[i], axis=1)
        d[i] = np.inf
        ok += int(y[int(np.argmin(d))] == y[i])
    return ok / len(y)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", default="a_cat__x__a_dog")
    ap.add_argument("--seed-rng", type=int, default=0,
                    help="numpy seed for the shuffled-label null, for reproducibility")
    args = ap.parse_args()

    lab = json.loads(LABELS.read_text())
    if args.pair not in lab["composed"]:
        raise SystemExit(f"no labels for {args.pair} in {LABELS}")
    outcomes = {int(k): v for k, v in lab["composed"][args.pair].items()}

    seeds, pngs, y = [], [], []
    for s, composed in sorted(outcomes.items()):
        png_dir = IMAGE_ROOT / args.pair / f"seed_{s}" / IMAGE_TAG
        hits = sorted(png_dir.glob("*.png"))
        if not hits:
            continue
        seeds.append(s); pngs.append(hits[0]); y.append(int(composed))
    y = np.asarray(y)
    if len(seeds) < 6 or y.std() == 0:
        raise SystemExit(f"only {len(seeds)} usable seeds, or all one class")
    print(f"{args.pair}: {len(seeds)} seeds with both a cached label and an image")
    print(f"  composed: {sorted(s for s, v in zip(seeds, y) if v)}")
    print(f"  failed:   {sorted(s for s, v in zip(seeds, y) if not v)}\n")

    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    feats = []
    with torch.no_grad():
        for png in pngs:
            im = Image.open(png).convert("RGB")
            inputs = processor(images=im, return_tensors="pt").to(device)
            emb = model.get_image_features(**inputs)[0].float().cpu().numpy()
            feats.append(emb)
    X = np.stack(feats)   # [n_seeds, 512]

    real_acc = loo_nearest_neighbour(X, y)

    rng = np.random.default_rng(args.seed_rng)
    null_accs = np.empty(N_SHUFFLES)
    for i in range(N_SHUFFLES):
        y_shuf = rng.permutation(y)
        null_accs[i] = loo_nearest_neighbour(X, y_shuf)
    null_median = float(np.median(null_accs))
    percentile = float((null_accs <= real_acc).mean() * 100)

    print(f"CLIP-image leave-one-out accuracy   {real_acc:.3f}  ({int(round(real_acc*len(y)))}/{len(y)})")
    print(f"shuffled-label null, {N_SHUFFLES} draws, median   {null_median:.3f}")
    print(f"real result sits at the {percentile:.0f}th percentile of the null")
    if real_acc > null_median + 0.15:
        print("reads as a real signal: clearly above what shuffled labels manage")
    elif real_acc < null_median - 0.15:
        print("reads as below chance, same direction as the r_t-curve result")
    else:
        print("reads as indistinguishable from the shuffled-label null: no signal found")
    print("\nfor direct comparison, scripts/seed_curve_signature.py's own result on the "
         "same seeds and labels, using the correction's curves instead of images: "
         "25% real accuracy, below its shuffled-null median of 42-50%")

    out_json = {
        "pair": args.pair, "seeds": seeds, "composed": y.tolist(),
        "clip_model": CLIP_MODEL_ID, "image_tag": IMAGE_TAG,
        "real_loo_accuracy": real_acc, "n_shuffles": N_SHUFFLES,
        "null_median": null_median, "null_accuracies": null_accs.tolist(),
        "percentile_of_real_in_null": percentile,
        "comparison": {"seed_curve_signature_real_accuracy": 0.25,
                      "seed_curve_signature_null_median_range": [0.42, 0.50]},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"seed_clip_signature_{args.pair}.json").write_text(json.dumps(out_json, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    ax.hist(null_accs, bins=np.linspace(0, 1, 21), color="0.75", edgecolor="white",
           label=f"shuffled-label null, {N_SHUFFLES} draws")
    ax.axvline(real_acc, color="#1f77b4", lw=2.2,
              label=f"real labels, CLIP image space ({real_acc:.2f})")
    ax.axvline(0.25, color="#d62728", lw=1.6, ls="--",
              label="real labels, correction curves (0.25, already run)")
    ax.set_xlabel("leave-one-out accuracy", fontsize=9, family="serif")
    ax.set_ylabel("count of shuffles", fontsize=9, family="serif")
    ax.set_title(f"{args.pair.replace('__x__', ' x ').replace('a_','').replace('an_','')}: "
                f"does CLIP see what the correction's shape could not?",
                fontsize=9, family="serif", loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    out_png = OUT_DIR / f"seed_clip_signature_{args.pair}.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"\nfigure     {out_png}")
    print(f"sidecar    {OUT_DIR / f'seed_clip_signature_{args.pair}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
