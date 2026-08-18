#!/usr/bin/env python
"""When do the Mono and PoE paths part company, in a space that knows what a
fused animal is?

Raw-latent distance is noise-dominated early in the run, so the fork step
measured there could be an artefact of latent geometry. This reads the same
question through the scorer's two validated embedding spaces instead. For each
cell it takes the decoded per-step estimates of the finished picture (the
``frames/`` folder that ``decode_trajectory_frames.py`` writes beside each
``latent_trajectory.pt``), embeds every step with CLIP and with DINOv2
ViT-S/14, and computes at each step the cosine distance between the
correction-on arm (``call__rall``, which reproduces Mono) and the
correction-off arm (``call__roff``, pure PoE).

What the hypothesis says these curves should do, written before this ran:
near-flat before step ``FLAT_BEFORE_STEP``, steepest rise inside
``RISE_WINDOW`` (the fork-step measurement on raw latents put 15 of 19 cells
in 13..20), committed well before the end. The script reports, per cell and
per space, the step of steepest rise and whether it lands in the window, so
the check cannot be re-aimed after seeing the curves.

Reads frames only, no sampling and no VAE. Cost is the two embedding models
over ~50 frames per arm per cell.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/trajectory_divergence.py
    CUDA_VISIBLE_DEVICES=1 python scripts/trajectory_divergence.py \
        --cells 'a_cat__x__a_dog/*'
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CROSS_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cross")
OUT_DIR = Path(
    "/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses"
    "/trajectory_divergence"
)

ARM_ON = "call__rall"    # correction on at every step; reproduces Mono
ARM_OFF = "call__roff"   # correction off everywhere; pure PoE

# The pre-registered shape checks. The fork step measured on raw latents is 16,
# with 15 of 19 cells between 13 and 20. Written here, in the source, before
# the first run of this script.
FLAT_BEFORE_STEP = 10    # mean distance over steps 0..9 should be small
RISE_WINDOW = (13, 20)   # steepest rise should land here for most cells
# The rise step is read off a 5-step rolling median of the per-step increments,
# same smoothing width as F3, so one jittery step cannot claim the rise.
SMOOTH = 5


def frames_of(cell: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in sorted((cell / "frames").glob("step_*.png")):
        m = re.match(r"step_(\d+)\.png", p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def rolling_median(x: np.ndarray, w: int) -> np.ndarray:
    half = w // 2
    return np.array([
        np.median(x[max(0, i - half): i + half + 1]) for i in range(len(x))
    ])


def clip_embed_paths(paths: list[Path]) -> np.ndarray:
    from poe_repair.experiments.residual_diagnostics.metrics import clip_image_embed
    chunks = [clip_image_embed(paths[i:i + 32]) for i in range(0, len(paths), 32)]
    import torch
    return torch.cat(chunks).numpy()


_dino = None


def dino_embed_paths(paths: list[Path]) -> np.ndarray:
    """L2-normalised CLS embeddings under DINOv2 ViT-S/14, same recipe as
    scripts/cross_seed_lora_pooling/smoke_dino_distance.py."""
    import torch
    from PIL import Image
    from torchvision import transforms

    global _dino
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if _dino is None:
        model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", trust_repo=True,
        ).to(device=device, dtype=torch.float32)
        model.eval()
        _dino = model
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    feats = []
    for i in range(0, len(paths), 32):
        batch = torch.stack([
            preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + 32]
        ]).to(device)
        with torch.no_grad():
            feats.append(_dino(batch).float().cpu().numpy())
    out = np.concatenate(feats)
    return out / np.linalg.norm(out, axis=-1, keepdims=True).clip(min=1e-8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(CROSS_ROOT))
    ap.add_argument("--cells", help="glob over <pair>/seed_N")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    root = Path(args.root) / "pairs"
    seeds = sorted(
        s for pair in root.iterdir() if pair.is_dir()
        for s in pair.iterdir()
        if s.is_dir() and (s / ARM_ON / "frames").is_dir()
        and (s / ARM_OFF / "frames").is_dir()
    )
    if args.cells:
        seeds = [s for s in seeds
                 if fnmatch.fnmatch(str(s.relative_to(root)), args.cells)]
    if not seeds:
        print(f"no cells with decoded frames for both arms under {root}",
              file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for cell in seeds:
        on, off = frames_of(cell / ARM_ON), frames_of(cell / ARM_OFF)
        steps = sorted(set(on) & set(off))
        if len(steps) < 20:
            print(f"  {cell.relative_to(root)}: only {len(steps)} shared "
                  f"steps decoded, skipping", file=sys.stderr)
            continue
        paths = [on[s] for s in steps] + [off[s] for s in steps]
        row = {"pair": cell.parent.name, "seed": cell.name, "steps": steps}
        for space, embed in (("clip", clip_embed_paths),
                             ("dino", dino_embed_paths)):
            emb = embed(paths)
            a, b = emb[: len(steps)], emb[len(steps):]
            dist = 1.0 - (a * b).sum(axis=-1)

            # The raw pairwise distance is dominated early by the texture of
            # two mush images, so the timing check runs on the anchored read:
            # project every frame onto the axis from this cell's own PoE
            # endpoint to its Mono endpoint (the manifold-walk construction,
            # extended over time). Mush is largely orthogonal to that axis.
            e_off, e_on = b[-1], a[-1]
            axis = e_on - e_off
            axis_len2 = float((axis * axis).sum())
            pos_on = ((a - e_off) @ axis) / axis_len2
            pos_off = ((b - e_off) @ axis) / axis_len2
            sep = pos_on - pos_off
            smooth = rolling_median(sep, SMOOTH)
            inc = np.diff(smooth)
            rise_step = steps[int(np.argmax(inc)) + 1]
            early = float(np.mean(
                [d for s, d in zip(steps, sep) if s < FLAT_BEFORE_STEP]))
            row[space] = {
                "distance": [round(float(d), 5) for d in dist],
                "final": round(float(dist[-1]), 5),
                "pos_on": [round(float(v), 4) for v in pos_on],
                "pos_off": [round(float(v), 4) for v in pos_off],
                "separation": [round(float(v), 4) for v in sep],
                "early_sep_mean": round(early, 4),
                "rise_step": int(rise_step),
                "rise_in_window": bool(
                    RISE_WINDOW[0] <= rise_step <= RISE_WINDOW[1]),
            }
        results.append(row)
        print(f"  {row['pair']}/{row['seed']}: "
              + ", ".join(
                  f"{sp}: rise@{row[sp]['rise_step']}"
                  f"{'✓' if row[sp]['rise_in_window'] else '✗'}"
                  f" early_sep={row[sp]['early_sep_mean']:+.2f}"
                  for sp in ("clip", "dino")), flush=True)

    blob = {
        "arms": {"on": ARM_ON, "off": ARM_OFF},
        "distance": "1 - cosine, L2-normalised embeddings of decoded x0 frames",
        "flat_before_step": FLAT_BEFORE_STEP,
        "rise_window": list(RISE_WINDOW),
        "smooth": SMOOTH,
        "cells": results,
    }
    out_json = out_dir / "divergence.json"
    out_json.write_text(json.dumps(blob, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for col, space in enumerate(("clip", "dino")):
        ax = axes[0][col]
        for row in results:
            ax.plot(row["steps"], row[space]["separation"], lw=1.2, alpha=0.8,
                    label=f"{row['pair'].replace('a_', '').replace('an_', '')}"
                          f"/{row['seed']}")
        ax.set_title(f"{space.upper()}: arm separation along the "
                     f"PoE→Mono endpoint axis")
        ax.set_ylabel("pos(on) − pos(off), 1 = full axis")
        bx = axes[1][col]
        for row in results:
            d = np.array(row[space]["distance"])
            bx.plot(row["steps"], d / max(row[space]["final"], 1e-8),
                    lw=1.0, alpha=0.6)
        bx.set_title(f"{space.upper()}: raw pairwise distance / final "
                     f"(noise-dominated early)")
        bx.set_xlabel("denoising step (noise → image)")
        bx.set_ylabel("cosine distance / final")
        for a in (ax, bx):
            a.axvspan(RISE_WINDOW[0], RISE_WINDOW[1], color="0.85", zorder=0)
            a.axvline(16, color="0.5", lw=0.8, ls="--")
    axes[0][0].legend(fontsize=6, ncol=2)
    fig.suptitle("Fork band 13–20 shaded, raw-latent fork step 16 dashed")
    fig.tight_layout()
    fig.savefig(out_dir / "divergence_eyeball.png", dpi=150)

    n = len(results)
    for space in ("clip", "dino"):
        hits = sum(r[space]["rise_in_window"] for r in results)
        print(f"{space}: steepest rise inside steps {RISE_WINDOW[0]}–"
              f"{RISE_WINDOW[1]} for {hits}/{n} cells")
    print(f"wrote {out_json} and divergence_eyeball.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
