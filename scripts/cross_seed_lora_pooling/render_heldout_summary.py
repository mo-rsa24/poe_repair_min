#!/usr/bin/env python3
"""Render one summary figure per cross-seed-LoRA held-out-pair eval.

Layout per pair:
  Left column   : PoE (top, green border) and Mono (bottom, red border)
                  from the eval cache at a chosen reference seed.
  Right column  : 2x2 grid of the four held-out-pair samples
                  (seeds 9, 10, 11, 12 in row-major order),
                  each labelled "seed NN" beneath.

Inputs are discovered by scanning for ``manifest.json`` files under
``heldout_pair`` subtrees in the cross_seed_lora_pooling output
directories. Both layouts are supported:

    <run-dir>/samples/heldout_pair/<eval_pair>/final/manifest.json
    outputs/cross_seed_lora_pooling/<train_pair>/heldout_pair/<eval_pair>/manifest.json

For each manifest:
  - PoE and Mono pulled from
    $POE_REPAIR_TRAINING_CACHE/heldout/<eval_pair>/seed_<ref_seed>/{poe,mono}.png
  - Samples pulled from each manifest entry's ``png`` field.
  - Output PNG written to <out-dir>/<train_pair>__heldout__<eval_pair>.png
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs")
DEFAULT_REPO_OUTPUTS = REPO_ROOT / "outputs"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "presentation" / "heldout_summary"

GRID_SEEDS = (9, 10, 11, 12)
GREEN = "#2ca02c"
RED = "#d62728"
BORDER_LW = 10


def discover_manifests(search_roots: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("manifest.json"):
            if "heldout_pair" not in path.parts:
                continue
            real = path.resolve()
            if real in seen:
                continue
            try:
                with path.open() as fh:
                    m = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            if not {"train_pair_slug", "pair_slug", "samples"}.issubset(m):
                continue
            seen.add(real)
            found.append(path)
    return found


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def draw_panel(ax, img: Image.Image, border_color: str | None) -> None:
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        if border_color is None:
            spine.set_visible(False)
        else:
            spine.set_edgecolor(border_color)
            spine.set_linewidth(BORDER_LW)


def render_one(
    manifest_path: Path,
    eval_cache_root: Path,
    out_dir: Path,
    ref_seed: int,
) -> Path | None:
    with manifest_path.open() as fh:
        m = json.load(fh)

    train_pair = m["train_pair_slug"]
    eval_pair = m["pair_slug"]
    joint_prompt = m.get("joint_prompt", eval_pair.replace("__x__", " x "))
    ckpt = m.get("checkpoint", "")

    samples_by_seed = {s["seed"]: Path(s["png"]) for s in m["samples"]}
    missing_samples = [s for s in GRID_SEEDS if s not in samples_by_seed]
    if missing_samples:
        print(f"[skip] {eval_pair}: missing sample seeds {missing_samples}")
        return None

    poe_path = eval_cache_root / "heldout" / eval_pair / f"seed_{ref_seed}" / "poe.png"
    mono_path = eval_cache_root / "heldout" / eval_pair / f"seed_{ref_seed}" / "mono.png"
    if not poe_path.exists() or not mono_path.exists():
        print(f"[skip] {eval_pair}: missing PoE/Mono refs at {poe_path.parent}")
        return None
    for seed, p in samples_by_seed.items():
        if not p.exists():
            print(f"[skip] {eval_pair}: missing sample png {p}")
            return None

    poe_img = load_image(poe_path)
    mono_img = load_image(mono_path)
    sample_imgs = {seed: load_image(samples_by_seed[seed]) for seed in GRID_SEEDS}

    fig = plt.figure(figsize=(11.5, 7.0), constrained_layout=False)
    fig.suptitle(
        f'"{joint_prompt}"   ·   train pair = {train_pair}   ·   eval pair = {eval_pair}',
        fontsize=11,
    )

    outer = fig.add_gridspec(
        nrows=1, ncols=2, width_ratios=[1.0, 2.05],
        wspace=0.06, left=0.03, right=0.985, top=0.90, bottom=0.05,
    )

    left = outer[0, 0].subgridspec(nrows=2, ncols=1, hspace=0.18)
    ax_poe = fig.add_subplot(left[0, 0])
    ax_mono = fig.add_subplot(left[1, 0])
    draw_panel(ax_poe, poe_img, GREEN)
    draw_panel(ax_mono, mono_img, RED)
    ax_poe.set_title(f"PoE (vanilla, seed {ref_seed})", fontsize=10, color=GREEN, pad=6)
    ax_mono.set_title(f"Mono (seed {ref_seed})", fontsize=10, color=RED, pad=6)

    right = outer[0, 1].subgridspec(nrows=2, ncols=2, hspace=0.22, wspace=0.06)
    for idx, seed in enumerate(GRID_SEEDS):
        r, c = divmod(idx, 2)
        ax = fig.add_subplot(right[r, c])
        draw_panel(ax, sample_imgs[seed], None)
        ax.set_xlabel(f"seed {seed:02d}", fontsize=10, labelpad=4)

    fig.text(
        0.5, 0.015,
        f"checkpoint: {ckpt}" if ckpt else "",
        ha="center", fontsize=7, color="#666666",
    )

    out_name = f"{train_pair}__heldout__{eval_pair}.png"
    out_path = out_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[ok]   {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(prog="render_heldout_summary")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--ref-seed", type=int, default=9,
                    help="Seed whose PoE/Mono refs are shown on the left column.")
    ap.add_argument(
        "--eval-cache-root", type=Path,
        default=Path(os.environ.get(
            "POE_REPAIR_TRAINING_CACHE",
            str(DEFAULT_DATASETS_ROOT / "training_cache"),
        )),
        help="Root containing heldout/<pair>/seed_N/{poe,mono}.png.",
    )
    ap.add_argument(
        "--search-root", type=Path, action="append", default=None,
        help="Roots to scan for manifest.json files (repeatable). "
             "Default scans both repo outputs/ and /datasets/.../outputs/.",
    )
    args = ap.parse_args()

    roots = args.search_root or [
        DEFAULT_REPO_OUTPUTS / "cross_seed_lora_pooling",
        DEFAULT_DATASETS_ROOT / "cross_seed_lora_pooling",
    ]
    manifests = discover_manifests(roots)
    if not manifests:
        print(f"no manifests under: {[str(r) for r in roots]}")
        return
    print(f"found {len(manifests)} manifest(s)")

    for path in manifests:
        try:
            render_one(path, args.eval_cache_root, args.out_dir, args.ref_seed)
        except Exception as exc:  # pragma: no cover
            print(f"[err]  {path}: {exc}")


if __name__ == "__main__":
    main()
