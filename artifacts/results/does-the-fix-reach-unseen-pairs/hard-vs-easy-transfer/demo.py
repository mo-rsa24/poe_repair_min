"""Demonstration: hard vs easy transfer, Phase 1 pooled LoRA, step 60000.

Query (verbatim): "could you /demonstrate 'Hard (lowest compose-rate as unseen
pairs): frog×toad (0.9375), eagle×hawk (0.9375), seal×walrus (0.9375)
Easy (perfect transfer): leopard×jaguar (1.0), and, from the 11 with no
direct read, ones like wolf×husky or lion×tiger as animal-family-diverse
easy cases' and create a (e.g qualitative /pair-figure) of each of them in
a single figure that shows mono|PoE|LoRA as 3 columns of a single row"

Read-only: composes the existing triptych PNGs the trainer already rendered
inline during Phase 1 training (compose_triptych in
poe_repair/experiments/cross_pair_lora_pooling/_inline_sampling.py). Does
not re-render or call any model code — the images already exist on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from poe_repair import paths

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = paths.resolve(paths.DOES_THE_FIX_REACH_UNSEEN_PAIRS) / "pooled_lora/phase1_r8_100k"
STEP_DIR = RUN_DIR / "samples/per_epoch/epoch_1200_step_060000"
COMPOSE_RATE_PATH = RUN_DIR / "compose_rate.json"
OUT_DIR = Path(__file__).resolve().parent

# (quadrant, pair_slug, seed, display_label) — ordered hard -> easy, per the
# query's own hard/easy split. Seed = lowest-numbered cached seed available
# for that pair (first in the trainer's own render order), NOT cherry-picked
# for outcome. in_in (train) pairs start seeds at 01; out_out (held-out)
# pairs start at 09 (see build_context / seed_pool.train_pool vs held_out).
ROWS = [
    ("out_out", "a_frog__x__a_toad", 9, "Frog x Toad"),
    ("out_out", "an_eagle__x__a_hawk", 9, "Eagle x Hawk"),
    ("out_out", "a_seal__x__a_walrus", 9, "Seal x Walrus"),
    ("out_out", "a_leopard__x__a_jaguar", 9, "Leopard x Jaguar"),
    ("in_in", "a_wolf__x__a_husky", 1, "Wolf x Husky"),
    ("in_in", "a_lion__x__a_tiger", 1, "Lion x Tiger"),
]

STEP = 60000

# The cached triptychs carry baked-in header strips (26 px title +
# 22 px panel labels with "@ step N", see compose_triptych); crop them
# off and draw clean column captions once instead.
BAKED_STRIP_H = 26 + 22
TRIP_PAD = 8
TRIP_THUMB = 320


def load_compose_rate() -> dict:
    return json.loads(COMPOSE_RATE_PATH.read_text())


def rate_label(compose_rate: dict, quadrant: str, pair_slug: str) -> str:
    if quadrant == "in_in":
        return "training pair, not held out"
    per_pair = compose_rate["per_step_heldout_pair"].get(pair_slug, {})
    v = per_pair.get(str(STEP))
    if v is None:
        return "no compose-rate at this step"
    return f"held-out compose-rate {v:.4f}"


def cmp_path(quadrant: str, pair_slug: str, seed: int) -> Path:
    return STEP_DIR / f"{quadrant}__{pair_slug}__seed{seed:02d}__cmp.png"


def build_figure(rows: list[tuple[str, str, int, str]]) -> Image.Image:
    tiles = []
    missing = []
    for quadrant, pair_slug, seed, label in rows:
        p = cmp_path(quadrant, pair_slug, seed)
        if not p.exists():
            missing.append(str(p))
            continue
        tiles.append((label, quadrant, pair_slug, p))

    if missing:
        raise FileNotFoundError(f"missing triptych files: {missing}")

    sample = Image.open(tiles[0][3])
    tile_w, tile_h = sample.size
    tile_h -= BAKED_STRIP_H
    row_label_w = 260
    pad = 10
    header_h = 30
    W = row_label_w + tile_w + 2 * pad
    H = header_h + len(tiles) * (tile_h + pad) + pad

    canvas = Image.new("RGB", (W, H), (14, 14, 14))
    draw = ImageDraw.Draw(canvas)
    try:
        font_label = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
    except Exception:
        font_label = ImageFont.load_default()

    for i, col in enumerate(["Mono (target)", "PoE", "LoRA"]):
        x = row_label_w + TRIP_PAD + i * (TRIP_THUMB + TRIP_PAD)
        draw.text((x, 8), col, fill=(200, 200, 200), font=font_label)

    y = header_h
    for label, quadrant, pair_slug, p in tiles:
        draw.text((pad, y + tile_h // 2 - 8), label,
                  fill=(255, 255, 255), font=font_label)
        im = Image.open(p).convert("RGB")
        im = im.crop((0, BAKED_STRIP_H, im.width, im.height))
        canvas.paste(im, (row_label_w, y))
        y += tile_h + pad

    return canvas


def main() -> int:
    print("=" * 70)
    print("Hard vs easy transfer, Phase 1 pooled LoRA, step 60000")
    print("=" * 70)
    compose_rate = load_compose_rate()
    print(f"{'pair':<22} {'quadrant':<8} {'seed':<5} {'label'}")
    for quadrant, pair_slug, seed, label in ROWS:
        p = cmp_path(quadrant, pair_slug, seed)
        exists = "OK" if p.exists() else "MISSING"
        sub = rate_label(compose_rate, quadrant, pair_slug)
        print(f"{label:<22} {quadrant:<8} {seed:<5} {sub}  [{exists}]")

    fig = build_figure(ROWS)
    out_path = OUT_DIR / "hard_vs_easy_transfer.png"
    fig.save(out_path)
    print()
    print(f"wrote {out_path}  ({fig.size[0]}x{fig.size[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
