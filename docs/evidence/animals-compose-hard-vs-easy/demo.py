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

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = REPO_ROOT / "outputs/animals_compose_transfer/pooled_lora/phase1_r8_100k"
STEP_DIR = RUN_DIR / "samples/per_epoch/epoch_1200_step_060000"
COMPOSE_RATE_PATH = RUN_DIR / "compose_rate.json"
OUT_DIR = Path(__file__).resolve().parent

# (quadrant, pair_slug, seed, display_label) — ordered hard -> easy, per the
# query's own hard/easy split. Seed = lowest-numbered cached seed available
# for that pair (first in the trainer's own render order), NOT cherry-picked
# for outcome. in_in (train) pairs start seeds at 01; out_out (held-out)
# pairs start at 09 (see build_context / seed_pool.train_pool vs held_out).
ROWS = [
    ("out_out", "a_frog__x__a_toad", 9, "frog x toad"),
    ("out_out", "an_eagle__x__a_hawk", 9, "eagle x hawk"),
    ("out_out", "a_seal__x__a_walrus", 9, "seal x walrus"),
    ("out_out", "a_leopard__x__a_jaguar", 9, "leopard x jaguar"),
    ("in_in", "a_wolf__x__a_husky", 1, "wolf x husky"),
    ("in_in", "a_lion__x__a_tiger", 1, "lion x tiger"),
]

STEP = 60000


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

    compose_rate = load_compose_rate()

    sample = Image.open(tiles[0][3])
    tile_w, tile_h = sample.size
    row_label_w = 260
    pad = 10
    title_h = 46
    W = row_label_w + tile_w + 2 * pad
    H = title_h + len(tiles) * (tile_h + pad) + pad

    canvas = Image.new("RGB", (W, H), (14, 14, 14))
    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        font_label = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
        font_sub = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        font_title = font_label = font_sub = ImageFont.load_default()

    draw.text((pad, 12), "Transfer quality across held-out pairs, step 60000",
              fill=(255, 255, 255), font=font_title)

    y = title_h
    for label, quadrant, pair_slug, p in tiles:
        sub = rate_label(compose_rate, quadrant, pair_slug)
        draw.text((pad, y + tile_h // 2 - 18), label,
                  fill=(255, 255, 255), font=font_label)
        draw.text((pad, y + tile_h // 2 + 4), sub,
                  fill=(180, 180, 180), font=font_sub)
        im = Image.open(p).convert("RGB")
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
