#!/usr/bin/env python
"""The two ablation grids as renders: one row per held-out cell, one column per configuration.

The bar figures beside these say how far each configuration moved toward the joint-prompt render;
these say what that looks like. Every tile is a real render of that cell through that checkpoint,
from the same starting noise as its neighbours, produced by `scripts/figure_candidates.py`.

The first two columns are the references the measure is defined against: the joint-prompt render the
correction aims at, and the plain product it repairs. Each column header carries the run, the layers
it adapts and its trainable-parameter share, and each tile carries that cell's DINOv2 drift, so the
picture and the number sit together.

    python scripts/adapter_ablation_grids.py <tiles_dir> <measures.json> <out_dir>
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

MEASURE = "eval/tracking/embedding_drift/dino"

# One grid per seed: these pairs are the rows, and the seed is fixed for the whole image, so a
# starting noise that draws a sketch or a photograph is a property of the sheet rather than a
# difference between rows. `DRIFT_CELLS` names the cells the runs log a drift for; every other
# tile is a render with no logged number.
PAIRS = [
    ("a_cat__x__a_dog", "cat and dog"),
    ("a_tiger__x__a_dog", "tiger and dog"),
    ("a_cat__x__a_fox", "cat and fox"),
    ("a_chess_board__x__an_hourglass", "chess board and hourglass"),
    ("a_picnic_basket__x__a_watermelon", "picnic basket and watermelon"),
]
SEEDS = [2, 9, 11, 12, 42]
DRIFT_CELLS = {("a_cat__x__a_dog", 9): "out_out/a_cat__x__a_dog/seed_09",
               ("a_cat__x__a_dog", 10): "out_out/a_cat__x__a_dog/seed_10"}
# Seed 2 is in the training seed pool, so its starting noise was seen during training on other
# pairs; the pairs on these sheets are held out either way. Seeds 9 to 16 are held out as well.
TRAIN_SEEDS = {1, 2, 3, 4, 5, 6, 7, 8, 18, 26, 35, 42}

# (tile name, run folder name, header lines). A run of None is a reference column with no measure.
LAYERS_COLUMNS = [
    ("mono", None, ["joint prompt", "the target the correction aims at"]),
    ("poe", None, ["plain product", "what the correction repairs"]),
    ("ours_crossw25_w50", "v57-08-x0-w25", ["cross-attention, rank 32", "trained on steps 0 to 25"]),
    ("ours_crossseeds_w50", "v57-09-x0-seeds", ["cross-attention, rank 32", "every cached seed"]),
    ("ours_crosscontrast_w50", "v57-10-x0-contrast", ["cross-attention, rank 32", "contrast term at 0.01"]),
    ("ours_crossself_w50", "v57-11b-x0-self-decay", ["cross + self-attention, rank 32", "mixed: also decays its rate"]),
]
RANK_COLUMNS = [
    ("mono", None, ["joint prompt", "the target the correction aims at"]),
    ("poe", None, ["plain product", "what the correction repairs"]),
    ("ours_rank8best_w50", "phase1_r8_200k", ["rank 8, cross-attention", "0.19% of the model, step 105k"]),
    ("ours_rank16best_w50", "phase1_r16_100k", ["rank 16, cross-attention", "0.38% of the model, step 50k"]),
    ("ours_rank32best_w50", "phase1_r32_100k", ["rank 32, cross-attention", "0.77% of the model, step 30k"]),
]

T, GAP, LEFT, HEAD, FOOT = 300, 8, 268, 104, 26


def font(size: int):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def drift(measures: dict, run: str | None, cell: str) -> float | None:
    if run is None or run not in measures:
        return None
    v = measures[run]["eval"].get(f"{MEASURE}/{cell}")
    return None if v is None else float(v[0])


def build(tiles: Path, measures: dict, columns, out: Path, seed: int, title: str,
          subtitle: str) -> dict:
    W = LEFT + len(columns) * (T + GAP)
    H = HEAD + len(PAIRS) * (T + GAP + FOOT) + 34
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    f_title, f_head, f_cell, f_row = font(21), font(12), font(12), font(13)
    seen = " (a seed the pool trained on, with other pairs)" if seed in TRAIN_SEEDS else " (a held-out seed)"
    d.text((14, 10), f"{title} — seed {seed}{seen}", fill="black", font=f_title)
    for k, line in enumerate(textwrap.wrap(subtitle, 190)):
        d.text((14, 38 + k * 16), line, fill="#444444", font=f_head)

    record = {"measure": MEASURE, "seed": seed, "rows": [], "columns": [], "tiles": {}}
    for j, (tile, run, head) in enumerate(columns):
        x = LEFT + j * (T + GAP)
        for k, line in enumerate(head):
            d.text((x + 4, HEAD - 36 + k * 16), line, fill="black" if k == 0 else "#444444", font=f_head)
        record["columns"].append({"tile": tile, "run": run, "header": head,
                                  "wandb": (measures.get(run) or {}).get("wandb")})

    for i, (slug, label) in enumerate(PAIRS):
        y = HEAD + i * (T + GAP + FOOT)
        cell = DRIFT_CELLS.get((slug, seed))
        for k, line in enumerate(textwrap.wrap(label, 30)):
            d.text((10, y + T // 2 - 18 + k * 17), line, fill="black", font=f_row)
        record["rows"].append({"pair": slug, "seed": seed, "cell": cell,
                               "drift_logged": cell is not None})
        for j, (tile, run, _) in enumerate(columns):
            x = LEFT + j * (T + GAP)
            p = tiles / slug / f"seed{seed:02d}" / f"{tile}.png"
            if p.exists():
                canvas.paste(Image.open(p).convert("RGB").resize((T, T), Image.LANCZOS), (x, y))
                record["tiles"].setdefault(f"{slug}/seed_{seed:02d}", {})[tile] = str(p)
            else:
                d.rectangle((x, y, x + T, y + T), outline="#cccccc")
                d.text((x + 10, y + T // 2), "not rendered", fill="#999999", font=f_cell)
            v = drift(measures, run, cell) if cell else None
            if v is not None:
                d.text((x + 4, y + T + 5), f"drift {v:+.3f}", fill="#222222", font=f_cell)
    d.text((14, H - 26),
           "drift = the render's distance to the joint-prompt column minus its distance to the "
           "plain-product column, in DINOv2 space; lower is nearer the target. It is logged only "
           "for the cat-and-dog cells at seeds 9 and 10; every other tile is a render with no "
           "logged number.",
           fill="#444444", font=f_head)
    canvas.save(out)
    return record


if __name__ == "__main__":
    tiles, measures_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    measures = json.loads(measures_path.read_text())
    seeds = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else SEEDS
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        g1 = out_dir / f"which-layers-carry-the-adapter-seed{seed:02d}.png"
        r1 = build(tiles, measures, LAYERS_COLUMNS, g1, seed,
                   "Which layers carry the adapter",
                   "five held-out pairs from one starting noise, all adapters rank 32 on the same "
                   "pool, correction on all 50 steps; the cross-and-self column also decays its "
                   "learning rate, so that comparison is mixed")
        (g1.with_suffix(".json")).write_text(json.dumps(r1, indent=1))

        g2 = out_dir / f"how-much-rank-the-correction-needs-seed{seed:02d}.png"
        r2 = build(tiles, measures, RANK_COLUMNS, g2, seed,
                   "How much rank the correction needs",
                   "cross-attention only, each rank at its best surviving checkpoint; rank 8's "
                   "early checkpoints were deleted, so its earliest survivor is step 105k, already "
                   "past the peak this project reported, and the comparison is mixed in loss and "
                   "in training length as well")
        (g2.with_suffix(".json")).write_text(json.dumps(r2, indent=1))
        print("wrote", g1.name, "and", g2.name)
