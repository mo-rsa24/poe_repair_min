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
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

MEASURE = "eval/tracking/embedding_drift/dino"

# The first two rows are the cells the runs log a drift for, so those tiles carry their number.
# The rest are pairs held out of every pool here, shown as renders only.
ROWS = [
    ("a_cat__x__a_dog", 9, "out_out/a_cat__x__a_dog/seed_09", "cat and dog, seed 9"),
    ("a_cat__x__a_dog", 10, "out_out/a_cat__x__a_dog/seed_10", "cat and dog, seed 10"),
    ("a_tiger__x__a_dog", 9, None, "tiger and dog, seed 9"),
    ("a_cat__x__a_fox", 9, None, "cat and fox, seed 9"),
    ("a_chess_board__x__an_hourglass", 9, None, "chess board and hourglass, seed 9"),
    ("a_picnic_basket__x__a_watermelon", 9, None, "picnic basket and watermelon, seed 9"),
]

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
    ("ours_rank8_w50", "phase1_r8_200k", ["rank 8, cross-attention", "0.19% of the model, 200k steps"]),
    ("ours_rank16_w50", "phase1_r16_100k", ["rank 16, cross-attention", "0.38% of the model, 100k steps"]),
    ("ours_rank32_w50", "phase1_r32_100k", ["rank 32, cross-attention", "0.77% of the model, 100k steps"]),
]

T, GAP, LEFT, HEAD, FOOT = 300, 8, 210, 62, 26


def font(size: int):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def drift(measures: dict, run: str | None, cell: str) -> float | None:
    if run is None or run not in measures:
        return None
    v = measures[run]["eval"].get(f"{MEASURE}/{cell}")
    return None if v is None else float(v[0])


def build(tiles: Path, measures: dict, columns, out: Path, title: str, subtitle: str) -> dict:
    W = LEFT + len(columns) * (T + GAP)
    H = HEAD + len(ROWS) * (T + GAP + FOOT) + 34
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    f_title, f_head, f_cell, f_row = font(21), font(12), font(12), font(13)
    d.text((14, 10), title, fill="black", font=f_title)
    d.text((14, 36), subtitle, fill="#444444", font=f_head)

    record = {"measure": MEASURE, "rows": [], "columns": [], "tiles": {}}
    for j, (tile, run, head) in enumerate(columns):
        x = LEFT + j * (T + GAP)
        for k, line in enumerate(head):
            d.text((x + 4, HEAD - 34 + k * 15), line, fill="black" if k == 0 else "#444444", font=f_head)
        record["columns"].append({"tile": tile, "run": run, "header": head,
                                  "wandb": (measures.get(run) or {}).get("wandb")})

    for i, (slug, seed, cell, label) in enumerate(ROWS):
        y = HEAD + i * (T + GAP + FOOT)
        d.text((10, y + T // 2 - 8), label, fill="black", font=f_row)
        record["rows"].append({"pair": slug, "seed": seed, "cell": cell,
                               "drift_logged": cell is not None})
        for j, (tile, run, _) in enumerate(columns):
            x = LEFT + j * (T + GAP)
            p = tiles / slug / f"seed{seed:02d}" / f"{tile}.png"
            if p.exists():
                canvas.paste(Image.open(p).convert("RGB").resize((T, T), Image.LANCZOS), (x, y))
                record["tiles"].setdefault(cell or f"{slug}/seed_{seed:02d}", {})[tile] = str(p)
            else:
                d.rectangle((x, y, x + T, y + T), outline="#cccccc")
                d.text((x + 10, y + T // 2), "not rendered", fill="#999999", font=f_cell)
            v = drift(measures, run, cell) if cell else None
            if v is not None:
                d.text((x + 4, y + T + 5), f"drift {v:+.3f}", fill="#222222", font=f_cell)
    d.text((14, H - 26),
           "drift = the render's distance to the joint-prompt column minus its distance to the "
           "plain-product column, in DINOv2 space; lower is nearer the target. It is logged only "
           "for the two cat-and-dog cells; the other rows are held out of every pool here and are "
           "shown as renders only.",
           fill="#444444", font=f_head)
    canvas.save(out)
    return record


if __name__ == "__main__":
    tiles, measures_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    measures = json.loads(measures_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    g1 = out_dir / "which-layers-carry-the-adapter-renders.png"
    r1 = build(tiles, measures, LAYERS_COLUMNS, g1,
               "Which layers carry the adapter: the same six held-out cells through each configuration",
               "all adapters rank 32 on the same pool, correction on all 50 steps; the cross-and-self "
               "column also decays its learning rate, so that comparison is mixed")
    (g1.with_suffix(".json")).write_text(json.dumps(r1, indent=1))

    g2 = out_dir / "how-much-rank-the-correction-needs-renders.png"
    r2 = build(tiles, measures, RANK_COLUMNS, g2,
               "How much rank the correction needs: the same six held-out cells at three ranks",
               "cross-attention only; these runs differ in loss and in training length as well as in "
               "rank, and each is shown at its last checkpoint, so the comparison is mixed; "
               "tiger is a word these adapters trained on, in the pair lion and tiger")
    (g2.with_suffix(".json")).write_text(json.dumps(r2, indent=1))
    print("wrote", g1, "and", g2)
