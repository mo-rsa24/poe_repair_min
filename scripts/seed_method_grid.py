#!/usr/bin/env python
"""Seeds down the page, methods across: one grid per pair, from the tiles seed_method_grid.sbatch rendered.

The joint prompt, the plain product and every adapter column in a row start from the same cached
noise for that seed; CO3 draws its own noise from the same seed number, so its tile is not
pixel-comparable with the rest of the row. A dotted rule separates the published methods from the
adapters.

    python scripts/seed_method_grid.py <tiles_dir> <co3_dir> <out_dir> [--picked]

`--picked` builds only the five hand-picked pair-and-seed rows in `PICKED`, reward KL K0 alone as
the adapter, no row labels.

`tiles_dir` is the pulled `ablation_grid` tree, `co3_dir` the pulled `showcase/co3_seed_grid` tree.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_study_grids import grid  # noqa: E402

SEEDS = [1, 2, 4, 9, 10, 11, 12, 13, 14, 15]
PAIRS = [("an_elephant__x__a_penguin", "elephant and penguin"),
         ("a_cat__x__a_dog", "cat and dog"),
         ("a_tiger__x__a_dog", "tiger and dog"),
         ("a_cat__x__a_teapot", "cat and teapot"),
         ("a_horse__x__a_bicycle", "horse and bicycle"),
         ("a_picnic_basket__x__a_watermelon", "picnic basket and watermelon"),
         ("a_camel__x__a_forest", "camel and forest")]
# (tile key, column heading): the key is the name the sbatch gave the tile.
ADAPTERS = [("v57-11b-x0-self-decay_s050000", "v57 self, decay", "50k"),
            ("v58-01-plain_s047500", "v58 plain", "47.5k"),
            ("v58-02-contrast_s027500", "v58 contrast", "27.5k"),
            ("v58-03-self_s030000", "v58 self", "30k"),
            ("v58-05-self-contrast_s015000", "v58 self + contrast", "15k"),
            ("v58f-01-plain_s020000", "v58f plain", "20k"),
            ("v58f-05-self-contrast_s025000", "v58f self + contrast", "25k"),
            ("R1_s050000", "R1 out-of-span only", "50k, experts 3.0"),
            ("R2a_s050000", "R2a on-policy states", "50k"),
            ("R2b_s050000", "R2b plain states", "50k"),
            ("refl-03-self-contrast-15k_s001250", "reward refl-03", "1,250 steps"),
            ("refl-kl-K0_s002000", "reward KL K0", "2,000 steps"),
            ("refl-kl-K1_s002000", "reward KL K1", "2,000 steps"),
            ("refl-kl-K2_s002000", "reward KL K2", "2,000 steps")]


# One hand-picked seed per pair, and only reward KL K0 as the adapter: the short figure.
PICKED = [("a_horse__x__a_bicycle", 2), ("a_camel__x__a_forest", 12), ("a_cat__x__a_teapot", 10),
          ("a_tiger__x__a_dog", 13), ("a_picnic_basket__x__a_watermelon", 10)]


def resolver(tiles: Path, co3: Path, slug: str, seed: int):
    def resolve(key: str):
        if key == "co3":
            hit = sorted((co3 / slug).rglob(f"seed-{seed}/*.png")) if (co3 / slug).exists() else []
            return hit[0] if hit else None
        return tiles / slug / f"seed{seed:02d}" / f"{key}.png"
    return resolve


def picked(tiles: Path, co3: Path, out_dir: Path):
    columns = [("mono", ["Joint prompt", "one prompt names both"]),
               ("poe", ["Plain product", "one prompt per concept"]),
               ("co3", ["CO3", "own noise"]),
               ("ours_refl-kl-K0_s002000_w50", ["Adapter", "the product through it"])]
    # Empty labels in the side-label layout leave a thin blank margin instead of the label column.
    rows = [("", resolver(tiles, co3, slug, seed)) for slug, seed in PICKED]
    out = out_dir / "five-picked-pairs.png"
    record = grid(rows, columns, out, "", "", rule_before=3, vertical_labels=True)
    # grid() keys tiles by row label, and every label here is empty, so name the rows in the sidecar.
    record["rows"] = [f"{slug} seed {seed}" for slug, seed in PICKED]
    record["tiles"] = {f"{slug} seed {seed}": {k: str(p) for k, _ in columns if (p := resolver(tiles, co3, slug, seed)(k))}
                       for slug, seed in PICKED}
    out.with_suffix(".json").write_text(json.dumps(record, indent=1))


def main():
    tiles, co3, out_dir = (Path(p) for p in sys.argv[1:4])
    out_dir.mkdir(parents=True, exist_ok=True)
    if "--picked" in sys.argv[4:]:
        return picked(tiles, co3, out_dir)
    columns = [("mono", ["Joint prompt", "one prompt names both"]),
               ("poe", ["Plain product", "one prompt per concept"]),
               ("co3", ["CO3", "own noise"])]
    columns += [(f"ours_{k}_w50", [h, sub]) for k, h, sub in ADAPTERS]

    for slug, name in PAIRS:
        rows = [(f"seed {s}", resolver(tiles, co3, slug, s)) for s in SEEDS]
        grid(rows, columns, out_dir / f"{slug.replace('__x__', '-and-').replace('_', '-')}.png",
             f"{name}: ten seeds, every method",
             "Rows are seeds. Every column except CO3 starts a row from the same cached noise, so a "
             "difference along a row is the method. Adapter columns are the product through that run's "
             "adapter at the step named under it, correction on all 50 steps.",
             rule_before=3)


if __name__ == "__main__":
    main()
