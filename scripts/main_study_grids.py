#!/usr/bin/env python
"""The main study's comparison grids, one image per question, from renders already on disk.

Each grid is rows of cells against columns of methods, every tile a real render, and a row always
starts from the same noise so a difference across a row is the method rather than the seed. A
dotted rule marks where published methods end and this project's adapter begins.

    python scripts/main_study_grids.py <tiles_dir> <co3_dir> <models_dir> <out_dir>

`tiles_dir` is the pulled `ablation_grid` tree (`<pair>/seed<NN>/<column>.png`), `co3_dir` the CO3
renders for one prompt (`seed-<N>/<prompt>.png`), `models_dir` the four-model tree
(`<model>/<pair>/seed_<N>/{joint,poe}.png`).
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

T, GAP, LEFT, HEAD, FOOT = 300, 8, 250, 132, 22
OURS = "ours_v58self_w50"


def font(size: int):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def dotted_line(d: ImageDraw.ImageDraw, x: int, y0: int, y1: int) -> None:
    for y in range(y0, y1, 12):
        d.line((x, y, x, min(y + 6, y1)), fill="#444444", width=2)


def grid(rows, columns, out: Path, title: str, subtitle: str, rule_before: int | None = None) -> dict:
    """rows: (label, resolver) where resolver(column_key) returns a path or None."""
    W = LEFT + len(columns) * (T + GAP) + 24
    H = HEAD + len(rows) * (T + GAP) + 40
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    f_title, f_sub, f_head, f_row = font(21), font(12), font(13), font(13)
    d.text((14, 12), title, fill="black", font=f_title)
    for k, line in enumerate(textwrap.wrap(subtitle, 150)):
        d.text((14, 42 + k * 17), line, fill="#444444", font=f_sub)

    record = {"title": title, "columns": [c[0] for c in columns], "rows": [], "tiles": {}}
    for j, (key, head) in enumerate(columns):
        x = LEFT + j * (T + GAP) + (24 if rule_before is not None and j >= rule_before else 0)
        for k, line in enumerate(head):
            d.text((x + 4, HEAD - 34 + k * 15), line,
                   fill="black" if k == 0 else "#444444", font=f_head if k == 0 else f_sub)
    if rule_before is not None:
        x = LEFT + rule_before * (T + GAP) + 10
        dotted_line(d, x, HEAD - 40, H - 44)

    for i, (label, resolve) in enumerate(rows):
        y = HEAD + i * (T + GAP)
        for k, line in enumerate(textwrap.wrap(label, 28)):
            d.text((8, y + T // 2 - 18 + k * 17), line, fill="black", font=f_row)
        record["rows"].append(label)
        for j, (key, _) in enumerate(columns):
            x = LEFT + j * (T + GAP) + (24 if rule_before is not None and j >= rule_before else 0)
            p = resolve(key)
            if p and Path(p).exists():
                canvas.paste(Image.open(p).convert("RGB").resize((T, T), Image.LANCZOS), (x, y))
                record["tiles"].setdefault(label, {})[key] = str(p)
            else:
                d.rectangle((x, y, x + T, y + T), outline="#cccccc")
                d.text((x + 10, y + T // 2), "not rendered", fill="#999999", font=f_sub)
    canvas.save(out)
    (out.with_suffix(".json")).write_text(json.dumps(record, indent=1))
    print("wrote", out.name)
    return record


if __name__ == "__main__":
    tiles, co3, models, out_dir = (Path(p) for p in sys.argv[1:5])
    out_dir.mkdir(parents=True, exist_ok=True)

    def cell(pair: str, seed: int):
        return lambda key: tiles / pair / f"seed{seed:02d}" / f"{key}.png"

    three = [("mono", ["Joint prompt", "one prompt names both"]),
             ("poe", ["Plain product", "one prompt per concept, combined"]),
             (OURS, ["Ours", "the same product, through the adapter"])]

    # 1. The fix is not about animals.
    grid([("animals: cat and dog, seed 9", cell("a_cat__x__a_dog", 9)),
          ("animals: tiger and dog, seed 9", cell("a_tiger__x__a_dog", 9)),
          ("objects: chess board and hourglass, seed 9", cell("a_chess_board__x__an_hourglass", 9)),
          ("objects: picnic basket and watermelon, seed 9", cell("a_picnic_basket__x__a_watermelon", 9)),
          ("object and place: lighthouse and stormy sea, seed 9", cell("a_lighthouse__x__a_stormy_sea", 9)),
          ("object and place: tent and snowy mountain, seed 9", cell("a_tent__x__a_snowy_mountain", 9))],
         three, out_dir / "the-fix-is-not-about-animals.png",
         "Three kinds of pair, one method per column",
         "Every row starts from the same noise in all three columns. The plain product is the "
         "composition this work repairs; ours is that same product run through the adapter.")

    # 2. Where the joint prompt itself fails.
    grid([("chess board and hourglass, seed 10", cell("a_chess_board__x__an_hourglass", 10)),
          ("chess board and hourglass, seed 12", cell("a_chess_board__x__an_hourglass", 12)),
          ("cat and fox, seed 9", cell("a_cat__x__a_fox", 9)),
          ("cat and fox, seed 12", cell("a_cat__x__a_fox", 12)),
          ("dog and dog, seed 1", cell("a_dog__x__a_dog", 1)),
          ("elephant and penguin, seed 9", cell("an_elephant__x__a_penguin", 9))],
         three, out_dir / "where-the-joint-prompt-itself-fails.png",
         "Cells where the target is wrong: the joint prompt drops a concept or adds one",
         "The joint prompt is what the adapter was trained to match, so these rows ask whether the "
         "correction can beat its own target: the hourglass is missing, the cat is a second fox, "
         "and a person appears beside one dog.")

    # 3. The same concept twice.
    grid([("dog and dog, seed 1", cell("a_dog__x__a_dog", 1)),
          ("dog and dog, seed 2", cell("a_dog__x__a_dog", 2)),
          ("dog and dog, seed 9", cell("a_dog__x__a_dog", 9)),
          ("cat and cat, seed 9", cell("a_cat__x__a_cat", 9)),
          ("tiger and tiger, seed 9", cell("a_tiger__x__a_tiger", 9))],
         three, out_dir / "the-same-concept-twice.png",
         "One concept asked for twice, so the two experts agree",
         "Nothing here contends for the same region on semantic grounds, and the plain product "
         "still returns a single animal. Whether the adapter returns two says what it learned: "
         "plurality itself, or a way of separating two different concepts.")

    # 4. What the adapter costs on the prompt it never targets.
    grid([("cat and dog, seed 9", cell("a_cat__x__a_dog", 9)),
          ("tiger and dog, seed 9", cell("a_tiger__x__a_dog", 9)),
          ("chess board and hourglass, seed 9", cell("a_chess_board__x__an_hourglass", 9)),
          ("picnic basket and watermelon, seed 9", cell("a_picnic_basket__x__a_watermelon", 9))],
         [("mono", ["Joint prompt, base model", "the model as shipped"]),
          ("monolora_v58self", ["Joint prompt, through the adapter", "same prompt, same noise"])],
         out_dir / "what-the-adapter-costs-on-a-joint-prompt.png",
         "The adapter is never meant to touch a joint prompt. This is what it does to one anyway",
         "Both columns run the single prompt naming both concepts, from the same starting noise. "
         "A difference here is damage the adapter does outside the composition it was trained to "
         "repair.")

    # 5. Published models on the joint prompt, then ours.
    def model_tile(pair: str, seed: int):
        def resolve(key: str):
            if key == "co3":
                folder = co3.parent / pair.replace("__x__", "_and_")
                hit = list(folder.glob(f"seed-{seed}/*.png")) if folder.exists() else []
                if not hit:
                    hit = list(co3.glob(f"seed-{seed}/*.png"))
                return hit[0] if hit else None
            if key.startswith("ours"):
                return tiles / pair / f"seed{seed:02d}" / f"{key}.png"
            if key == "sdxl_poe":
                return models / "sdxl" / pair / f"seed_{seed}" / "poe.png"
            return models / key / pair / f"seed_{seed}" / "joint.png"
        return resolve

    grid([("cat and dog, seed 42", model_tile("a_cat__x__a_dog", 42)),
          ("butterfly and flower meadow, seed 42", model_tile("a_butterfly__x__a_flower_meadow", 42)),
          ("camel and forest, seed 42", model_tile("a_camel__x__a_forest", 42))],
         [("sd14", ["SD 1.4", "joint prompt"]),
          ("sd21", ["SD 2.1", "joint prompt"]),
          ("sd35", ["SD 3.5", "joint prompt"]),
          ("sdxl", ["SDXL", "joint prompt"]),
          ("sdxl_poe", ["SDXL", "plain product"]),
          ("co3", ["CO3", "product with its corrector"]),
          (OURS, ["Ours", "product through the adapter"])],
         out_dir / "published-models-then-ours.png",
         "The same request of five published settings, and of this work",
         "Everything left of the dotted rule is a published model or method; the column to its "
         "right is this project's adapter on the plain product. CO3 starts from its own noise, so "
         "its tile is not pixel-comparable with the rest of the row.",
         rule_before=6)
