#!/usr/bin/env python
"""The paper's main grid: four pairs, four methods, the best available adapter tile per row.

Columns are the joint prompt, the plain product, CO3, and the product through the adapter. The
first, second and fourth columns of a row share the starting noise; CO3 draws its own. Each
"ours" tile is picked per row; the run and step behind each pick are the `ROWS` list below, and
the tiles carry no chip. Row labels are rotated to read bottom-up, which keeps the left margin to one line of text.

    python scripts/paper_main_grid.py <tiles_dir> <co3_dir> <out.png>

`tiles_dir` holds `<pair>/seed<NN>/<key>.png`; `co3_dir` holds `<pair>/.../seed-<N>/*.png` (CO3 nests its own algorithm and prompt folders).
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

T, GAP, LEFT, HEAD = 300, 8, 44, 40
ROWS = [  # (label, pair slug, seed, ours key)
    ("cat and dog", "a_cat__x__a_dog", 9, "ours_crossself_w50"),
    ("elephant and penguin", "an_elephant__x__a_penguin", 10, "ours_v58-05-self-contrast_s015000_rawA_w50"),
    ("tent and snowy mountain", "a_tent__x__a_snowy_mountain", 9, "ours_v58self_w50"),
    ("lighthouse and stormy sea", "a_lighthouse__x__a_stormy_sea", 9, "ours_v58self_w50"),
]
COLUMNS = [("mono", "Joint prompt", "one prompt names both"),
           ("poe", "Plain product", "one prompt per concept, combined"),
           ("co3", "CO3", "product with its corrector, own noise"),
           ("ours", "Ours", "the same product, through the adapter")]


def font(size: int):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def main():
    tiles, co3, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    W = LEFT + len(COLUMNS) * (T + GAP)
    H = HEAD + len(ROWS) * (T + GAP)
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    f_h, f_s, f_r = font(20), font(13), font(22)
    for j, (_, title, sub) in enumerate(COLUMNS):
        x = LEFT + j * (T + GAP)
        d.text((x, 8), title, fill="black", font=f_h)
    for i, (label, slug, seed, ours_key) in enumerate(ROWS):
        y = HEAD + i * (T + GAP)
        # the row label, rotated to read bottom-up along the tile's left edge
        lab = Image.new("RGBA", (T, 32), (255, 255, 255, 0))
        ImageDraw.Draw(lab).text((0, 2), label, fill="black", font=f_r)
        canvas.paste(lab.rotate(90, expand=True), (LEFT - 40, y), lab.rotate(90, expand=True))
        for j, (key, _, _) in enumerate(COLUMNS):
            x = LEFT + j * (T + GAP)
            if key == "co3":
                folder = co3 / slug
                hits = sorted(folder.rglob(f"seed-{seed}/*.png")) if folder.exists() else []
                p = hits[0] if hits else None
            elif key == "ours":
                p = tiles / slug / f"seed{seed:02d}" / f"{ours_key}.png"
            else:
                p = tiles / slug / f"seed{seed:02d}" / f"{key}.png"
            if p is not None and p.exists():
                canvas.paste(Image.open(p).convert("RGB").resize((T, T), Image.LANCZOS), (x, y))
            else:
                d.rectangle((x, y, x + T, y + T), outline="#cccccc")
                d.text((x + T // 2 - 38, y + T // 2 - 8), "not rendered", fill="#999999", font=f_s)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"wrote {out} {canvas.size}")


if __name__ == "__main__":
    main()
