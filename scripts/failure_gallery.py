#!/usr/bin/env python
"""The failure gallery: one panel per way a render goes wrong, each with the telling patch zoomed.

Every panel is a real render already on disk, named in `context/world/how-a-render-fails.md`. The
box drawn on the render is where the inset came from, so the look the caption names can be seen
rather than taken on trust: a face that is two animals at once, a person the prompt never asked
for, the third animal in the corner.

    python scripts/failure_gallery.py <tiles_dir> <out.png>

`tiles_dir` holds the renders pulled from the cluster, in the layout `scripts/across_adapters.py`
and the training runs write them.
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# (file, crop box on the 1024 render, what it looks like, what causes it, which render it is,
# and optionally the corner the inset sits in: "tl", "tr", "bl", "br". Without one it goes to the
# corner farthest from the box, which is right unless that corner holds something worth seeing.
PANELS = [
    ("across_adapters_seed09/00.png", (330, 330, 720, 720),
     "One fused animal",
     "no correction at all: plain product-of-experts composition",
     "cat and dog, seed 9, no adapter"),
    ("v57w10-01-base/samples/per_epoch/epoch_0200_step_010000/out_out__a_cat__x__a_dog__seed09.png",
     (120, 150, 450, 480),
     "Invented people",
     "the plain loss trained past its peak on a broad pool",
     "cat and dog, seed 9, v57w10-01-base at step 10,000"),
    ("across_adapters_seed09/01.png", (250, 250, 650, 650),
     "Grey mush, no animal in it",
     "trained far past its peak",
     "cat and dog, seed 9, rank 8 at step 450,000"),
    ("across_adapters_seed11/16.png", (300, 280, 750, 730),
     "Flat line art of one animal",
     "only the empty branch was free to move: it can restyle a picture, not place a second animal",
     "cat and dog, seed 11, only the empty branch adapted"),
    ("across_adapters_seed09/08.png", (830, 600, 1014, 784),
     "A third animal in the corner",
     "plurality learned without identity",
     "cat and dog, seed 9, 43 cells trained on all 50 steps", "bl"),
    ("across_adapters_seed11/18.png", (520, 180, 800, 460),
     "Two animals of the wrong species",
     "identity lost while plurality holds: a horse stands where the dog was asked for",
     "cat and dog, seed 11, picture target with a cached empty branch"),
]

TILE, ZOOM_FRAC, GAP, CAP, HEAD = 420, 0.42, 14, 96, 74
BOX = "#e8352e"


def font(size: int):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def panel(src: Path, crop: tuple[int, int, int, int], corner: str | None = None) -> Image.Image:
    """The render at tile size, with its crop drawn on it and blown up in the corner."""
    img = Image.open(src).convert("RGB")
    scale = TILE / img.width
    tile = img.resize((TILE, TILE), Image.LANCZOS)
    d = ImageDraw.Draw(tile)
    box = tuple(int(round(v * scale)) for v in crop)
    d.rectangle(box, outline=BOX, width=3)

    z = int(TILE * ZOOM_FRAC)
    inset = img.crop(crop).resize((z, z), Image.LANCZOS)
    # Named corner where the panel asks for one; otherwise the corner farthest from the box, so the
    # inset never covers what it zooms.
    named = {"tl": (4, 4), "tr": (TILE - z - 4, 4),
             "bl": (4, TILE - z - 4), "br": (TILE - z - 4, TILE - z - 4)}
    if corner:
        x0, y0 = named[corner]
    else:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        anchors = {named["tl"]: (0, 0), named["tr"]: (TILE, 0),
                   named["bl"]: (0, TILE), named["br"]: (TILE, TILE)}
        (x0, y0), _ = max(anchors.items(), key=lambda kv: (kv[1][0] - cx) ** 2 + (kv[1][1] - cy) ** 2)
    tile.paste(inset, (x0, y0))
    d.rectangle((x0, y0, x0 + z - 1, y0 + z - 1), outline=BOX, width=3)
    return tile


if __name__ == "__main__":
    tiles, out = Path(sys.argv[1]), Path(sys.argv[2])
    cols = 3
    rows = (len(PANELS) + cols - 1) // cols
    W = cols * TILE + (cols + 1) * GAP
    H = HEAD + rows * (TILE + CAP + GAP)
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    f_title, f_sub, f_look, f_cause = font(25), font(13), font(17), font(12)
    d.text((GAP, 14), "How a corrected render goes wrong", fill="black", font=f_title)
    d.text((GAP, 46),
           "Six looks, each from a render already on disk. The red box is the patch shown enlarged "
           "in the corner of the same panel.", fill="#444444", font=f_sub)

    record = {"panels": []}
    for i, (rel, crop, look, cause, which, *rest) in enumerate(PANELS):
        r, c = divmod(i, cols)
        x = GAP + c * (TILE + GAP)
        y = HEAD + r * (TILE + CAP + GAP)
        src = tiles / rel
        if not src.exists():
            d.rectangle((x, y, x + TILE, y + TILE), outline="#cccccc")
            d.text((x + 12, y + TILE // 2), f"missing: {rel}", fill="#999999", font=f_cause)
            continue
        canvas.paste(panel(src, crop, rest[0] if rest else None), (x, y))
        d.text((x, y + TILE + 8), look, fill="black", font=f_look)
        for k, line in enumerate(textwrap.wrap(cause, 62)):
            d.text((x, y + TILE + 32 + k * 15), line, fill="#333333", font=f_cause)
        d.text((x, y + TILE + 32 + 15 * max(1, len(textwrap.wrap(cause, 62)))),
               which, fill="#777777", font=f_cause)
        record["panels"].append({"look": look, "cause": cause, "render": which,
                                 "file": str(src), "crop": list(crop),
                                 "inset_corner": rest[0] if rest else "auto"})
    canvas.save(out)
    (out.with_suffix(".json")).write_text(json.dumps(record, indent=1))
    print("wrote", out, "with", len(record["panels"]), "panels")
