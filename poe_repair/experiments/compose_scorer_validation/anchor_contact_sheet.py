"""Build a per-pair anchor contact sheet (compose-scorer plan 01, task 4).

Three anchors side by side per pair: A-alone, B-alone, joint, each labelled with
its prompt. One PNG per validation pair under the pair's anchors dir.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from poe_repair import paths

REPO = Path(__file__).resolve().parents[3]
ANCHOR_ROOT = paths.resolve(paths.COMPOSE_SCORER_VALIDATION) / "anchors"
PAIRS = ["a_cat__x__a_dog", "a_wolf__x__a_husky"]
EDGE = 384
PAD = 12
LABEL_H = 28


def _thumb(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail((EDGE, EDGE))
    canvas = Image.new("RGB", (EDGE, EDGE), (0, 0, 0))
    canvas.paste(im, ((EDGE - im.width) // 2, (EDGE - im.height) // 2))
    return canvas


def build_for_pair(pair: str) -> Path:
    d = ANCHOR_ROOT / pair
    manifest = json.loads((d / "anchors_manifest.json").read_text())
    prompts = manifest["prompts"]
    order = ["a_alone", "b_alone", "joint"]

    cols = len(order)
    canvas_w = cols * EDGE + (cols + 1) * PAD
    canvas_h = EDGE + 2 * PAD + LABEL_H
    canvas = Image.new("RGB", (canvas_w, canvas_h), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)

    for i, key in enumerate(order):
        x = PAD + i * (EDGE + PAD)
        canvas.paste(_thumb(d / f"anchor_{key}.png"), (x, PAD))
        label = f"{key}: '{prompts[key]}'"
        draw.text((x + 4, PAD + EDGE + 6), label, fill=(255, 255, 255))

    out = d / "anchor_contact_sheet.png"
    canvas.save(out)
    return out


def main() -> int:
    for pair in PAIRS:
        out = build_for_pair(pair)
        assert out.exists() and out.stat().st_size > 0, f"empty sheet {out}"
        print(f"wrote {out}  ({out.stat().st_size}B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
