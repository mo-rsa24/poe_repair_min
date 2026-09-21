#!/usr/bin/env python
"""8 seeds x 3 columns: Mono (joint prompt) | PoE, LoRA off | PoE + LoRA. Pure compositing,
no GPU -- every panel is already rendered under OUT_ROOT by dog_x_dog_probe.py's mono/poe/probe
stages. This is the grid the plan's "What happens (visual)" section describes.

Usage:
    python scripts/showcase/dog_x_dog_grid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dog_x_dog_probe import HELD_OUT_SEEDS, OUT_ROOT  # noqa: E402

COLUMNS = [
    ("Mono (joint prompt)", OUT_ROOT / "mono"),
    ("PoE, LoRA off", OUT_ROOT / "poe"),
    ("PoE + LoRA", OUT_ROOT),
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main() -> int:
    thumb = 320
    col_header_h = 40
    row_label_w = 90

    grid = Image.new(
        "RGB",
        (row_label_w + thumb * len(COLUMNS), col_header_h + thumb * len(HELD_OUT_SEEDS)),
        "white",
    )
    draw = ImageDraw.Draw(grid)
    header_font = _font(18)
    row_font = _font(16)

    for c, (label, _) in enumerate(COLUMNS):
        x = row_label_w + c * thumb
        draw.text((x + 8, 10), label, fill="black", font=header_font)

    missing = []
    for r, seed in enumerate(HELD_OUT_SEEDS):
        y = col_header_h + r * thumb
        draw.text((8, y + thumb // 2 - 8), f"seed {seed}", fill="black", font=row_font)
        for c, (_, col_dir) in enumerate(COLUMNS):
            x = row_label_w + c * thumb
            img_path = col_dir / f"seed_{seed}.png"
            if img_path.exists():
                panel = Image.open(img_path).convert("RGB").resize((thumb, thumb))
            else:
                panel = Image.new("RGB", (thumb, thumb), "lightgray")
                missing.append(str(img_path))
            grid.paste(panel, (x, y))

    out_path = OUT_ROOT / "grid_mono_poe_lora.png"
    grid.save(str(out_path))
    print(f"[dog_x_dog_grid] wrote {out_path} "
          f"({len(HELD_OUT_SEEDS)} rows x {len(COLUMNS)} columns)")
    if missing:
        print(f"[dog_x_dog_grid] {len(missing)} panel(s) missing, filled gray:")
        for m in missing:
            print(f"  {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
