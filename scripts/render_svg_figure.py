"""Rasterise a hand-authored SVG figure to the PNG and PDF the paper uses.

    <co3-python> scripts/render_svg_figure.py paper/iclr/figures/lora-training.svg

Writes <stem>.png (scale 1.6) and <stem>.pdf beside the source.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cairosvg


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    src = Path(argv[1]).resolve()
    cairosvg.svg2png(url=str(src), write_to=str(src.with_suffix(".png")), scale=1.6)
    cairosvg.svg2pdf(url=str(src), write_to=str(src.with_suffix(".pdf")))
    print(f"wrote {src.with_suffix('.png')}")
    print(f"wrote {src.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
