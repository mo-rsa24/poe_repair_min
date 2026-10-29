"""The SD 3.5 Medium version of the x sheet: references, x alone, and PoE with x, with the two diagnostic columns
that apply the arithmetic to the CLIP part only and the T5 part only. Reads the tiles scripts/sd35_and_vector_render.py wrote.

    python3 scripts/sd35_x_sheet.py <tiles_dir> <out.png>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from and_vector_sheet import sheet  # noqa: E402

COLUMNS = [
    ("a", ["{a}", "alone"], []),
    ("b", ["{b}", "alone"], []),
    ("joint", ["Joint prompt", "{joint}"], []),
    ("poe", ["PoE", "{a}, {b}"], []),
    ("x", ["x alone", ""], []),
    ("poe_x", ["PoE", "{a}, {b}, x"], []),
    ("poe_x_clip", ["PoE", "{a}, {b}, x (CLIP only)"], []),
    ("poe_x_t5", ["PoE", "{a}, {b}, x (T5 only)"], []),
]

if __name__ == "__main__":
    tiles, out = Path(sys.argv[1]), Path(sys.argv[2])
    sheet(COLUMNS, 'Stable Diffusion 3.5 Medium: a third expert x beside "dog" and "cat"', tiles, out,
          "dog", "cat", "dog and cat", list(range(1, 9)))
