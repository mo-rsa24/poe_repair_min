"""The "and" vector sheet with only the references and x (the vector the repo's code calls x_fix) as the third expert.
Reuses the tiles scripts/and_vector_sheet.py rendered; only the columns and their labels change.

    python3 scripts/and_vector_x_sheet.py <tiles_dir> <out.png>
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
    ("x_fix", ["x alone", ""], []),
    ("poe_x_fix", ["PoE", "{a}, {b}, x"], []),
]

if __name__ == "__main__":
    tiles, out = Path(sys.argv[1]), Path(sys.argv[2])
    sheet(COLUMNS, 'A third expert x beside "dog" and "cat"', tiles, out, "dog", "cat", "dog and cat", list(range(1, 9)))
