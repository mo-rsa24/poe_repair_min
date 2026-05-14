"""Validate and summarise dataset/labels.json after hand-labelling.

Reads ``dataset/labels.json`` (which the user copies from
``labels.template.json`` and fills in by hand). Reports:

  - Schema validity (every cell has a ``teacher`` set to one of
    ``mono`` | ``co3`` | ``none``).
  - Counts per pair × teacher.
  - Total working cells (i.e. teacher != "none") — these are the ones
    that will produce per-step residuals when we run the data-generation
    step next.

Exits non-zero if any cell still has ``teacher: null`` or an invalid
value.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS = REPO_ROOT / "dataset" / "labels.json"

VALID_TEACHERS = {"mono", "co3", "none"}


def main() -> int:
    if not LABELS.exists():
        print(f"[labels] {LABELS} not found.")
        print("        Copy dataset/labels.template.json -> dataset/labels.json")
        print("        and fill in 'teacher' per cell.")
        return 2

    data = json.loads(LABELS.read_text())
    cells = data.get("cells", [])
    if not cells:
        print("[labels] no cells in labels.json")
        return 2

    bad_rows: list[dict] = []
    by_pair: dict[str, Counter] = defaultdict(Counter)
    by_split: dict[str, Counter] = defaultdict(Counter)
    total = Counter()

    for c in cells:
        t = c.get("teacher")
        slug = c.get("slug", "?")
        split = c.get("split", "?")
        if t not in VALID_TEACHERS:
            bad_rows.append(c)
            continue
        by_pair[slug][t] += 1
        by_split[split][t] += 1
        total[t] += 1

    print(f"[labels] {LABELS}")
    print(f"  total cells: {len(cells)}")
    print(f"  unlabelled / invalid: {len(bad_rows)}")
    if bad_rows:
        for c in bad_rows[:8]:
            print(f"    {c.get('slug','?'):25s} seed={c.get('seed','?')}  teacher={c.get('teacher')!r}")
        if len(bad_rows) > 8:
            print(f"    ... and {len(bad_rows)-8} more")
    print()
    print("  by pair (split, slug):")
    for split, slug in sorted(
        {(c.get("split","?"), c.get("slug","?")) for c in cells}
    ):
        cnts = by_pair[slug]
        print(
            f"    [{split:8s}] {slug:25s}  "
            f"mono={cnts['mono']}  co3={cnts['co3']}  none={cnts['none']}"
        )
    print()
    print("  by split:")
    for split in sorted(by_split):
        cnts = by_split[split]
        print(
            f"    {split:10s}  mono={cnts['mono']}  co3={cnts['co3']}  none={cnts['none']}"
        )
    print()
    n_working = total["mono"] + total["co3"]
    print(f"  working teachers: {n_working} / {len(cells)}  "
          f"(mono={total['mono']}, co3={total['co3']}, skipped={total['none']})")

    if bad_rows:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
