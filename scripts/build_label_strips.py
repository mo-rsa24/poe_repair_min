"""Build label-helper artifacts for ``dataset/``.

Produces:

  dataset/strips/<pair_slug>.png
      One image per pair: rows = seeds, columns = PoE | Mono | CO3.
      Used for a fast visual scan when hand-labelling the working teacher.

  dataset/labels.template.json
      A skeleton with all 25 cells listed. Copy to dataset/labels.json and
      fill in ``teacher`` per cell:
          "mono"  : Mono image is a clean co-occurrence (preferred).
          "co3"   : Mono failed; CO3 image is clean.
          "none"  : neither is clean — skip the cell.
      Optional ``notes`` field for free-form comments.

Re-run after changing pairs / seeds / source priority. Idempotent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poe_repair.figures._common import image_grid  # noqa: E402
DATASET_ROOT = REPO_ROOT / "dataset"
CELLS_ROOT = DATASET_ROOT / "cells"
STRIPS_ROOT = DATASET_ROOT / "strips"
LABELS_TEMPLATE = DATASET_ROOT / "labels.template.json"
LABELS = DATASET_ROOT / "labels.json"


def main() -> None:
    scope = json.loads((DATASET_ROOT / "scope.json").read_text())
    pairs = (
        [("held_out", a, b) for a, b in scope["held_out_pairs"]]
        + [("train", a, b) for a, b in scope["train_pairs"]]
    )
    seeds: list[int] = scope["seeds"]
    methods: list[str] = scope["methods"]
    method_labels = {"poe": "PoE", "mono": "Mono", "co3": "CO3"}

    STRIPS_ROOT.mkdir(parents=True, exist_ok=True)

    import re
    def _clean(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")

    template_cells: list[dict] = []
    for split, prompt_a, prompt_b in pairs:
        slug = f"{_clean(prompt_a)}__x__{_clean(prompt_b)}"
        rows: list[list[Path]] = []
        for seed in seeds:
            cell_dir = CELLS_ROOT / slug / f"seed_{seed}"
            row: list[Path] = []
            for method in methods:
                p = cell_dir / f"{method}.png"
                if not p.exists():
                    raise FileNotFoundError(
                        f"missing {p}; rerun scripts/build_dataset.py first"
                    )
                row.append(p)
            rows.append(row)
            template_cells.append({
                "pair": [prompt_a, prompt_b],
                "slug": slug,
                "split": split,
                "seed": int(seed),
                "teacher": None,
                "notes": "",
            })

        out_path = STRIPS_ROOT / f"{slug}.png"
        image_grid(
            rows, out_path,
            col_labels=[method_labels[m] for m in methods],
            row_labels=[f"seed {s}" for s in seeds],
            title=f"{prompt_a} × {prompt_b}  ({split})",
            panel_size=2.6,
        )
        print(f"[strips] wrote {out_path}")

    rule = (
        "Per cell, set teacher to 'mono' if mono.png shows two distinct, "
        "anatomically-correct, co-occurring subjects; else 'none' (skip). "
        "CO3 is no longer used as a teacher in the POC round."
    )
    LABELS_TEMPLATE.write_text(json.dumps(
        {"rule": rule, "scope": scope, "cells": template_cells},
        indent=2,
    ))
    print(f"[strips] wrote {LABELS_TEMPLATE}")

    # Merge: if labels.json exists, preserve user-set teacher/notes for
    # cells that already appear there; insert new cells with teacher=null.
    if LABELS.exists():
        existing = json.loads(LABELS.read_text())
        existing_by_key = {
            (c["slug"], int(c["seed"])): c for c in existing.get("cells", [])
        }
        merged_cells: list[dict] = []
        n_preserved = 0
        n_added = 0
        for cell in template_cells:
            key = (cell["slug"], int(cell["seed"]))
            if key in existing_by_key:
                prev = existing_by_key[key]
                cell["teacher"] = prev.get("teacher")
                cell["notes"] = prev.get("notes", "")
                n_preserved += 1
            else:
                n_added += 1
            merged_cells.append(cell)
        LABELS.write_text(json.dumps(
            {"rule": rule, "scope": scope, "cells": merged_cells},
            indent=2,
        ))
        print(
            f"[strips] merged {LABELS}: preserved {n_preserved} existing "
            f"labels, added {n_added} new cells (teacher=null)"
        )
    else:
        print(
            f"[strips] no existing labels.json; copy "
            f"{LABELS_TEMPLATE.name} -> labels.json and fill in teacher "
            "per cell"
        )
    print(f"[strips] strips dir: {STRIPS_ROOT}")


if __name__ == "__main__":
    main()
