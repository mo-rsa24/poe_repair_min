#!/usr/bin/env python
"""Score the mechanism re-probe: does the value-channel finding replicate?

Plan 02 tasks 4 and 6. Reads the sweep's captured cells and answers Goal 6:
does the LoRA change WHAT a word paints more than WHERE it looks, across
held-out pairs and seeds, or only on the single cat/dog seed-9 cell it was
found on?

The comparison is the PATTERN change, not ||on-off||/||off||. The raw norm is
dominated by a ~25% uniform dimming of the attention weights and gives the
opposite answer on the same data. See
docs/evidence/mechanism-reprobe/measure-fairness.md.

Writes verdict.json next to the cells. Cache-free, no GPU.

    python scripts/mechanism_study/reprobe_table.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from poe_repair import paths

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from poe_repair.experiments.mechanism_study.value_probe import (  # noqa: E402
    gain_and_pattern,
)

DEFAULT_ROOT = paths.resolve(paths.CONTENT_CHANGE_RELATIVE_TO_ATTENTION_CHANGE)
# Pre-registered before the sweep ran, so the bar is not moved to fit the data.
# Support needs the effect present in most cells AND a clear median, not one
# strong pair carrying a weak average.
MIN_MEDIAN_RATIO = 1.2
MIN_FRACTION_ABOVE_ONE = 0.75


def read_cell(d: Path) -> list[dict]:
    """One cell's per-token, per-step rows. Recomputes rather than trusting."""
    manifest = d / "value_probe_manifest.json"
    if not manifest.exists():
        return []
    meta = json.loads(manifest.read_text())
    token_names = list(meta.get("token_map", {}))
    rows = []
    for f in sorted(d.glob("step_*_valuemaps.pt")):
        r = torch.load(f, map_location="cpu", weights_only=False)
        names = token_names or [
            k[: -len("_off_weight")] for k in r if k.endswith("_off_weight")
        ]
        for nm in names:
            w = gain_and_pattern(r.get(f"{nm}_off_weight"), r.get(f"{nm}_on_weight"))
            c = gain_and_pattern(r.get(f"{nm}_off_content"), r.get(f"{nm}_on_content"))
            if w is None or c is None:
                continue
            rows.append({
                "pair": d.parent.name, "seed": int(d.name.split("_")[1]),
                "step": int(r["step_index"]), "token": nm,
                "weight_gain": w["gain"], "weight_pattern": w["pattern"],
                "content_gain": c["gain"], "content_pattern": c["pattern"],
                "noise_floor": max(w["noise_floor"], c["noise_floor"]),
                "ratio": c["pattern"] / max(w["pattern"], 1e-12),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--pool", default=str(paths.resolve(paths.DOES_THE_FIX_REACH_UNSEEN_PAIRS) / "pair_pool.yaml"))
    args = ap.parse_args()

    if not args.root.is_dir():
        print(f"no cells at {args.root}. Run the sweep first:\n"
              f"  sbatch scripts/mechanism_study/value_probe_sweep.sbatch",
              file=sys.stderr)
        return 2

    rows = []
    for cell in sorted(args.root.glob("*/seed_*")):
        rows.extend(read_cell(cell))
    if not rows:
        print(f"no captured cells under {args.root}", file=sys.stderr)
        return 2

    cells = {(r["pair"], r["seed"]) for r in rows}
    pairs = sorted({r["pair"] for r in rows})
    print(f"{len(cells)} cells, {len(pairs)} pairs, {len(rows)} token-step rows\n")

    # Per pair, so one strong pair carrying the average is visible.
    by_pair: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_pair[r["pair"]].append(r["ratio"])
    print(f"{'pair':<30}{'cells':>6}{'median':>9}{'IQR':>17}{'above 1':>9}")
    per_pair = {}
    for p in pairs:
        a = np.array(by_pair[p])
        n_cells = len({r["seed"] for r in rows if r["pair"] == p})
        frac = float((a > 1).mean())
        per_pair[p] = {"median": float(np.median(a)), "n_rows": len(a),
                       "n_cells": n_cells, "fraction_above_one": frac}
        iqr = f"{np.percentile(a,25):.2f} to {np.percentile(a,75):.2f}"
        print(f"{p:<30}{n_cells:>6}{np.median(a):>9.2f}{iqr:>17}{frac:>8.0%}")

    allr = np.array([r["ratio"] for r in rows])
    median = float(np.median(allr))
    frac = float((allr > 1).mean())
    print(f"\noverall: median {median:.2f}x, {frac:.0%} of rows above 1 "
          f"({(allr > 1).sum()}/{len(allr)})")

    # Is the pattern signal above the per-cell noise floor at all?
    floors = np.array([r["noise_floor"] for r in rows])
    pats = np.array([max(r["weight_pattern"], r["content_pattern"]) for r in rows])
    print(f"  pattern vs shuffled-map noise floor: "
          f"{float(np.median(floors / np.maximum(pats, 1e-12))):.1f}x headroom")

    replicates = median >= MIN_MEDIAN_RATIO and frac >= MIN_FRACTION_ABOVE_ONE
    verdict = "replicates" if replicates else "does not replicate"
    print(f"\nPRE-REGISTERED BAR: median >= {MIN_MEDIAN_RATIO}x "
          f"AND >= {MIN_FRACTION_ABOVE_ONE:.0%} of rows above 1")
    print(f"VERDICT: {verdict}")
    if replicates:
        print("  The mechanism section proceeds: the LoRA changes what the word")
        print("  paints more than where it looks, across held-out pairs and seeds.")
    else:
        print("  Goal 6 not met. The mechanism section shrinks to the negative")
        print("  paragraph, as the plan provides for. This is a result, not a")
        print("  failure: do not loosen the bar to rescue it.")

    out = {
        "verdict": verdict, "replicates": bool(replicates),
        "median_ratio": median, "fraction_above_one": frac,
        "bar": {"min_median_ratio": MIN_MEDIAN_RATIO,
                "min_fraction_above_one": MIN_FRACTION_ABOVE_ONE},
        "n_cells": len(cells), "n_pairs": len(pairs), "n_rows": len(rows),
        "per_pair": per_pair,
        "measure": "content_pattern / weight_pattern, after removing the best "
                   "single rescale from each map (see measure-fairness.md)",
        "rows": rows,
    }
    (args.root / "verdict.json").write_text(json.dumps(out, indent=2))
    print(f"\nwritten: {args.root / 'verdict.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
