#!/usr/bin/env python
"""Is the corrected image two animals, or two animals plus a quality cost?

A dose that raises the compose rate is not worth much if it also degrades the
image. This script scores paired outputs from the same cell (uncorrected vs
corrected) and reports the gap on two axes the compose rate cannot see:

  detector confidence   mean confidence of the kept animal boxes. A drop means
                        the objects got less recognisable even if there are now
                        two of them.
  instance count        how many animals each side shows. Reported so a "gain"
                        that actually produced three animals is visible rather
                        than counted as success.

Expected gap is about zero: the correction should buy composition without
costing quality. A large negative gap is a finding, not a failure of the
script.

Reads sampled images; it does not sample.

Usage:
    python scripts/quality_control.py
    python scripts/quality_control.py --root outputs/interaction_term/dose/pairs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.plot_dose_curves import require_validated_scorer  # noqa: E402

OUT_DIR = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses")
DEFAULT_ROOTS = (
    Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose/pairs"),
    Path("outputs/interaction_term/dose/pairs"),
)
SCORER_CONTRACT = Path("outputs/compose_scorer/scorer_validated.json")
LAM_RE = re.compile(r"lam(\d{3})")


def find_pairs(roots, baseline: float, corrected: float):
    """Cells that have BOTH the baseline and the corrected dose rendered."""
    found = []
    for root in roots:
        if not root.is_dir():
            continue
        for pair_dir in sorted(root.iterdir()):
            if not pair_dir.is_dir():
                continue
            for seed_dir in sorted(pair_dir.glob("seed_*")):
                imgs = {}
                for run_dir in sorted(seed_dir.glob("teacher_residual_const_lam*")):
                    m = LAM_RE.search(run_dir.name)
                    png = run_dir / f"{run_dir.name}.png"
                    if m and png.exists():
                        imgs[int(m.group(1)) / 100.0] = png
                if baseline in imgs and corrected in imgs:
                    found.append((pair_dir.name,
                                  int(seed_dir.name.split("_")[1]),
                                  imgs[baseline], imgs[corrected]))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, action="append", dest="roots")
    ap.add_argument("--baseline", type=float, default=0.0, help="uncorrected dose")
    ap.add_argument("--corrected", type=float, default=1.0, help="corrected dose")
    ap.add_argument("--contract", type=Path, default=SCORER_CONTRACT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    contract = require_validated_scorer(args.contract)
    print(f"scorer: {contract['method']} via {contract['detector']}\n")

    cells = find_pairs(args.roots or DEFAULT_ROOTS, args.baseline, args.corrected)
    if not cells:
        print(
            f"no cell has both lambda={args.baseline} and lambda={args.corrected} "
            "rendered. This script scores images, it does not sample.\n"
            "Produce both with scripts/interaction_term_inject.py.",
            file=sys.stderr,
        )
        return 2

    from poe_repair.experiments.compose_scorer.detection_scorer import count_instances

    rows = []
    for pair, seed, base_png, corr_png in cells:
        nb, boxes_b = count_instances(base_png)
        nc, boxes_c = count_instances(corr_png)
        cb = float(np.mean([b["confidence"] for b in boxes_b])) if boxes_b else 0.0
        cc = float(np.mean([b["confidence"] for b in boxes_c])) if boxes_c else 0.0
        rows.append({"pair": pair, "seed": seed,
                     "n_baseline": int(nb), "n_corrected": int(nc),
                     "conf_baseline": cb, "conf_corrected": cc,
                     "conf_gap": cc - cb})
        print(f"  {pair} seed {seed}: "
              f"instances {nb}->{nc}   confidence {cb:.3f}->{cc:.3f} "
              f"({cc - cb:+.3f})")

    gaps = np.array([r["conf_gap"] for r in rows])
    over = sum(1 for r in rows if r["n_corrected"] > 2)
    print(f"\nconfidence gap over {len(rows)} cells: "
          f"median {np.median(gaps):+.3f}, mean {gaps.mean():+.3f}")
    print(f"  cells where the corrected image shows more than 2 animals: {over}")
    verdict = ("no quality cost" if abs(np.median(gaps)) < 0.05 else
               "quality cost" if np.median(gaps) < 0 else "quality gain")
    print(f"  reading: {verdict}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "quality_control.json").write_text(json.dumps({
        "baseline_lambda": args.baseline, "corrected_lambda": args.corrected,
        "median_conf_gap": float(np.median(gaps)),
        "over_count_cells": over, "verdict": verdict, "cells": rows,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
