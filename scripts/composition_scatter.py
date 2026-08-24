#!/usr/bin/env python
"""Does the correction size predict which prompt types PoE fails on?

Plan 07's headline. Group pairs by composition type (two animals, an animal and
a scene, an attribute binding, and so on), then plot the pre-registered
correction-size measure against the compose rate for each group. The prediction
is that the groups order along a falling curve: types needing a larger
correction are the ones PoE fails on.

This script will not choose the normalisation for you. Comparing correction
size across prompt types is exactly the slicing choice that produced one
retraction in this result family, so the measure has to be committed in writing
first (plan 01, report/normalization_preregistration.md) and this script reads it
from there. No memo, no plot.

Usage:
    python scripts/composition_scatter.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from poe_repair import paths

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.experiments.interaction_term.cache import (  # noqa: E402
    CACHE_ROOT,
    load_cell,
)

OUT_DIR = paths.resolve(paths.CACHE_ANALYSES)
PREREG = Path("report/normalization_preregistration.md")
DOSE_SCORES = paths.resolve(paths.HOW_MUCH_CORRECTION_IS_NEEDED) / "dose_curves.json"
# The committed choice is parsed from the memo, not guessed here.
MEASURES = {
    "relative_norm": "||r_t|| / ||eps_PoE||",
    "distance_fraction": "fraction of the PoE->Mono distance",
}


def read_prereg(path: Path) -> str:
    """Return the committed measure name, or explain why we cannot proceed."""
    if not path.exists():
        raise SystemExit(
            f"no normalization memo at {path}.\n"
            "Correction size is not comparable across prompt types without a "
            "committed measure, and choosing one after seeing the plot is how "
            "the 95% delta-field number had to be retracted. Write the memo "
            "first: plan 01 (plans/does-the-correction-cause-composition/plans/01-preregister-"
            "normalization.md).\n"
            f"Recognised measures: {', '.join(MEASURES)}"
        )
    text = path.read_text()
    found = [k for k in MEASURES if re.search(rf"\b{k}\b", text)]
    if len(found) != 1:
        raise SystemExit(
            f"{path} must name exactly one committed measure "
            f"({', '.join(MEASURES)}); found {found or 'none'}."
        )
    return found[0]


def measure_for_cell(cell, measure: str) -> float:
    """Correction size for one cell under the committed measure."""
    r = cell.r_t().flatten(1).norm(dim=1)
    if measure == "relative_norm":
        return float((r / cell.eps_poe().flatten(1).norm(dim=1)).median())
    if measure == "distance_fraction":
        gap = (cell.eps_mono() - cell.eps_poe()).flatten(1).norm(dim=1)
        return float((r / gap.clamp_min(1e-12)).median())
    raise ValueError(measure)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prereg", type=Path, default=PREREG)
    ap.add_argument("--scores", type=Path, default=DOSE_SCORES,
                    help="dose_curves.json, for the compose rate per pair")
    ap.add_argument("--groups", type=Path,
                    help="JSON mapping pair slug -> composition type")
    ap.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    measure = read_prereg(args.prereg)
    print(f"committed measure: {measure} ({MEASURES[measure]})")

    if not args.scores.exists():
        raise SystemExit(
            f"no compose rates at {args.scores}. Run scripts/plot_dose_curves.py "
            "first: the scatter needs a rate per pair to plot against."
        )
    if not args.groups:
        raise SystemExit(
            "--groups is required: a JSON mapping each pair slug to its "
            "composition type. Plan 07 owns that assignment; this script will "
            "not infer types from slugs, because a wrong grouping would be "
            "invisible in the plot."
        )

    groups = json.loads(args.groups.read_text())
    scores = json.loads(args.scores.read_text())
    by_pair: dict[str, list[int]] = {}
    for row in scores["scores"]:
        by_pair.setdefault(row["pair"], []).append(row["compose"])

    rows = []
    for pair, flags in sorted(by_pair.items()):
        if pair not in groups:
            print(f"  skipping {pair}: no composition type in --groups")
            continue
        seeds = {r["seed"] for r in scores["scores"] if r["pair"] == pair}
        sizes = []
        for seed in sorted(seeds):
            try:
                sizes.append(measure_for_cell(
                    load_cell(pair, seed, root=args.cache_root), measure))
            except FileNotFoundError:
                continue
        if not sizes:
            continue
        rows.append({"pair": pair, "type": groups[pair],
                     "correction_size": float(np.mean(sizes)),
                     "compose_rate": float(np.mean(flags))})
        print(f"  {pair:<36} {groups[pair]:<18} "
              f"size {rows[-1]['correction_size']:.4f}  "
              f"rate {rows[-1]['compose_rate']:.0%}")

    if len(rows) < 3:
        raise SystemExit(
            f"only {len(rows)} pairs have both a correction size and a compose "
            "rate; a scatter needs more before it says anything."
        )

    x = np.array([r["correction_size"] for r in rows])
    y = np.array([r["compose_rate"] for r in rows])
    # Spearman: the prediction is about ordering, not a linear fit.
    rx, ry = x.argsort().argsort(), y.argsort().argsort()
    rho = float(np.corrcoef(rx, ry)[0, 1])
    print(f"\nSpearman rho (correction size vs compose rate): {rho:+.3f} "
          f"over {len(rows)} pairs")
    print("  prediction is a NEGATIVE rho: bigger correction needed, "
          "lower compose rate")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "composition_scatter.json").write_text(json.dumps(
        {"measure": measure, "spearman_rho": rho, "pairs": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
