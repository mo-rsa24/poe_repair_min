#!/usr/bin/env python
"""Does the compose rate rise with the dose?

Plan 03's headline. Score every sampled image in a dose sweep with the
validated compose scorer, then plot compose rate against lambda, one row per
control arm. The claim is supported if the oracle row rises with lambda while
the norm-matched random control stays at the floor.

Scoring uses the instrument validated in the compose-scorer scope:
count distinct animal instances with GroundingDINO, compose iff count >= 2.
That validation is what makes this readable, so this script refuses to run if
the scorer is not marked validated rather than silently using an unvetted read.

Reads sampled images; it does not sample. Produce them with
scripts/interaction_term_inject.py across a lambda grid.

Usage:
    python scripts/plot_dose_curves.py
    python scripts/plot_dose_curves.py --root outputs/interaction_term/dose/pairs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose")
DEFAULT_ROOTS = (
    Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose/pairs"),
    Path("outputs/interaction_term/dose/pairs"),
)
SCORER_CONTRACT = Path("outputs/compose_scorer/scorer_validated.json")
LAM_RE = re.compile(r"lam(\d{3})")


def require_validated_scorer(path: Path) -> dict:
    """Refuse to score with an uncertified instrument."""
    if not path.exists():
        raise SystemExit(
            f"no scorer contract at {path}. The compose scorer must be "
            "validated before any dose curve is readable "
            "(see plans/compose-scorer)."
        )
    contract = json.loads(path.read_text())
    if not contract.get("pass"):
        raise SystemExit(f"scorer at {path} is not marked validated: refusing.")
    return contract


def find_images(roots) -> dict[tuple[str, int], dict[float, Path]]:
    """Map (pair, seed) -> {lambda: image path} from the sweep layout."""
    out: dict[tuple[str, int], dict[float, Path]] = defaultdict(dict)
    for root in roots:
        if not root.is_dir():
            continue
        for pair_dir in sorted(root.iterdir()):
            if not pair_dir.is_dir():
                continue
            for seed_dir in sorted(pair_dir.glob("seed_*")):
                seed = int(seed_dir.name.split("_")[1])
                for run_dir in sorted(seed_dir.glob("teacher_residual_const_lam*")):
                    m = LAM_RE.search(run_dir.name)
                    if not m:
                        continue
                    png = run_dir / f"{run_dir.name}.png"
                    if png.exists():
                        out[(pair_dir.name, seed)][int(m.group(1)) / 100.0] = png
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, action="append", dest="roots")
    ap.add_argument("--contract", type=Path, default=SCORER_CONTRACT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    contract = require_validated_scorer(args.contract)
    print(f"scorer: {contract['method']} via {contract['detector']}")
    print(f"  rule: {contract['compose_rule']}\n")

    cells = find_images(args.roots or DEFAULT_ROOTS)
    if not cells:
        looked = args.roots or DEFAULT_ROOTS
        print(
            "no dose-sweep images found. This script scores images, it does "
            "not sample. Produce them with:\n"
            "  for L in 0 0.25 0.5 0.75 1; do \\\n"
            "    python scripts/interaction_term_inject.py --pair <slug> "
            "--seed <n> --lambda $L; done\n"
            "Looked under:\n" + "\n".join(f"  {p}" for p in looked),
            file=sys.stderr,
        )
        return 2

    # Import late: this pulls in torch and the detector weights.
    from poe_repair.experiments.compose_scorer.detection_scorer import count_instances

    by_lambda: dict[float, list[int]] = defaultdict(list)
    rows = []
    for (pair, seed), lam_map in sorted(cells.items()):
        for lam, png in sorted(lam_map.items()):
            n, _ = count_instances(png)
            composed = int(n >= 2)
            by_lambda[lam].append(composed)
            rows.append({"pair": pair, "seed": seed, "lambda": lam,
                         "n_instances": int(n), "compose": composed})
            print(f"  {pair} seed {seed} lam {lam:.2f}: "
                  f"{n} instances -> {'COMPOSE' if composed else 'blend'}")

    lams = sorted(by_lambda)
    rates = [float(np.mean(by_lambda[l])) for l in lams]
    n_cells = len(cells)

    print(f"\ncompose rate by dose ({n_cells} cells):")
    for l, r in zip(lams, rates):
        print(f"  lambda {l:.2f}: {r:.0%}  (n={len(by_lambda[l])})")

    # Area under the dose curve: one number summarising "does it rise".
    auc = float(np.trapezoid(rates, lams) / max(lams[-1] - lams[0], 1e-12)) if len(lams) > 1 else float("nan")
    print(f"\nAUC (mean compose rate across the dose range): {auc:.3f}")
    if len(lams) > 1:
        print(f"  lambda={lams[0]:.2f}: {rates[0]:.0%}   "
              f"lambda={lams[-1]:.2f}: {rates[-1]:.0%}   "
              f"rise: {rates[-1] - rates[0]:+.0%}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "dose_curves.json").write_text(json.dumps({
        "scorer": contract["method"], "n_cells": n_cells,
        "lambdas": lams, "compose_rate": rates, "auc": auc, "scores": rows,
    }, indent=2))

    if not args.no_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(lams, rates, "o-", color="tab:blue", lw=2, label="oracle r_t")
        ax.set_xlabel("dose (lambda)")
        ax.set_ylabel("compose rate")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"Compose rate vs dose ({n_cells} cells, {len(lams)} doses)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        # The control rows belong here too; plan 03 adds them once the
        # norm-matched random and direction-flipped arms are sampled.
        fig.tight_layout()
        fig.savefig(args.out_dir / "dose_curves.png", dpi=150)
        print(f"figure: {args.out_dir / 'dose_curves.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
