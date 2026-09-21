#!/usr/bin/env python
"""Count the animals in the over-correction panels (lambda above 1.0).

The dose sweep stopped at lambda 1.0, where every point averages 32 cells
(8 pairs x 4 seeds). The over-correction cells are one cell each: cat x dog,
seed 9, four arms, at lambda 1.25, 1.50, 1.75 and 2.00. Scoring them into
dose_curves.json would silently mix 32-cell points with 1-cell points in the
same curve, so they get their own file and never touch a rate.

What lands: overdose_counts.json, one record per panel carrying the detected
instance count, next to dose_curves.json. The figure reads counts from it; no
compose rate is computed here, because one cell cannot carry one.

    python scripts/score_overdose_cells.py --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poe_repair import paths

DOSE = paths.resolve(paths.HOW_MUCH_CORRECTION_IS_NEEDED)
PAIRS = DOSE / "pairs"
SCORER_CONTRACT = paths.resolve(paths.COMPOSE_SCORER_VALIDATION) / "scorer_validated.json"

LAMBDAS = (1.25, 1.50, 1.75, 2.00)
ROWS = ("oracle", "wrong_pair", "wrong_seed", "wrong_step")


def run_dir(pair: str, seed: int, lam: float, row: str) -> Path:
    tag = f"teacher_residual_const_lam{int(round(lam * 100)):03d}"
    if row != "oracle":
        tag = f"{tag}_{row}"
    return PAIRS / pair / f"seed_{seed}" / tag


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", default="a_cat__x__a_dog")
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=DOSE / "overdose_counts.json")
    args = ap.parse_args()

    contract = json.loads(SCORER_CONTRACT.read_text())
    if not contract.get("pass"):
        raise SystemExit(f"scorer at {SCORER_CONTRACT} is not marked validated: refusing.")
    print(f"scorer: {contract['method']} via {contract['detector']}")
    print(f"  rule: {contract['compose_rule']}\n")

    from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances

    device = None
    if args.device:
        import torch
        device = torch.device(args.device)

    records = []
    for lam in LAMBDAS:
        for row in ROWS:
            d = run_dir(args.pair, args.seed, lam, row)
            png = d / f"{d.name}.png"
            if not png.exists():
                raise SystemExit(f"no image at {png}")
            n, _ = count_instances(png, device=device)
            records.append({"row": row, "pair": args.pair, "seed": args.seed,
                            "lambda": lam, "n_instances": int(n),
                            "compose": int(n >= 2)})
            print(f"  lambda={lam:<5} {row:<11} count={n}")

    args.out.write_text(json.dumps({
        "scorer": contract["method"],
        "note": "one cell per record; no compose rate is defined here",
        "lambdas": list(LAMBDAS),
        "scores": records,
    }, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
