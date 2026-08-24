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
# Fault one, fixed in the source rather than on a command line so a later change
# shows up in a diff. The dose sweep ran seeds 9 to 12 (see the SEEDS list in
# scripts/mechanism_study/run_dose_sweep.sh). Earlier quick tests left cells at
# other seeds under the same output root, and they exist only at lambda 0 and 1,
# so including them scored the two end strengths over more cells than the three
# middle ones. That is not a fair curve, so the collection is pinned here.
SWEEP_SEEDS = (9, 10, 11, 12)
ROW_RE = re.compile(
    r"lam\d{3}_(random|wrong_pair|wrong_seed|wrong_step|mean_others)$")
ROWS = ("oracle", "random", "wrong_pair", "wrong_seed", "wrong_step", "mean_others")


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


def find_images(roots, seeds=SWEEP_SEEDS) -> dict[str, dict[tuple[str, int], dict[float, Path]]]:
    """Map row -> (pair, seed) -> {lambda: image path}.

    `seeds` restricts collection to the sweep's own seeds; pass None to take
    every seed present, which is what produced the unequal cell counts.

    The row is read from the directory suffix the injection CLI writes
    (`..._random`, `..._wrong_pair`, bare for the oracle). Pooling the three
    into one curve would average the controls into the claim, which is the
    whole thing this experiment exists to avoid.
    """
    out: dict[str, dict[tuple[str, int], dict[float, Path]]] = {
        r: defaultdict(dict) for r in ROWS
    }
    for root in roots:
        if not root.is_dir():
            continue
        for pair_dir in sorted(root.iterdir()):
            if not pair_dir.is_dir():
                continue
            for seed_dir in sorted(pair_dir.glob("seed_*")):
                seed = int(seed_dir.name.split("_")[1])
                if seeds is not None and seed not in seeds:
                    continue
                for run_dir in sorted(seed_dir.glob("teacher_residual_const_lam*")):
                    m = LAM_RE.search(run_dir.name)
                    if not m:
                        continue
                    png = run_dir / f"{run_dir.name}.png"
                    if not png.exists():
                        continue
                    rm = ROW_RE.search(run_dir.name)
                    row = rm.group(1) if rm else "oracle"
                    lam = int(m.group(1)) / 100.0
                    out[row][(pair_dir.name, seed)][lam] = png
                    # lambda=0 injects nothing, so the oracle image IS the
                    # control image. Share it rather than sampling it 3x.
                    if lam == 0.0 and row == "oracle":
                        for other in ROWS[1:]:
                            out[other][(pair_dir.name, seed)][0.0] = png
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, action="append", dest="roots")
    ap.add_argument("--contract", type=Path, default=SCORER_CONTRACT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--device", default=None,
                    help="torch device for the detector, e.g. cpu. Default: "
                         "let the detector choose, which means GPU if visible. "
                         "Use cpu when another process holds the card.")
    ap.add_argument("--all-seeds", action="store_true",
                    help="collect every seed under the root instead of only the "
                         "sweep's seeds (%s). Off by default: see SWEEP_SEEDS."
                         % ", ".join(map(str, SWEEP_SEEDS)))
    args = ap.parse_args()

    contract = require_validated_scorer(args.contract)
    print(f"scorer: {contract['method']} via {contract['detector']}")
    print(f"  rule: {contract['compose_rule']}\n")

    seeds = None if args.all_seeds else SWEEP_SEEDS
    print(f"  seeds: {'every seed present' if seeds is None else seeds}\n")
    by_row = find_images(args.roots or DEFAULT_ROOTS, seeds=seeds)
    if not any(by_row[r] for r in ROWS):
        looked = args.roots or DEFAULT_ROOTS
        print(
            "no dose-sweep images found. This script scores images, it does "
            "not sample. Produce them with:\n"
            "  bash scripts/mechanism_study/run_dose_sweep.sh\n"
            "Looked under:\n" + "\n".join(f"  {p}" for p in looked),
            file=sys.stderr,
        )
        return 2

    from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances

    device = None
    if args.device:
        import torch
        device = torch.device(args.device)

    scored: dict[str, dict[float, list[int]]] = {r: defaultdict(list) for r in ROWS}
    rows_out = []
    for row in ROWS:
        for (pair, seed), lam_map in sorted(by_row[row].items()):
            for lam, png in sorted(lam_map.items()):
                n, _ = count_instances(png, device=device)
                composed = int(n >= 2)
                scored[row][lam].append(composed)
                rows_out.append({"row": row, "pair": pair, "seed": seed,
                                 "lambda": lam, "n_instances": int(n),
                                 "compose": composed})

    lams = sorted({l for r in ROWS for l in scored[r]})
    print(f"\ncompose rate by dose and row\n")
    print(f"  {'lambda':>7}" + "".join(f"{r:>14}" for r in ROWS))
    curves = {r: [] for r in ROWS}
    for lam in lams:
        line = f"  {lam:>7.2f}"
        for r in ROWS:
            v = scored[r].get(lam, [])
            rate = float(np.mean(v)) if v else float("nan")
            curves[r].append(rate)
            line += f"{rate:>13.0%}" + ("*" if not v else " ")
        print(line)
    print("  (* = no cells at that dose)")

    aucs = {}
    for r in ROWS:
        y = np.array(curves[r], dtype=float)
        ok = ~np.isnan(y)
        aucs[r] = (float(np.trapezoid(y[ok], np.array(lams)[ok])
                         / max(lams[-1] - lams[0], 1e-12)) if ok.sum() > 1
                   else float("nan"))
    print(f"\n  AUC   " + "".join(f"{aucs[r]:>13.3f} " for r in ROWS))

    o = np.array(curves["oracle"], dtype=float)
    rise = o[-1] - o[0] if len(o) > 1 else float("nan")
    ctl_rise = max(
        (np.array(curves[r], dtype=float)[-1] - np.array(curves[r], dtype=float)[0])
        for r in ("random", "wrong_pair")
        if len(curves[r]) > 1 and not np.isnan(curves[r][-1])
    ) if any(len(curves[r]) > 1 for r in ("random", "wrong_pair")) else float("nan")
    print(f"\n  oracle rises {rise:+.0%} from lambda {lams[0]} to {lams[-1]}")
    print(f"  best control rises {ctl_rise:+.0%}")
    if not np.isnan(ctl_rise):
        if rise > 0.3 and ctl_rise < 0.15:
            print("\n  SUPPORTS the causal claim: the oracle rises and both")
            print("  controls stay near the floor. Injecting the pair's OWN r_t")
            print("  is what helps, not injecting something of that magnitude.")
        elif ctl_rise >= rise * 0.6:
            print("\n  NULL: a control rises about as much as the oracle. The")
            print("  effect is not specific to this pair's correction. Goal 1's")
            print("  null arm; do not loosen the threshold to rescue it.")
        else:
            print("\n  MIXED: the oracle leads but a control is not at the floor.")

    # The two newer rows are not floor-expected controls; each has a reading
    # declared before its sweep ran (see run_dose_sweep_controls2.sh):
    #   wrong_seed  composes -> r_t is a function of prompt + noise level, not
    #               of the particular run, which is what makes it learnable;
    #               stays fused -> the adapter story owes an explanation.
    #   wrong_step  stays fused -> the correction's content is needed at the
    #               right time (F4's premise from the injection side);
    #               composes -> only the total amount matters.
    for r in ("wrong_seed", "wrong_step"):
        y = np.array(curves.get(r, []), dtype=float)
        if len(y) > 1 and not np.isnan(y[-1]):
            print(f"  {r} rises {y[-1] - y[0]:+.0%} (own reading, not a floor control)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "dose_curves.json").write_text(json.dumps({
        "scorer": contract["method"], "lambdas": lams,
        "curves": curves, "auc": aucs,
        "n_cells": {r: len(by_row[r]) for r in ROWS},
        "scores": rows_out,
    }, indent=2))

    if not args.no_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        style = {"oracle": ("tab:blue", "o-", "the pair's own r_t"),
                 "random": ("tab:red", "s--", "norm-matched random"),
                 "wrong_pair": ("tab:orange", "^--", "another pair's r_t"),
                 "wrong_seed": ("tab:green", "v--", "same pair, another seed's r_t"),
                 "wrong_step": ("tab:purple", "d--", "own r_t, steps deranged"),
                 "mean_others": ("tab:brown", "P--",
                                 "mean r_t of the pair's other runs")}
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for r in ROWS:
            if not by_row[r]:
                continue
            c, m, lab = style[r]
            y = np.array(curves[r], dtype=float)
            ok = ~np.isnan(y)
            n = len(by_row[r])
            ax.plot(np.array(lams)[ok], y[ok], m, color=c, lw=2,
                    label=f"{lab} (n={n})")
        ax.set_xlabel("dose (lambda)")
        ax.set_ylabel("compose rate")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title("Compose rate vs dose, with both controls")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_dir / "dose_curves.png", dpi=150)
        print(f"\nfigure: {args.out_dir / 'dose_curves.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
