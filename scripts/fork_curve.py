#!/usr/bin/env python
"""When do the PoE and Mono paths part company?

Both samplers start from the same noise. If the interaction term matters in a
narrow band rather than everywhere, the distance between the two trajectories
should stay near zero, then turn upward at some step. That elbow is the
timing claim, measured rather than assumed, and it is the number this script
prints.

Reads saved latent trajectories. It does not sample: the trajectories come from
teacher-residual runs at lambda=0 (PoE) and lambda=1 (Mono) on the same cell,
written by `--save-trajectory`. Generating them is plan 03/05's job, not this
script's.

Usage:
    python scripts/fork_curve.py --pair a_cat__x__a_dog --seed 9
    python scripts/fork_curve.py --root outputs/residual_diagnostics/existence/pairs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from poe_repair import paths

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = paths.resolve(paths.CACHE_ANALYSES)
DEFAULT_ROOTS = (
    paths.resolve(paths.HOW_MUCH_CORRECTION_IS_NEEDED) / "pairs",
    paths.resolve(paths.RESIDUAL_BETWEEN_MONO_AND_POE) / "existence" / "pairs",
    # artifacts/diagnostics/residual_diagnostics/existence/pairs never existed on
    # either filesystem (checked 2026-08-24); kept as a literal since no family
    # constant maps to it and it is a defensive extra candidate, not load-bearing.
    Path("artifacts/diagnostics/residual_diagnostics/existence/pairs"),
)
POE_DIR = "teacher_residual_const_lam000"
MONO_DIR = "teacher_residual_const_lam100"


def find_cells(roots) -> list[tuple[str, int, Path, Path]]:
    """Locate (pair, seed) cells that have BOTH a PoE and a Mono trajectory."""
    found = []
    for root in roots:
        if not root.is_dir():
            continue
        for pair_dir in sorted(root.iterdir()):
            if not pair_dir.is_dir():
                continue
            for seed_dir in sorted(pair_dir.glob("seed_*")):
                poe = seed_dir / POE_DIR / "latent_trajectory.pt"
                mono = seed_dir / MONO_DIR / "latent_trajectory.pt"
                if poe.exists() and mono.exists():
                    found.append(
                        (pair_dir.name, int(seed_dir.name.split("_")[1]), poe, mono)
                    )
    return found


def elbow_index(d: np.ndarray) -> int:
    """Step where the distance curve turns upward.

    Uses the point of maximum distance from the straight line joining the
    curve's endpoints: the standard knee construction, and it needs no
    threshold to be chosen by hand.
    """
    n = len(d)
    if n < 3:
        return 0
    x = np.arange(n, dtype=float)
    y = (d - d.min()) / max(d.max() - d.min(), 1e-12)
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    num = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0)
    return int(np.argmax(num / np.hypot(y1 - y0, x1 - x0)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--root", type=Path, action="append", dest="roots")
    ap.add_argument("--min-steps", type=int, default=40,
                    help="skip trajectories shorter than this: a 20-step smoke "
                         "run pooled with 50-step runs distorts the elbow")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    cells = find_cells(args.roots or DEFAULT_ROOTS)
    if args.pair:
        cells = [c for c in cells if c[0] == args.pair]
    if args.seed is not None:
        cells = [c for c in cells if c[1] == args.seed]

    if not cells:
        looked = args.roots or DEFAULT_ROOTS
        print(
            "no cells have both a PoE and a Mono trajectory.\n"
            "This script reads trajectories, it does not sample. Produce them "
            "with:\n"
            "  python scripts/interaction_term_inject.py --pair <slug> --seed <n> "
            "--lambda 0\n"
            "  python scripts/interaction_term_inject.py --pair <slug> --seed <n> "
            "--lambda 1\n"
            "(both write latent_trajectory.pt). Looked under:\n"
            + "\n".join(f"  {p}" for p in looked),
            file=sys.stderr,
        )
        return 2

    rows = []
    for pair, seed, poe_p, mono_p in cells:
        poe = torch.load(poe_p, map_location="cpu", weights_only=True)["trajectories"].float()
        mono = torch.load(mono_p, map_location="cpu", weights_only=True)["trajectories"].float()
        n = min(poe.shape[0], mono.shape[0])
        if n < args.min_steps:
            print(f"  skipping {pair} seed {seed}: only {n} steps "
                  f"(< --min-steps {args.min_steps}); a short smoke run pooled "
                  f"with full ones distorts the elbow")
            continue
        d = (poe[:n] - mono[:n]).flatten(1).norm(dim=1).numpy()
        k = elbow_index(d)
        rows.append({"pair": pair, "seed": seed, "elbow_step": k,
                     "distance": d.tolist()})
        print(f"{pair} seed {seed}: elbow at step {k} of {n}   "
              f"d(0)={d[0]:.2f}  d(elbow)={d[k]:.2f}  d(end)={d[-1]:.2f}")

    elbows = [r["elbow_step"] for r in rows]
    print(f"\nelbow at step {int(np.median(elbows))} (median over {len(rows)} cells)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "fork_curve.json").write_text(json.dumps(
        {"median_elbow_step": int(np.median(elbows)), "cells": rows}, indent=2))

    if not args.no_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        for r in rows:
            ax.plot(r["distance"], color="0.7", lw=0.8)
        med = int(np.median(elbows))
        ax.axvline(med, color="tab:red", ls="--", lw=1.5,
                   label=f"median elbow, step {med}")
        ax.set_xlabel("denoising step")
        ax.set_ylabel("distance between PoE and Mono paths")
        ax.set_title(f"Where the two paths part company ({len(rows)} cells)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out_dir / "fork_curve.png", dpi=150)
        print(f"figure: {args.out_dir / 'fork_curve.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
