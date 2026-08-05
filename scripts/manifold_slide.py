#!/usr/bin/env python
"""Does raising the dose slide the sample along the PoE-to-Mono axis?

Plan 06's manifold corroboration. The compose rate is a yes/no read: it says
whether an image tipped over into two animals, not how far it moved. This
projects each dose's trajectory onto the axis joining the PoE endpoint to the
Mono endpoint, so a partial dose shows as a partial slide.

Projection is 0 at PoE and 1 at Mono by construction. If the account is right,
the projection should rise monotonically with lambda: the correction moves the
sample along that axis rather than off in some third direction. How much of the
motion is off-axis is reported too, since a large off-axis component would mean
the axis is the wrong summary.

Reads saved trajectories; it does not sample. Needs at least the lambda=0 and
lambda=1 endpoints plus one intermediate dose for the slide to be visible.

Usage:
    python scripts/manifold_slide.py --root outputs/interaction_term/dose/pairs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses")
DEFAULT_ROOTS = (
    Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose/pairs"),
    Path("outputs/interaction_term/dose/pairs"),
)
LAM_RE = re.compile(r"lam(\d{3})")


def find_trajectories(roots) -> dict[tuple[str, int], dict[float, Path]]:
    out: dict[tuple[str, int], dict[float, Path]] = defaultdict(dict)
    for root in roots:
        if not root.is_dir():
            continue
        for pair_dir in sorted(root.iterdir()):
            if not pair_dir.is_dir():
                continue
            for seed_dir in sorted(pair_dir.glob("seed_*")):
                for run_dir in sorted(seed_dir.glob("teacher_residual_const_lam*")):
                    m = LAM_RE.search(run_dir.name)
                    traj = run_dir / "latent_trajectory.pt"
                    if m and traj.exists():
                        seed = int(seed_dir.name.split("_")[1])
                        out[(pair_dir.name, seed)][int(m.group(1)) / 100.0] = traj
    return out


def final_latent(p: Path) -> torch.Tensor:
    return torch.load(p, map_location="cpu", weights_only=True)["trajectories"][-1].float().flatten()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, action="append", dest="roots")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    cells = find_trajectories(args.roots or DEFAULT_ROOTS)
    usable = {k: v for k, v in cells.items() if 0.0 in v and 1.0 in v}
    if not usable:
        looked = args.roots or DEFAULT_ROOTS
        print(
            "no cell has both the lambda=0 and lambda=1 trajectories needed to "
            "define the axis. This script reads trajectories, it does not "
            "sample.\n"
            "  for L in 0 0.25 0.5 0.75 1; do \\\n"
            "    python scripts/interaction_term_inject.py --pair <slug> "
            "--seed <n> --lambda $L; done\n"
            "Looked under:\n" + "\n".join(f"  {p}" for p in looked),
            file=sys.stderr,
        )
        return 2

    rows = []
    by_lambda: dict[float, list[float]] = defaultdict(list)
    for (pair, seed), lam_map in sorted(usable.items()):
        poe, mono = final_latent(lam_map[0.0]), final_latent(lam_map[1.0])
        axis = mono - poe
        denom = float(axis.dot(axis))
        if denom < 1e-12:
            print(f"  {pair} seed {seed}: PoE and Mono endpoints coincide, "
                  "no axis to project onto")
            continue
        for lam, p in sorted(lam_map.items()):
            v = final_latent(p) - poe
            along = float(v.dot(axis)) / denom
            off = float((v - along * axis).norm() / max(v.norm().item(), 1e-12))
            by_lambda[lam].append(along)
            rows.append({"pair": pair, "seed": seed, "lambda": lam,
                         "projection": along, "off_axis_fraction": off})
            print(f"  {pair} seed {seed} lam {lam:.2f}: "
                  f"projection {along:+.3f}   off-axis {off:.1%}")

    if not rows:
        print("no usable cells", file=sys.stderr)
        return 2

    lams = sorted(by_lambda)
    means = [float(np.mean(by_lambda[l])) for l in lams]
    print(f"\nprojection onto the PoE->Mono axis ({len(usable)} cells):")
    for l, m in zip(lams, means):
        print(f"  lambda {l:.2f}: {m:+.3f}")
    monotone = all(b >= a - 1e-6 for a, b in zip(means, means[1:]))
    print(f"  monotone in lambda: {'yes' if monotone else 'NO'}")
    if len(lams) < 3:
        print("  (only the endpoints: add an intermediate dose to see a slide)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifold_slide.json").write_text(json.dumps({
        "lambdas": lams, "mean_projection": means, "monotone": monotone,
        "cells": rows,
    }, indent=2))

    if not args.no_figure and len(lams) > 1:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(lams, means, "o-", color="tab:purple", lw=2)
        ax.plot([0, 1], [0, 1], ls=":", color="0.6", label="exact slide")
        ax.set_xlabel("dose (lambda)")
        ax.set_ylabel("projection onto PoE -> Mono axis")
        ax.set_title(f"How far the dose moves the sample ({len(usable)} cells)")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_dir / "manifold_slide.png", dpi=150)
        print(f"figure: {args.out_dir / 'manifold_slide.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
