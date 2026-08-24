#!/usr/bin/env python
"""Does the correction point somewhere consistent, or just anywhere?

At every step there are two predictions available: PoE's, and Mono's. The
scope's account says the interaction term is what turns the first into the
second. This script measures whether it does so consistently, using two
distributions read straight from the cache.

  climb        ||r_t|| as a fraction of ||eps_PoE||, per step. How large is the
               correction relative to the thing being corrected?
  alignment    cosine between r_t at consecutive steps. If the correction points
               a consistent direction over time, this stays high; if it thrashes,
               it sits near zero.

Alignment is the more informative of the two: a correction with a stable
direction is one a low-rank adapter could learn, and a thrashing one is not.

Cache-only, no GPU.

Usage:
    python scripts/climb.py --pair a_cat__x__a_dog
    python scripts/climb.py --all --max-pairs 20
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

from poe_repair.experiments.interaction_term.cache import (  # noqa: E402
    CACHE_ROOT,
    load_cell,
)
from poe_repair.experiments.interaction_term.pool import load_pool  # noqa: E402
from scripts.snr_collapse import iter_cells  # noqa: E402

OUT_DIR = paths.resolve(paths.CACHE_ANALYSES)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", action="append", dest="pairs")
    ap.add_argument("--all", action="store_true",
                    help="scan the cache dir; mixes experiments, prefer --pool")
    ap.add_argument("--pool", nargs="?", const=str(paths.resolve(paths.DOES_THE_FIX_REACH_UNSEEN_PAIRS) / "pair_pool.yaml"),
                    help="restrict to one experiment's declared pairs")
    ap.add_argument("--max-pairs", type=int)
    ap.add_argument("--max-seeds", type=int, default=2)
    ap.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    if not args.all and not args.pairs and not args.pool:
        ap.error("give --pool, --all, or --pair")

    pairs = args.pairs
    if args.pool:
        pool = load_pool(args.pool)
        print(pool.summary())
        pairs = pool.train + pool.heldout()

    cells = list(iter_cells(args.cache_root, pairs, args.max_seeds))
    if args.max_pairs:
        keep, out = set(), []
        for slug, seed in cells:
            if slug not in keep and len(keep) >= args.max_pairs:
                continue
            keep.add(slug)
            out.append((slug, seed))
        cells = out
    if not cells:
        print("no cached cells matched")
        return 2

    climbs, aligns, rows = [], [], []
    for slug, seed in cells:
        c = load_cell(slug, seed, root=args.cache_root)
        r = c.r_t().flatten(1)                        # [T, D]
        climb = (r.norm(dim=1) / c.eps_poe().flatten(1).norm(dim=1)).numpy()
        nxt = torch.nn.functional.cosine_similarity(r[:-1], r[1:], dim=1).numpy()
        climbs.append(climb)
        aligns.append(nxt)
        rows.append({"pair": slug, "seed": int(seed),
                     "climb_median": float(np.median(climb)),
                     "alignment_median": float(np.median(nxt))})

    climb_all = np.concatenate(climbs)
    align_all = np.concatenate(aligns)
    n_pairs = len({s for s, _ in cells})

    print(f"{n_pairs} pairs, {len(cells)} cells\n")
    print("correction size as a fraction of ||eps_PoE||:")
    print(f"  median {np.median(climb_all):.1%}   "
          f"IQR {np.percentile(climb_all, 25):.1%} to "
          f"{np.percentile(climb_all, 75):.1%}")
    print("\nstep-to-step direction agreement (cosine, consecutive r_t):")
    print(f"  median {np.median(align_all):.3f}   "
          f"IQR {np.percentile(align_all, 25):.3f} to "
          f"{np.percentile(align_all, 75):.3f}")
    # A random direction pair in D dims has cosine ~ 1/sqrt(D), indistinguishable
    # from zero here, so anything well above that is real structure.
    d = climbs[0].shape[0] and 4 * 128 * 128
    print(f"  random-direction floor for {d} dims: ~{1/np.sqrt(d):.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "climb.json").write_text(json.dumps({
        "n_pairs": n_pairs, "n_cells": len(cells),
        "climb_median": float(np.median(climb_all)),
        "alignment_median": float(np.median(align_all)),
        "random_direction_floor": float(1 / np.sqrt(d)),
        "cells": rows,
    }, indent=2))

    if not args.no_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
        for cl in climbs:
            ax1.plot(cl, color="0.75", lw=0.7)
        ax1.plot(np.median(np.stack([c[:min(map(len, climbs))] for c in climbs]),
                           axis=0), color="tab:blue", lw=2)
        ax1.set_xlabel("denoising step")
        ax1.set_ylabel("||r_t|| / ||eps_PoE||")
        ax1.set_title("How big the correction is")

        for al in aligns:
            ax2.plot(al, color="0.75", lw=0.7)
        ax2.plot(np.median(np.stack([a[:min(map(len, aligns))] for a in aligns]),
                           axis=0), color="tab:orange", lw=2)
        ax2.axhline(0, color="k", lw=0.8, ls=":")
        ax2.set_ylim(-1, 1)
        ax2.set_xlabel("denoising step")
        ax2.set_ylabel("cosine, consecutive steps")
        ax2.set_title("Whether it keeps pointing the same way")

        fig.suptitle(f"Correction size and direction stability ({n_pairs} pairs)")
        fig.tight_layout()
        fig.savefig(args.out_dir / "climb.png", dpi=150)
        print(f"\nfigure: {args.out_dir / 'climb.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
