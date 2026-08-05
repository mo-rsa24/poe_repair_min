#!/usr/bin/env python
"""Can the text alone predict which pairs PoE will fail on?

Plan 06's language-space corroboration. Two probes, both read from the prompt
embeddings the cache already stores, so neither needs the UNet.

  L1  additivity gap. How far is the joint prompt's embedding from the sum of
      the two solo embeddings (minus the unconditional)? That sum is what PoE
      implicitly assumes the joint means. A large gap says the assumption is
      wrong for this pair before a single image is sampled.

  L3  shared binding direction. Is the per-pair additivity residual pointing
      the same way across pairs? If one direction explains most of them, that
      is the language-space twin of the low-rank claim about r_t.

L2 (chimera readback via a lambda sweep in language space) needs sampled
outputs and is not implemented here; plan 06 owns it.

Cache-only, no GPU.

Usage:
    python scripts/language_probes.py --probe l1 --probe l3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.experiments.interaction_term.cache import (  # noqa: E402
    CACHE_ROOT,
    cell_dir,
)
from poe_repair.experiments.interaction_term.pool import load_pool  # noqa: E402
from scripts.snr_collapse import iter_cells  # noqa: E402

OUT_DIR = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses")


def load_embeddings(pair: str, seed: int, root: Path) -> dict:
    """The cached prompt embeddings for one cell."""
    p = cell_dir(pair, seed, root=root) / "embeddings.pt"
    if not p.exists():
        raise FileNotFoundError(p)
    return torch.load(p, map_location="cpu", weights_only=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", action="append", dest="probes",
                    choices=("l1", "l3"), help="repeatable; default both")
    ap.add_argument("--pair", action="append", dest="pairs")
    ap.add_argument("--pool", nargs="?", const="outputs/animals_compose_transfer/pair_pool.yaml",
                    help="restrict to one experiment's declared pairs")
    ap.add_argument("--max-pairs", type=int, default=30)
    ap.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    probes = args.probes or ["l1", "l3"]

    # One seed per pair: the embeddings depend on the prompt, not the seed.
    pairs = args.pairs
    if args.pool:
        pool = load_pool(args.pool)
        print(pool.summary())
        pairs = pool.train + pool.heldout()
    cells = list(iter_cells(args.cache_root, pairs, 1))[: args.max_pairs]
    if not cells:
        print("no cached cells matched")
        return 2

    keys = None
    rows, residuals = [], []
    for pair, seed in cells:
        try:
            emb = load_embeddings(pair, seed, args.cache_root)
        except FileNotFoundError:
            continue
        if keys is None:
            keys = sorted(emb.keys())
        need = ("pool_a", "pool_b", "pool_j", "pool_uncond")
        if not all(k in emb for k in need):
            print(f"embeddings.pt lacks {need}; found {keys}", file=sys.stderr)
            print("L1/L3 need the solo, joint, and unconditional pooled "
                  "embeddings. Plan 06 owns extending the cache builder.",
                  file=sys.stderr)
            return 2
        a, b = emb["pool_a"].float(), emb["pool_b"].float()
        j, e = emb["pool_j"].float(), emb["pool_uncond"].float()
        # What PoE implicitly assumes the joint prompt means.
        assumed = a + b - e
        resid = (j - assumed).flatten()
        gap = float(resid.norm() / max(j.flatten().norm().item(), 1e-12))
        rows.append({"pair": pair, "seed": int(seed), "additivity_gap": gap})
        residuals.append(resid.numpy())
        print(f"  {pair:<40} additivity gap {gap:.4f}")

    if not rows:
        print("no cell had usable embeddings", file=sys.stderr)
        return 2

    result: dict = {"n_pairs": len(rows), "cells": rows}

    if "l1" in probes:
        g = np.array([r["additivity_gap"] for r in rows])
        print(f"\nL1 additivity gap over {len(g)} pairs: "
              f"median {np.median(g):.4f}, range {g.min():.4f} to {g.max():.4f}")
        result["l1_median_gap"] = float(np.median(g))

    if "l3" in probes and len(residuals) > 2:
        R = np.stack(residuals)
        R = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
        # Fraction of energy on the first singular direction: how much of the
        # per-pair residual is one shared direction rather than per-pair noise.
        s = np.linalg.svd(R - R.mean(0, keepdims=True), compute_uv=False)
        share = float(s[0] ** 2 / max((s ** 2).sum(), 1e-12))
        # Same-shape random floor, since a small sample concentrates by chance.
        rng = np.random.default_rng(0)
        G = rng.normal(size=R.shape)
        G = G / np.linalg.norm(G, axis=1, keepdims=True)
        sg = np.linalg.svd(G - G.mean(0, keepdims=True), compute_uv=False)
        floor = float(sg[0] ** 2 / max((sg ** 2).sum(), 1e-12))
        print(f"\nL3 shared binding direction over {len(R)} pairs:")
        print(f"  first direction holds {share:.1%} of residual energy "
              f"(random floor {floor:.1%}, ratio {share/floor:.1f}x)")
        result["l3_top_direction_share"] = share
        result["l3_random_floor"] = floor

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "language_probes.json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
