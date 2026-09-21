"""Score every cached cell by the size of the correction it teaches, with no eye and no renders.

The adapter is trained on r_t = eps_J - eps_PoE, the interaction term PoE drops. That residual is
computed from the four cached noise predictions, not from the picture, so a cell's value as
training data is how large and how learnable its r_t is, not whether its render shows two animals.

This matters because the two can disagree completely. A cell whose joint render shows two cheetahs
still carries a large r_t if PoE would have drawn only one: it teaches "put a second instance in",
which is exactly what cat x dog needs. A cell whose render shows a clean cat and dog but whose PoE
render is nearly identical carries r_t near zero and teaches nothing at all.

Two numbers per cell, both means over a step window:

  rel   ||r_t|| / ||eps_PoE||        how much of the PoE prediction the correction rewrites
  cos   cos(r_t, eps_J - eps_uncond) whether the correction points along the joint's own direction
                                     rather than being noise

Run:
  python3 scripts/showcase/score_cells_by_residual.py --pairs-from <cells.json> --out <json>
  python3 scripts/showcase/score_cells_by_residual.py --all --limit 400 --out <json>
"""
import argparse, json, os, sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from poe_repair._sdxl.metrics import guided_eps, poe_eps            # noqa: E402

CACHE = "/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache"


def cell_dir(pair, seed):
    for split in ("train", "heldout"):
        d = os.path.join(CACHE, split, pair, "seed_%d" % seed)
        if os.path.isdir(d):
            return d
    return None


def score_cell(d, gs, lo, hi):
    """Mean relative residual size and direction agreement over steps [lo, hi)."""
    rels, coss = [], []
    for idx in range(lo, hi):
        f = os.path.join(d, "residuals", "step_%03d.pt" % idx)
        if not os.path.exists(f):
            continue
        t = torch.load(f, map_location="cpu")
        eu = t["eps_uncond"].float()
        ej = guided_eps(t["eps_j_raw"].float(), eu, gs)
        ep = poe_eps(guided_eps(t["eps_a_raw"].float(), eu, gs),
                     guided_eps(t["eps_b_raw"].float(), eu, gs), eu)
        r = ej - ep
        rn, pn = r.norm(), ep.norm()
        if pn > 0:
            rels.append(float(rn / pn))
        joint_dir = ej - eu
        denom = rn * joint_dir.norm()
        if denom > 0:
            coss.append(float((r * joint_dir).sum() / denom))
    if not rels:
        return None
    return {"rel": sum(rels) / len(rels), "cos": sum(coss) / len(coss), "n_steps": len(rels)}


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pairs-from", help="JSON {pair: [seeds]} to score exactly")
    src.add_argument("--all", action="store_true", help="score every cached cell")
    ap.add_argument("--window", nargs=2, type=int, default=[0, 25],
                    metavar=("LO", "HI"), help="step range to average over")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many cells (0 = no cap)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.pairs_from:
        want = {k: [int(x) for x in v] for k, v in json.load(open(args.pairs_from)).items()}
    else:
        want = {}
        root = os.path.join(CACHE, "train")
        for p in sorted(os.listdir(root)):
            if "__x__" not in p:
                continue
            seeds = sorted(int(x.split("_")[1]) for x in os.listdir(os.path.join(root, p))
                           if x.startswith("seed_") and x.split("_")[1].isdigit())
            # the evaluation seeds share their starting noise across pairs; never score them in
            seeds = [s for s in seeds if not (9 <= s <= 16)]
            if seeds:
                want[p] = seeds

    lo, hi = args.window
    out, n = {}, 0
    for pair, seeds in want.items():
        for s in seeds:
            d = cell_dir(pair, s)
            if not d:
                continue
            meta_p = os.path.join(d, "meta.json")
            gs = json.load(open(meta_p)).get("guidance_scale", 7.5) if os.path.exists(meta_p) else 7.5
            sc = score_cell(d, float(gs), lo, hi)
            if sc:
                out.setdefault(pair, {})[str(s)] = sc
                n += 1
                if n % 50 == 0:
                    print(f"  scored {n} cells", flush=True)
            if args.limit and n >= args.limit:
                break
        if args.limit and n >= args.limit:
            break
    json.dump({"window": [lo, hi], "cells": out}, open(args.out, "w"), indent=2, sort_keys=True)
    print(f"scored {n} cells across {len(out)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
