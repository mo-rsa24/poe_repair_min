#!/usr/bin/env python
"""Does the subspace test predict whether the LoRA transfers?

Two independent readings of one question, on one split (11 training pairs,
6 unseen transfer pairs, run 1d3qy31e / lora_step_100000.pt).

  GEOMETRY   Fit a subspace to the training pairs' cached r_t. Measure how much
             of the unseen pairs' r_t falls inside it. Cache-only, no GPU.
  BEHAVIOUR  What the trained LoRA actually did on those same unseen pairs,
             read from compose_rate.json.

If the geometry predicts behaviour, the cheap test can stand in for the
expensive one. If it does not, the geometry is measuring the wrong thing and
should not be used to decide anything.

Read-only. Calls the real cache loader; scores nothing itself.

    python evidence/subspace-vs-transfer/demo.py
    python evidence/subspace-vs-transfer/demo.py --sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from poe_repair import paths

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from poe_repair.experiments.interaction_term.cache import CACHE_ROOT, load_cell
from poe_repair.experiments.interaction_term.pool import load_pool

COMPOSE_RATE = paths.resolve(paths.DOES_THE_FIX_REACH_UNSEEN_PAIRS) / "pooled_lora/phase1_r8_100k/compose_rate.json"
FAIL_RATE = paths.resolve(paths.DOES_THE_FIX_REACH_UNSEEN_PAIRS) / "fail_rate.md"
OUT = Path("docs/evidence/subspace-vs-transfer")
KS = (1, 2, 4, 8, 16, 32, 64)
MIN_STEPS = 2       # below this a cell is an eval stub with zeroed eps


def cells_for(slug: str, root: Path, max_seeds: int) -> list[int]:
    """Seeds with a real trajectory for this pair, full cells preferred."""
    found: list[int] = []
    for split in ("train", "heldout"):
        d = root / split / slug
        if not d.is_dir():
            continue
        for sd in sorted(d.glob("seed_*"), key=lambda p: int(p.name.split("_")[1])):
            if len(found) >= max_seeds:
                break
            if len(list((sd / "residuals").glob("step_*.pt"))) < MIN_STEPS:
                continue
            s = int(sd.name.split("_")[1])
            if s not in found:
                found.append(s)
    return found


def stack_r_t(slugs, root: Path, max_seeds: int, stride: int):
    """[N, D] fp32 of r_t vectors, plus per-slug row ranges."""
    rows, spans, used = [], {}, []
    n = 0
    for slug in slugs:
        seeds = cells_for(slug, root, max_seeds)
        if not seeds:
            continue
        start = n
        for seed in seeds:
            v = load_cell(slug, seed, root=root).r_t()[::stride].flatten(1)
            rows.append(v)
            n += v.shape[0]
        spans[slug] = (start, n)
        used.append(slug)
    if not rows:
        return torch.empty(0), {}, []
    return torch.cat(rows, 0), spans, used


def captured_fraction(x: torch.Tensor, basis: torch.Tensor, mean: torch.Tensor) -> float:
    """Fraction of x's energy lying inside the subspace spanned by basis."""
    c = x - mean
    total = float((c ** 2).sum())
    return float(((c @ basis.T) ** 2).sum()) / max(total, 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default=str(paths.resolve(paths.DOES_THE_FIX_REACH_UNSEEN_PAIRS) / "pair_pool.yaml"))
    ap.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    ap.add_argument("--max-seeds", type=int, default=3)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--sweep", action="store_true",
                    help="also report reference and control pairs")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()

    pool = load_pool(args.pool)
    print("=" * 72)
    print("1. THE SPLIT")
    print("=" * 72)
    print(f"  {pool.summary()}")
    print(f"  train    : {', '.join(pool.train)}")
    print(f"  transfer : {', '.join(pool.transfer)}")
    if args.sweep:
        print(f"  reference: {', '.join(pool.reference)}")
        print(f"  control  : {', '.join(pool.control)}")
    print("  Same split as run 1d3qy31e (lora_step_100000.pt).")

    # ---- geometry ----------------------------------------------------------
    train, _, train_used = stack_r_t(
        pool.train, args.cache_root, args.max_seeds, args.stride)
    if train.numel() == 0:
        print("no training cells found", file=sys.stderr)
        return 2
    mean = train.mean(0, keepdim=True)
    Vh = torch.linalg.svd(train - mean, full_matrices=False)[2]

    roles = ["transfer"] + (["reference", "control"] if args.sweep else [])
    held_slugs = pool.heldout(roles=tuple(roles))
    held, spans, held_used = stack_r_t(
        held_slugs, args.cache_root, args.max_seeds, args.stride)

    print()
    print("=" * 72)
    print("2. GEOMETRY: does the training subspace contain the unseen pairs?")
    print("=" * 72)
    print(f"  subspace fitted on {train.shape[0]} vectors from "
          f"{len(train_used)} training pairs, {train.shape[1]} dims each")
    print()
    print(f"  {'k':>4}  {'train':>8}  {'unseen':>8}   ratio")
    geo = {}
    for k in KS:
        if k > Vh.shape[0]:
            break
        b = Vh[:k]
        tr = captured_fraction(train, b, mean)
        hd = captured_fraction(held, b, mean)
        geo[k] = {"train": tr, "heldout": hd}
        print(f"  {k:>4}  {tr:>7.1%}  {hd:>7.1%}   {tr/max(hd,1e-12):>5.1f}x")

    per_pair_geo = {
        slug: captured_fraction(held[a:b], Vh[:64], mean)
        for slug, (a, b) in spans.items()
    }

    # ---- behaviour ---------------------------------------------------------
    print()
    print("=" * 72)
    print("3. BEHAVIOUR: what the trained LoRA actually did on those pairs")
    print("=" * 72)
    if not COMPOSE_RATE.exists():
        print(f"  no compose_rate.json at {COMPOSE_RATE}; behaviour unavailable")
        return 2
    cr = json.loads(COMPOSE_RATE.read_text())["per_step_heldout_pair"]
    steps = sorted({s for v in cr.values() for s in v}, key=int)
    last = steps[-1]
    print(f"  vanilla PoE composes 0% on every one of these pairs "
          f"(fail_rate.md, 8 seeds).")
    print(f"  with the adapter, at eval step {last}:")
    print()
    print(f"  {'pair':<30}{'compose':>9}{'in 64-dim subspace':>21}")
    rows = []
    for slug in held_used:
        rate = cr.get(slug, {}).get(last)
        g = per_pair_geo.get(slug, float("nan"))
        if rate is None:
            continue
        rows.append({"pair": slug, "compose_rate": rate, "geometry_k64": g})
        print(f"  {slug:<30}{rate:>9.1%}{g:>20.1%}")

    # ---- the comparison ----------------------------------------------------
    print()
    print("=" * 72)
    print("4. DO THEY AGREE?")
    print("=" * 72)
    transfer_rows = [r for r in rows if r["pair"] in pool.transfer]
    mean_rate = float(np.mean([r["compose_rate"] for r in transfer_rows]))
    mean_geo = float(np.mean([r["geometry_k64"] for r in transfer_rows]))
    print(f"  transfer pairs, mean compose rate with the adapter : {mean_rate:.1%}")
    print(f"  transfer pairs, mean energy in the 64-dim subspace : {mean_geo:.1%}")
    print()
    if mean_rate > 0.8 and mean_geo < 0.2:
        print("  THEY DISAGREE.")
        print("  The adapter transfers almost perfectly to pairs it never saw.")
        print("  The geometry says those pairs' corrections lie almost entirely")
        print("  outside the training subspace.")
        print()
        print("  Both cannot be describing the same thing. The behaviour is the")
        print("  ground truth here: it is what the method actually does. So the")
        print("  geometric read is measuring something that does not determine")
        print("  transfer, and must not be used to predict it.")
    elif mean_rate > 0.8 and mean_geo > 0.5:
        print("  THEY AGREE (shared): high transfer, high subspace overlap.")
    elif mean_rate < 0.3 and mean_geo < 0.2:
        print("  THEY AGREE (not shared): low transfer, low subspace overlap.")
    else:
        print("  Mixed. Neither clearly confirms nor refutes the other.")

    if len(transfer_rows) > 2:
        x = np.array([r["geometry_k64"] for r in transfer_rows])
        y = np.array([r["compose_rate"] for r in transfer_rows])
        if x.std() > 1e-9 and y.std() > 1e-9:
            rho = float(np.corrcoef(x.argsort().argsort(),
                                    y.argsort().argsort())[0, 1])
            print(f"\n  across the {len(transfer_rows)} transfer pairs, rank "
                  f"correlation between subspace overlap and compose rate: "
                  f"{rho:+.2f}")
            print("  (if geometry predicted transfer this would be strongly "
                  "positive)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "result.json").write_text(json.dumps({
        "train_pairs": train_used, "heldout_pairs": held_used,
        "train_vectors": int(train.shape[0]),
        "heldout_vectors": int(held.shape[0]),
        "geometry_at_k": {str(k): v for k, v in geo.items()},
        "per_pair": rows,
        "mean_compose_rate_transfer": mean_rate,
        "mean_geometry_transfer": mean_geo,
        "eval_step": last,
    }, indent=2))
    print(f"\n  record: {args.out_dir / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
