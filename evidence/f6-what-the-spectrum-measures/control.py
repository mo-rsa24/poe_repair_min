#!/usr/bin/env python
"""What is F6's spectrum actually measuring?

F6 stacks the cached corrections, takes the singular values, and reports how much
energy sits in the top k directions against a same-shape Gaussian floor. The stack
beats that floor by 11 to 14 times, which has been read as "the corrections share a
small set of directions, so a small adapter can learn them".

That floor gives every row the same expected norm. Real rows do not have the same
norm: ||r_t|| tracks the noise level and spans a 4 to 5 times range across the run,
which is F3's whole subject. A stack of rows pointing in unrelated directions still
piles its energy into the top few singular directions if some rows are far larger
than others. So the floor cannot tell shared directions from uneven sizes.

Three measurements separate them, all cache-only, no GPU:

  1. NORM-MATCHED FLOOR. Random directions carrying the real ||r_t|| values. The
     excess over this floor, not over the Gaussian one, is what shared direction
     structure would look like.
  2. WITHIN one cell, unit rows. 50 steps of a single run, every row scaled to unit
     norm. Does one run's own correction turn smoothly through its steps?
  3. ACROSS cells, unit rows. One step from each of 50 different cells, same shape,
     same normalisation. Do different runs and pairs share directions?

Writes result.json beside this file.

    python evidence/f6-what-the-spectrum-measures/control.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from poe_repair import paths

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from poe_repair.experiments.interaction_term.cache import (  # noqa: E402
    CACHE_ROOT,
    load_cell,
)
from poe_repair.experiments.interaction_term.pool import load_pool  # noqa: E402

KS = (1, 2, 4, 8, 16, 32)
ROWS_PER_STACK = 50
N_WITHIN_CELLS = 6
STEP_ROTATION = 7   # so the ACROSS rows span noise levels instead of sharing one


def energy_at_k(M: torch.Tensor) -> dict[int, float]:
    """Fraction of energy in the top k directions of a centred stack."""
    s = torch.linalg.svdvals(M - M.mean(dim=0, keepdim=True)).numpy() ** 2
    return {k: float(s[:k].sum() / s.sum()) for k in KS}


def floor_at_k(shape, norms: torch.Tensor | None) -> dict[int, float]:
    """Random directions, either equal-norm or carrying the supplied norms."""
    g = torch.randn(shape, generator=torch.Generator().manual_seed(0))
    g = torch.nn.functional.normalize(g, dim=1)
    if norms is not None:
        g = g * norms[:, None]
    return energy_at_k(g)


def cells_in_pool() -> list[tuple[str, int]]:
    pool = load_pool(str(paths.resolve(paths.DOES_THE_FIX_REACH_UNSEEN_PAIRS) / "pair_pool.yaml"))
    out = []
    for slug in pool.train:
        for split in ("train", "heldout"):
            d = CACHE_ROOT / split / slug
            if not d.is_dir():
                continue
            for sd in sorted(d.glob("seed_*"), key=lambda p: int(p.name.split("_")[1])):
                if len(list((sd / "residuals").glob("step_*.pt"))) >= 40:
                    out.append((slug, int(sd.name.split("_")[1])))
    return out


def report(name: str, ek: dict[int, float], fl: dict[int, float]) -> dict:
    print(f"\n{name}")
    print(f"  {'k':>3} {'r_t':>7} {'floor':>7} {'ratio':>7}")
    for k in KS:
        print(f"  {k:>3} {ek[k]:>6.1%} {fl[k]:>6.1%} {ek[k]/fl[k]:>6.1f}x")
    return {"energy_at_k": {str(k): ek[k] for k in KS},
            "floor_at_k": {str(k): fl[k] for k in KS},
            "ratio_at_k": {str(k): ek[k] / fl[k] for k in KS}}


def main() -> int:
    cells = cells_in_pool()
    print(f"{len(cells)} cells in the pool's training split")

    # 1. The pooled stack F6 draws, against both floors.
    pooled = torch.cat(
        [load_cell(s, sd, root=CACHE_ROOT).r_t()[::10].flatten(1).float()
         for s, sd in cells], dim=0)
    norms = pooled.norm(dim=1)
    print(f"\npooled stack: {pooled.shape[0]} rows x {pooled.shape[1]} dims")
    print(f"||r_t||: min {norms.min():.1f}, median {norms.median():.1f}, "
          f"max {norms.max():.1f}, max/median {norms.max()/norms.median():.1f}x")
    ek_pooled = energy_at_k(pooled)
    out = {
        "pooled_rows": int(pooled.shape[0]),
        "row_norm_min": float(norms.min()),
        "row_norm_median": float(norms.median()),
        "row_norm_max": float(norms.max()),
        "pooled_vs_gaussian": report(
            "pooled stack against the equal-norm Gaussian floor (what F6 draws)",
            ek_pooled, floor_at_k(pooled.shape, None)),
        "pooled_vs_norm_matched": report(
            "pooled stack against random directions carrying the real ||r_t||",
            ek_pooled, floor_at_k(pooled.shape, norms)),
    }

    # 2. WITHIN one cell, magnitude removed.
    within = []
    for slug, seed in cells[:N_WITHIN_CELLS]:
        r = load_cell(slug, seed, root=CACHE_ROOT).r_t().flatten(1)[:ROWS_PER_STACK].float()
        r = torch.nn.functional.normalize(r, dim=1)
        within.append(report(f"WITHIN one cell, unit rows: {slug} seed {seed}",
                             energy_at_k(r), floor_at_k(r.shape, None)))
    out["within_cell_unit_rows"] = within

    # 3. ACROSS cells, magnitude removed, same shape as each WITHIN stack.
    rows = []
    for i, (slug, seed) in enumerate(cells[:ROWS_PER_STACK]):
        r = load_cell(slug, seed, root=CACHE_ROOT).r_t().flatten(1).float()
        j = (i * STEP_ROTATION) % ROWS_PER_STACK
        rows.append(r[j:j + 1])
    across = torch.nn.functional.normalize(torch.cat(rows, dim=0), dim=1)
    out["across_cells_unit_rows"] = report(
        f"ACROSS {across.shape[0]} different cells, one step each, unit rows",
        energy_at_k(across), floor_at_k(across.shape, None))

    print("\n" + "=" * 60)
    print(f"  {'k':>3} {'WITHIN one cell':>18} {'ACROSS cells':>15}   (unit rows)")
    for k in KS:
        w = float(np.median([d["ratio_at_k"][str(k)] for d in within]))
        a = out["across_cells_unit_rows"]["ratio_at_k"][str(k)]
        print(f"  {k:>3} {w:>17.1f}x {a:>14.1f}x")
    out["within_cell_median_ratio_at_k"] = {
        str(k): float(np.median([d["ratio_at_k"][str(k)] for d in within])) for k in KS}

    dest = Path(__file__).resolve().parent / "result.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
