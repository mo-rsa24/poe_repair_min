#!/usr/bin/env python
"""Compute the candidate correction-size measures, and show why one is void.

Backs `report/normalization_preregistration.md`. Re-running this reproduces every
number in that memo.

Three candidates:

  relative_norm     ||r_t|| / ||eps_PoE||          the committed choice
  distance_fraction ||r_t|| / ||eps_Mono - eps_PoE||   identically 1, see below
  step_relative     ||r_t|| / ||x_{t+1} - x_t||    rejected, sampler-dependent

`distance_fraction` is the interesting one. It reads exactly 1.000000 for every
step of every pair, because r_t IS eps_Mono - eps_PoE, so it divides a quantity
by itself. This script prints it anyway rather than quietly dropping it: a
measure that cannot vary is worth seeing once, so the wording does not come
back later in a disguised form.

Cache-only, no GPU.

    python scripts/normalization_candidates.py
    python scripts/normalization_candidates.py --pair a_frog__x__a_toad
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from poe_repair import paths

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.experiments.interaction_term.cache import (  # noqa: E402
    CACHE_ROOT,
    load_cell,
)

OUT_DIR = paths.resolve(paths.CACHE_ANALYSES)
# One training pair, one unseen transfer pair, the known-failure reference.
DEFAULT_PAIRS = (
    "a_wolf__x__a_husky",
    "a_leopard__x__a_jaguar",
    "a_cat__x__a_dog",
)
MIN_STEPS = 2


def first_seed(slug: str, root: Path) -> int | None:
    """First seed of this pair with a real trajectory, not an eval stub."""
    for split in ("train", "heldout"):
        d = root / split / slug
        if not d.is_dir():
            continue
        for sd in sorted(d.glob("seed_*"), key=lambda p: int(p.name.split("_")[1])):
            if len(list((sd / "residuals").glob("step_*.pt"))) >= MIN_STEPS:
                return int(sd.name.split("_")[1])
    return None


def candidates(cell) -> dict[str, np.ndarray]:
    """The three per-step measures for one cell."""
    r = cell.r_t().flatten(1).norm(dim=1)
    poe = cell.eps_poe().flatten(1).norm(dim=1)
    gap = (cell.eps_mono() - cell.eps_poe()).flatten(1).norm(dim=1)
    dx = (cell.x_t[1:] - cell.x_t[:-1]).flatten(1).norm(dim=1)
    return {
        "relative_norm": (r / poe).numpy(),
        "distance_fraction": (r / gap.clamp_min(1e-12)).numpy(),
        "step_relative": (r[:-1] / dx.clamp_min(1e-12)).numpy(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", action="append", dest="pairs")
    ap.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    pairs = args.pairs or list(DEFAULT_PAIRS)

    rows = []
    print(f"{'pair':<26}{'seed':>5}{'relative_norm':>15}"
          f"{'distance_frac':>15}{'step_relative':>15}")
    for slug in pairs:
        seed = first_seed(slug, args.cache_root)
        if seed is None:
            print(f"  {slug}: no cached trajectory", file=sys.stderr)
            continue
        c = candidates(load_cell(slug, seed, root=args.cache_root))
        med = {k: float(np.median(v)) for k, v in c.items()}
        iqr = {k: float(np.percentile(v, 75) - np.percentile(v, 25))
               for k, v in c.items()}
        rows.append({"pair": slug, "seed": seed, "median": med, "iqr": iqr})
        print(f"{slug:<26}{seed:>5}{med['relative_norm']:>15.4f}"
              f"{med['distance_fraction']:>15.6f}{med['step_relative']:>15.4f}")

    if not rows:
        return 2

    df = np.array([r["median"]["distance_fraction"] for r in rows])
    print(f"\ndistance_fraction: min {df.min():.6f}, max {df.max():.6f}")
    if np.allclose(df, 1.0, atol=1e-6):
        print("  VOID. It is identically 1 because r_t IS eps_Mono - eps_PoE,")
        print("  so this divides a quantity by itself. It cannot vary, cannot")
        print("  distinguish pairs, and cannot fail. Not a measure.")

    print("\nwithin-pair spread (IQR over steps), lower is more stable:")
    for r in rows:
        print(f"  {r['pair']:<26} relative_norm {r['iqr']['relative_norm']:.4f}"
              f"   step_relative {r['iqr']['step_relative']:.4f}")
    print("\nstep_relative rejected: its denominator is the sampler step size,")
    print("which plan 08 varies, so the measure would move with the schedule.")
    print("\nCOMMITTED: relative_norm  (report/normalization_preregistration.md)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "normalization_candidates.json").write_text(
        json.dumps({"committed": "relative_norm", "pairs": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
