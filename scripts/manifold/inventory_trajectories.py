"""Plan-15 step 1 — inventory cached trajectories for the manifold-geometry
measurements on cat × dog under the cross-seed LoRA artefact.

Walks the filesystem (no sampling, no model loads) and reports, per
(seed, condition, lambda), whether the cached z_t trajectory and the
per-step guided eps_t are present. Writes a JSON gap manifest and prints
a human-readable summary.

Conditions inventoried:
    - mono            CFG on "a cat and a dog"
    - solo_a          CFG on "a cat"            (informational, not used by Claim A)
    - solo_b          CFG on "a dog"            (informational, not used by Claim A)
    - poe_no_lora     A∧B joint, lambda not applicable
    - poe_plus_lora   PoE + λ·R, lambda ∈ {0.00, 0.50, 1.00}

The bank-filter for plan-15 excludes seeds {1, 4, 6, 10}; this script
reports them as bank-excluded but still inventories the underlying
files so the supplementary panel (LoRA on mono-failed seeds) can find
them later.

Output:
    --out <path>.json   structured manifest, one entry per
                        (seed, condition, lambda); has_z, has_eps,
                        z_path, eps_path. Top-level summary keys list
                        the missing combinations for backfill planning.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from poe_repair import paths as poe_paths

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Config — paths and expected layouts
# ---------------------------------------------------------------------------

DEFAULT_PAIR_SLUG = "a_cat__x__a_dog"
DEFAULT_SEEDS = tuple(range(1, 13))          # 1..12
DEFAULT_BANK_EXCLUDED = (1, 4, 6, 10)        # mono failed co-occurrence
DEFAULT_LAMBDAS = (0.00, 0.50, 1.00)
DEFAULT_CONDITIONS = ("mono", "solo_a", "solo_b", "poe_no_lora", "poe_plus_lora")

# Expected cache locations. These are *candidate* paths — the script
# checks all of them and reports which (if any) exist.
CROSS_SEED_ROOT = REPO_ROOT / "outputs" / "cross_seed_lora_pooling" / DEFAULT_PAIR_SLUG
SEED42_LORA_ROOT = poe_paths.resolve(poe_paths.ONE_PAIR_ONE_SEED) / "a_cat__x__a_dog" / "seed_42" / "run__local"

# The seed-42 single-seed LoRA cached mds bank — informational only.
SEED42_MDS_STATIC = SEED42_LORA_ROOT / "mds_cache" / "static"
SEED42_MDS_CELLS = SEED42_LORA_ROOT / "mds_cache" / "cells"

# Plan-15's target storage layout (manifold_cache). May or may not exist.
MANIFOLD_CACHE = REPO_ROOT / "outputs" / "manifold_cache" / DEFAULT_PAIR_SLUG


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    seed: int
    condition: str
    lambda_value: float | None
    has_z: bool
    has_eps: bool
    z_path: str | None
    eps_path: str | None
    bank_excluded: bool

    def key(self) -> str:
        if self.lambda_value is None:
            return f"seed={self.seed:02d} {self.condition}"
        return f"seed={self.seed:02d} {self.condition} λ={self.lambda_value:.2f}"


# ---------------------------------------------------------------------------
# Path probing
# ---------------------------------------------------------------------------

def _first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.is_file():
            return p
    return None


def _z_candidates(seed: int, condition: str, lam: float | None) -> list[Path]:
    """Plausible cached z_t locations for this (seed, condition, lambda)."""
    out: list[Path] = []
    # 1) Plan-15 target layout
    if condition == "mono":
        out.append(MANIFOLD_CACHE / "mono_bank" / "z_t" / f"seed_{seed:03d}.npy")
    elif condition == "poe_no_lora":
        out.append(MANIFOLD_CACHE / "poe_no_lora" / "z_t" / f"seed_{seed:03d}.npy")
    elif condition == "poe_plus_lora" and lam is not None:
        lam_s = f"lambda_{lam:.2f}"
        out.append(MANIFOLD_CACHE / "poe_plus_lora" / lam_s / "z_t"
                   / f"seed_{seed:03d}.npy")
    elif condition == "solo_a":
        out.append(MANIFOLD_CACHE / "solo_a" / "z_t" / f"seed_{seed:03d}.npy")
    elif condition == "solo_b":
        out.append(MANIFOLD_CACHE / "solo_b" / "z_t" / f"seed_{seed:03d}.npy")

    # 2) Seed-42 single-seed LoRA legacy cache (z_t only, no eps).
    #    Only matches seed==42. We include it because if the user wants a
    #    fallback n=1 analysis, this is where the data lives.
    if seed == 42:
        if condition in ("mono", "solo_a", "solo_b"):
            out.append(SEED42_MDS_STATIC / f"{condition}.npy")
        elif condition == "poe_plus_lora" and lam is not None:
            # The seed-42 cache enumerates many epochs; we only check the
            # final epoch (1600) here, matching plan-15's headline.
            out.append(SEED42_MDS_CELLS / "epoch_1600" / f"lambda_{lam:.2f}.npy")
        elif condition == "poe_no_lora":
            # Equivalent to lambda=0.0 in the lora_residual_inject sampler.
            # Cached under that name.
            out.append(SEED42_MDS_CELLS / "epoch_1600" / "lambda_0.00.npy")

    return out


def _eps_candidates(seed: int, condition: str, lam: float | None) -> list[Path]:
    """Plausible cached eps_t (velocity) locations."""
    out: list[Path] = []
    # 1) Plan-15 target layout
    if condition == "mono":
        out.append(MANIFOLD_CACHE / "mono_bank" / "eps_t" / f"seed_{seed:03d}.npy")
    elif condition == "poe_no_lora":
        out.append(MANIFOLD_CACHE / "poe_no_lora" / "eps_t" / f"seed_{seed:03d}.npy")
    elif condition == "poe_plus_lora" and lam is not None:
        lam_s = f"lambda_{lam:.2f}"
        out.append(MANIFOLD_CACHE / "poe_plus_lora" / lam_s / "eps_t"
                   / f"seed_{seed:03d}.npy")
    elif condition == "solo_a":
        out.append(MANIFOLD_CACHE / "solo_a" / "eps_t" / f"seed_{seed:03d}.npy")
    elif condition == "solo_b":
        out.append(MANIFOLD_CACHE / "solo_b" / "eps_t" / f"seed_{seed:03d}.npy")

    # 2) The cross-seed sample_heldout pipeline emits eps records when run
    #    with --record-eps, at outputs/cross_seed_lora_pooling/.../eps_records/
    if condition == "poe_plus_lora" and lam is not None:
        out.append(CROSS_SEED_ROOT / "heldout_pair" / "a_wolf__x__a_husky"
                   / "eps_records" / f"eps_seed_{seed:02d}.pt")
        # Generic location (in case sampling was done into the cat-dog tree
        # directly, without the heldout_pair detour):
        out.append(CROSS_SEED_ROOT / "eps_records" / f"eps_seed_{seed:02d}.pt")

    return out


def probe(seed: int, condition: str, lam: float | None,
          bank_excluded_set: frozenset[int]) -> Cell:
    z_path = _first_existing(_z_candidates(seed, condition, lam))
    eps_path = _first_existing(_eps_candidates(seed, condition, lam))
    return Cell(
        seed=seed,
        condition=condition,
        lambda_value=lam,
        has_z=z_path is not None,
        has_eps=eps_path is not None,
        z_path=str(z_path) if z_path else None,
        eps_path=str(eps_path) if eps_path else None,
        bank_excluded=(seed in bank_excluded_set and condition == "mono"),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarise(cells: list[Cell]) -> dict:
    """Aggregate counts for the JSON header."""
    by_cond: dict[str, dict[str, int]] = {}
    for c in cells:
        d = by_cond.setdefault(c.condition, {"total": 0, "has_z": 0, "has_eps": 0,
                                              "bank_excluded": 0})
        d["total"] += 1
        d["has_z"] += int(c.has_z)
        d["has_eps"] += int(c.has_eps)
        d["bank_excluded"] += int(c.bank_excluded)
    missing_z = [c.key() for c in cells if not c.has_z]
    missing_eps = [c.key() for c in cells if not c.has_eps]
    return {
        "per_condition": by_cond,
        "missing_z": missing_z,
        "missing_eps": missing_eps,
        "n_total": len(cells),
        "n_complete": sum(1 for c in cells if c.has_z and c.has_eps),
    }


def print_table(cells: list[Cell]) -> None:
    print()
    print(f"{'seed':>4}  {'condition':<14} {'λ':>5}  z   eps  bank   path-hint")
    print("-" * 90)
    for c in cells:
        lam = "—" if c.lambda_value is None else f"{c.lambda_value:.2f}"
        z = "✓" if c.has_z else "·"
        e = "✓" if c.has_eps else "·"
        bank = "EXCL" if c.bank_excluded else ("kept" if c.condition == "mono" else "")
        hint = ""
        if c.z_path:
            hint = c.z_path.replace(str(REPO_ROOT) + "/", "")
        elif c.eps_path:
            hint = c.eps_path.replace(str(REPO_ROOT) + "/", "")
        print(f"{c.seed:>4}  {c.condition:<14} {lam:>5}  {z}   {e}   {bank:>4}   {hint}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                    help="comma-separated seed list (default: 1..12)")
    ap.add_argument("--bank-excluded",
                    default=",".join(str(s) for s in DEFAULT_BANK_EXCLUDED),
                    help="seeds excluded from the mono bank per plan-15 §2")
    ap.add_argument("--lambdas",
                    default=",".join(f"{x:.2f}" for x in DEFAULT_LAMBDAS),
                    help="comma-separated λ list for PoE+λR")
    ap.add_argument("--conditions",
                    default=",".join(DEFAULT_CONDITIONS),
                    help="comma-separated condition list")
    ap.add_argument("--out",
                    default=str(REPO_ROOT / "outputs" / "manifold_cache"
                                / "inventory.json"),
                    help="write JSON gap manifest here")
    ap.add_argument("--quiet", action="store_true",
                    help="skip the human-readable table; only write JSON")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    bank_excluded = frozenset(int(s) for s in args.bank_excluded.split(",") if s.strip())
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    cells: list[Cell] = []
    for seed in seeds:
        for cond in conditions:
            if cond == "poe_plus_lora":
                for lam in lambdas:
                    cells.append(probe(seed, cond, lam, bank_excluded))
            else:
                cells.append(probe(seed, cond, None, bank_excluded))

    if not args.quiet:
        print_table(cells)

    summary = summarise(cells)
    summary["seeds"] = seeds
    summary["bank_excluded"] = sorted(bank_excluded)
    summary["lambdas"] = lambdas
    summary["conditions"] = conditions
    summary["cells"] = [asdict(c) for c in cells]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print()
    print(f"wrote inventory → {out_path}")
    print(f"  cells total      = {summary['n_total']}")
    print(f"  cells complete   = {summary['n_complete']}   (have both z and eps)")
    print(f"  missing z_t      = {len(summary['missing_z'])}")
    print(f"  missing eps_t    = {len(summary['missing_eps'])}")
    if summary["n_complete"] < summary["n_total"]:
        print()
        print("Backfill is required before plan-15 can run on this artefact.")
        print("See: scripts/manifold/sample_with_trajectory.py (to be written).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
