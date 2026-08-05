#!/usr/bin/env python
"""Bulk-load smoke over the training cache.

Checks every cached residual file for the four eps keys, the expected shape,
fp16 dtype, and NaN/Inf. Prints "N/N ok" when clean, otherwise names every bad
file. Read-only: opens tensors, writes nothing.

A quietly dropped cell biases every average computed later, so this never skips
silently — a file that cannot be read is reported, not passed over.

Usage:
    python scripts/cache_smoke.py --all
    python scripts/cache_smoke.py --pair a_cat__x__a_dog
    python scripts/cache_smoke.py --pair a_cat__x__a_dog --seed 9
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

CACHE_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/training_cache")
EPS_KEYS = ("eps_a_raw", "eps_b_raw", "eps_j_raw", "eps_uncond")
EXPECTED_SHAPE = (1, 4, 128, 128)
EXPECTED_DTYPE = torch.float16


def iter_cells(root: Path, pair: str | None, seed: int | None):
    """Yield (split, pair_slug, seed_dir) for every cell under the cache."""
    for split in ("train", "heldout"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for pair_dir in sorted(split_dir.iterdir()):
            if not pair_dir.is_dir():
                continue
            if pair is not None and pair_dir.name != pair:
                continue
            for seed_dir in sorted(pair_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                if seed is not None and seed_dir.name != f"seed_{seed}":
                    continue
                yield split, pair_dir.name, seed_dir


def check_step_file(path: Path) -> list[str]:
    """Return a list of problems with one residual file; empty means clean."""
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # unreadable is a finding, not a crash
        return [f"unreadable: {type(exc).__name__}: {exc}"]

    problems: list[str] = []
    for key in EPS_KEYS:
        if key not in sd:
            problems.append(f"missing key {key}")
            continue
        t = sd[key]
        if tuple(t.shape) != EXPECTED_SHAPE:
            problems.append(f"{key} shape {tuple(t.shape)} != {EXPECTED_SHAPE}")
        if t.dtype != EXPECTED_DTYPE:
            problems.append(f"{key} dtype {t.dtype} != {EXPECTED_DTYPE}")
        # NaN/Inf in fp16 survives the cast to fp32, so check after upcasting.
        f = t.float()
        if torch.isnan(f).any():
            problems.append(f"{key} has NaN")
        if torch.isinf(f).any():
            problems.append(f"{key} has Inf")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="scan every cached pair")
    ap.add_argument("--pair", help="restrict to one pair slug")
    ap.add_argument("--seed", type=int, help="restrict to one seed")
    ap.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    ap.add_argument("--json", type=Path, help="also write a machine-readable report")
    args = ap.parse_args()

    if not args.all and args.pair is None:
        ap.error("give --all or --pair")
    if not args.cache_root.is_dir():
        print(f"FAIL: no cache at {args.cache_root}", file=sys.stderr)
        return 2

    pairs_seen: set[str] = set()
    bad_pairs: set[str] = set()
    n_cells = n_files = 0
    findings: list[dict] = []

    for split, pair_slug, seed_dir in iter_cells(
        args.cache_root, args.pair, args.seed
    ):
        pairs_seen.add(pair_slug)
        n_cells += 1
        res_dir = seed_dir / "residuals"
        if not res_dir.is_dir():
            bad_pairs.add(pair_slug)
            findings.append(
                {"pair": pair_slug, "split": split, "seed_dir": str(seed_dir),
                 "file": None, "problems": ["no residuals/ directory"]}
            )
            continue
        step_files = sorted(res_dir.glob("step_*.pt"))
        if not step_files:
            bad_pairs.add(pair_slug)
            findings.append(
                {"pair": pair_slug, "split": split, "seed_dir": str(seed_dir),
                 "file": None, "problems": ["residuals/ holds no step_*.pt"]}
            )
            continue
        for f in step_files:
            n_files += 1
            problems = check_step_file(f)
            if problems:
                bad_pairs.add(pair_slug)
                findings.append(
                    {"pair": pair_slug, "split": split, "seed_dir": str(seed_dir),
                     "file": str(f), "problems": problems}
                )

    n_pairs = len(pairs_seen)
    n_ok = n_pairs - len(bad_pairs)

    for entry in findings:
        where = entry["file"] or entry["seed_dir"]
        for p in entry["problems"]:
            print(f"BAD  {where}: {p}")

    print(f"{n_ok}/{n_pairs} ok   ({n_cells} cells, {n_files} step files)")
    if bad_pairs:
        print(f"pairs with problems: {', '.join(sorted(bad_pairs))}")

    # 18 train + 58 heldout directories are only 70 distinct pairs: six slugs
    # are cached under both splits. Anything averaging "per pair" over the
    # directory listing double-counts those six, so say it out loud.
    if args.all:
        by_split: dict[str, set[str]] = {}
        for split, pair_slug, _ in iter_cells(args.cache_root, None, None):
            by_split.setdefault(split, set()).add(pair_slug)
        overlap = sorted(set.intersection(*by_split.values())) if len(by_split) > 1 else []
        dirs = sum(len(v) for v in by_split.values())
        print(
            "pair directories: "
            + " + ".join(f"{len(v)} {k}" for k, v in sorted(by_split.items()))
            + f" = {dirs}, but {n_pairs} distinct slugs"
        )
        if overlap:
            print(f"cached under both splits ({len(overlap)}): {', '.join(overlap)}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"pairs_total": n_pairs, "pairs_ok": n_ok, "cells": n_cells,
             "step_files": n_files,
             "pair_dirs": {k: sorted(v) for k, v in by_split.items()} if args.all else None,
             "in_both_splits": overlap if args.all else None,
             "findings": findings}, indent=2,
        ))

    return 0 if not bad_pairs else 1


if __name__ == "__main__":
    raise SystemExit(main())
