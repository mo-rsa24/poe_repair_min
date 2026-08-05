#!/usr/bin/env python
"""Is the interaction term small, and is it the same small thing across pairs?

Two questions, one script.

Smallness: stack the cached r_t vectors and take their singular values. If the
energy piles into the first few components, r_t lives in a low-dimensional
subspace. The honest comparison is against a Gaussian of the same shape, since
any finite sample of high-dimensional vectors concentrates somewhat by chance.
Beating that floor is the claim; slow decay narrows it rather than killing it.

Sharedness: fit the subspace on training pairs only, then measure how much of
the held-out pairs' energy it captures. A train-fitted subspace explaining
held-out energy at small k is what "shared across pairs" means operationally.
Without this split the spectrum only says r_t is compressible, not that it is
the *same* r_t.

Cache-only, no GPU.

Usage:
    python scripts/spectrum.py --pair a_cat__x__a_dog        # one pair, no split
    python scripts/spectrum.py --all --max-pairs 20
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
    load_cell,
)

OUT_DIR = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses")
KS = (1, 2, 4, 8, 16, 32, 64)


def collect(root: Path, split: str, pairs: list[str] | None, max_pairs: int | None,
            max_seeds: int, stride: int) -> tuple[torch.Tensor, list[str]]:
    """Stack r_t vectors from one split. Returns [N, D] fp32 and the pair list."""
    split_dir = root / split
    slugs = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    if pairs:
        slugs = [s for s in slugs if s in pairs]
    if max_pairs:
        slugs = slugs[:max_pairs]

    rows: list[torch.Tensor] = []
    used: list[str] = []
    for slug in slugs:
        seeds = sorted(
            int(d.name.split("_")[1]) for d in (split_dir / slug).iterdir()
            if d.is_dir() and d.name.startswith("seed_")
        )[:max_seeds]
        for seed in seeds:
            c = load_cell(slug, seed, root=root)
            # Subsample steps: adjacent steps are highly correlated, so taking
            # every one inflates the row count without adding rank.
            rows.append(c.r_t()[::stride].flatten(1))
        if seeds:
            used.append(slug)
    if not rows:
        return torch.empty(0), []
    return torch.cat(rows, dim=0), used


def energy_at_k(s: np.ndarray) -> dict[int, float]:
    """Fraction of total energy in the top-k singular directions."""
    e = s ** 2
    total = float(e.sum())
    return {k: float(e[:k].sum() / total) for k in KS if k <= len(e)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", action="append", dest="pairs")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=12)
    ap.add_argument("--max-seeds", type=int, default=1)
    ap.add_argument("--stride", type=int, default=5, help="step subsampling")
    ap.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    if not args.all and not args.pairs:
        ap.error("give --all or --pair")

    train, train_pairs = collect(
        args.cache_root, "train", args.pairs, args.max_pairs,
        args.max_seeds, args.stride,
    )
    if train.numel() == 0:
        print("no training cells matched")
        return 2
    print(f"train: {train.shape[0]} vectors x {train.shape[1]} dims "
          f"from {len(train_pairs)} pairs")

    # Centre before the SVD: an uncentred stack reports its own mean as
    # component 1, which would flatter the low-rank claim for free.
    mean = train.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(train - mean, full_matrices=False)
    s = S.numpy()
    ek = energy_at_k(s)

    # Same-shape Gaussian floor: how concentrated does a random stack look?
    g = torch.randn(train.shape, generator=torch.Generator().manual_seed(0))
    sg = torch.linalg.svdvals(g - g.mean(dim=0, keepdim=True)).numpy()
    ekg = energy_at_k(sg)

    print("\nenergy in the top k directions (train stack, centred):")
    print(f"  {'k':>4}  {'r_t':>8}  {'gaussian':>9}  {'ratio':>6}")
    for k in sorted(ek):
        print(f"  {k:>4}  {ek[k]:>7.1%}  {ekg[k]:>8.1%}  {ek[k]/ekg[k]:>5.1f}x")
    print("  Read the ratio, not the raw percentage. With N vectors in D dims "
          "and N << D the\n  centred stack has rank N-1, so a high energy-at-k "
          "is partly forced by N alone:\n  random vectors give 56% at k=16 when "
          "N=30, but only 6% when N=300.")
    if train.shape[0] < 4 * max(k for k in ek):
        print(f"  WARNING: only {train.shape[0]} vectors for k up to "
              f"{max(ek)}. Raise --max-pairs or --max-seeds, or lower --stride, "
              "before quoting any of these numbers.")

    result = {
        "train_pairs": train_pairs, "train_vectors": int(train.shape[0]),
        "dims": int(train.shape[1]),
        "energy_at_k": {str(k): v for k, v in ek.items()},
        "gaussian_floor_at_k": {str(k): v for k, v in ekg.items()},
        "singular_values_head": s[:64].tolist(),
    }

    # Held-out projection: does the TRAIN subspace explain unseen pairs?
    held, held_pairs = collect(
        args.cache_root, "heldout", args.pairs, args.max_pairs,
        args.max_seeds, args.stride,
    )
    # A slug in both splits is not held out; excluding it keeps the test honest.
    overlap = sorted(set(held_pairs) & set(train_pairs))
    if overlap:
        print(f"\nexcluding {len(overlap)} pair(s) cached in both splits: "
              f"{', '.join(overlap)}")
        held, held_pairs = collect(
            args.cache_root, "heldout",
            [p for p in held_pairs if p not in train_pairs],
            args.max_pairs, args.max_seeds, args.stride,
        )

    if held.numel() > 0:
        hc = held - mean                       # the train mean, not its own
        total = float((hc ** 2).sum())
        print(f"\nheld-out energy captured by the TRAIN subspace "
              f"({held.shape[0]} vectors, {len(held_pairs)} pairs):")
        proj = {}
        for k in KS:
            if k > Vh.shape[0]:
                break
            basis = Vh[:k]                      # [k, D], orthonormal rows
            captured = float(((hc @ basis.T) ** 2).sum())
            proj[k] = captured / max(total, 1e-12)
            print(f"  k={k:>3}: {proj[k]:>6.1%}   (train {ek.get(k, float('nan')):.1%})")
        result["heldout_pairs"] = held_pairs
        result["heldout_projection_at_k"] = {str(k): v for k, v in proj.items()}
    else:
        print("\nno held-out pairs available for the projection test")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "spectrum.json").write_text(json.dumps(result, indent=2))

    if not args.no_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ks = sorted(ek)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ks, [ek[k] for k in ks], "o-", color="tab:blue",
                label="r_t, training pairs")
        ax.plot(ks, [ekg[k] for k in ks], "s--", color="0.6",
                label="same-shape Gaussian")
        if held.numel() > 0:
            ax.plot([k for k in ks if k in proj], [proj[k] for k in ks if k in proj],
                    "^-", color="tab:orange", label="held-out pairs, train subspace")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("k (number of directions)")
        ax.set_ylabel("fraction of energy captured")
        ax.set_ylim(0, 1)
        ax.set_title(f"Energy in the top k directions ({len(train_pairs)} train pairs)")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_dir / "spectrum.png", dpi=150)
        print(f"\nfigure: {args.out_dir / 'spectrum.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
