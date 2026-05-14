"""Stage a cat-dog-only overfit cache by symlinking existing cells.

Builds ``outputs/training_cache_overfit_catdog/`` so that:

    train/a_cat__x__a_dog/seed_{2, 3, 7, 9, 12}/   -> symlinks
    heldout/a_cat__x__a_dog/seed_42/               -> symlink

All cells point back at the existing cache under
``outputs/training_cache/heldout/a_cat__x__a_dog/``. No SDXL re-run is
needed because the cells were already built when cat-dog was held-out
in the original cache.

Idempotent: re-running just refreshes the symlinks.

Usage::

    python -m scripts.build_overfit_catdog_cache
"""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = (
    REPO_ROOT / "outputs" / "training_cache"
    / "heldout" / "a_cat__x__a_dog"
)
DEFAULT_TARGET_ROOT = REPO_ROOT / "outputs" / "training_cache_overfit_catdog"
PAIR_SLUG = "a_cat__x__a_dog"

DEFAULT_TRAIN_SEEDS = [2, 3, 7, 9, 12]
DEFAULT_HELDOUT_SEEDS = [42]


def _link_cell(source_seed_dir: Path, dest_seed_dir: Path) -> str:
    if not source_seed_dir.exists():
        raise FileNotFoundError(f"source cell missing: {source_seed_dir}")
    dest_seed_dir.parent.mkdir(parents=True, exist_ok=True)
    if dest_seed_dir.is_symlink() or dest_seed_dir.exists():
        dest_seed_dir.unlink()
    dest_seed_dir.symlink_to(source_seed_dir.resolve(), target_is_directory=True)
    return f"{dest_seed_dir} -> {source_seed_dir.resolve()}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-pair-dir", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    ap.add_argument("--train-seeds", nargs="*", type=int, default=DEFAULT_TRAIN_SEEDS)
    ap.add_argument("--heldout-seeds", nargs="*", type=int, default=DEFAULT_HELDOUT_SEEDS)
    args = ap.parse_args()

    src = args.source_pair_dir
    tgt = args.target_root
    print(f"[overfit-cache] source: {src}")
    print(f"[overfit-cache] target: {tgt}")

    if not src.exists():
        raise FileNotFoundError(f"source pair dir missing: {src}")

    overlap = sorted(set(args.train_seeds) & set(args.heldout_seeds))
    if overlap:
        raise ValueError(
            f"seeds {overlap} appear in both train and heldout — must be disjoint"
        )

    print(f"[overfit-cache] train seeds:    {args.train_seeds}")
    print(f"[overfit-cache] heldout seeds:  {args.heldout_seeds}")

    train_pair_dir = tgt / "train" / PAIR_SLUG
    heldout_pair_dir = tgt / "heldout" / PAIR_SLUG

    links: list[str] = []
    for seed in args.train_seeds:
        links.append(_link_cell(
            src / f"seed_{seed}", train_pair_dir / f"seed_{seed}",
        ))
    for seed in args.heldout_seeds:
        links.append(_link_cell(
            src / f"seed_{seed}", heldout_pair_dir / f"seed_{seed}",
        ))

    print()
    print("[overfit-cache] created symlinks:")
    for line in links:
        print(f"  {line}")
    print()
    print(f"[overfit-cache] done. Train this with --cache-root {tgt}")


if __name__ == "__main__":
    main()
