"""Single source of truth for the cross-pair pool split.

Loads ``artifacts/_shared/cross_pair_pool_configs/pair_pool.yaml`` and asserts no
overlap between train and held-out pairs at load time. Mirrors the
seed-pool loader's contract for the pair axis.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from poe_repair import paths


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POOL_PATH = paths.resolve(paths.GROUP_POOL_CONFIGS) / "pair_pool.yaml"


@dataclass(frozen=True)
class PairPool:
    train: tuple[str, ...]
    heldout: tuple[str, ...]
    source_path: Path

    def assert_disjoint(self) -> None:
        overlap = set(self.train) & set(self.heldout)
        if overlap:
            raise RuntimeError(
                f"pair-pool leak: {sorted(overlap)} present in both "
                f"train and heldout (source={self.source_path})"
            )

    def all_pairs(self) -> tuple[str, ...]:
        return tuple(self.train) + tuple(self.heldout)

    def persist_alongside(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "pair_pool.json"
        target.write_text(json.dumps({
            "train": list(self.train),
            "heldout": list(self.heldout),
            "source_path": str(self.source_path),
        }, indent=2))
        return target


def load_pair_pool(path: Path | None = None) -> PairPool:
    p = Path(path) if path is not None else DEFAULT_POOL_PATH
    if not p.exists():
        raise FileNotFoundError(f"pair_pool.yaml not found at {p}")
    raw = yaml.safe_load(p.read_text())
    pool = PairPool(
        train=tuple(str(s) for s in raw["train"]),
        heldout=tuple(str(s) for s in raw["heldout"]),
        source_path=p,
    )
    pool.assert_disjoint()
    return pool


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pair_pool")
    ap.add_argument("--pair-pool", default=None,
                    help=f"path to pair_pool.yaml (default: {DEFAULT_POOL_PATH})")
    ap.add_argument("--check-only", action="store_true",
                    help="load and assert disjoint; non-zero exit on leak")
    args = ap.parse_args(argv)
    pool = load_pair_pool(args.pair_pool)
    print(f"train   : {list(pool.train)} (n={len(pool.train)})")
    print(f"heldout : {list(pool.heldout)} (n={len(pool.heldout)})")
    print("overlap : (assertion passed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
