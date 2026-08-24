"""Cross-pair-aware seed-pool loader.

Same shape as ``cross_seed_lora_pooling.seed_pool.SeedPool`` but the
``pair_slug`` field is optional — the cross-pair YAML shares one seed
list across all pairs in the pair pool.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from poe_repair import paths


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POOL_PATH = paths.resolve(paths.GROUP_POOL_CONFIGS) / "seed_pool.yaml"


@dataclass(frozen=True)
class SeedPool:
    train_pool: tuple[int, ...]
    held_out: tuple[int, ...]
    source_path: Path
    pair_slug: str = ""        # empty = cross-pair (no per-pair binding)

    def assert_disjoint(self) -> None:
        overlap = set(self.train_pool) & set(self.held_out)
        if overlap:
            raise RuntimeError(
                f"seed-pool leak: {sorted(overlap)} present in both "
                f"train_pool and held_out (source={self.source_path})"
            )

    def subset(self, k: int) -> tuple[int, ...]:
        if k > len(self.train_pool):
            raise ValueError(
                f"k={k} exceeds train_pool size={len(self.train_pool)}; "
                f"extend seed_pool.yaml or pick smaller k"
            )
        return tuple(self.train_pool[:k])

    def persist_alongside(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "seed_pool.json"
        target.write_text(json.dumps({
            "pair_slug": self.pair_slug,
            "train_pool": list(self.train_pool),
            "held_out": list(self.held_out),
            "source_path": str(self.source_path),
        }, indent=2))
        return target


def load_seed_pool(path: Path | None = None) -> SeedPool:
    p = Path(path) if path is not None else DEFAULT_POOL_PATH
    if not p.exists():
        raise FileNotFoundError(f"seed_pool.yaml not found at {p}")
    raw = yaml.safe_load(p.read_text())
    pool = SeedPool(
        train_pool=tuple(int(s) for s in raw["train_pool"]),
        held_out=tuple(int(s) for s in raw["held_out"]),
        source_path=p,
        pair_slug=str(raw.get("pair_slug", "")),
    )
    pool.assert_disjoint()
    return pool


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="seed_pool")
    ap.add_argument("--seed-pool-path", default=None,
                    help=f"path to seed_pool.yaml (default: {DEFAULT_POOL_PATH})")
    ap.add_argument("--check-only", action="store_true",
                    help="load and assert disjoint; non-zero exit on leak")
    args = ap.parse_args(argv)
    pool = load_seed_pool(args.seed_pool_path)
    print(f"train_pool: {list(pool.train_pool)} (n={len(pool.train_pool)})")
    print(f"held_out  : {list(pool.held_out)} (n={len(pool.held_out)})")
    print(f"pair_slug : {pool.pair_slug!r} (empty = cross-pair)")
    print("overlap   : (assertion passed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
