"""Single source of truth for the cross-seed pool split.

Loads ``outputs/cross_seed_lora_pooling/seed_pool.yaml`` and asserts no
overlap between train and held-out at load time. Every script that
trains or evaluates a pooled LoRA must go through ``load_seed_pool``
so the same split is enforced uniformly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POOL_PATH = (
    REPO_ROOT / "outputs" / "cross_seed_lora_pooling" / "seed_pool.yaml"
)


@dataclass(frozen=True)
class SeedPool:
    pair_slug: str
    train_pool: tuple[int, ...]
    held_out: tuple[int, ...]
    source_path: Path

    def assert_disjoint(self) -> None:
        overlap = set(self.train_pool) & set(self.held_out)
        if overlap:
            raise RuntimeError(
                f"seed-pool leak: {sorted(overlap)} present in both "
                f"train_pool and held_out (source={self.source_path})"
            )

    def subset(self, k: int) -> tuple[int, ...]:
        """First k seeds from the train pool (deterministic ordering)."""
        if k > len(self.train_pool):
            raise ValueError(
                f"k={k} exceeds train_pool size={len(self.train_pool)}; "
                f"extend seed_pool.yaml or pick smaller k"
            )
        return tuple(self.train_pool[:k])

    def persist_alongside(self, out_dir: Path) -> Path:
        """Write a JSON copy into the run's output dir for later audit."""
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
        pair_slug=str(raw["pair_slug"]),
        train_pool=tuple(int(s) for s in raw["train_pool"]),
        held_out=tuple(int(s) for s in raw["held_out"]),
        source_path=p,
    )
    pool.assert_disjoint()
    return pool


if __name__ == "__main__":
    pool = load_seed_pool()
    print(f"pair_slug : {pool.pair_slug}")
    print(f"train_pool: {list(pool.train_pool)} (n={len(pool.train_pool)})")
    print(f"held_out  : {list(pool.held_out)}   (n={len(pool.held_out)})")
    print(f"overlap   : (assertion passed)")
