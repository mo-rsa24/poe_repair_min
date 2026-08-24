"""Per-pair prompt registry for the cross-pair LoRA pooling experiment.

Loads ``artifacts/_shared/cross_pair_pool_configs/pair_prompts.yaml``, which maps
each pair slug to its (prompt_a, prompt_b, joint_prompt). Verifies
every slug in a given ``PairPool`` has an entry.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml

from poe_repair.experiments.cross_pair_lora_pooling.pair_pool import (
    load_pair_pool,
)
from poe_repair import paths


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPTS_PATH = paths.resolve(paths.GROUP_POOL_CONFIGS) / "pair_prompts.yaml"


@dataclass(frozen=True)
class PairPrompts:
    prompt_a: str
    prompt_b: str
    joint_prompt: str


def load_pair_prompts(
    path: Path | None = None,
    *,
    pair_pool=None,
) -> dict[str, PairPrompts]:
    p = Path(path) if path is not None else DEFAULT_PROMPTS_PATH
    if not p.exists():
        raise FileNotFoundError(f"pair_prompts.yaml not found at {p}")
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"pair_prompts.yaml must be a top-level mapping, got {type(raw)}"
        )
    out: dict[str, PairPrompts] = {}
    for slug, entry in raw.items():
        if not isinstance(entry, dict):
            raise RuntimeError(
                f"pair_prompts.yaml: entry for {slug!r} is not a mapping"
            )
        try:
            out[str(slug)] = PairPrompts(
                prompt_a=str(entry["prompt_a"]),
                prompt_b=str(entry["prompt_b"]),
                joint_prompt=str(entry["joint_prompt"]),
            )
        except KeyError as exc:
            raise RuntimeError(
                f"pair_prompts.yaml: entry for {slug!r} missing key {exc}"
            ) from exc
    if pair_pool is not None:
        missing = [s for s in pair_pool.all_pairs() if s not in out]
        if missing:
            raise RuntimeError(
                f"pair_prompts.yaml missing entries for: {missing} "
                f"(source={p})"
            )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pair_prompts")
    ap.add_argument("--prompts", default=None,
                    help=f"path to pair_prompts.yaml (default: {DEFAULT_PROMPTS_PATH})")
    ap.add_argument("--pair-pool", default=None,
                    help="optional pair_pool.yaml to cross-check coverage")
    ap.add_argument("--check-only", action="store_true",
                    help="load + (optionally) verify coverage; non-zero exit on gap")
    args = ap.parse_args(argv)
    pool = load_pair_pool(args.pair_pool) if args.pair_pool else None
    prompts = load_pair_prompts(args.prompts, pair_pool=pool)
    for slug, p in prompts.items():
        print(f"{slug}: a={p.prompt_a!r}  b={p.prompt_b!r}  j={p.joint_prompt!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
