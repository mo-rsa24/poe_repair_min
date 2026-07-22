"""Build the training-cache for a taxonomy group's representative pair.

Resolves the group (G1, G2, G3, G4, G6) to the pair fixed in
``plans/LORA_TAXONOMY_PLAN.md`` and builds one cache cell per requested
seed under ``$POE_REPAIR_TRAINING_CACHE/<split>/<pair_slug>/seed_<N>/``.

The SDXL pipeline is loaded once per invocation and reused across all
seeds in the same process — much cheaper than spawning a fresh process
per seed.

G5 is intentionally not in the registry — see Plan 09 for the deferral
rationale. ``--group g5`` aborts.

Examples
--------
Build seeds 1..12 for G4 on GPU 0::

    CUDA_VISIBLE_DEVICES=0 python -m scripts.build_group_cache \
        --group g4

Shard the same 12 seeds across 4 GPUs (run each line in its own tmux
window after exporting ``POE_REPAIR_TRAINING_CACHE``)::

    CUDA_VISIBLE_DEVICES=0 python -m scripts.build_group_cache --group g4 --seeds 1,2,3
    CUDA_VISIBLE_DEVICES=1 python -m scripts.build_group_cache --group g4 --seeds 4,5,6
    CUDA_VISIBLE_DEVICES=0 python -m scripts.build_group_cache --group g4 --seeds 7,8,9
    CUDA_VISIBLE_DEVICES=1 python -m scripts.build_group_cache --group g4 --seeds 10,11,12

Rebuild even if the sentinel exists::

    python -m scripts.build_group_cache --group g1 --seeds 42 --overwrite
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from poe_repair.run import make_ctx
from poe_repair.training_cache import DEFAULT_CACHE_ROOT
from scripts.build_training_cache import build_cell


@dataclass(frozen=True)
class GroupSpec:
    label: str
    prompt_a: str
    prompt_b: str
    joint_prompt: str


GROUP_REGISTRY: dict[str, GroupSpec] = {
    "g1": GroupSpec(
        label="co-occurrence",
        prompt_a="a dolphin",
        prompt_b="an ocean wave",
        joint_prompt="a dolphin and an ocean wave",
    ),
    "g2": GroupSpec(
        label="factorization",
        prompt_a="a dog",
        prompt_b="oil painting style",
        joint_prompt="a dog in oil painting style",
    ),
    "g3": GroupSpec(
        label="object_plus_scene",
        prompt_a="a mailbox",
        prompt_b="a snowfield",
        joint_prompt="a mailbox in a snowfield",
    ),
    "g4": GroupSpec(
        label="dual_object",
        prompt_a="a typewriter",
        prompt_b="a cactus",
        joint_prompt="a typewriter and a cactus",
    ),
    "g6": GroupSpec(
        label="concept_collision",
        prompt_a="a cat",
        prompt_b="a dog",
        joint_prompt="a cat and a dog",
    ),
}

DEFERRED_GROUPS = {
    "g5": (
        "G5 (concept-pair entanglement) is deferred from the LoRA "
        "taxonomy arc — the entangled third concept is not specified by "
        "the joint prompt, so the reference trajectory is ambiguous and "
        "the cache target would be untrustworthy. See "
        "plans/LORA_TAXONOMY_PLAN.md and plans/09-lora-taxonomy-single-seed.md."
    ),
}


def _parse_seeds(raw: str) -> list[int]:
    if raw.strip().lower() in {"all", "default"}:
        return list(range(1, 13))
    seeds: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(chunk))
    if not seeds:
        raise ValueError(f"no seeds parsed from {raw!r}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds in {raw!r}")
    return seeds


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--group", required=True,
        help="Taxonomy group label: g1, g2, g3, g4, or g6. "
             "G5 is deferred and aborts.",
    )
    ap.add_argument(
        "--seeds", default="1-12",
        help="Comma-separated seeds and/or ranges, e.g. '1,2,3' or "
             "'1-12' or '42'. 'all' / 'default' → seeds 1..12.",
    )
    ap.add_argument("--split", default="heldout", choices=("heldout", "train"))
    ap.add_argument(
        "--output-root", type=Path, default=DEFAULT_CACHE_ROOT,
        help=f"Cache root. Defaults to $POE_REPAIR_TRAINING_CACHE or "
             f"{DEFAULT_CACHE_ROOT}.",
    )
    ap.add_argument("--num-inference-steps", type=int, default=None)
    ap.add_argument("--guidance-scale", type=float, default=None)
    ap.add_argument(
        "--overwrite", action="store_true",
        help="Rebuild even if the per-cell sentinel exists.",
    )
    args = ap.parse_args()

    group = args.group.strip().lower()
    if group in DEFERRED_GROUPS:
        print(f"[error] {DEFERRED_GROUPS[group]}", file=sys.stderr)
        return 2
    if group not in GROUP_REGISTRY:
        print(
            f"[error] unknown group {args.group!r}. "
            f"Known: {sorted(GROUP_REGISTRY)}; deferred: {sorted(DEFERRED_GROUPS)}.",
            file=sys.stderr,
        )
        return 2

    spec = GROUP_REGISTRY[group]
    seeds = _parse_seeds(args.seeds)

    print(
        f"[plan] group={group} ({spec.label}) "
        f"pair=({spec.prompt_a!r} × {spec.prompt_b!r}) "
        f"seeds={seeds} split={args.split} out_root={args.output_root}"
    )

    t_ctx = time.time()
    ctx = make_ctx(
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )
    print(f"[ctx ] loaded SDXL in {time.time() - t_ctx:.1f}s")

    for seed in seeds:
        t0 = time.time()
        cell_dir = build_cell(
            prompt_a=spec.prompt_a,
            prompt_b=spec.prompt_b,
            joint_prompt_text=spec.joint_prompt,
            seed=seed,
            split=args.split,
            out_root=args.output_root,
            overwrite=args.overwrite,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            ctx=ctx,
        )
        print(f"[done] seed={seed} dt={time.time() - t0:.1f}s -> {cell_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
