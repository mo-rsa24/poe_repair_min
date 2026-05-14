"""Build dataset/ folder for the idea-2 proof-of-concept.

Collect-first-generate-later. This script does not run any sampler. It walks
known asset locations on disk, symlinks the PoE / Mono / CO3 image we want
for each (pair, seed) cell into ``dataset/cells/<pair>/seed_<n>/``, and
writes ``dataset/manifest.json`` + ``dataset/missing.json`` so we know
exactly what still needs to be generated.

Scope (edit ``HELD_OUT_PAIRS``, ``TRAIN_PAIRS``, ``SEEDS`` below to change):

  Held-out pair  : a cat × a dog
  Training pairs : a cow × a horse, a lion × a horse, a lion × a dog,
                   a tiger × a dog
  Seeds          : 1, 2, 3, 4, 5, 6, 7, 42
  Methods        : poe, mono, co3

For each (pair, seed, method) the script tries each source path in priority
order and uses the first one that exists. Assets are symlinked, not copied,
so the dataset folder stays cheap and reflects upstream changes.

Run: ``python scripts/build_dataset.py``  (idempotent)
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "dataset"
PILOT_ROOT = Path("/datasets/mmolefe/neurips2026/pilot_5seeds")
OUTPUTS_ROOT = REPO_ROOT / "outputs"
ARCHIVE_ROOT = OUTPUTS_ROOT / "_archive"


HELD_OUT_PAIRS: list[tuple[str, str]] = [
    ("a cat", "a dog"),
]
TRAIN_PAIRS: list[tuple[str, str]] = [
    ("a cow", "a horse"),
    ("a lion", "a horse"),
    ("a lion", "a dog"),
    ("a tiger", "a dog"),
]
SEEDS: list[int] = [1, 2, 3, 4, 8, 9, 10, 11, 12, 42]
METHODS: list[str] = ["poe", "mono"]


def slugify(prompt_a: str, prompt_b: str) -> str:
    import re
    def _clean(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")
    return f"{_clean(prompt_a)}__x__{_clean(prompt_b)}"


def candidate_sources(slug: str, seed: int, method: str) -> list[Path]:
    """Priority-ordered candidate source paths for one cell asset.

    The first path that exists on disk is used. Order is "most canonical
    first" — pilot for PoE/Mono, e1_held_out for CO3, archive as fallback.
    """
    cands: list[Path] = []
    pilot_cell = (
        PILOT_ROOT
        / f"seed_{seed}"
        / "group6_coherent_collision"
        / slug
    )
    e1_cell = OUTPUTS_ROOT / "e1_held_out" / "pairs" / slug / f"seed_{seed}"
    teacher_res_cell = (
        OUTPUTS_ROOT / "e_teacher_residual" / "pairs" / slug / f"seed_{seed}"
    )

    # Priority: in-repo (current outputs first, then archive) before pilot.
    # We prefer SDXL from this repo's pipeline so CFG / scheduler / steps
    # match what teacher_residual.run_teacher_residual would produce. Pilot
    # is also SDXL (model_id confirmed) but its summary.json doesn't pin
    # the inference settings, so it's a last resort.
    archive_cells_co3 = [
        ARCHIVE_ROOT / "e_perp_gated",
        ARCHIVE_ROOT / "e3_mono_methods",
        ARCHIVE_ROOT / "e6_mono_composite",
        ARCHIVE_ROOT / "e_router_endpoint",
        ARCHIVE_ROOT / "e5_vlm_reward",
        ARCHIVE_ROOT / "e6_5b_rollout",
        ARCHIVE_ROOT / "e6_5c_co3_vlm",
        ARCHIVE_ROOT / "e3_5_hparams",
    ]
    archive_cells_mono = [
        ARCHIVE_ROOT / "e3_mono_methods",
        ARCHIVE_ROOT / "e_perp_gated",
        ARCHIVE_ROOT / "e6_mono_composite",
        ARCHIVE_ROOT / "e_router_endpoint",
        ARCHIVE_ROOT / "e5_vlm_reward",
        ARCHIVE_ROOT / "e6_5b_rollout",
        ARCHIVE_ROOT / "e6_5c_co3_vlm",
    ]

    if method == "poe":
        cands += [
            e1_cell / "poe" / "poe.png",
            teacher_res_cell / "poe" / "poe.png",
            pilot_cell / "poe.png",
        ]
    elif method == "mono":
        cands += [
            e1_cell / "mono_literal" / "mono_literal.png",
            teacher_res_cell / "mono_literal" / "mono_literal.png",
        ]
        cands += [
            root / "pairs" / slug / f"seed_{seed}"
                / "mono_literal" / "mono_literal.png"
            for root in archive_cells_mono
        ]
        cands.append(pilot_cell / "monolithic.png")
    elif method == "co3":
        cands += [
            e1_cell / "co3_literal" / "co3_literal.png",
        ]
        cands += [
            root / "pairs" / slug / f"seed_{seed}"
                / "co3_literal" / "co3_literal.png"
            for root in archive_cells_co3
        ]
    return cands


def link_or_skip(src: Path, dst: Path) -> str:
    """Create a relative symlink dst -> src. If dst already correctly points
    to src, do nothing. Returns one of {'created','exists','replaced'}.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        try:
            current = dst.resolve()
        except OSError:
            current = None
        if current == src.resolve():
            return "exists"
        dst.unlink()
        dst.symlink_to(src)
        return "replaced"
    dst.symlink_to(src)
    return "created"


def main() -> None:
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    all_pairs = [
        ("held_out", a, b) for a, b in HELD_OUT_PAIRS
    ] + [
        ("train", a, b) for a, b in TRAIN_PAIRS
    ]

    manifest_rows: list[dict] = []
    missing: list[dict] = []

    for split, prompt_a, prompt_b in all_pairs:
        slug = slugify(prompt_a, prompt_b)
        for seed in SEEDS:
            cell_dir = DATASET_ROOT / "cells" / slug / f"seed_{seed}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            # Drop orphan symlinks for methods no longer in METHODS.
            for stale in cell_dir.glob("*.png"):
                if stale.stem not in METHODS:
                    stale.unlink()
            sources_record: dict[str, str | None] = {}
            for method in METHODS:
                cands = candidate_sources(slug, seed, method)
                chosen: Path | None = None
                for c in cands:
                    if c.exists():
                        chosen = c
                        break
                dst = cell_dir / f"{method}.png"
                if chosen is None:
                    if dst.is_symlink() or dst.exists():
                        dst.unlink()
                    sources_record[method] = None
                    missing.append({
                        "split": split, "pair": [prompt_a, prompt_b],
                        "slug": slug, "seed": seed, "method": method,
                    })
                else:
                    link_or_skip(chosen, dst)
                    sources_record[method] = str(chosen)
                manifest_rows.append({
                    "split": split, "pair": [prompt_a, prompt_b],
                    "slug": slug, "seed": seed, "method": method,
                    "source": sources_record[method],
                    "linked_path": str(dst),
                    "available": chosen is not None,
                })
            (cell_dir / "sources.json").write_text(
                json.dumps(sources_record, indent=2)
            )

    scope = {
        "held_out_pairs": [list(p) for p in HELD_OUT_PAIRS],
        "train_pairs": [list(p) for p in TRAIN_PAIRS],
        "seeds": SEEDS,
        "methods": METHODS,
        "n_cells_total": (len(HELD_OUT_PAIRS) + len(TRAIN_PAIRS)) * len(SEEDS),
        "n_assets_total": (len(HELD_OUT_PAIRS) + len(TRAIN_PAIRS)) * len(SEEDS) * len(METHODS),
    }
    (DATASET_ROOT / "scope.json").write_text(json.dumps(scope, indent=2))
    (DATASET_ROOT / "manifest.json").write_text(json.dumps({
        "scope": scope,
        "rows": manifest_rows,
    }, indent=2))
    (DATASET_ROOT / "missing.json").write_text(json.dumps({
        "n_missing": len(missing),
        "by_split": {
            "held_out": [m for m in missing if m["split"] == "held_out"],
            "train": [m for m in missing if m["split"] == "train"],
        },
        "by_method": {
            method: [m for m in missing if m["method"] == method]
            for method in METHODS
        },
        "all": missing,
    }, indent=2))

    n_have = sum(1 for r in manifest_rows if r["available"])
    n_total = len(manifest_rows)
    print(f"[build_dataset] wrote {DATASET_ROOT}")
    print(f"  scope:    {scope['n_cells_total']} cells, {scope['n_assets_total']} assets")
    print(f"  have:     {n_have} / {n_total}")
    print(f"  missing:  {len(missing)}")
    by_method = {m: 0 for m in METHODS}
    for r in missing:
        by_method[r["method"]] += 1
    print(f"  missing by method: {by_method}")
    by_split = {"held_out": 0, "train": 0}
    for r in missing:
        by_split[r["split"]] += 1
    print(f"  missing by split:  {by_split}")


if __name__ == "__main__":
    main()
