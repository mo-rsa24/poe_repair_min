"""Build inspector manifest for the LoRA SDXL training chain.

Walks a canonical sequence of resume-linked runs and emits a flat JSON manifest
mapping (epoch, lambda) -> decoded.png path. Later runs in the chain override
earlier ones when epochs overlap (resume probes write the same weights at the
overlap boundary, so either copy is fine — last one wins is simplest).

Usage:
    python scripts/build_lora_manifest.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PAIR_ROOT = REPO_ROOT / "outputs/lora/a_cat__x__a_dog/seed_42"
DEFAULT_MONO_PATH = REPO_ROOT / "outputs/training_cache/heldout/a_cat__x__a_dog/seed_42/mono.png"

# Canonical chain — order matters: later runs override earlier ones at overlaps.
DEFAULT_CHAIN = [
    "m5_perarm_long2",
    "m5_perarm_resume100_plus200",
    "m5_perarm_resume300_plus500",
    "m5_perarm_resume800_plus800",
]

EPOCH_RE = re.compile(r"^epoch_(\d+)$")
LAMBDA_RE = re.compile(r"^lambda_(\d+\.\d+)$")


def scan_run(run_dir: Path) -> dict[int, dict[str, str]]:
    """Return {epoch: {lambda_str: rel_decoded_png}} for a single run."""
    probes_root = run_dir / "probes"
    if not probes_root.is_dir():
        return {}
    out: dict[int, dict[str, str]] = {}
    for epoch_dir in sorted(probes_root.iterdir()):
        m = EPOCH_RE.match(epoch_dir.name)
        if not m or not epoch_dir.is_dir():
            continue
        epoch = int(m.group(1))
        cells: dict[str, str] = {}
        for lam_dir in sorted(epoch_dir.iterdir()):
            lm = LAMBDA_RE.match(lam_dir.name)
            if not lm or not lam_dir.is_dir():
                continue
            decoded = lam_dir / "decoded.png"
            if not decoded.is_file():
                continue
            cells[lm.group(1)] = str(decoded.relative_to(REPO_ROOT))
        if cells:
            out[epoch] = cells
    return out


def build_manifest(pair_root: Path, chain: list[str], mono_path: Path | None) -> dict:
    merged: dict[int, dict[str, str]] = {}
    sources: dict[int, str] = {}
    for run_name in chain:
        run_dir = pair_root / run_name
        if not run_dir.is_dir():
            print(f"[warn] missing run dir: {run_dir}")
            continue
        run_cells = scan_run(run_dir)
        for epoch, cells in run_cells.items():
            existing = merged.get(epoch, {})
            existing.update(cells)
            merged[epoch] = existing
            sources[epoch] = run_name

    epochs = sorted(merged.keys())
    all_lambdas = sorted({lam for cells in merged.values() for lam in cells})

    mono_rel: str | None = None
    if mono_path is not None and mono_path.is_file():
        mono_rel = str(mono_path.relative_to(REPO_ROOT))
    elif mono_path is not None:
        print(f"[warn] mono image not found: {mono_path}")

    return {
        "pair_slug": pair_root.parent.name,
        "seed": int(pair_root.name.split("_")[-1]),
        "chain": chain,
        "epochs": epochs,
        "lambdas": all_lambdas,
        "cells": {str(e): merged[e] for e in epochs},
        "cell_source_run": {str(e): sources[e] for e in epochs},
        "mono_path": mono_rel,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-root", default=str(DEFAULT_PAIR_ROOT))
    ap.add_argument("--chain", nargs="+", default=DEFAULT_CHAIN)
    ap.add_argument("--mono-path", default=str(DEFAULT_MONO_PATH),
                    help="oracle mono reference (epoch-independent); set to '' to omit")
    ap.add_argument("--output", default=None,
                    help="default: <pair-root>/inspector_manifest.json")
    args = ap.parse_args()

    pair_root = Path(args.pair_root)
    mono_path = Path(args.mono_path) if args.mono_path else None
    manifest = build_manifest(pair_root, list(args.chain), mono_path)

    out_path = Path(args.output) if args.output else pair_root / "inspector_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_path}")
    print(f"  epochs: {len(manifest['epochs'])} "
          f"({manifest['epochs'][0]} -> {manifest['epochs'][-1]})")
    print(f"  lambdas: {manifest['lambdas']}")
    n_cells = sum(len(c) for c in manifest["cells"].values())
    print(f"  cells: {n_cells}")
    print(f"  mono: {manifest['mono_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
