"""Build inspector manifest for the consolidated LoRA SDXL run.

Scans a single ``results/`` directory and emits a flat JSON manifest
mapping (epoch, lambda) -> decoded.png path. Expects the layout produced
by the cleanup commit C6::

    outputs/lora/<pair>/seed_<n>/results/
        checkpoints/lora_step_*.pt
        thumbnails/thumbnails_epoch_*.png
        probes/epoch_*/lambda_*/decoded.png
        figures/{cumulative_grid,curve_vqa_vs_lambda}.png
        history.json
        resume_log.json
        config.json
        lora_attach.json

Usage::

    python scripts/build_lora_manifest.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from poe_repair import paths

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RESULTS = paths.resolve(paths.ONE_PAIR_ONE_SEED) / "a_cat__x__a_dog/seed_42/run__local"

EPOCH_RE = re.compile(r"^epoch_(\d+)$")
LAMBDA_RE = re.compile(r"^lambda_(\d+\.\d+)$")


def scan_probes(probes_root: Path) -> dict[int, dict[str, str]]:
    """Return ``{epoch: {lambda_str: rel_decoded_png}}``."""
    out: dict[int, dict[str, str]] = {}
    if not probes_root.is_dir():
        return out
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


def build_manifest(results_root: Path) -> dict:
    probes = scan_probes(results_root / "probes")
    epochs = sorted(probes.keys())
    all_lambdas = sorted({lam for cells in probes.values() for lam in cells})
    return {
        "results_root": str(results_root.relative_to(REPO_ROOT)),
        "epochs": epochs,
        "lambdas": all_lambdas,
        "cells": {str(e): probes[e] for e in epochs},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=str(DEFAULT_RESULTS))
    ap.add_argument(
        "--output", default=None,
        help="default: <results-root>/inspector_manifest.json",
    )
    args = ap.parse_args()

    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        # Prepend the repo root without resolve(), so that any `results`
        # symlink stays as `.../results/...` in the recorded paths instead
        # of being rewritten to the underlying run-dir name.
        results_root = REPO_ROOT / results_root
    manifest = build_manifest(results_root)

    out_path = (
        Path(args.output) if args.output
        else results_root / "inspector_manifest.json"
    )
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_path}")
    if manifest["epochs"]:
        print(
            f"  epochs: {len(manifest['epochs'])} "
            f"({manifest['epochs'][0]} -> {manifest['epochs'][-1]})"
        )
    print(f"  lambdas: {manifest['lambdas']}")
    n_cells = sum(len(c) for c in manifest["cells"].values())
    print(f"  cells: {n_cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
