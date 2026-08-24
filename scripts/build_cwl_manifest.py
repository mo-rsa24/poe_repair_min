"""Rebuild conditioning_window_lora inspector manifests from on-disk cells.

Walks ``outputs/conditioning_window_lora/<pair>/seed_<n>/<mode>/`` and
emits one ``results/inspector_manifest.json`` per mode. Each manifest is
3D-keyed: ``cells[<epoch>][<lambda_tag>] -> {ckpt_path, lambda_value,
schedules: [...]}``.

Useful when summary.json files have stale absolute ``image_path`` fields
(e.g. after a directory rename or migration) — this rebuilder derives
the canonical relative path from the file's own location on disk.

Usage::

    python scripts/build_cwl_manifest.py
    python scripts/build_cwl_manifest.py --run-dir outputs/conditioning_window_lora/a_cat__x__a_dog/seed_42
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from poe_repair import paths

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = paths.resolve(paths.CFG_WINDOW_WITH_LORA) / "a_cat__x__a_dog/seed_42"

EPOCH_RE = re.compile(r"^epoch_(\d+)$")
LAMBDA_RE = re.compile(r"^lambda_(\d+\.\d+)$")


def _relpath(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def scan_cell(cell_dir: Path) -> list[dict]:
    schedules_root = cell_dir / "schedules"
    if not schedules_root.is_dir():
        return []
    out: list[dict] = []
    for sched_dir in sorted(schedules_root.iterdir()):
        if not sched_dir.is_dir():
            continue
        summary = sched_dir / "summary.json"
        image = sched_dir / "image.png"
        if not summary.is_file() or not image.is_file():
            continue
        d = json.loads(summary.read_text())
        rec = {
            "id": d.get("id", sched_dir.name),
            "family": d.get("family", ""),
            "mask": d.get("mask", ""),
            "num_on": int(d.get("num_on", 0)),
            "image_path": _relpath(image),
            "delta_norm_per_step": list(d.get("delta_norm_per_step", [])),
            "elapsed_s": float(d.get("elapsed_s", 0.0) or 0.0),
            "sanity": bool(d.get("sanity", False)),
        }
        out.append(rec)
    return out


def scan_mode(mode_dir: Path) -> tuple[dict[int, dict[str, dict]], set[int], set[str]]:
    cells: dict[int, dict[str, dict]] = {}
    epochs: set[int] = set()
    lambdas: set[str] = set()
    for epoch_dir in sorted(mode_dir.iterdir()):
        m = EPOCH_RE.match(epoch_dir.name)
        if not m or not epoch_dir.is_dir():
            continue
        epoch = int(m.group(1))
        for lam_dir in sorted(epoch_dir.iterdir()):
            lm = LAMBDA_RE.match(lam_dir.name)
            if not lm or not lam_dir.is_dir():
                continue
            lam_tag = lm.group(1)
            schedules = scan_cell(lam_dir)
            if not schedules:
                continue
            ckpt_guess = (
                paths.resolve(paths.ONE_PAIR_ONE_SEED)
                / f"a_cat__x__a_dog/seed_42/run__local/checkpoints/lora_step_{epoch:06d}.pt"
            )
            cells.setdefault(epoch, {})[lam_tag] = {
                "ckpt_path": (
                    _relpath(ckpt_guess) if ckpt_guess.is_file()
                    else f"lora_step_{epoch:06d}.pt (not found)"
                ),
                "lambda_value": float(lam_tag),
                "schedules": schedules,
            }
            epochs.add(epoch)
            lambdas.add(lam_tag)
    return cells, epochs, lambdas


def write_manifest(mode_dir: Path, mode_name: str) -> Path | None:
    cells, epochs, lambdas = scan_mode(mode_dir)
    if not cells:
        return None
    sanity_path = mode_dir / "results" / "sanity" / "sanity.json"
    sanity = None
    if sanity_path.is_file():
        try:
            sanity = json.loads(sanity_path.read_text())
        except Exception:
            sanity = None
    manifest = {
        "experiment": "conditioning_window_lora",
        "composition_mode": mode_name,
        "epochs": sorted(epochs),
        "lambdas": sorted(lambdas),
        "cells": {str(e): v for e, v in cells.items()},
        "sanity": sanity,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "prompt": "a cat and a dog",
        "prompt_a": "a cat",
        "prompt_b": "a dog",
        "pair_slug": "a_cat__x__a_dog",
        "seed": 42,
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
    }
    out = mode_dir / "results" / "inspector_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    total = sum(len(v) for v in cells.values())
    print(
        f"[{mode_name}] {out} — epochs={sorted(epochs)} "
        f"lambdas={sorted(lambdas)} cells={total}"
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"run dir not found: {run_dir}")
        return 1
    any_written = False
    for mode_dir in sorted(run_dir.iterdir()):
        if not mode_dir.is_dir():
            continue
        if mode_dir.name in {"with_prompt", "always"}:
            if write_manifest(mode_dir, mode_dir.name):
                any_written = True
    if not any_written:
        print("[warn] no cells found")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
