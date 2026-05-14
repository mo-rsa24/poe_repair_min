"""Verify training-cache integrity in one shot.

Walks ``outputs/training_cache/{train,heldout}/<pair>/<seed>/`` and
checks every cell for:

  - ``meta.json`` exists and is parseable
  - ``embeddings.pt`` has all expected keys with non-empty tensors
  - ``residuals/`` contains exactly ``step_000.pt`` through
    ``step_(num_steps-1).pt``
  - each step file has the expected keys and consistent tensor shapes
  - ``step_index`` field matches the filename
  - ``timestep`` is monotonically descending across steps (DDIM convention)
  - ``mono.png`` and ``poe.png`` exist and are readable
  - regenerated-vs-curated pixel diff in ``meta.json`` is below a warning
    threshold (default 5/255 mean-abs-diff)

Cross-cell:
  - all cells within a split share the same ``num_inference_steps`` and
    the same x_t latent shape
  - ``manifest.json`` (if present) lists exactly the cells on disk

Exit code is 0 if no cell is in FAIL state, 1 otherwise. WARN does not
fail. Use ``--full`` to spot-check every step file instead of the first /
middle / last; ``--json`` to emit a machine-readable summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_ROOT = REPO_ROOT / "outputs" / "training_cache"

EXPECTED_STEP_KEYS: set[str] = {
    "x_t", "timestep", "step_index",
    "eps_a_raw", "eps_b_raw", "eps_j_raw", "eps_uncond",
}
EXPECTED_EMB_KEYS: set[str] = {
    "seq_a", "pool_a", "seq_b", "pool_b",
    "seq_j", "pool_j", "seq_uncond", "pool_uncond",
    "init_latents", "euler_init_noise_sigma",
}
EPS_KEYS = ("eps_a_raw", "eps_b_raw", "eps_j_raw", "eps_uncond")

PIXEL_DIFF_WARN_THRESHOLD = 5.0  # uint8 scale; ~2% mean-abs pixel diff


# ---------------------------------------------------------------------------
# Per-cell checks
# ---------------------------------------------------------------------------


def _load_step(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _check_step_file(
    cell_dir: Path,
    step_index: int,
    *,
    issues: list[str],
    warnings: list[str],
    ref_x_t_shape: tuple[int, ...] | None,
    prev_timestep: int | None,
) -> tuple[tuple[int, ...] | None, int | None]:
    """Validate one step file. Returns (x_t_shape, timestep) for chaining."""
    path = cell_dir / "residuals" / f"step_{step_index:03d}.pt"
    if not path.exists():
        issues.append(f"step_{step_index:03d}.pt missing")
        return ref_x_t_shape, prev_timestep
    try:
        step = _load_step(path)
    except Exception as exc:
        issues.append(f"step_{step_index:03d}.pt unloadable: {exc}")
        return ref_x_t_shape, prev_timestep

    missing = EXPECTED_STEP_KEYS - set(step.keys())
    if missing:
        issues.append(f"step_{step_index:03d}.pt missing keys: {sorted(missing)}")
        return ref_x_t_shape, prev_timestep

    saved_index = int(step["step_index"])
    if saved_index != step_index:
        issues.append(
            f"step_{step_index:03d}.pt has step_index={saved_index} (filename mismatch)"
        )

    x_t = step["x_t"]
    x_t_shape = tuple(x_t.shape)
    if ref_x_t_shape is None:
        ref_x_t_shape = x_t_shape
    elif x_t_shape != ref_x_t_shape:
        issues.append(
            f"step_{step_index:03d}.pt x_t shape {x_t_shape} != reference {ref_x_t_shape}"
        )

    for key in EPS_KEYS:
        eps_shape = tuple(step[key].shape)
        if eps_shape != x_t_shape:
            issues.append(
                f"step_{step_index:03d}.pt {key} shape {eps_shape} != x_t {x_t_shape}"
            )

    cur_t = int(step["timestep"])
    if prev_timestep is not None and cur_t > prev_timestep:
        issues.append(
            f"step {step_index} timestep {cur_t} > previous {prev_timestep} "
            "(DDIM expects descending)"
        )
    return ref_x_t_shape, cur_t


def verify_cell(cell_dir: Path, *, full: bool) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []

    meta_path = cell_dir / "meta.json"
    if not meta_path.exists():
        return {
            "path": str(cell_dir.relative_to(REPO_ROOT)),
            "status": "fail",
            "issues": ["meta.json missing"],
            "warnings": [],
            "num_steps": None, "split": None, "pair_slug": None, "seed": None,
        }
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as exc:
        return {
            "path": str(cell_dir.relative_to(REPO_ROOT)),
            "status": "fail",
            "issues": [f"meta.json unparseable: {exc}"],
            "warnings": [],
            "num_steps": None, "split": None, "pair_slug": None, "seed": None,
        }

    num_steps = meta.get("num_inference_steps")
    if num_steps is None:
        issues.append("meta.json missing num_inference_steps; assuming 50")
        num_steps = 50
    num_steps = int(num_steps)

    # --- embeddings.pt -----------------------------------------------------
    emb_path = cell_dir / "embeddings.pt"
    if not emb_path.exists():
        issues.append("embeddings.pt missing")
    else:
        try:
            emb = torch.load(emb_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            issues.append(f"embeddings.pt unloadable: {exc}")
        else:
            missing = EXPECTED_EMB_KEYS - set(emb.keys())
            if missing:
                issues.append(f"embeddings.pt missing keys: {sorted(missing)}")
            for key in ("seq_a", "seq_b", "seq_j", "seq_uncond",
                        "pool_a", "pool_b", "pool_j", "pool_uncond",
                        "init_latents"):
                if key in emb and isinstance(emb[key], torch.Tensor) and emb[key].numel() == 0:
                    issues.append(f"embeddings.pt[{key}] is empty")

    # --- residuals/ --------------------------------------------------------
    residuals_dir = cell_dir / "residuals"
    if not residuals_dir.exists():
        issues.append("residuals/ directory missing")
    else:
        expected = {f"step_{i:03d}.pt" for i in range(num_steps)}
        actual = {p.name for p in residuals_dir.glob("step_*.pt")}
        missing_files = expected - actual
        extra_files = actual - expected
        if missing_files:
            sample = sorted(missing_files)[:3]
            issues.append(
                f"residuals missing {len(missing_files)} files (e.g. {sample})"
            )
        if extra_files:
            warnings.append(
                f"residuals has {len(extra_files)} unexpected files: "
                f"{sorted(extra_files)[:3]}"
            )

        if full:
            indices = list(range(num_steps))
        elif num_steps >= 3:
            indices = [0, num_steps // 2, num_steps - 1]
        else:
            indices = list(range(num_steps))

        ref_shape: tuple[int, ...] | None = None
        prev_t: int | None = None
        for i in indices:
            ref_shape, prev_t = _check_step_file(
                cell_dir, i,
                issues=issues, warnings=warnings,
                ref_x_t_shape=ref_shape, prev_timestep=prev_t,
            )

    # --- images ------------------------------------------------------------
    for name in ("mono.png", "poe.png"):
        p = cell_dir / name
        if not p.exists():
            issues.append(f"{name} missing")
            continue
        try:
            with Image.open(p) as img:
                img.load()
        except Exception as exc:
            issues.append(f"{name} unreadable: {exc}")

    # --- verification field ------------------------------------------------
    ver = meta.get("verification", {})
    poe_diff = ver.get("regenerated_vs_curated_poe_mean_abs_pixel_diff")
    mono_diff = ver.get("regenerated_vs_curated_mono_mean_abs_pixel_diff")
    if poe_diff is not None and poe_diff > PIXEL_DIFF_WARN_THRESHOLD:
        warnings.append(
            f"poe pixel-diff vs curated: {poe_diff:.2f} "
            f"(>{PIXEL_DIFF_WARN_THRESHOLD})"
        )
    if mono_diff is not None and mono_diff > PIXEL_DIFF_WARN_THRESHOLD:
        warnings.append(
            f"mono pixel-diff vs curated: {mono_diff:.2f} "
            f"(>{PIXEL_DIFF_WARN_THRESHOLD})"
        )

    status = "fail" if issues else ("warn" if warnings else "ok")
    return {
        "path": str(cell_dir.relative_to(REPO_ROOT)),
        "status": status,
        "issues": issues,
        "warnings": warnings,
        "num_steps": num_steps,
        "split": meta.get("split"),
        "pair_slug": meta.get("pair_slug"),
        "seed": meta.get("seed"),
        "x_t_shape": list(ref_shape) if "ref_shape" in dir() and ref_shape else None,
    }


# ---------------------------------------------------------------------------
# Cross-cell + manifest checks
# ---------------------------------------------------------------------------


def check_cross_cell(results: list[dict]) -> list[str]:
    notes: list[str] = []
    by_split: dict[str, list[dict]] = {}
    for r in results:
        if r["status"] == "fail" or r["split"] is None:
            continue
        by_split.setdefault(r["split"], []).append(r)
    for split, rows in by_split.items():
        steps = {r["num_steps"] for r in rows}
        if len(steps) > 1:
            notes.append(
                f"split={split}: inconsistent num_inference_steps across cells: {steps}"
            )
        shapes = {tuple(r["x_t_shape"]) for r in rows if r.get("x_t_shape")}
        if len(shapes) > 1:
            notes.append(
                f"split={split}: inconsistent x_t shapes across cells: {shapes}"
            )
    return notes


def check_manifest(cache_root: Path, results: list[dict]) -> list[str]:
    notes: list[str] = []
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.exists():
        notes.append(f"manifest.json missing at {manifest_path}")
        cohort_files = sorted(cache_root.glob("manifest_*.json"))
        if cohort_files:
            names = [p.name for p in cohort_files]
            notes.append(
                f"  found cohort manifests {names}; run "
                "`python -m scripts.build_training_cache --consolidate`"
            )
        return notes
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        notes.append(f"manifest.json unparseable: {exc}")
        return notes

    on_disk = {
        (r["split"], r["pair_slug"], r["seed"])
        for r in results
        if r["status"] != "fail" and r["split"] is not None
    }
    in_manifest: set[tuple[str, str, int]] = set()
    for split, items in manifest.get("splits", {}).items():
        for item in items:
            in_manifest.add((split, item["pair_slug"], int(item["seed"])))

    only_disk = on_disk - in_manifest
    only_manifest = in_manifest - on_disk
    if only_disk:
        sample = sorted(only_disk)[:3]
        notes.append(
            f"on disk but not in manifest: {len(only_disk)} cells (e.g. {sample})"
        )
    if only_manifest:
        sample = sorted(only_manifest)[:3]
        notes.append(
            f"in manifest but not on disk: {len(only_manifest)} cells (e.g. {sample})"
        )
    return notes


# ---------------------------------------------------------------------------
# Discovery + main
# ---------------------------------------------------------------------------


def discover_cells(cache_root: Path) -> list[Path]:
    cells: list[Path] = []
    for split_dir in sorted(cache_root.iterdir()):
        if not split_dir.is_dir() or split_dir.name not in {"train", "heldout"}:
            continue
        for pair_dir in sorted(split_dir.iterdir()):
            if not pair_dir.is_dir():
                continue
            for seed_dir in sorted(pair_dir.iterdir()):
                if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
                    cells.append(seed_dir)
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    ap.add_argument("--full", action="store_true",
                    help="Spot-check every step file (slower).")
    ap.add_argument("--json", action="store_true",
                    help="Emit a machine-readable JSON summary on stdout.")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-cell OK lines.")
    args = ap.parse_args()

    if not args.cache_root.exists():
        print(f"ERROR: cache root does not exist: {args.cache_root}")
        return 1

    cells = discover_cells(args.cache_root)
    if not cells:
        print(f"No cells found under {args.cache_root}/{{train,heldout}}/")
        return 1

    print(f"Verifying {len(cells)} cells under {args.cache_root.relative_to(REPO_ROOT)}")
    if args.full:
        print("(--full: checking every step file)")
    print()

    results: list[dict] = []
    fail_count = warn_count = ok_count = 0
    for cell_dir in cells:
        r = verify_cell(cell_dir, full=args.full)
        results.append(r)
        if r["status"] == "fail":
            fail_count += 1
            label = "[FAIL]"
        elif r["status"] == "warn":
            warn_count += 1
            label = "[WARN]"
        else:
            ok_count += 1
            label = "[ ok ]"
        if r["status"] == "ok" and args.quiet:
            continue
        print(f"{label} {r['path']}")
        for issue in r["issues"]:
            print(f"        ! {issue}")
        for w in r["warnings"]:
            print(f"        ~ {w}")

    print()
    print("--- Summary ---")
    print(f"OK:   {ok_count}")
    print(f"WARN: {warn_count}")
    print(f"FAIL: {fail_count}")

    cross_notes = check_cross_cell(results)
    manifest_notes = check_manifest(args.cache_root, results)
    if cross_notes or manifest_notes:
        print()
        print("--- Cross-cell / manifest notes ---")
        for n in cross_notes + manifest_notes:
            print(f"  {n}")

    if args.json:
        print()
        print(json.dumps({
            "cells": results,
            "summary": {"ok": ok_count, "warn": warn_count, "fail": fail_count},
            "cross_notes": cross_notes,
            "manifest_notes": manifest_notes,
        }, indent=2))

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
