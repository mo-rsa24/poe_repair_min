"""Per-quadrant contact sheets for the cross-pair × cross-seed LoRA run.

Renders one PNG per quadrant (``in_in``, ``in_out``, ``out_in``,
``out_out``). Each grid:

    rows    = pairs in that quadrant
              (train pairs first, then heldout pairs)
    columns = [PoE, pooled-LoRA (this run), per-group LoRA (Plan 10),
               per-pair LoRA (Plan 09), mono]

Optional reference columns (Plans 09 / 10) are dropped from the grid if
no matching PNG is found anywhere — sparser sheet, less placeholder
noise. PoE/mono are read directly from the training-cache cell
directory (``seed_N/poe.png`` and ``seed_N/mono.png``).

Also writes ``quadrant_table.csv`` with the n_cells skeleton; the
``n_recognisable`` column is left blank for human eyeball.

Usage::

    python -m poe_repair.experiments.cross_pair_lora_pooling.contact_sheet \\
        --pooled-run artifacts/rung4-scale/cross_pair/all_groups/main \\
        --pair-pool artifacts/_shared/cross_pair_pool_configs/pair_pool.yaml \\
        --seed-pool-path artifacts/_shared/cross_pair_pool_configs/seed_pool.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from poe_repair.experiments.cross_pair_lora_pooling.pair_pool import (
    load_pair_pool,
)
from poe_repair.experiments.cross_pair_lora_pooling.seed_pool import (
    load_seed_pool,
)
from poe_repair.training_cache import DEFAULT_CACHE_ROOT, CellPath
from poe_repair import paths


log = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]

QUADRANTS = ("in_in", "in_out", "out_in", "out_out")
COLUMNS_ALL = ("poe", "pooled_lora", "plan10_lora", "plan09_lora", "mono")
COLUMN_LABEL = {
    "poe": "PoE",
    "pooled_lora": "pooled-LoRA",
    "plan10_lora": "Plan 10 (per-group)",
    "plan09_lora": "Plan 09 (per-pair)",
    "mono": "mono (ceiling)",
}


def _ref_paths(pair: str, seed: int, cache_root: Path) -> dict[str, Path | None]:
    """Locate PoE / mono PNGs for one (pair, seed) cell from the cache."""
    out: dict[str, Path | None] = {"poe": None, "mono": None}
    try:
        cell = CellPath.from_root(pair, int(seed), cache_root=cache_root)
    except FileNotFoundError:
        return out
    for key in ("poe", "mono"):
        cand = cell.root / f"{key}.png"
        if cand.exists():
            out[key] = cand
    return out


def _plan10_path(pair: str, seed: int) -> Path | None:
    """Best-effort discovery of Plan 10's per-group LoRA PNG for this cell."""
    base = paths.resolve(paths.HELD_OUT_SEEDS_INDEX) / pair
    if not base.exists():
        return None
    # Plan 10 typically names samples sample_seed_NN.png under a heldout-pair
    # subtree. Search a couple of well-known patterns; first hit wins.
    patterns = [
        f"**/heldout*/seed_{seed:02d}/lora.png",
        f"**/heldout*/sample_seed_{seed:02d}.png",
        f"**/samples/**/sample_seed_{seed:02d}.png",
    ]
    for pat in patterns:
        hits = sorted(base.glob(pat))
        if hits:
            return hits[0]
    return None


def _plan09_path(pair: str, seed: int) -> Path | None:
    """Best-effort discovery of Plan 09's per-pair LoRA PNG for this cell."""
    base = paths.resolve(paths.RESULTS_BY_PAIR)
    candidates = [
        base / pair / f"seed_{seed}" / "lora.png",
        base / pair / f"seed_{seed:02d}" / "lora.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: glob.
    if not base.exists():
        return None
    hits = sorted(base.glob(f"{pair}/**/lora.png"))
    return hits[0] if hits else None


def _load_cells(samples_root: Path) -> list[dict]:
    f = samples_root / "cells.jsonl"
    if not f.exists():
        raise FileNotFoundError(
            f"{f} not found — run sample_crossbar first"
        )
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


def _thumb(path: Path | None, edge: int, missing_label: str) -> Image.Image:
    if path is None or not Path(path).exists():
        img = Image.new("RGB", (edge, edge), (32, 32, 32))
        d = ImageDraw.Draw(img)
        d.text((6, edge // 2 - 8), missing_label, fill=(160, 160, 160))
        return img
    im = Image.open(path).convert("RGB")
    im.thumbnail((edge, edge))
    canvas = Image.new("RGB", (edge, edge), (0, 0, 0))
    ox = (edge - im.width) // 2
    oy = (edge - im.height) // 2
    canvas.paste(im, (ox, oy))
    return canvas


def _draw_label(img: Image.Image, text: str, *, fill=(255, 255, 255)) -> None:
    d = ImageDraw.Draw(img)
    d.text((6, 6), text, fill=fill)


def _build_grid(
    *,
    cells_for_quadrant: list[dict],
    pair_pool,
    cache_root: Path,
    edge: int,
    quadrant: str,
) -> tuple[Image.Image, list[str]]:
    # Determine row order: train pairs first, then heldout pairs.
    pair_order = list(pair_pool.train) + list(pair_pool.heldout)
    pairs_present = [p for p in pair_order
                     if any(c["pair_slug"] == p for c in cells_for_quadrant)]
    # Within a pair, sort by seed ascending.
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for c in cells_for_quadrant:
        by_pair[c["pair_slug"]].append(c)
    for p in by_pair:
        by_pair[p].sort(key=lambda c: int(c["seed"]))

    # Decide which optional columns to include (presence-based).
    has_plan10 = any(
        _plan10_path(c["pair_slug"], int(c["seed"])) is not None
        for c in cells_for_quadrant
    )
    has_plan09 = any(
        _plan09_path(c["pair_slug"], int(c["seed"])) is not None
        for c in cells_for_quadrant
    )
    columns: list[str] = ["poe", "pooled_lora"]
    if has_plan10:
        columns.append("plan10_lora")
    if has_plan09:
        columns.append("plan09_lora")
    columns.append("mono")
    log.info("quadrant=%s pairs=%d cols=%s (plan10=%s plan09=%s)",
             quadrant, len(pairs_present), columns, has_plan10, has_plan09)

    # Each row is a (pair, seed) cell. Build them all in order.
    rows: list[tuple[str, int, list[Path | None]]] = []
    for pair in pairs_present:
        for cell in by_pair[pair]:
            seed = int(cell["seed"])
            refs = _ref_paths(pair, seed, cache_root)
            row_paths: list[Path | None] = []
            for col in columns:
                if col == "poe":
                    row_paths.append(refs["poe"])
                elif col == "mono":
                    row_paths.append(refs["mono"])
                elif col == "pooled_lora":
                    row_paths.append(Path(cell["png_path"]))
                elif col == "plan10_lora":
                    row_paths.append(_plan10_path(pair, seed))
                elif col == "plan09_lora":
                    row_paths.append(_plan09_path(pair, seed))
            rows.append((pair, seed, row_paths))

    header_h = 28
    label_w = 220
    n_rows = len(rows)
    n_cols = len(columns)
    canvas_w = label_w + n_cols * edge
    canvas_h = header_h + n_rows * edge
    canvas = Image.new("RGB", (canvas_w, canvas_h), (16, 16, 16))
    d = ImageDraw.Draw(canvas)
    # Column headers.
    for ci, col in enumerate(columns):
        x = label_w + ci * edge + 6
        d.text((x, 6), COLUMN_LABEL[col], fill=(220, 220, 220))
    # Rows.
    for ri, (pair, seed, paths) in enumerate(rows):
        y = header_h + ri * edge
        d.text((6, y + edge // 2 - 8),
               f"{pair}\nseed {seed}",
               fill=(220, 220, 220))
        for ci, p in enumerate(paths):
            tile = _thumb(p, edge, missing_label=f"no {columns[ci]}")
            canvas.paste(tile, (label_w + ci * edge, y))

    return canvas, columns


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("CROSS_PAIR_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(prog="contact_sheet")
    ap.add_argument("--pooled-run", required=True,
                    help="run dir containing samples/cells.jsonl")
    ap.add_argument("--pair-pool", required=True)
    ap.add_argument("--seed-pool-path", required=True)
    ap.add_argument("--thumb-edge", type=int, default=320)
    ap.add_argument("--cache-root", default=None)
    args = ap.parse_args(argv)

    pair_pool = load_pair_pool(args.pair_pool)
    _ = load_seed_pool(args.seed_pool_path)  # validates only

    pooled_run = Path(args.pooled_run)
    samples_root = pooled_run / "samples"
    cells = _load_cells(samples_root)
    cache_root = Path(args.cache_root) if args.cache_root else DEFAULT_CACHE_ROOT

    by_q: dict[str, list[dict]] = defaultdict(list)
    for c in cells:
        by_q[c["quadrant"]].append(c)

    # CSV skeleton.
    csv_path = pooled_run / "quadrant_table.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quadrant", "n_cells", "n_recognisable", "pct", "notes"])
        for q in QUADRANTS:
            w.writerow([q, len(by_q.get(q, [])), "", "", ""])
    log.info("wrote %s", csv_path)

    # One PNG per quadrant that actually has cells.
    for q in QUADRANTS:
        cells_q = by_q.get(q, [])
        if not cells_q:
            log.info("quadrant=%s has no cells; skipping sheet", q)
            continue
        canvas, cols = _build_grid(
            cells_for_quadrant=cells_q,
            pair_pool=pair_pool,
            cache_root=cache_root,
            edge=int(args.thumb_edge),
            quadrant=q,
        )
        out_path = pooled_run / f"contact_sheet_{q}.png"
        canvas.save(out_path)
        log.info("wrote %s (%dx%d, cols=%s)",
                 out_path, canvas.width, canvas.height, cols)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
