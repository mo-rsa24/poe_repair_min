"""Phase 1 — qualitative grid: pairs × {Solo A, Solo B, PoE, Mono}.

Single figure showing every (pair, seed) cell as a row, with columns for
each subject's solo rendering, vanilla PoE, and the mono oracle. Establishes
the failure-mode contrast: PoE on the collision pair vs. mono on the same
seed, plus the cooperative pair as control.

Run:
    python -m poe_repair.phases.phase1
    python -m poe_repair.phases.phase1 --pair-filter a_cat__x__a_dog
"""

from __future__ import annotations

import argparse
from pathlib import Path

from poe_repair.config import RunConfig
from poe_repair.figures._common import image_grid
from poe_repair.run import MethodCtx, make_ctx, run_method
from poe_repair.runtime import discover_pairs


COLUMNS: list[tuple[str, str]] = [
    ("Solo A", "solo_a"),
    ("Solo B", "solo_b"),
    ("PoE", "poe"),
    ("Mono", "mono"),
]


def phase1(
    *,
    pair_filter: list[str] | None = None,
    seed_filter: list[int] | None = None,
    ctx: MethodCtx | None = None,
) -> Path:
    cfg = RunConfig()
    ctx = ctx or make_ctx()
    cells = discover_pairs(
        cfg.paths.pilot_dir, pair_filter=pair_filter, seed_filter=seed_filter
    )
    if not cells:
        raise RuntimeError(f"No pair-seed cells found at {cfg.paths.pilot_dir}.")

    rows: list[list[Path]] = [
        [run_method(method, cell, ctx) for _, method in COLUMNS]
        for cell in cells
    ]

    out = cfg.paths.output_root / "figures" / "phase1__qualitative_grid.png"
    written = image_grid(
        rows,
        out,
        col_labels=[label for label, _ in COLUMNS],
        row_labels=[
            f"{cell.regime}\n{cell.pair_slug}\nseed {cell.seed}" for cell in cells
        ],
        title="Phase 1 — qualitative comparison",
    )
    print(f"[phase1] wrote {written}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 1 — qualitative grid (Solo A | Solo B | PoE | Mono)."
    )
    ap.add_argument("--pair-filter", nargs="*", default=None)
    ap.add_argument("--seed-filter", nargs="*", type=int, default=None)
    args = ap.parse_args()
    phase1(pair_filter=args.pair_filter, seed_filter=args.seed_filter)


if __name__ == "__main__":
    main()
