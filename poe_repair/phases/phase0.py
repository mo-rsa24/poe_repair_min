"""Phase 0 — PoE + Mono baselines and a side-by-side figure per cell.

For each (pair, seed), runs PoE and Mono via the shared dispatcher and
writes a Mono | PoE strip under
``outputs/figures/phase0__<slug>__seed_<n>.png``.

Run:
    python -m poe_repair.phases.phase0
    python -m poe_repair.phases.phase0 --pair-filter a_cat__x__a_dog
"""

from __future__ import annotations

import argparse
from pathlib import Path

from poe_repair.config import RunConfig
from poe_repair.figures._common import image_grid
from poe_repair.run import MethodCtx, make_ctx, run_method
from poe_repair.runtime import discover_pairs


def phase0(
    *,
    pair_filter: list[str] | None = None,
    seed_filter: list[int] | None = None,
    ctx: MethodCtx | None = None,
) -> list[Path]:
    cfg = RunConfig()
    ctx = ctx or make_ctx()
    cells = discover_pairs(
        cfg.paths.pilot_dir, pair_filter=pair_filter, seed_filter=seed_filter
    )
    if not cells:
        raise RuntimeError(f"No pair-seed cells found at {cfg.paths.pilot_dir}.")

    figures: list[Path] = []
    for cell in cells:
        mono = run_method("mono", cell, ctx)
        poe = run_method("poe", cell, ctx)
        out = (
            cfg.paths.output_root
            / "figures"
            / f"phase0__{cell.pair_slug}__seed_{cell.seed}.png"
        )
        figures.append(
            image_grid(
                [[mono, poe]],
                out,
                col_labels=["Mono", "PoE"],
                title=f"{cell.pair_slug} — seed {cell.seed}",
            )
        )
        print(f"[phase0] wrote {figures[-1]}")
    return figures


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 0 — Mono vs PoE side-by-side.")
    ap.add_argument("--pair-filter", nargs="*", default=None)
    ap.add_argument("--seed-filter", nargs="*", type=int, default=None)
    args = ap.parse_args()
    phase0(pair_filter=args.pair_filter, seed_filter=args.seed_filter)


if __name__ == "__main__":
    main()
