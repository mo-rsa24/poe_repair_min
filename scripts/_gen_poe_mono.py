"""Inner driver for gen_extra_seeds.sh.

Runs only PoE + Mono (literal e_J) on the given (pairs, seeds), writing to
``outputs/e1_held_out/pairs/<slug>/seed_<n>/{poe,mono_literal}/`` so
``scripts/build_dataset.py`` picks them up automatically.

This is the economical alternative to running the full e1_held_out grid
(which also runs sched-M2 × {literal, synth} and CO3) — about a third of
the cost per cell.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poe_repair.composers import mono as cmp_mono  # noqa: E402
from poe_repair.composers import poe as cmp_poe  # noqa: E402
from poe_repair.experiments._eval_common import cell_for  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pairs", nargs="+", required=True,
        help='Pair specs like "a cat|a dog".',
    )
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--exp-name", default="e1_held_out")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    pairs = []
    for spec in args.pairs:
        a, _, b = spec.partition("|")
        if not a or not b:
            raise ValueError(f"bad --pairs entry {spec!r}")
        pairs.append((a, b))

    ctx = make_ctx()
    for prompt_a, prompt_b in pairs:
        for seed in args.seeds:
            cell = cell_for(prompt_a, prompt_b, seed)
            print(f"[gen] {cell.pair_slug} seed={seed}")
            cmp_poe.run(
                cell, ctx, exp_name=args.exp_name, overwrite=args.overwrite,
            )
            cmp_mono.run(
                cell, ctx, anchor_source="literal",
                exp_name=args.exp_name, overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
