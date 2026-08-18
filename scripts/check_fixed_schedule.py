#!/usr/bin/env python
"""Does driving lambda through `adaptive_schedule` reproduce the window sweep?

The dose-matched experiments set lambda per step through the sampler's
`adaptive_schedule` hook instead of through `correction_window`. That is a
different code path, and nothing has ever checked the two agree.

So: run one cell with a FixedSchedule that returns exactly what
`correction_window=(0,10)` at full strength would have produced, and compare the
result against the image the window sweep already wrote for that same cell.

If they match, the hook is a faithful way to express a window and the 16-cell
experiment can be trusted. If they do not, every dose-matched cell would differ
from the sweep for reasons having nothing to do with dose, and the comparison
between the two experiments would be meaningless.

This is the canary the dose-matched runs sit behind. Run it before them.

    python scripts/check_fixed_schedule.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from poe_repair.composers import teacher_residual as cmp_tr  # noqa: E402
from poe_repair.experiments.interaction_term.cell import cell_from_slug  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from interaction_term_dose_matched import FixedSchedule, STEPS  # noqa: E402

WINDOW_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/window")
PAIR = "a_cat__x__a_dog"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--window", default="0,10")
    ap.add_argument("--max-mean-abs", type=float, default=1.0,
                    help="grey levels of 255. The two paths batch identically, so "
                         "the honest expectation is 0; a small nonzero would be "
                         "nondeterminism, and anything larger is a real difference.")
    args = ap.parse_args()

    w0, w1 = (int(x) for x in args.window.split(","))
    reference = WINDOW_ROOT / "pairs" / PAIR / f"seed_{args.seed}" / \
        f"teacher_residual_const_lam100_w{w0}-{w1}"
    hits = sorted(reference.glob("*.png"))
    if not hits:
        raise SystemExit(f"no reference image under {reference}. This check needs "
                         f"the window sweep's own output for the same cell.")
    ref_png = hits[0]

    # Exactly what correction_window=(w0,w1) at lambda_max=1 produces: full dose
    # inside, nothing outside. Expressed as a per-step list instead of as bounds.
    lam = np.zeros(STEPS, dtype=float)
    lam[w0:w1] = 1.0
    print(f"reference   {ref_png}")
    print(f"schedule    lambda 1.0 on steps {w0} to {w1 - 1}, 0 elsewhere")

    cell = cell_from_slug(PAIR, args.seed)
    ctx = make_ctx(num_inference_steps=STEPS)
    got = cmp_tr.run(
        cell, ctx,
        lambda_max=1.0,
        adaptive_schedule=FixedSchedule(lam),
        method_name_override=f"check_fixed_schedule_w{w0}-{w1}",
        exp_name="interaction_term/canary",
        overwrite=True,
    )
    print(f"produced    {got}")

    a = np.asarray(Image.open(ref_png).convert("RGB"), dtype=np.float64)
    b = np.asarray(Image.open(got).convert("RGB"), dtype=np.float64)
    if a.shape != b.shape:
        print(f"FAILED: shapes differ, {a.shape} against {b.shape}", file=sys.stderr)
        return 1
    diff = np.abs(a - b)
    mean_abs, max_abs = float(diff.mean()), float(diff.max())
    print(f"difference  mean {mean_abs:.4f}, max {max_abs:.1f} grey levels of 255")

    if mean_abs > args.max_mean_abs:
        print(
            f"FAILED: driving lambda through adaptive_schedule does not reproduce "
            f"correction_window for the same cell. The dose-matched experiments "
            f"would differ from the window sweep for reasons unrelated to dose, "
            f"so their comparison would mean nothing. Diagnose before running them.",
            file=sys.stderr,
        )
        return 1
    print("PASSED: the schedule hook expresses a window faithfully. The "
          "dose-matched cells are comparable with the window sweep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
