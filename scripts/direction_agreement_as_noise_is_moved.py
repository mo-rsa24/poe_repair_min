#!/usr/bin/env python
"""The correction is smooth in the state early, chaotic late.

D2 shows two runs from completely different starting noise sharing no
correction at all. That is one point on a scale. This walks the scale: start
from seed 9's noise, move a fraction of the way toward seed 13's, and measure
how much of the correction still agrees with seed 9's run.

Two curves, because the answer splits by when in the run you look:

    first 3 steps    decays gently and levels off around +0.35
    steps 10 to 49   gone by the time the noise has moved a tenth of the way

Read with F4, which measured that the correction only works when it arrives in
the first ten steps, this says the part that matters is the part that behaves
like a learnable function of the state, and the part that is chaotic is the
part that does not matter.

The leftmost point is not 1.0 and that is a measurement, not a fault. Rerunning
the identical starting noise still diverges from the cached run: fp16
nondeterminism compounds through the sampler, drifting the trajectory 0.24% by
step 10, 0.93% by step 20 and 17.1% at step 49. So the leftmost point is the
ceiling this comparison can reach, and every other point is read against it.

Numbers come from `scripts/rt_noise_interpolation.py`, which walks each
interpolated start with run_teacher_residual at lambda 0 and saves the
per-step correction. This script only draws them.

    python scripts/direction_agreement_as_noise_is_moved.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from poe_repair import paths

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SRC = paths.resolve(paths.DIRECTION_WALL) / "noise_interpolation.json"
OUT_DIR = Path("paper/iclr/figures")
FIG_NAME = "direction-agreement-as-the-starting-noise-is-moved"
EARLY, LATE = "#1f77b4", "#d62728"
# Measured by rerunning seed 9 from its own starting noise and comparing the
# trajectory with the cached one, step by step. Quoted in the caption so the
# leftmost point is never read as a failed reproduction.
DRIFT = {10: 0.0024, 20: 0.0093, 49: 0.1713}


def main() -> int:
    if not SRC.exists():
        raise SystemExit(
            f"no measurements at {SRC}. Produce them first:\n"
            "  CUDA_VISIBLE_DEVICES=0 python scripts/rt_noise_interpolation.py")
    d = json.loads(SRC.read_text())
    rows = d["rows"]
    x = np.array([r["fraction"] for r in rows])
    early = np.array([r["r_t_agreement_first3"] for r in rows])
    late = np.array([r["r_t_agreement_late"] for r in rows])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    })
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.axhline(0.0, color="0.8", lw=0.6, zorder=1)
    ax.plot(x, early, "o-", color=EARLY, lw=1.6, ms=4.5, zorder=3,
            label="first 3 steps")
    ax.plot(x, late, "s--", color=LATE, lw=1.6, ms=4, zorder=3,
            label="steps 10 to 49")

    ax.annotate("same starting noise", (x[0], early[0]), xytext=(0.07, 0.84),
                textcoords="data", fontsize=7, color="0.35",
                arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6))
    ax.annotate("the part every run shares", (0.85, early[-1]),
                xytext=(0.52, 0.19), textcoords="data", fontsize=7,
                color=EARLY,
                arrowprops=dict(arrowstyle="-", color=EARLY, lw=0.6))

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.15, 1.05)
    ax.set_xlabel("how far the starting noise was moved toward another run")
    ax.set_ylabel("correction agreement (cosine)")
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon=False, loc="upper right", handlelength=1.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{FIG_NAME}.{ext}", dpi=300)
    plt.close(fig)

    (OUT_DIR / f"{FIG_NAME}.json").write_text(json.dumps({
        "source": str(SRC), "pair": d["pair"],
        "from_seed": d["from_seed"], "to_seed": d["to_seed"],
        "curves": {"first3": early.tolist(), "late": late.tolist(),
                   "fraction": x.tolist()},
        "ceiling_note": "the leftmost point reruns the identical starting "
                        "noise and still reaches only +0.996 early and +0.427 "
                        "late, because fp16 nondeterminism compounds through "
                        "the sampler: measured trajectory drift "
                        + ", ".join(f"{v:.2%} at step {k}"
                                    for k, v in DRIFT.items()),
        "caption_cap": "the late curve's collapse mixes two things that this "
                       "figure cannot separate: the correction changing with "
                       "the state, and the two trajectories being at "
                       "genuinely different states by then. The early curve "
                       "carries the claim.",
    }, indent=2))
    print(f"wrote {OUT_DIR / FIG_NAME}.png and .pdf")
    print(f"early: {np.round(early, 3)}")
    print(f"late:  {np.round(late, 3)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
