#!/usr/bin/env python
"""One trained adapter composes on pairs it never saw.

Everything else in the figure set measures the oracle correction, which is
computed from the joint prompt and therefore cannot be shipped: needing the
joint prompt is the problem the paper is trying to solve. This is the figure
of the thing that can be shipped. A single rank-8 adapter, pooled over 11
training pairs, evaluated on 8 pairs it never trained on.

Two panels sharing the training-step axis.

    (a) compose rate for the pairs the adapter trained on and for the pairs it
        did not, with the uncorrected PoE floor drawn as the reference. The
        gap between the two curves is what "transfer" means here, and it is
        small.
    (b) the same held-out result unpooled, one curve per pair, so a reader can
        see that the aggregate is not carried by one easy pair.

Read the caveats in the sidecar before quoting anything. The training ran to
step 100000 and only checkpoints to 60000 were scored, so every number here
carries its step and none of them is the final word.

Reads the scored run only; it does not sample or train.

    python scripts/adapter_transfers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RUN = Path("outputs/animals_compose_transfer/pooled_lora/phase1_r8_100k")
OUT_DIR = Path("paper/iclr/figures")
FIG_NAME = "compose-rate-as-the-lora-trains"

# Measured in the dose sweep over 32 cells with nothing injected. Its single
# positive is a detector mistake (frog x toad seed 10: three boxes on one fused
# body), and the scorer's rule is not changed to remove it, so the floor is
# drawn as measured and the caption carries the note.
POE_FLOOR = 0.03
# The pair that composes without any adapter, so its perfect score is not
# evidence of transfer and the figure says so on the line itself.
CONTROL_PAIR = "an_elephant__x__a_penguin"
RUNNING = "a_cat__x__a_dog"


def pretty(slug: str) -> str:
    def strip(side: str) -> str:
        side = side.replace("_", " ")
        for art in ("an ", "a "):
            if side.startswith(art):
                return side[len(art):]
        return side
    return " x ".join(strip(s) for s in slug.split("__x__"))


def main() -> int:
    src = RUN / "compose_rate.json"
    if not src.exists():
        raise SystemExit(f"no scored run at {src}")
    d = json.loads(src.read_text())
    pool = json.loads((RUN / "pair_pool.json").read_text())
    quad = d["per_step_quadrant"]
    per_pair = d["per_step_heldout_pair"]

    steps = sorted(int(s) for s in quad)
    in_dist = [quad[str(s)]["in_in"]["compose_rate"] for s in steps]
    held = [quad[str(s)]["out_out"]["compose_rate"] for s in steps]
    n_in = quad[str(steps[-1])]["in_in"]["n"]
    n_out = quad[str(steps[-1])]["out_out"]["n"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7,
    })
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4))
    ks = np.array(steps) / 1000.0

    ax = axes[0]
    ax.axhline(POE_FLOOR, color="0.55", lw=1.0, ls=":", zorder=1)
    ax.text(ks[0], POE_FLOOR + 0.04, "PoE, no adapter", ha="left",
            fontsize=6.5, color="0.45")
    ax.plot(ks, in_dist, "o-", color="0.35", lw=1.5, ms=4,
            label=f"pairs it trained on (n={n_in})")
    ax.plot(ks, held, "s-", color="#1f77b4", lw=1.7, ms=4.5,
            label=f"pairs it never saw (n={n_out})")
    ax.set_ylabel("compose rate")
    ax.set_title("a. trained on, and never seen", fontsize=8, loc="left")
    ax.legend(frameon=False, loc="center right", handlelength=1.6,
              borderaxespad=0.4)

    ax = axes[1]
    for slug, curve in per_pair.items():
        y = [curve[str(s)] for s in steps]
        if slug == CONTROL_PAIR:
            colour, lw, z = "#d62728", 1.4, 4
        elif slug == RUNNING:
            colour, lw, z = "#1f77b4", 1.4, 4
        else:
            colour, lw, z = "0.65", 0.9, 2
        ax.plot(ks, y, "-", color=colour, lw=lw, zorder=z)
    # Labels sit in the empty lower half rather than beside their lines: the
    # curves crowd the top of the panel and colour alone identifies them.
    ax.text(ks[0], 0.40, "elephant x penguin, which composes\nwithout any "
            "adapter at all", ha="left", va="center", fontsize=6.5,
            color="#d62728")
    ax.text(ks[0], 0.22, "cat x dog", ha="left", va="center", fontsize=6.5,
            color="#1f77b4")
    ax.text(ks[0], 0.10, "the other six held-out pairs", ha="left",
            va="center", fontsize=6.5, color="0.55")
    ax.set_title("b. each held-out pair on its own", fontsize=8, loc="left")

    for ax in axes:
        ax.set_xlim(ks[0] - 3, ks[-1] + 3)
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0%", "50%", "100%"])
        ax.set_xlabel("training step (thousands)")
        ax.grid(alpha=0.22, linewidth=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(w_pad=1.6)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{FIG_NAME}.{ext}", dpi=300)
    plt.close(fig)

    (OUT_DIR / f"{FIG_NAME}.json").write_text(json.dumps({
        "run": str(RUN), "scored_steps": steps,
        "train_pairs": pool["train"], "heldout_pairs": pool["heldout"],
        "in_distribution": dict(zip(map(str, steps), in_dist)),
        "held_out": dict(zip(map(str, steps), held)),
        "per_heldout_pair": per_pair,
        "n_cells": {"in_distribution": n_in, "held_out": n_out},
        "poe_floor": POE_FLOOR,
        "caption_caps": [
            "training ran to step 100000 and only checkpoints to 60000 were "
            "scored, so every number carries its step and 60000 is the "
            "best-read rather than the final word",
            "elephant x penguin is the control pair: it composes without any "
            "adapter, so its 1.0 is not evidence of transfer and the figure "
            "labels it on the line",
            "the PoE floor is the dose sweep's lambda=0 rate over 32 cells. "
            "Its single positive is a detector mistake on frog x toad seed 10 "
            "and the scorer's rule was not changed to remove it",
            "held-out here means pairs the adapter never trained on, not a "
            "leave-one-pair-out sweep. The 15-run version is a separate "
            "experiment and is not what this figure shows",
        ],
    }, indent=2))
    print(f"wrote {OUT_DIR / FIG_NAME}.png and .pdf")
    print(f"  step 60000: trained-on {in_dist[-1]:.3f} (n={n_in}), "
          f"never-seen {held[-1]:.3f} (n={n_out}), PoE floor {POE_FLOOR:.2f}")
    worst = min(per_pair, key=lambda k: per_pair[k][str(steps[-1])])
    print(f"  weakest held-out pair at 60000: {pretty(worst)} "
          f"{per_pair[worst][str(steps[-1])]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
