#!/usr/bin/env python
"""F8b: what the learned adapter costs against the correction it imitates.

The paper's other figures measure r_t, the correction computed from the joint
prompt. That is an oracle: obtaining it needs the very prompt the method exists
to avoid. This puts the oracle and the shippable adapter on one axis, over the
same eight pairs, so the reader can see the price of not having the answer.

Four points per pair:

    PoE               nothing injected, the failure the paper starts from
    oracle at 0.75    the correction at its largest informative dose
    oracle at 1.0     drawn hollow, because at full dose the corrected
                      prediction reproduces the joint prediction exactly (1.9
                      grey levels of 255). It is the joint render, so it is the
                      target rather than a method, and it is on the figure only
                      to mark where the ceiling sits
    adapter           the pooled rank-8 LoRA at training step 60000, which
                      needs no joint prompt at test time

The comparison is pair-matched, not cell-matched, and the figure says so. The
oracle arm ran seeds 9 to 12 (4 cells per pair) and the adapter arm seeds 9 to
16 (16 cells per pair). Both are held-out seeds of the same eight pairs, so
the pairs are the sampling unit here; a cell-matched version needs the adapter
rescored per seed and is not what this shows.

Reads two already-scored files, samples nothing.

    python scripts/make_f8b.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DOSE = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose/"
            "dose_curves.json")
LORA = Path("outputs/animals_compose_transfer/pooled_lora/phase1_r8_100k/"
            "compose_rate.json")
OUT_DIR = Path("paper/iclr/figures")
FIG_NAME = "F8b-adapter-against-the-oracle"
STEP = "60000"

POE_C, O75_C, O1_C, AD_C = "0.45", "#ff7f0e", "#ff7f0e", "#1f77b4"
CONTROL_PAIR = "an_elephant__x__a_penguin"


def pretty(slug: str) -> str:
    def strip(side: str) -> str:
        side = side.replace("_", " ")
        for art in ("an ", "a "):
            if side.startswith(art):
                return side[len(art):]
        return side
    return " x ".join(strip(s) for s in slug.split("__x__"))


def main() -> int:
    for f in (DOSE, LORA):
        if not f.exists():
            raise SystemExit(f"missing {f}")
    dose = json.loads(DOSE.read_text())
    lora = json.loads(LORA.read_text())["per_step_heldout_pair"]

    by = collections.defaultdict(list)
    for s in dose["scores"]:
        if s["row"] == "oracle":
            by[(s["pair"], float(s["lambda"]))].append(s["compose"])

    pairs = [p for p in sorted(lora) if (p, 0.75) in by]
    rows = []
    for p in pairs:
        rows.append({
            "pair": p,
            "poe": float(np.mean(by[(p, 0.0)])),
            "oracle_075": float(np.mean(by[(p, 0.75)])),
            "oracle_1": float(np.mean(by[(p, 1.0)])),
            "adapter": float(lora[p][STEP]),
        })
    rows.sort(key=lambda r: r["adapter"] - r["oracle_075"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 6.8})
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    y = np.arange(len(rows))[::-1]

    for yi, r in zip(y, rows):
        ax.plot([r["poe"], max(r["oracle_075"], r["adapter"])], [yi, yi],
                color="0.88", lw=3, zorder=1, solid_capstyle="round")
    ax.plot([r["poe"] for r in rows], y, "o", color=POE_C, ms=4.5, zorder=3,
            label="PoE, nothing injected")
    ax.plot([r["oracle_1"] for r in rows], y, "o", mfc="none", mec=O1_C,
            mew=1.1, ms=7, zorder=3,
            label="oracle at $\\lambda$=1 (the joint render itself)")
    ax.plot([r["oracle_075"] for r in rows], y, "^", color=O75_C, ms=5,
            zorder=4, label="oracle correction at $\\lambda$=0.75")
    ax.plot([r["adapter"] for r in rows], y, "s", color=AD_C, ms=5, zorder=5,
            label="adapter, no joint prompt at test time")

    labels = [pretty(r["pair"]) + ("  (control)" if r["pair"] == CONTROL_PAIR
                                   else "") for r in rows]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(-0.04, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("compose rate")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    # Above the axes: every row runs the full width, so any in-axes legend
    # would sit on top of the data it explains.
    leg = ax.legend(frameon=False, ncol=2, handlelength=1.0, labelspacing=0.3,
                    columnspacing=0.8, loc="lower left", fontsize=6.4,
                    bbox_to_anchor=(-0.02, 1.02), borderaxespad=0.0)
    ax.text(0.5, -0.30,
            "oracle: 4 seeds per pair.  adapter: 16 cells per pair.  "
            "pair-matched, not cell-matched.",
            transform=ax.transAxes, fontsize=6, color="0.45", ha="center")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{FIG_NAME}.{ext}", dpi=300)
    plt.close(fig)

    beats = sum(1 for r in rows if r["adapter"] >= r["oracle_075"])
    (OUT_DIR / f"{FIG_NAME}.json").write_text(json.dumps({
        "sources": {"oracle": str(DOSE), "adapter": str(LORA)},
        "adapter_checkpoint_step": int(STEP), "rows": rows,
        "adapter_at_or_above_oracle_075": f"{beats} of {len(rows)} pairs",
        "caption_caps": [
            "pair-matched, not cell-matched: the oracle arm ran seeds 9-12 "
            "(4 cells per pair) and the adapter arm seeds 9-16 (16 cells per "
            "pair). The pair is the sampling unit; a cell-matched version "
            "needs the adapter rescored per seed",
            "oracle at lambda=1 reproduces the joint prediction exactly (1.9 "
            "grey levels of 255), so it is the target rather than a method "
            "and is drawn hollow",
            "the adapter is at training step 60000; steps 70000 to 100000 "
            "were never scored",
            "elephant x penguin composes without any intervention, so its row "
            "is a control and not evidence for either arm",
        ],
    }, indent=2))
    print(f"wrote {OUT_DIR / FIG_NAME}.png and .pdf")
    print(f"  adapter at or above the oracle's 0.75 dose: {beats} of {len(rows)} pairs")
    for r in rows:
        print(f"  {pretty(r['pair']):24} PoE {r['poe']:.2f}  "
              f"oracle.75 {r['oracle_075']:.2f}  oracle1 {r['oracle_1']:.2f}  "
              f"adapter {r['adapter']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
