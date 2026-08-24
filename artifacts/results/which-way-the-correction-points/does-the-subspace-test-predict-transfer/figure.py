#!/usr/bin/env python
"""Figure: the geometry says one thing, the trained adapter says another."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OUT = Path("docs/evidence/subspace-vs-transfer")
R = json.loads((OUT / "result.json").read_text())
TRANSFER = [r for r in R["per_pair"]
            if r["pair"] not in ("a_cat__x__a_dog", "an_elephant__x__a_penguin")]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# Left: the two readings, per pair
names = [r["pair"].replace("__x__", " x ").replace("a_", "").replace("an_", "")
         for r in TRANSFER]
y = np.arange(len(TRANSFER))
h = 0.38
ax1.barh(y + h / 2, [r["compose_rate"] for r in TRANSFER], h,
         color="tab:green", label="adapter composes")
ax1.barh(y - h / 2, [r["geometry_k64"] for r in TRANSFER], h,
         color="tab:red", label="correction inside training subspace")
ax1.set_yticks(y)
ax1.set_yticklabels(names, fontsize=8)
ax1.set_xlim(0, 1.05)
ax1.set_xlabel("fraction")
ax1.set_title("Six pairs the adapter never saw")
ax1.set_ylim(-0.6, len(TRANSFER) - 0.4)
ax1.legend(frameon=False, fontsize=8, loc="upper center",
           bbox_to_anchor=(0.5, -0.16), ncol=2)
ax1.grid(axis="x", alpha=0.3)

# Right: energy-at-k, train vs unseen, the geometric claim on its own
geo = R["geometry_at_k"]
ks = sorted(int(k) for k in geo)
ax2.plot(ks, [geo[str(k)]["train"] for k in ks], "o-", color="tab:blue",
         lw=2, label="training pairs")
ax2.plot(ks, [geo[str(k)]["heldout"] for k in ks], "^-", color="tab:red",
         lw=2, label="unseen pairs")
ax2.axhline(R["mean_compose_rate_transfer"], color="tab:green", ls="--", lw=1.6,
            label=f"adapter compose rate ({R['mean_compose_rate_transfer']:.0%})")
ax2.set_xscale("log", base=2)
ax2.set_xlabel("k (directions kept)")
ax2.set_ylabel("fraction of energy captured")
ax2.set_ylim(0, 1.05)
ax2.set_title("The subspace test misses what the adapter uses")
ax2.legend(frameon=False, fontsize=8, loc="center right")
ax2.grid(alpha=0.3)

fig.suptitle("Geometry says not shared, the adapter transfers anyway", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "geometry_vs_transfer.png", dpi=150)
print(f"figure: {OUT / 'geometry_vs_transfer.png'}")
