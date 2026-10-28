import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INDIGO = "#4527a0"
INK = "#212121"

repo = Path(__file__).resolve().parents[4]
with (repo / "rebuild/out/walk-numbers.json").open() as f:
    numbers = json.load(f)

values = np.array([
    numbers["p2.clean_product_mean"],
    numbers["p2.noised_product_mean"],
    numbers["p2.product_of_noised_mean"],
])
labels = ["clean product", "product,\nthen noise", "noise,\nthen product"]

fig, ax = plt.subplots(figsize=(7.4, 4.2))
bars = ax.bar(
    np.arange(3),
    values,
    color=[INDIGO, INDIGO, "white"],
    edgecolor=[INDIGO, INDIGO, INDIGO],
    width=0.58,
)

for bar, a, lw, h in zip(bars, [0.45, 0.75, 1.0], [1.5, 1.5, 2.0], [None, None, "//"]):
    bar.set_alpha(a)
    bar.set_linewidth(lw)
    if h:
        bar.set_hatch(h)

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.035,
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=11,
        weight="bold",
        color=INK,
    )

ax.text(
    2,
    values[2] + 0.12,
    "summed noised scores\npoint here",
    ha="center",
    va="bottom",
    color=INK,
    fontsize=9,
    weight="bold",
)
ax.set_xticks(np.arange(3), labels)
ax.set_ylabel("mean position")
ax.set_ylim(0, 1.12)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Multiplying and noising do not commute")
fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=160)
