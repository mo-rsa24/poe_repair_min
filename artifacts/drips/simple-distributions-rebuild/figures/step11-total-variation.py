import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ANNOTATION = "#212121"
DARK_GREY = "#455a64"
MID_GREY = "#78909c"
LIGHT_GREY = "#b0bec5"

root = Path(__file__).resolve().parents[4]
phase1 = json.loads((root / "rebuild/out/phase1/phase1.json").read_text())
rows = {row["panel"]: row for row in phase1["rows"]}
panels = [
    ("① bar(scale=0.2)", rows["bar(scale=0.2)"]),
    ("② toy_gmm(std=.03)", rows["toy_gmm(std=.03)"]),
]
labels = ["with last\nnoise", "without last\nnoise", "sampling\nbaseline"]
colors = [DARK_GREY, MID_GREY, LIGHT_GREY]

fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0), sharey=True)
for ax, (title, row) in zip(axes, panels):
    vals = [row["tv"], row["tv_last_step_mean"], row["floor"]]
    bars = ax.bar(labels, vals, color=colors, width=0.64, edgecolor="white")
    for index, (bar, value) in enumerate(zip(bars, vals)):
        if index == 0:
            y = value + 0.009
            colour = ANNOTATION
        else:
            y = value - 0.011
            colour = "white" if index == 1 else ANNOTATION
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                f"{value:.3f}", ha="center", fontsize=10,
                fontweight="bold", color=colour)
    ax.set_title(title)
    ax.set_ylim(0, 0.23)
    ax.spines[["top", "right"]].set_visible(False)
    ax.add_patch(Ellipse((1.5, min(vals[1], vals[2]) + 0.004), 1.9,
                         abs(vals[1] - vals[2]) + 0.020, fill=False,
                         edgecolor=ANNOTATION, linewidth=1.8, linestyle="--"))
    ax.annotate("compare with baseline", xy=(1.5, max(vals[1], vals[2]) + 0.002),
                xytext=(0.55, 0.208),
                arrowprops=dict(arrowstyle="-|>", color=ANNOTATION),
                color=ANNOTATION, fontsize=9, fontweight="bold")
axes[0].set_ylabel(r"windowed total variation $D_W$")
fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=160)
