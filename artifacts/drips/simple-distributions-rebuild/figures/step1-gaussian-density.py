import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

DARK_GREY = "#37474f"
ANNOTATION = "#212121"

root = Path(__file__).resolve().parents[4]
values = json.loads((root / "rebuild/out/walk-numbers.json").read_text())
peak = values["p1.normal_pdf_0_mu0_sd0.5"]
one_sd = values["p1.normal_pdf_0.5_mu0_sd0.5"]
sd = 0.5
area_without_constant = 0.5 * math.sqrt(2 * math.pi)

xs = [-2.0 + i * 0.01 for i in range(401)]
normalized = [peak * math.exp(-(x * x) / (2 * sd * sd)) for x in xs]
without_constant = [math.exp(-(x * x) / (2 * sd * sd)) for x in xs]

fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), sharex=True, sharey=True)

axes[0].plot(xs, normalized, color=DARK_GREY, linewidth=2.5)
axes[0].scatter([-0.5, 0.5], [one_sd, one_sd], color=ANNOTATION, zorder=3)
axes[0].add_patch(Ellipse((0, one_sd), 1.3, 0.11, fill=False, edgecolor=ANNOTATION,
                          linewidth=1.8, linestyle="--"))
axes[0].annotate("same height: 0.4839", xy=(0.5, one_sd), xytext=(0.7, 0.67),
                 arrowprops=dict(arrowstyle="->", color=ANNOTATION), color=ANNOTATION, fontsize=9)
axes[0].annotate("peak: 0.7979", xy=(0, peak), xytext=(-1.65, 0.98),
                 arrowprops=dict(arrowstyle="-|>", color=ANNOTATION), color=ANNOTATION,
                 fontsize=10, fontweight="bold")
axes[0].text(0, 0.08, "area = 1", ha="center", color="#37474f", fontweight="bold")
axes[0].set_title("① Gaussian density, constant kept")

axes[1].plot(xs, without_constant, color=DARK_GREY, linewidth=2.5)
axes[1].annotate("peak rises to 1", xy=(0, 1), xytext=(-1.7, 1.12),
                 arrowprops=dict(arrowstyle="-|>", color=ANNOTATION), color=ANNOTATION,
                 fontsize=10, fontweight="bold")
axes[1].text(0, 0.08, f"area = {area_without_constant:.4f}", ha="center",
             color="#37474f", fontweight="bold")
axes[1].set_title("② Front constant removed")

for ax in axes:
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("position x")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("density height")
fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=160)
