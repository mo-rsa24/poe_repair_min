import json
from pathlib import Path

import matplotlib.pyplot as plt

GREEN = "#2e7d32"
ORANGE = "#ef6c00"
ANNOTATION = "#212121"

root = Path(__file__).resolve().parents[4]
values = json.loads((root / "rebuild/out/walk-numbers.json").read_text())
steps = [0, 25, 50, 75, 99]
kept = [values[f"p1.abar_{t}"] for t in steps]
noise = [values[f"p1.sqrt_1m_abar_{t}"] for t in steps]

fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.plot(steps, kept, marker="o", linewidth=2.5, color=GREEN)
ax.plot(steps, noise, marker="o", linewidth=2.5, color=ORANGE)

for t, value in zip(steps, kept):
    ax.text(t, value + 0.055, f"{value:.4f}" if value else "0.0",
            ha="center", color=GREEN, fontsize=8)
for t, value in zip(steps, noise):
    if t == 0:
        ax.text(t + 3, value + 0.035, f"{value:.4f}", ha="left",
                color=ORANGE, fontsize=8)
    else:
        ax.text(t, value - 0.09, f"{value:.4f}" if value < 1 else "1.0",
                ha="center", color=ORANGE, fontsize=8)

ax.text(79, 0.07, "kept share  ᾱₜ", color=GREEN, fontweight="bold", ha="right")
ax.text(79, 1.045, "noise level  √(1−ᾱₜ)", color=ORANGE, fontweight="bold", ha="right")
ax.set_xlabel("code step t")
ax.set_ylabel("schedule value")
ax.set_xticks(steps)
ax.set_ylim(0, 1.12)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=160)
