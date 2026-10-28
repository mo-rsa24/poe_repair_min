from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

INK = "#212121"
VARIANCE = "#37474f"

repo = Path(__file__).resolve().parents[4]
with (repo / "rebuild/out/walk-numbers.json").open() as f:
    values = json.load(f)

SIG = 0.05
steps = np.array([0, 25, 50, 75, 99])
abar = np.array([values[f"p1.abar_{t}"] for t in steps])
variance = 1 - abar + abar * SIG**2
variance[0] = values["p3.noised_ideal_var_t0_sig0.05"]
variance[2] = values["p3.noised_ideal_var_t50_sig0.05"]
variance[4] = values["p3.noised_ideal_var_t99_sig0.05"]

fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.plot(steps, variance, color=VARIANCE, marker="o", linewidth=2.5, markersize=7)
ax.fill_between(steps, 0, variance, color=VARIANCE, alpha=0.08)

for t, value in zip(steps, variance):
    ax.annotate(f"{value:.4g}", (t, value), xytext=(0, 10), textcoords="offset points",
                ha="center", color=INK, weight="bold")

ax.text(3, 0.32, "narrow clean kernel", color=INK, weight="bold")
ax.set_xlabel(r"diffusion step $t$")
ax.set_ylabel("exact component variance $v_t$\npoints joined by straight segments")
ax.set_xticks(steps)
ax.set_ylim(0, 1.12)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=160)
