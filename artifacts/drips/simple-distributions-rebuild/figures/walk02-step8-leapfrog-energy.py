from pathlib import Path
import json
import matplotlib.pyplot as plt

BROWN = "#6d4c41"
INK = "#212121"

root = Path(__file__).resolve().parents[4]
with (root / "rebuild/out/walk-numbers.json").open() as f:
    values = json.load(f)

before = values["p2.leapfrog_H_before"]
after = values["p2.leapfrog_H_after"]
delta = after - before

fig, ax = plt.subplots(figsize=(6.8, 4.2))
xs = [0, 1]
ax.plot(xs, [before, after], color=BROWN, marker="o", ms=9, lw=2)
for x, value in zip(xs, [before, after]):
    ax.annotate(f"{value:.5f}", (x, value), xytext=(0, 10), textcoords="offset points",
                ha="center", color=INK, weight="bold")
ax.annotate(f"change: +{delta:.5f}, about 0.4% of H", (0.5, (before + after) / 2),
            xytext=(0.5, after + 0.0035), ha="center", color=INK,
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
ax.set_xticks(xs, ["before\n(x=1, v=0.5)", "after\n(x'=1.105, v'=0.18425)"])
ax.set_xlim(-0.4, 1.4)
ax.set(ylabel="Hamiltonian energy H (axis zoomed: starts at 0.615)", ylim=(0.615, 0.633))
ax.set_title("One leapfrog step nearly keeps total energy")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=160)
