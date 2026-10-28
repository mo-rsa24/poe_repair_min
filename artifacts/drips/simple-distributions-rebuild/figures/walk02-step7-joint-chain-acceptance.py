from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

INK = "#212121"

root = Path(__file__).resolve().parents[4]
with (root / "rebuild/out/walk-numbers.json").open() as f:
    values = json.load(f)

n = np.array([1, 10, 100, 4000])
acceptance = np.array([
    values["p2.batch_mala_joint_accept_n1"],
    values["p2.batch_mala_joint_accept_n10"],
    values["p2.batch_mala_joint_accept_n100"],
    values["p2.batch_mala_joint_accept_n4000"],
])

fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.plot(n, acceptance, color="#37474f", marker="o", lw=2.5, ms=7)
for x, y in zip(n, acceptance):
    ax.annotate(f"{y:.3g}", (x, y), xytext=(6, 8), textcoords="offset points",
                ha="left", color=INK, weight="bold")
ax.set_xscale("log")
ax.set_xticks(n, ["1", "10", "100", "4000"])
ax.set(xlabel="samples forced into one joint chain, n (log scale)",
       ylabel="share of 200 moves accepted", ylim=(-0.04, 0.7))
ax.set_title("One batch-wide decision freezes as the batch grows")
ax.annotate("nearly no joint moves survive", (10, acceptance[1]), xytext=(35, 0.24),
            arrowprops=dict(arrowstyle="->", color=INK), color=INK)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=160)
