from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

INDIGO = "#4527a0"
INK = "#212121"

root = Path(__file__).resolve().parents[4]
with (root / "rebuild/out/walk-numbers.json").open() as f:
    values = json.load(f)

ss = np.linspace(0.001, 1.7, 500)
stationary = 2 / (2 - ss)
toy_ss = 0.9
toy_formula = values["p2.ula_stationary_var_ss0.9"]
toy_sim = values["p2.ula_simulated_var_ss0.9"]

fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.plot(ss, stationary, color="#37474f", lw=2.5)
ax.axhline(1, color=INK, ls="--", lw=1.5)
ax.scatter([toy_ss], [toy_formula], color="#37474f", s=65, zorder=4)
ax.scatter([toy_ss], [toy_sim], color=INK, marker="x", s=80, lw=2.2, zorder=5)
ax.annotate("target variance: 1", (1.34, 1), xytext=(1.08, 1.22),
            arrowprops=dict(arrowstyle="->", color=INK), color=INK)
ax.annotate(f"formula: {toy_formula:.4f}\nsimulation: {toy_sim:.3f}",
            (toy_ss, toy_formula), xytext=(0.15, 3.6),
            arrowprops=dict(arrowstyle="->", color=INK), color=INK, weight="bold")
ax.set(xlabel="ULA step size ss", ylabel="stationary variance", xlim=(0, 1.7), ylim=(0.8, 7.0))
ax.set_title("A finite ULA step widens the stationary law")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=160)
