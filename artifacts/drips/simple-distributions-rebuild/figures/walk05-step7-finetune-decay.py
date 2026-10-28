from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

INDIGO = "#4527a0"
NEAR_BLACK = "#212121"

repo = Path(__file__).resolve().parents[4]
data_path = repo / "rebuild" / "out" / "phase5" / "finetune.json"
with data_path.open() as f:
    data = json.load(f)

checkpoints = data["checkpoints"]
steps = [row["steps"] for row in checkpoints]
series = [
    ("grid", [row["exact"]["interior"] for row in checkpoints], "o", "-"),
    ("tuned MALA", [row["MALA, tuned"]["interior"] for row in checkpoints], "s", "--"),
    ("UHA", [row["UHA"]["interior"] for row in checkpoints], "^", ":"),
]

fig, ax = plt.subplots(figsize=(8.6, 4.8))
for label, values, marker, linestyle in series:
    ax.plot(
        steps,
        values,
        color=INDIGO,
        marker=marker,
        linestyle=linestyle,
        linewidth=2,
        markersize=5,
        alpha=1.0 if label == "grid" else 0.72,
    )

label_x = 11
for label, values, _, _ in series:
    y = values[steps.index(10)]
    offset = {"grid": -0.011, "tuned MALA": 0.0, "UHA": 0.011}[label]
    ax.text(label_x, y + offset, label, color=INDIGO, fontsize=9, va="center")

start = checkpoints[0]["exact"]["interior"]
step30 = next(row["exact"]["interior"] for row in checkpoints if row["steps"] == 30)

ax.add_patch(
    Ellipse(
        (420, 0.0012),
        width=520,
        height=0.009,
        fill=False,
        edgecolor=NEAR_BLACK,
        linewidth=1.8,
        linestyle="--",
    )
)
ax.text(300, 0.016, "all at most 0.001\nfrom step 300", color=NEAR_BLACK, fontsize=9)

for x, y, text in [(0, start, "0.159"), (30, step30, "0.001")]:
    ax.annotate(
        text,
        (x, y),
        xytext=(14, 6) if x == 30 else (0, 9),
        textcoords="offset points",
        ha="left" if x == 30 else "center",
        color=NEAR_BLACK,
        fontsize=9,
        weight="bold",
    )

ax.set_xscale("symlog", linthresh=10)
ax.set_xlim(-1, 12000)
ax.set_ylim(-0.006, 0.18)
ax.set_xlabel("fine-tune step (linear from 0 to 10, log scale above)")
ax.set_ylabel("interior $A\\cap B$ mass\n(share of the composed target)")
ax.set_title("Pretrained interior support disappears during fine-tuning")
ax.grid(axis="y", color="#e0e0e0", linewidth=0.8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=160)
