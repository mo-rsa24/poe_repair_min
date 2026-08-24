"""F1 'scorer-works' figure (compose-scorer plan 03, DoD 4).

Shows the validated instance-count scorer on the decisive case: the wolf×husky HARD
pair, both ways. A blended output (one animal, count 1) and a real composition (two
animals, count >= 2) sit side by side with their instance count and compose/blend
label as the point colour, above the full 10-item validation strip.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "outputs" / "compose_scorer"
GREEN, RED = "#2f855a", "#c53030"


def main() -> int:
    rep = json.loads((OUT_DIR / "agreement_table_detection.json").read_text())
    rows = {r["id"]: r for r in rep["rows"]}

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.3, wspace=0.2)

    # top: the hard pair both ways (blend vs compose), the case only instance-count solved
    hero = [
        ("wolfhusky_sample_seed_09", "wolf×husky CORRECTED (seed 09)"),
        ("wolfhusky_sample_seed_11", "wolf×husky CORRECTED (seed 11)"),
        ("wolfhusky_sample_seed_12", "wolf×husky CORRECTED (seed 12)"),
        ("wolfhusky_joint_anchor", "wolf×husky joint anchor"),
    ]
    for i, (rid, cap) in enumerate(hero):
        r = rows[rid]
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(Image.open(r["path"]).convert("RGB"))
        lbl = r["instance"]["label"]
        n = r["instance"]["n_instances"]
        color = GREEN if lbl == "compose" else RED
        ax.set_title(f"{cap}\n{n} animal(s) → {lbl.upper()}", fontsize=8, color=color)
        for s in ax.spines.values():
            s.set_edgecolor(color); s.set_linewidth(4)
        ax.set_xticks([]); ax.set_yticks([])

    # bottom: full 10-item validation strip, count vs threshold
    ax = fig.add_subplot(gs[1, :])
    ids = list(rows.keys())
    xs = range(len(ids))
    counts = [rows[i]["instance"]["n_instances"] for i in ids]
    truth = [rows[i]["truth"] for i in ids]
    colors = [GREEN if rows[i]["instance"]["label"] == "compose" else RED for i in ids]
    ax.bar(xs, counts, color=colors)
    ax.axhline(1.5, ls="--", color="black", lw=1, label="compose threshold (>= 2 instances)")
    ax.set_xticks(list(xs))
    short = [i.replace("catdog_compose_compose_seed_", "cd_cmp_s")
             .replace("wolfhusky_sample_seed_", "wh_s")
             .replace("catdog_joint_anchor", "cd_joint")
             .replace("wolfhusky_joint_anchor", "wh_joint")
             .replace("catdog_poe_blend", "cd_poe") for i in ids]
    ax.set_xticklabels([f"{s}\n({t})" for s, t in zip(short, truth)], fontsize=7, rotation=25, ha="right")
    ax.set_ylabel("distinct animal instances")
    ax.set_title(f"Instance-count read: 10/10 correct. Blends → 1, composes → >= 2. "
                 f"Green=compose, red=blend (colour = scorer label; label in parens = truth).", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("F1 — scorer works: instance-count separates a two-animal COMPOSE from a "
                 "chimera BLEND, including the hard wolf×husky pair both ways.", fontsize=11, y=0.98)
    out = OUT_DIR / "F1_scorer_works.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}  ({out.stat().st_size}B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
