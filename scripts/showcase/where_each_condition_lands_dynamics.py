#!/usr/bin/env python
"""Static reads of the animated "where does each condition land" page.

Reads artifacts/scenes/where-each-condition-lands/public/data.json (the per-run,
per-saved-step (x, y) on the cloud axes written by
where_each_condition_lands_frames_embed.py) and the decoded frames on /datasets.

Writes into artifacts/results/where-does-each-condition-land/:
    both-ness-over-denoising-steps.png   y = both-ness (projection toward the joint-prompt
                                         centroid, DINOv2 cosine units), x = denoising step;
                                         one thin line per seed, one thick line per condition
                                         mean, for the joint prompt, PoE and PoE + 1.2 x correction
    commit-and-fork-steps.json           per seed: the commit step of each run (first saved step
                                         after which both-ness stays within COMMIT_TOL of its
                                         final value) and the fork step between PoE and the
                                         corrected run (first saved step at which the two are
                                         further apart in the plane than FORK_MIN_DIST)
    frames-seed-15-strip.png             the decoded running estimate at seven saved steps for
                                         all six conditions of one seed, from
                                         /datasets/.../where_each_condition_lands/frames/
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "artifacts/scenes/where-each-condition-lands/public/data.json"
OUT = REPO / "artifacts/results/where-does-each-condition-land"
FRAMES = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/where_each_condition_lands/frames")

COMMIT_TOL = 0.10      # both-ness units; the run has committed once it stays this close to its end
FORK_MIN_DIST = 0.05   # plane distance between the PoE and corrected run that counts as a fork
STRIP_SEED = 15
STRIP_STEPS = (0, 5, 10, 20, 30, 40, 50)

COLORS = {"joint": "#7570b3", "poe": "#111111", "lora_1.2": "#e7298a",
          "solo_a": "#d95f02", "solo_b": "#1b9e77", "lora_1.0": "#666666"}
LABELS = {"solo_a": "a cat alone", "solo_b": "a dog alone", "joint": '"a cat and a dog"',
          "poe": "PoE, no correction", "lora_1.0": "PoE + 1.0 x correction",
          "lora_1.2": "PoE + 1.2 x correction"}


def main() -> int:
    d = json.loads(DATA.read_text())
    steps = d["steps"]
    seeds = d["seeds"]
    tracks = {c: {int(s): np.asarray(v) for s, v in d["tracks"][c].items()} for c in d["tracks"]}

    # ---- commit and fork steps ------------------------------------------------
    rows = []
    for s in seeds:
        row = {"seed": s}
        for c in ("joint", "poe", "lora_1.2"):
            y = tracks[c][s][:, 1]
            final = y[-1]
            commit = None
            for i, st in enumerate(steps):
                if np.all(np.abs(y[i:] - final) <= COMMIT_TOL):
                    commit = st
                    break
            row[f"commit_step_{c}"] = commit
            row[f"final_both_ness_{c}"] = round(float(final), 4)
        dist = np.linalg.norm(tracks["poe"][s] - tracks["lora_1.2"][s], axis=1)
        fork = next((st for st, dd in zip(steps, dist) if dd > FORK_MIN_DIST), None)
        row["fork_step_poe_vs_lora_1.2"] = fork
        row["plane_dist_poe_vs_lora_1.2_per_step"] = [round(float(x), 4) for x in dist]
        rows.append(row)

    summary = {}
    for c in ("joint", "poe", "lora_1.2"):
        vals = [r[f"commit_step_{c}"] for r in rows if r[f"commit_step_{c}"] is not None]
        summary[f"commit_step_{c}"] = {"median": float(np.median(vals)) if vals else None,
                                        "min": min(vals) if vals else None,
                                        "max": max(vals) if vals else None,
                                        "n_defined": len(vals)}
    forks = [r["fork_step_poe_vs_lora_1.2"] for r in rows if r["fork_step_poe_vs_lora_1.2"] is not None]
    summary["fork_step_poe_vs_lora_1.2"] = {"median": float(np.median(forks)) if forks else None,
                                            "min": min(forks) if forks else None,
                                            "max": max(forks) if forks else None,
                                            "n_defined": len(forks)}
    (OUT / "commit-and-fork-steps.json").write_text(json.dumps({
        "source": str(DATA.relative_to(REPO)),
        "steps_saved": steps,
        "COMMIT_TOL": COMMIT_TOL, "FORK_MIN_DIST": FORK_MIN_DIST,
        "definitions": {
            "commit_step": "first saved step after which both-ness stays within COMMIT_TOL of its final value",
            "fork_step": "first saved step at which the PoE run and the lambda 1.2 run are further apart in the plane than FORK_MIN_DIST",
        },
        "summary": summary, "per_seed": rows,
    }, indent=2))
    print("summary", json.dumps(summary, indent=1))

    # ---- both-ness over steps -------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for c in ("joint", "poe", "lora_1.2"):
        ys = np.stack([tracks[c][s][:, 1] for s in seeds])
        for y in ys:
            ax.plot(steps, y, color=COLORS[c], lw=0.8, alpha=0.35)
        ax.plot(steps, ys.mean(0), color=COLORS[c], lw=2.8, label=f"{LABELS[c]} (mean of 8 seeds)")
    ax.axhline(0, color="#bbbbbb", lw=0.8)
    ax.axhline(d["axes"]["joint_centroid_both_ness"], color=COLORS["joint"], lw=0.8, ls="--")
    ax.text(steps[-1], d["axes"]["joint_centroid_both_ness"] + 0.01, "joint-prompt centroid",
            color=COLORS["joint"], ha="right", fontsize=9)
    ax.set_xlabel("denoising step (0 = pure noise, 50 = finished image); one point per saved frame")
    ax.set_ylabel("both-ness of the running estimate: solo midpoint (0) toward the joint-prompt centroid")
    ax.set_title("cat x dog, held-out seeds 9 to 16: both-ness of each run's running estimate over the 50 steps\n"
                 "thin lines one seed each, thick lines the condition mean; from the scene's data.json",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "both-ness-over-denoising-steps.png", dpi=160)

    # ---- frame strip for one seed ---------------------------------------------
    conds = ("solo_a", "solo_b", "joint", "poe", "lora_1.0", "lora_1.2")
    T, pad, lab, left = 192, 6, 22, 150
    W = left + len(STRIP_STEPS) * (T + pad) + pad
    H = lab + len(conds) * (T + pad) + pad
    im = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(im)
    for j, st in enumerate(STRIP_STEPS):
        dr.text((left + pad + j * (T + pad), 6), f"step {st}", fill="black")
    for i, c in enumerate(conds):
        y0 = lab + pad + i * (T + pad)
        dr.text((6, y0 + T // 2 - 6), LABELS[c], fill="black")
        for j, st in enumerate(STRIP_STEPS):
            p = FRAMES / c / f"seed_{STRIP_SEED}" / f"step_{st:03d}.png"
            if not p.exists():
                dr.rectangle([left + pad + j * (T + pad), y0, left + pad + j * (T + pad) + T, y0 + T], outline="#999")
                continue
            im.paste(Image.open(p).convert("RGB").resize((T, T)), (left + pad + j * (T + pad), y0))
    im.save(OUT / f"frames-seed-{STRIP_SEED}-strip.png")
    print("wrote", OUT / "both-ness-over-denoising-steps.png", OUT / f"frames-seed-{STRIP_SEED}-strip.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
