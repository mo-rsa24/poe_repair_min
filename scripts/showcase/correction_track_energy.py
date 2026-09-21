#!/usr/bin/env python
"""Rung 4 of "what the correction is made of": how far each run's running estimate travels
before it settles, and which animal it is at every saved step.

Reads artifacts/results/where-does-each-condition-land/frames-dino-feats.npz: the (672, 384)
L2-normalised DINOv2 ViT-S/14 CLS embeddings of the decoded running estimate (Tweedie mean) of
six conditions x eight seeds x fourteen saved steps (0, 2, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45,
49, 50), plus the cloud axes u1 (cat to dog), u2 (both-ness) and their origin.

Per run (condition, seed):
    kinetic energy   sum over the 13 consecutive saved segments of ||z_{k+1} - z_k||^2
    its floor        ||z_50 - z_0||^2 / 13, what a constant-speed straight line would score
    path length      sum of ||z_{k+1} - z_k||;  its floor is ||z_50 - z_0||
    the same four numbers in the 2-d cloud-axes plane
    which-animal     cos(z_k, cat centroid) - cos(z_k, dog centroid) per saved step, centroids
                     from the step-50 frames of the cat-alone and dog-alone runs (normalised)
    flips            number of sign changes of which-animal after step 10, and the last one's step

Writes into artifacts/results/what-the-correction-is-made-of/:
    track-kinetic-energy.png            x: the floor, y: the kinetic energy, one point per run,
                                        colour per condition, diagonal = a straight constant-speed track
    which-animal-over-steps.png         y: which-animal score, x: saved step; thin line per seed,
                                        thick mean per condition; seed 15 marked
    track-energy-and-which-animal.json  every number above, one row per run, plus per-condition medians
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import correction_span_common as C  # noqa: E402

FEATS = C.REPO / "artifacts/results/where-does-each-condition-land/frames-dino-feats.npz"
FLIP_AFTER_STEP = 10
FLIP_MIN_MAGNITUDE = 0.1   # a sign change counts as a real flip only if the score exceeds this on both sides
COLORS = {"solo_a": "#d95f02", "solo_b": "#1b9e77", "joint": "#7570b3",
          "poe": "#111111", "lora_1.0": "#888888", "lora_1.2": "#e7298a"}
LABELS = {"solo_a": "a cat alone", "solo_b": "a dog alone", "joint": '"a cat and a dog"',
          "poe": "PoE, no correction", "lora_1.0": "PoE + 1.0 x correction", "lora_1.2": "PoE + 1.2 x correction"}


def main() -> int:
    z = np.load(FEATS)
    feats, order, u1, u2, origin = z["feats"], z["order"], z["u1"], z["u2"], z["origin"]
    keys = [tuple(o.split("|")) for o in order]
    idx = {(c, int(s), int(k)): i for i, (c, s, k) in enumerate(keys)}
    conds = [c for c in COLORS if any(kk[0] == c for kk in idx)]
    seeds = sorted({kk[1] for kk in idx})
    steps = sorted({kk[2] for kk in idx})
    nseg = len(steps) - 1

    cat = feats[[idx[("solo_a", s, 50)] for s in seeds]].mean(0); cat /= np.linalg.norm(cat)
    dog = feats[[idx[("solo_b", s, 50)] for s in seeds]].mean(0); dog /= np.linalg.norm(dog)

    rows = []
    for c in conds:
        for s in seeds:
            Z = feats[[idx[(c, s, k)] for k in steps]]                    # (14, 384), unit rows
            P = np.stack([(Z - origin) @ u1, (Z - origin) @ u2], 1)      # (14, 2), the cloud-axes plane
            d = np.diff(Z, axis=0); dp = np.diff(P, axis=0)
            late = [i for i in range(nseg) if steps[i] >= FLIP_AFTER_STEP]      # segments starting at or after step 10
            Zl = Z[[i for i, k in enumerate(steps) if k >= FLIP_AFTER_STEP]]
            straight = float(np.linalg.norm(Z[-1] - Z[0])); straight_p = float(np.linalg.norm(P[-1] - P[0]))
            which = Z @ cat - Z @ dog
            after = [(k, w) for k, w in zip(steps, which) if k >= FLIP_AFTER_STEP]
            signs = np.sign([w for _, w in after])
            changes = [after[i][0] for i in range(1, len(after)) if signs[i] != signs[i - 1] and signs[i] != 0]
            real = [after[i][0] for i in range(1, len(after))
                    if signs[i] != signs[i - 1] and abs(after[i][1]) >= FLIP_MIN_MAGNITUDE
                    and abs(after[i - 1][1]) >= FLIP_MIN_MAGNITUDE]
            rows.append({
                "condition": c, "seed": s,
                "kinetic_energy": float((d ** 2).sum()), "kinetic_floor": straight ** 2 / nseg,
                "path_length": float(np.linalg.norm(d, axis=1).sum()), "straight_line": straight,
                "kinetic_energy_from_step_10": float((d[late] ** 2).sum()),
                "kinetic_floor_from_step_10": float(np.linalg.norm(Zl[-1] - Zl[0]) ** 2 / len(late)),
                "kinetic_energy_plane": float((dp ** 2).sum()), "kinetic_floor_plane": straight_p ** 2 / nseg,
                "path_length_plane": float(np.linalg.norm(dp, axis=1).sum()), "straight_line_plane": straight_p,
                "which_animal_per_step": [round(float(w), 4) for w in which],
                "which_animal_final": round(float(which[-1]), 4),
                "flips_after_step_10": len(changes), "last_flip_step": changes[-1] if changes else None,
                "real_flips_after_step_10": len(real), "last_real_flip_step": real[-1] if real else None,
            })

    def med(c, key):
        return float(np.median([r[key] for r in rows if r["condition"] == c]))
    summary = {c: {"kinetic_energy_median": med(c, "kinetic_energy"), "kinetic_floor_median": med(c, "kinetic_floor"),
                   "excess_over_floor_median": float(np.median([r["kinetic_energy"] - r["kinetic_floor"] for r in rows if r["condition"] == c])),
                   "kinetic_energy_from_step_10_median": med(c, "kinetic_energy_from_step_10"),
                   "kinetic_floor_from_step_10_median": med(c, "kinetic_floor_from_step_10"),
                   "wander_ratio_median": float(np.median([r["path_length"] / max(r["straight_line"], 1e-8) for r in rows if r["condition"] == c])),
                   "runs_with_a_flip_after_step_10": sum(r["flips_after_step_10"] > 0 for r in rows if r["condition"] == c),
                   "runs_with_a_real_flip_after_step_10": sum(r["real_flips_after_step_10"] > 0 for r in rows if r["condition"] == c),
                   "which_animal_final_mean": float(np.mean([r["which_animal_final"] for r in rows if r["condition"] == c]))}
               for c in conds}
    C.write_json(C.RESULTS / "track-energy-and-which-animal.json", {
        "source": str(FEATS.relative_to(C.REPO)), "steps": steps, "seeds": seeds, "n_segments": nseg,
        "FLIP_AFTER_STEP": FLIP_AFTER_STEP, "FLIP_MIN_MAGNITUDE": FLIP_MIN_MAGNITUDE,
        "definitions": {
            "space": "L2-normalised DINOv2 ViT-S/14 CLS embeddings of the decoded running estimate, 384-d; '_plane' = the cloud-axes plane (u1 cat to dog, u2 both-ness)",
            "kinetic_energy": "sum over the 13 saved segments of ||z_{k+1} - z_k||^2; saved steps are unevenly spaced",
            "kinetic_floor": "||z_last - z_first||^2 / 13, the score of a straight track at constant speed over the same segments",
            "kinetic_energy_from_step_10": "the same sum restricted to the segments starting at or after step 10 (8 segments: 10-15, ..., 49-50), with its own floor",
            "path_length / straight_line": "sum of segment lengths, and the straight-line distance it can never go below",
            "which_animal": "cos(z, cat-alone step-50 centroid) - cos(z, dog-alone step-50 centroid); positive is cat",
            "flips_after_step_10": "sign changes of which_animal over the saved steps at or after step 10",
            "real_flips_after_step_10": "the same, counting only sign changes where |which_animal| >= FLIP_MIN_MAGNITUDE on both sides, so a score hovering near 0 (both animals) does not count as flipping",
        },
        "summary": summary, "rows": rows,
    })

    # ---- kinetic energy per condition, floor as a tick under each point --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (ek, fk, title) in zip(axes, (("kinetic_energy", "kinetic_floor", "all 13 saved segments, step 0 to 50"),
                                          ("kinetic_energy_from_step_10", "kinetic_floor_from_step_10", "the 8 segments from step 10 to 50"))):
        for ci, c in enumerate(conds):
            rr = [r for r in rows if r["condition"] == c]
            xs = ci + np.linspace(-0.22, 0.22, len(rr))
            ax.scatter(xs, [r[ek] for r in rr], s=42, color=COLORS[c], zorder=3,
                       marker="s" if c == "poe" else ("D" if c.startswith("lora") else "o"))
            ax.scatter(xs, [r[fk] for r in rr], s=30, color="#999", marker="_", zorder=2)
            ax.hlines(np.median([r[ek] for r in rr]), ci - 0.3, ci + 0.3, color=COLORS[c], lw=2.2)
            for xx, r in zip(xs, rr):
                if r["seed"] == C.STRIP_SEED:
                    ax.annotate("15", (xx, r[ek]), fontsize=8, xytext=(3, 3), textcoords="offset points", color=COLORS[c])
        ax.set_xticks(range(len(conds))); ax.set_xticklabels([LABELS[c] for c in conds], rotation=20, ha="right", fontsize=8.5)
        ax.set_ylabel("kinetic energy: sum of squared displacements\nof the embedded running estimate")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, None)
    axes[0].text(0.02, 0.97, "grey tick = that run's floor (a straight track at constant speed); bar = condition median; '15' = seed 15",
                 transform=axes[0].transAxes, fontsize=8, va="top", color="#555")
    fig.suptitle("cat x dog, seeds 9 to 16: how much each run's running estimate moves in DINOv2 space (one point per run)", fontsize=11)
    fig.tight_layout(); fig.savefig(C.RESULTS / "track-kinetic-energy.png", dpi=160)

    # ---- which-animal over steps -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for c in ("solo_a", "solo_b", "joint", "poe", "lora_1.2"):
        W = np.array([r["which_animal_per_step"] for r in rows if r["condition"] == c])
        for r in [r for r in rows if r["condition"] == c]:
            ax.plot(steps, r["which_animal_per_step"], color=COLORS[c], lw=2.2 if r["seed"] == C.STRIP_SEED else 0.7,
                    alpha=0.9 if r["seed"] == C.STRIP_SEED else 0.3, ls="-" if r["seed"] != C.STRIP_SEED else "--")
        ax.plot(steps, W.mean(0), color=COLORS[c], lw=2.8, label=f"{LABELS[c]} (mean of 8)")
    ax.axhline(0, color="#999", lw=0.8)
    ax.axvline(FLIP_AFTER_STEP, color="#999", lw=0.8, ls=":")
    ax.text(FLIP_AFTER_STEP + 0.3, ax.get_ylim()[1] * 0.95, "flips counted from step 10", fontsize=8, color="#666")
    ax.set_xlabel("denoising step (0 = pure noise, 50 = finished image); one point per saved frame")
    ax.set_ylabel("which animal: cos to the cat-alone centroid minus cos to the dog-alone centroid\n(positive = cat, negative = dog)")
    ax.set_title("cat x dog, seeds 9 to 16: which animal each run's running estimate reads as, per saved step\n"
                 "thin lines one seed each, dashed = seed 15, thick = condition mean", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(C.RESULTS / "which-animal-over-steps.png", dpi=160)

    for c in conds:
        print(c, {k: round(v, 4) if isinstance(v, float) else v for k, v in summary[c].items()})
    print("seed 15 poe:", next(r for r in rows if r["condition"] == "poe" and r["seed"] == 15))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
