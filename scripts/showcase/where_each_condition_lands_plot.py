#!/usr/bin/env python
"""Where does each condition land: cat×dog, held-out seeds 9 to 16, in the
scorer's DINOv2 space.

One point per (seed, condition). Six conditions:
    solo_a     "a cat" alone
    solo_b     "a dog" alone
    joint      "a cat and a dog" as one prompt (Mono)
    poe        A + B − null, no correction               (rank-32 step-30050 run, λ 0)
    lora_1.0   PoE + 1.0 × correction, rank 32, step 30050
    lora_1.2   PoE + 1.2 × correction, rank 32, step 30050

Embedding: DINOv2 ViT-S/14 CLS, L2-normed, the same embedder the compose
scorer uses (scripts/build_lora_inspector_mds_semantic.DinoEmbedder).
Projection: one PCA fit on all 48 points, mean-centred. The plane's variance
share is printed on the figure because a 2-D picture of a 384-D cloud can
place two far-apart points side by side.

The numbers that matter are computed in the full 384-D space and written to
the sidecar: for every PoE and corrected point, its cosine distance to the
centroid of each of the three reference clouds, and the per-seed cosine
between the λ 1.0 arrow and the λ 1.2 arrow (are the two doses collinear).

Writes to artifacts/results/where-does-each-condition-land/:
    cat-x-dog-in-dino-space.png
    cat-x-dog-in-dino-space.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from scipy.spatial import ConvexHull

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.build_lora_inspector_mds_semantic import DinoEmbedder  # noqa: E402

SOLO_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/where_each_condition_lands")
R32_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/figure_r32_030050")
OUT_DIR = REPO_ROOT / "artifacts/results/where-does-each-condition-land"
NAME = "cat-x-dog-in-dino-space"
SEEDS = (9, 10, 11, 12, 13, 14, 15, 16)

CONDITIONS = {
    "solo_a":   ("a cat alone",            lambda s: SOLO_ROOT / "solo_a" / f"seed_{s}.png"),
    "solo_b":   ("a dog alone",            lambda s: SOLO_ROOT / "solo_b" / f"seed_{s}.png"),
    "joint":    ("\"a cat and a dog\"",    lambda s: SOLO_ROOT / "joint" / f"seed_{s}.png"),
    "poe":      ("PoE, no correction",     lambda s: R32_ROOT / "renders/full" / f"seed_{s}_lambda_0.0.png"),
    "lora_1.0": ("PoE + 1.0 × correction", lambda s: R32_ROOT / "renders/full" / f"seed_{s}_lambda_1.0.png"),
    "lora_1.2": ("PoE + 1.2 × correction", lambda s: R32_ROOT / "renders/full" / f"seed_{s}_lambda_1.2.png"),
}
CLOUDS = ("solo_a", "solo_b", "joint")
COLORS = {
    "solo_a": "#d95f02", "solo_b": "#1b9e77", "joint": "#7570b3",
    "poe": "#111111", "lora_1.0": "#666666", "lora_1.2": "#e7298a",
}


def load_batch(paths: list[Path]) -> torch.Tensor:
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        ims.append(torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1))
    return torch.stack(ims)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    return float(1.0 - a @ b)


def main() -> int:
    rows = []
    for cond, (_, pathfn) in CONDITIONS.items():
        for s in SEEDS:
            p = pathfn(s)
            if not p.exists():
                raise SystemExit(f"missing render: {p}")
            rows.append({"condition": cond, "seed": s, "png": str(p)})

    embedder = DinoEmbedder(device=torch.device("cpu"))
    feats = np.concatenate([
        embedder.embed_decoded_batch(load_batch([Path(r["png"]) for r in rows[i:i + 8]]))
        for i in range(0, len(rows), 8)
    ])  # (48, 384), L2-normed
    idx = {(r["condition"], r["seed"]): i for i, r in enumerate(rows)}

    # ---- one PCA over every point -------------------------------------------
    mu = feats.mean(0)
    u, sv, vt = np.linalg.svd(feats - mu, full_matrices=False)
    var = sv ** 2 / (sv ** 2).sum()
    xy = (feats - mu) @ vt[:2].T
    var2 = float(var[:2].sum())

    # ---- true-space numbers ---------------------------------------------------
    centroids = {c: feats[[idx[(c, s)] for s in SEEDS]].mean(0) for c in CLOUDS}
    for r in rows:
        f = feats[idx[(r["condition"], r["seed"])]]
        r["xy"] = [float(xy[idx[(r["condition"], r["seed"])], 0]), float(xy[idx[(r["condition"], r["seed"])], 1])]
        r["cos_dist_to_centroid"] = {c: round(cosine_distance(f, centroids[c]), 4) for c in CLOUDS}
        r["nearest_cloud"] = min(r["cos_dist_to_centroid"], key=r["cos_dist_to_centroid"].get)

    per_seed = []
    for s in SEEDS:
        p = feats[idx[("poe", s)]]; a = feats[idx[("lora_1.0", s)]]; b = feats[idx[("lora_1.2", s)]]
        d10, d12 = a - p, b - p
        cos = float(d10 @ d12 / (np.linalg.norm(d10) * np.linalg.norm(d12) + 1e-12))
        to_joint = centroids["joint"] - p
        cos_to_joint = float(d12 @ to_joint / (np.linalg.norm(d12) * np.linalg.norm(to_joint) + 1e-12))
        per_seed.append({
            "seed": s,
            "poe_nearest_cloud": rows[idx[("poe", s)]]["nearest_cloud"],
            "lora_1.2_nearest_cloud": rows[idx[("lora_1.2", s)]]["nearest_cloud"],
            "arrow_norm_1.0": round(float(np.linalg.norm(d10)), 4),
            "arrow_norm_1.2": round(float(np.linalg.norm(d12)), 4),
            "cos_arrow_1.0_vs_1.2": round(cos, 4),
            "cos_arrow_1.2_vs_poe_to_joint_centroid": round(cos_to_joint, 4),
        })

    # inter-cloud separation, so "overlap" is a number and not a look
    def spread(c):
        pts = feats[[idx[(c, s)] for s in SEEDS]]
        return float(np.mean([cosine_distance(x, centroids[c]) for x in pts]))
    cloud_stats = {
        c: {"mean_cos_dist_to_own_centroid": round(spread(c), 4)} for c in CLOUDS
    }
    cloud_gaps = {
        f"{a}__{b}": round(cosine_distance(centroids[a], centroids[b]), 4)
        for i, a in enumerate(CLOUDS) for b in CLOUDS[i + 1:]
    }

    # ---- draw -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 9))
    for c in CLOUDS:
        pts = xy[[idx[(c, s)] for s in SEEDS]]
        if len(pts) >= 3:
            hull = ConvexHull(pts)
            poly = pts[hull.vertices]
            ax.fill(poly[:, 0], poly[:, 1], color=COLORS[c], alpha=0.12, lw=0)
            ax.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]], color=COLORS[c], lw=1, alpha=0.6)
        ax.scatter(pts[:, 0], pts[:, 1], s=46, color=COLORS[c], label=CONDITIONS[c][0], zorder=3)
        cx, cy = pts.mean(0)
        ax.annotate(CONDITIONS[c][0], (cx, cy), color=COLORS[c], fontsize=11, ha="center",
                    va="center", weight="bold", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8))

    for s in SEEDS:
        p = xy[idx[("poe", s)]]; a = xy[idx[("lora_1.0", s)]]; b = xy[idx[("lora_1.2", s)]]
        ax.annotate("", xy=a, xytext=p, arrowprops=dict(arrowstyle="-|>", color=COLORS["lora_1.0"], lw=1.2), zorder=5)
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=COLORS["lora_1.2"], lw=1.2), zorder=5)
        ax.scatter(*p, s=70, color=COLORS["poe"], marker="s", zorder=6)
        ax.scatter(*b, s=70, color=COLORS["lora_1.2"], marker="D", zorder=6)
        ax.annotate(f"s{s}", p, fontsize=8, color=COLORS["poe"], xytext=(4, -9), textcoords="offset points")

    ax.scatter([], [], s=70, color=COLORS["poe"], marker="s", label="PoE, no correction (λ 0)")
    ax.scatter([], [], s=70, color=COLORS["lora_1.2"], marker="D", label="PoE + 1.2 × correction")
    ax.plot([], [], color=COLORS["lora_1.0"], lw=1.2, label="arrow: λ 0 → λ 1.0")
    ax.plot([], [], color=COLORS["lora_1.2"], lw=1.2, label="arrow: λ 1.0 → λ 1.2")

    # the contact sheet beside this file carries the images; none are inset here
    ax.set_xlabel(f"PC1 of DINOv2 CLS ({var[0] * 100:.0f}% of variance)")
    ax.set_ylabel(f"PC2 of DINOv2 CLS ({var[1] * 100:.0f}% of variance)")
    ax.set_title("cat × dog, held-out seeds 9 to 16: where each condition lands in DINOv2 space\n"
                 f"one PCA over all 48 renders, plane keeps {var2 * 100:.0f}% of variance; "
                 "correction = rank-32 LoRA, step 30050, held out on this pair",
                 fontsize=11)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.18)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{NAME}.png", dpi=160)
    (OUT_DIR / f"{NAME}.json").write_text(json.dumps({
        "pair": "a_cat__x__a_dog", "seeds": list(SEEDS),
        "embedding": "DINOv2 ViT-S/14 CLS, L2-normed (scorer embedder)",
        "projection": "PCA fit on all 48 points, mean-centred",
        "variance_share_pc1_pc2": [round(float(var[0]), 4), round(float(var[1]), 4)],
        "conditions": {k: v[0] for k, v in CONDITIONS.items()},
        "correction": {"checkpoint": str(R32_ROOT.parent / "phase1_r32_100k/checkpoints/lora_step_030050.pt"),
                       "rank": 32, "step": 30050, "lambdas": [1.0, 1.2], "window": "all 50 steps"},
        "cloud_stats": cloud_stats,
        "centroid_cosine_gaps": cloud_gaps,
        "per_seed": per_seed,
        "points": rows,
    }, indent=2))

    np.save(OUT_DIR / f"{NAME}-dino-feats.npy", feats)

    # ---- second view: axes defined by the reference clouds --------------------
    # u1: cat centroid -> dog centroid ("which animal").
    # u2: midpoint of the two solo centroids -> joint centroid ("both animals"),
    #     orthogonalised against u1. Origin = the solo midpoint.
    origin = 0.5 * (centroids["solo_a"] + centroids["solo_b"])
    u1 = centroids["solo_b"] - centroids["solo_a"]; u1 /= np.linalg.norm(u1)
    u2 = centroids["joint"] - origin; u2 -= (u2 @ u1) * u1; u2 /= np.linalg.norm(u2)
    ab = np.stack([(feats - origin) @ u1, (feats - origin) @ u2], 1)
    resid = np.linalg.norm((feats - origin) - ab[:, :1] * u1 - ab[:, 1:] * u2, axis=1)
    for r in rows:
        i = idx[(r["condition"], r["seed"])]
        r["cloud_axes"] = {"which_animal": round(float(ab[i, 0]), 4),
                           "both_ness": round(float(ab[i, 1]), 4),
                           "off_plane_norm": round(float(resid[i]), 4)}
    both_by_cond = {c: [round(float(ab[idx[(c, s)], 1]), 4) for s in SEEDS] for c in CONDITIONS}

    fig2, ax2 = plt.subplots(figsize=(11, 8))
    for c in CLOUDS:
        pts = ab[[idx[(c, s)] for s in SEEDS]]
        hull = ConvexHull(pts); poly = pts[hull.vertices]
        ax2.fill(poly[:, 0], poly[:, 1], color=COLORS[c], alpha=0.12, lw=0)
        ax2.scatter(pts[:, 0], pts[:, 1], s=46, color=COLORS[c], label=CONDITIONS[c][0], zorder=3)
        cx, cy = pts.mean(0)
        ax2.annotate(CONDITIONS[c][0], (cx, cy), color=COLORS[c], fontsize=11, ha="center", va="center",
                     weight="bold", zorder=4, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8))
    for s in SEEDS:
        p = ab[idx[("poe", s)]]; a = ab[idx[("lora_1.0", s)]]; b = ab[idx[("lora_1.2", s)]]
        ax2.annotate("", xy=a, xytext=p, arrowprops=dict(arrowstyle="-|>", color=COLORS["lora_1.0"], lw=1.2), zorder=5)
        ax2.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=COLORS["lora_1.2"], lw=1.2), zorder=5)
        ax2.scatter(*p, s=70, color=COLORS["poe"], marker="s", zorder=6)
        ax2.scatter(*b, s=70, color=COLORS["lora_1.2"], marker="D", zorder=6)
        ax2.annotate(f"s{s}", p, fontsize=8, color=COLORS["poe"], xytext=(4, -9), textcoords="offset points")
    ax2.scatter([], [], s=70, color=COLORS["poe"], marker="s", label="PoE, no correction (λ 0)")
    ax2.scatter([], [], s=70, color=COLORS["lora_1.2"], marker="D", label="PoE + 1.2 × correction")
    ax2.plot([], [], color=COLORS["lora_1.0"], lw=1.2, label="arrow: λ 0 → λ 1.0")
    ax2.plot([], [], color=COLORS["lora_1.2"], lw=1.2, label="arrow: λ 1.0 → λ 1.2")
    ax2.axhline(0, color="#bbbbbb", lw=0.8); ax2.axvline(0, color="#bbbbbb", lw=0.8)
    ax2.set_xlabel("which animal: cat centroid (left) to dog centroid (right), DINOv2 cosine units")
    ax2.set_ylabel("both-ness: solo midpoint (0) toward the joint-prompt centroid, orthogonal to x")
    ax2.set_title("cat × dog, held-out seeds 9 to 16: the same 48 renders on axes defined by the three reference clouds\n"
                  "off-plane residual per point is in the sidecar; correction = rank-32 LoRA, step 30050, held out on this pair",
                  fontsize=11)
    ax2.legend(loc="best", fontsize=9, framealpha=0.9)
    ax2.margins(0.15)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / f"{NAME}-cloud-axes.png", dpi=160)

    side = json.loads((OUT_DIR / f"{NAME}.json").read_text())
    side["cloud_axes"] = {
        "x": "unit vector from cat-alone centroid to dog-alone centroid",
        "y": "unit vector from the solo-centroid midpoint to the joint-prompt centroid, orthogonalised against x",
        "origin": "midpoint of the cat-alone and dog-alone centroids",
        "both_ness_by_condition": both_by_cond,
        "mean_off_plane_norm": round(float(resid.mean()), 4),
    }
    side["points"] = rows
    (OUT_DIR / f"{NAME}.json").write_text(json.dumps(side, indent=2))
    print("both-ness by condition (mean over seeds):",
          {c: round(float(np.mean(v)), 3) for c, v in both_by_cond.items()})

    print(f"plane variance {var2:.3f}; cloud gaps {cloud_gaps}; spreads {cloud_stats}")
    for r in per_seed:
        print(r)
    print(f"wrote {OUT_DIR / NAME}.png/.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
