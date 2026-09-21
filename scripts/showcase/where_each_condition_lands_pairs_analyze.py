#!/usr/bin/env python
"""Task 4.2 of plan 06 (scope 05): the landing figures on every held-out pair, judged
against the bar written in the plan before the render ran.

Reads <pairs-root>/<pair>/frames_manifest.json from where_each_condition_lands_pairs_render.py.
Per pair:
  * embeds the finished image of every (condition, seed) with the compose scorer's DINOv2
    ViT-S/14 CLS embedder, builds the cloud axes on that pair's own references (x: cat-like
    centroid to dog-like centroid, y: solo midpoint to joint centroid, orthogonalised), and
    writes the cloud-axes figure, the contact sheet and a sidecar;
  * embeds every saved frame, projects it onto the same axes, and writes the both-ness curves
    with the commit step per run (COMMIT_TOL) and the fork step per seed (FORK_MIN_DIST),
    same definitions as where_each_condition_lands_dynamics.py.
Then across pairs: the band test per pair (max PoE both-ness < min corrected both-ness), the
commit-step comparison, the pre-named exception, and the verdict per the plan's bar:
  support if BAND_AND_COMMIT_PASS_MIN of the counted pairs pass both halves,
  null if the bands overlap on BAND_OVERLAP_NULL_MIN or more counted pairs, inconclusive between.
Pairs whose two single-animal centroids sit closer than MIN_SOLO_GAP are excluded and named.

Writes to artifacts/results/where-does-each-condition-land/pairs/:
    <pair>/cloud-axes.png, <pair>/contact-sheet.png, <pair>/both-ness-over-steps.png, <pair>/sidecar.json
    bands-across-pairs.png, verdict.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.build_lora_inspector_mds_semantic import DinoEmbedder  # noqa: E402
from poe_repair.experiments.compose_scorer_validation.detection_scorer import (  # noqa: E402
    instance_score_to_dict, score_output_instances,
)

PAIRS_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/where_each_condition_lands/pairs")
PAIR_PROMPTS = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/pair_prompts.json")
OUT = REPO_ROOT / "artifacts/results/where-does-each-condition-land/pairs"
SEEDS = (9, 10, 11, 12, 13, 14, 15, 16)
REFS = ("solo_a", "solo_b", "joint")
ARMS = ("solo_a", "solo_b", "joint", "poe", "lora_1.2")
COLORS = {"solo_a": "#d95f02", "solo_b": "#1b9e77", "joint": "#7570b3", "poe": "#111111", "lora_1.2": "#e7298a"}
LABEL = {"solo_a": "A alone", "solo_b": "B alone", "joint": "joint prompt", "poe": "PoE, no correction",
         "lora_1.2": "PoE + 1.2 x correction"}

# ---- the bar, copied from the plan's "What has to pass before this runs" ----------------------
COMMIT_TOL = 0.10           # both-ness within this of its final value, for all later saved steps
FORK_MIN_DIST = 0.05        # plane distance at which PoE and corrected tracks count as apart
MIN_SOLO_GAP = 0.05         # cosine gap between the two single-animal centroids, else excluded
BAND_AND_COMMIT_PASS_MIN = 6
BAND_OVERLAP_NULL_MIN = 4
EXPECTED_EXCEPTION = "an_elephant__x__a_penguin"
SKIP_SCORER = False


def load01(p: Path) -> torch.Tensor:
    im = Image.open(p).convert("RGB")
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)


def embed(emb: DinoEmbedder, paths: list[Path]) -> np.ndarray:
    feats = []
    for i in range(0, len(paths), 8):
        batch = torch.stack([load01(p) for p in paths[i:i + 8]])
        feats.append(emb.embed_decoded_batch(batch))
    return np.concatenate(feats)


def cloud_axes(F: dict, seeds) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    cent = {c: np.stack([F[(c, s)] for s in seeds]).mean(0) for c in REFS}
    origin = 0.5 * (cent["solo_a"] + cent["solo_b"])
    u1 = cent["solo_b"] - cent["solo_a"]; u1 /= np.linalg.norm(u1)
    u2 = cent["joint"] - origin; u2 -= (u2 @ u1) * u1; u2 /= np.linalg.norm(u2)
    return origin, u1, u2, cent


def cosd(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    return float(1 - a @ b)


def commit_step(steps: list[int], y: list[float]) -> int:
    final = y[-1]
    for i, s in enumerate(steps):
        if all(abs(v - final) <= COMMIT_TOL for v in y[i:]):
            return s
    return steps[-1]


def analyze_pair(pair_dir: Path, emb: DinoEmbedder, out_dir: Path) -> dict | None:
    man_path = pair_dir / "frames_manifest.json"
    if man_path.exists():
        man = json.loads(man_path.read_text())
    else:
        # the render script writes every pair's manifest only at the very end; until then,
        # read the prompts from the pool and the saved steps from the files on disk
        prompts_all = json.loads(PAIR_PROMPTS.read_text())
        if pair_dir.name not in prompts_all:
            return None
        probe = sorted((pair_dir / "poe" / "seed_9").glob("step_*.png"))
        if not probe:
            return {"pair": pair_dir.name, "status": "no frames yet"}
        man = {"prompts": prompts_all[pair_dir.name],
               "frame_steps": [int(p.stem.split("_")[1]) for p in probe]}
    pr = man["prompts"]
    seeds = [s for s in SEEDS if all((pair_dir / c / f"seed_{s}" / "image_1024.png").exists() for c in ARMS)]
    if len(seeds) < 3:
        return {"pair": pair_dir.name, "status": f"only {len(seeds)} complete seeds"}
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- endpoints ----
    keys = [(c, s) for c in ARMS for s in seeds]
    paths = [pair_dir / c / f"seed_{s}" / "image_1024.png" for c, s in keys]
    feats = embed(emb, paths)
    F = {k: feats[i] for i, k in enumerate(keys)}
    origin, u1, u2, cent = cloud_axes(F, seeds)
    xy = {k: (float((F[k] - origin) @ u1), float((F[k] - origin) @ u2)) for k in keys}
    both = {c: [xy[(c, s)][1] for s in seeds] for c in ARMS}
    solo_gap = cosd(cent["solo_a"], cent["solo_b"])

    fig, ax = plt.subplots(figsize=(9, 7))
    for c in REFS:
        pts = np.array([xy[(c, s)] for s in seeds])
        if len(pts) >= 3:
            hull = ConvexHull(pts); poly = pts[hull.vertices]
            ax.fill(poly[:, 0], poly[:, 1], color=COLORS[c], alpha=0.12, lw=0)
        ax.scatter(pts[:, 0], pts[:, 1], s=40, color=COLORS[c], label=LABEL[c], zorder=3)
    for s in seeds:
        p, q = np.array(xy[("poe", s)]), np.array(xy[("lora_1.2", s)])
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=COLORS["lora_1.2"], lw=1.1), zorder=5)
        ax.scatter(*p, s=60, color=COLORS["poe"], marker="s", zorder=6)
        ax.scatter(*q, s=60, color=COLORS["lora_1.2"], marker="D", zorder=6)
        ax.annotate(f"s{s}", p, fontsize=7, xytext=(4, -8), textcoords="offset points")
    ax.scatter([], [], s=60, color=COLORS["poe"], marker="s", label=LABEL["poe"])
    ax.scatter([], [], s=60, color=COLORS["lora_1.2"], marker="D", label=LABEL["lora_1.2"])
    ax.axhline(0, color="#bbbbbb", lw=0.8); ax.axvline(0, color="#bbbbbb", lw=0.8)
    ax.set_xlabel(f"which animal: {pr['prompt_a']} centroid (left) to {pr['prompt_b']} centroid (right), DINOv2 cosine units")
    ax.set_ylabel("both-ness: solo midpoint (0) toward the joint-prompt centroid")
    ax.set_title(f"{pr['prompt_a']} x {pr['prompt_b']}, held-out seeds {seeds[0]} to {seeds[-1]}: where each condition lands\n"
                 f"rank-32 LoRA step 30050 at lambda 1.2; solo centroid gap {solo_gap:.2f}", fontsize=10)
    ax.legend(fontsize=8, loc="best"); ax.margins(0.15); fig.tight_layout()
    fig.savefig(out_dir / "cloud-axes.png", dpi=150); plt.close(fig)

    # contact sheet
    T = 192; pad = 6; lab = 18; left = 48
    W = left + len(ARMS) * (T + pad) + pad; H = lab + len(seeds) * (T + pad) + pad
    sheet = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(sheet)
    for j, c in enumerate(ARMS):
        d.text((left + pad + j * (T + pad), 4), LABEL[c].replace(" x ", " x "), fill="black")
    for i, s in enumerate(seeds):
        y0 = lab + pad + i * (T + pad); d.text((6, y0 + T // 2), f"s{s}", fill="black")
        for j, c in enumerate(ARMS):
            sheet.paste(Image.open(pair_dir / c / f"seed_{s}" / "step_050.png").convert("RGB").resize((T, T)),
                        (left + pad + j * (T + pad), y0))
    sheet.save(out_dir / "contact-sheet.png")

    # ---- dynamics from the saved frames ----
    steps = man["frame_steps"]
    tracks = {}
    for c in ("joint", "poe", "lora_1.2"):
        for s in seeds:
            fp = [pair_dir / c / f"seed_{s}" / f"step_{k:03d}.png" for k in steps]
            fe = embed(emb, fp)
            tracks[(c, s)] = [(float((f - origin) @ u1), float((f - origin) @ u2)) for f in fe]
    commit = {c: {s: commit_step(steps, [t[1] for t in tracks[(c, s)]]) for s in seeds} for c in ("joint", "poe", "lora_1.2")}
    fork = {}
    for s in seeds:
        a, b = np.array(tracks[("poe", s)]), np.array(tracks[("lora_1.2", s)])
        dist = np.linalg.norm(a - b, axis=1)
        idx = np.where(dist > FORK_MIN_DIST)[0]
        fork[s] = int(steps[idx[0]]) if len(idx) else None

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for c in ("joint", "poe", "lora_1.2"):
        ys = np.array([[t[1] for t in tracks[(c, s)]] for s in seeds])
        for row in ys:
            ax.plot(steps, row, color=COLORS[c], lw=0.8, alpha=0.3)
        ax.plot(steps, ys.mean(0), color=COLORS[c], lw=2.6, label=f"{LABEL[c]} (mean of {len(seeds)})")
    ax.axhline(float(np.mean(both["joint"])), color=COLORS["joint"], ls="--", lw=0.8)
    ax.axhline(0, color="#bbbbbb", lw=0.8)
    ax.set_xlabel("denoising step (0 = pure noise, 50 = finished image); one point per saved frame")
    ax.set_ylabel("both-ness of the running estimate")
    ax.set_title(f"{pr['prompt_a']} x {pr['prompt_b']}: both-ness over the 50 steps; median commit step "
                 f"joint {np.median(list(commit['joint'].values())):.0f}, corrected "
                 f"{np.median(list(commit['lora_1.2'].values())):.0f}, PoE {np.median(list(commit['poe'].values())):.0f}",
                 fontsize=10)
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out_dir / "both-ness-over-steps.png", dpi=150); plt.close(fig)

    # ---- post-hoc: the validated instance-count scorer on the same finished images ----
    # Not part of the bar. Recorded so a both-ness null on a near-identical pair can be read
    # beside what the compose rate's own rule says about the same pictures.
    inst = {}
    if not SKIP_SCORER:
        qa, qb = pr["prompt_a"].split()[-1], pr["prompt_b"].split()[-1]
        for c in ARMS:
            rows_ = []
            for s in seeds:
                d_ = instance_score_to_dict(score_output_instances(pair_dir / c / f"seed_{s}" / "image_1024.png", qa, qb))
                rows_.append({"seed": s, "n_instances": d_["n_instances"], "compose": d_["label"] == "compose"})
            inst[c] = {"compose_fraction": round(sum(r["compose"] for r in rows_) / len(rows_), 3), "per_seed": rows_}

    band_pass = max(both["poe"]) < min(both["lora_1.2"])
    commit_pass = np.median(list(commit["lora_1.2"].values())) <= np.median(list(commit["poe"].values()))
    corrected_below = float(np.mean(both["lora_1.2"])) < float(np.mean(both["poe"]))
    res = {
        "pair": pair_dir.name, "prompts": pr, "seeds": seeds, "status": "ok",
        "solo_centroid_cos_gap": round(solo_gap, 4),
        "counted": solo_gap >= MIN_SOLO_GAP and pair_dir.name != EXPECTED_EXCEPTION,
        "both_ness_by_condition": {c: [round(v, 4) for v in vs] for c, vs in both.items()},
        "both_ness_means": {c: round(float(np.mean(v)), 4) for c, v in both.items()},
        "band": {"max_poe": round(max(both["poe"]), 4), "min_corrected": round(min(both["lora_1.2"]), 4), "pass": bool(band_pass)},
        "commit_step": {c: {str(s): v for s, v in d_.items()} for c, d_ in commit.items()},
        "commit_median": {c: float(np.median(list(d_.values()))) for c, d_ in commit.items()},
        "commit_pass": bool(commit_pass),
        "fork_step_poe_vs_corrected": {str(s): v for s, v in fork.items()},
        "corrected_band_below_poe": bool(corrected_below),
        "instance_count_post_hoc": inst,
        "points": [{"condition": c, "seed": s, "xy": xy[(c, s)], "png": str(pair_dir / c / f"seed_{s}" / "image_1024.png")} for c, s in keys],
        "tracks": {f"{c}|{s}": v for (c, s), v in tracks.items()},
        "frame_steps": steps,
    }
    (out_dir / "sidecar.json").write_text(json.dumps(res, indent=2))
    ic = "" if not inst else (f"; instance-count compose fraction joint {inst['joint']['compose_fraction']:.2f} "
                              f"PoE {inst['poe']['compose_fraction']:.2f} corr {inst['lora_1.2']['compose_fraction']:.2f}")
    print(f"[{pair_dir.name}] band {'PASS' if band_pass else 'overlap'} (max PoE {max(both['poe']):.2f} vs min corr "
          f"{min(both['lora_1.2']):.2f}); commit medians joint {res['commit_median']['joint']:.0f} corr "
          f"{res['commit_median']['lora_1.2']:.0f} PoE {res['commit_median']['poe']:.0f}; solo gap {solo_gap:.2f}{ic}", flush=True)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-root", default=str(PAIRS_ROOT))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--skip-scorer", action="store_true", help="skip the post-hoc instance-count column")
    args = ap.parse_args(argv)
    global SKIP_SCORER
    SKIP_SCORER = bool(args.skip_scorer)
    root, out = Path(args.pairs_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    emb = DinoEmbedder(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    results = []
    for pair_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        r = analyze_pair(pair_dir, emb, out / pair_dir.name)
        if r is not None:
            results.append(r)
    ok = [r for r in results if r.get("status") == "ok"]
    counted = [r for r in ok if r["counted"]]
    excluded = [r["pair"] for r in ok if not r["counted"] and r["pair"] != EXPECTED_EXCEPTION]
    exception = next((r for r in ok if r["pair"] == EXPECTED_EXCEPTION), None)
    n_pass = sum(r["band"]["pass"] and r["commit_pass"] for r in counted)
    n_overlap = sum(not r["band"]["pass"] for r in counted)
    if n_pass >= BAND_AND_COMMIT_PASS_MIN:
        verdict = "support"
    elif n_overlap >= BAND_OVERLAP_NULL_MIN:
        verdict = "null"
    else:
        verdict = "inconclusive"
    surprises = [r["pair"] for r in ok if r["corrected_band_below_poe"]]

    # bands across pairs
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, r in enumerate(ok):
        for c, dx in (("poe", -0.15), ("lora_1.2", 0.15), ("joint", 0.0)):
            v = r["both_ness_by_condition"][c]
            ax.scatter([i + dx] * len(v), v, s=18, color=COLORS[c], alpha=0.8, zorder=3)
            ax.plot([i + dx - 0.08, i + dx + 0.08], [np.mean(v)] * 2, color=COLORS[c], lw=2.5)
    ax.set_xticks(range(len(ok)))
    ax.set_xticklabels([f"{r['prompts']['prompt_a']} x {r['prompts']['prompt_b']}" + ("\n(exception)" if r["pair"] == EXPECTED_EXCEPTION else "")
                        for r in ok], rotation=25, ha="right", fontsize=8)
    ax.axhline(0, color="#bbbbbb", lw=0.8)
    for c in ("poe", "lora_1.2", "joint"):
        ax.scatter([], [], color=COLORS[c], label=LABEL[c])
    ax.set_ylabel("both-ness of the finished image (per pair's own axes)")
    ax.set_title(f"held-out pairs: PoE band vs corrected band, one dot per seed, bar = mean; "
                 f"{n_pass} of {len(counted)} counted pairs pass both halves -> {verdict}", fontsize=10)
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "bands-across-pairs.png", dpi=150); plt.close(fig)

    (out / "verdict.json").write_text(json.dumps({
        "bar": {"COMMIT_TOL": COMMIT_TOL, "FORK_MIN_DIST": FORK_MIN_DIST, "MIN_SOLO_GAP": MIN_SOLO_GAP,
                "BAND_AND_COMMIT_PASS_MIN": BAND_AND_COMMIT_PASS_MIN, "BAND_OVERLAP_NULL_MIN": BAND_OVERLAP_NULL_MIN,
                "EXPECTED_EXCEPTION": EXPECTED_EXCEPTION},
        "verdict": verdict, "n_counted": len(counted), "n_pass_both_halves": n_pass, "n_band_overlap": n_overlap,
        "excluded_solo_gap_too_small": excluded, "surprises_corrected_below_poe": surprises,
        "exception": None if exception is None else {k: exception[k] for k in ("pair", "both_ness_means", "band", "commit_median")},
        "per_pair": [{**{k: r[k] for k in ("pair", "counted", "solo_centroid_cos_gap", "both_ness_means", "band", "commit_median", "commit_pass",
                                           "fork_step_poe_vs_corrected", "corrected_band_below_poe")},
                      "instance_count_compose_fraction_post_hoc": {c: v["compose_fraction"] for c, v in r["instance_count_post_hoc"].items()}}
                     for r in ok],
        "incomplete": [r for r in results if r.get("status") != "ok"],
    }, indent=2))
    print(f"[verdict] {verdict}: {n_pass}/{len(counted)} pass both halves, {n_overlap} overlap; excluded {excluded}; surprises {surprises}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
