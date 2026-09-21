#!/usr/bin/env python
"""Embed the per-step frames and project them onto the cloud axes, for the
animated "where does each condition land" page.

Reads <frames-root>/frames_manifest.json (from where_each_condition_lands_trajectories.py).
Embeds every frame with the scorer's DINOv2 ViT-S/14 CLS embedder. The axes are
the same construction as the endpoint figure, fitted on this run's own final
frames (step 50): x is the unit direction from the cat-alone centroid to the
dog-alone centroid, y the unit direction from the midpoint of those two to the
joint-prompt centroid, orthogonalised against x. Every frame of every run is
then projected onto those two fixed axes, so a point's motion over the slider is
motion in one fixed plane.

Writes into the scene folder:
    public/data.json                 steps, conditions, seeds, per-run per-step (x, y),
                                     endpoint hulls, per-frame both-ness, axis provenance
    public/frames/<cond>/seed_<n>/step_<kk>.png   160 px thumbnails
and beside the endpoint figure:
    artifacts/results/where-does-each-condition-land/frames-dino-feats.npz
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.build_lora_inspector_mds_semantic import DinoEmbedder  # noqa: E402

FRAMES_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/where_each_condition_lands/frames")
SCENE = REPO_ROOT / "artifacts/scenes/where-each-condition-lands"
RESULTS = REPO_ROOT / "artifacts/results/where-does-each-condition-land"
THUMB = 160

CONDITIONS = {
    "solo_a":   {"label": "a cat alone",            "color": "#d95f02", "kind": "reference"},
    "solo_b":   {"label": "a dog alone",            "color": "#1b9e77", "kind": "reference"},
    "joint":    {"label": "\"a cat and a dog\"",    "color": "#7570b3", "kind": "reference"},
    "poe":      {"label": "PoE, no correction",     "color": "#111111", "kind": "probe"},
    "lora_1.0": {"label": "PoE + 1.0 × correction", "color": "#888888", "kind": "probe"},
    "lora_1.2": {"label": "PoE + 1.2 × correction", "color": "#e7298a", "kind": "probe"},
}
CLOUDS = ("solo_a", "solo_b", "joint")


def load_batch(paths):
    ims = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
           for p in paths]
    return torch.stack(ims)


def main() -> int:
    man = json.loads((FRAMES_ROOT / "frames_manifest.json").read_text())
    steps = man["frame_steps"]
    seeds = man["seeds"]
    runs = {(r["condition"], r["seed"]): r for r in man["runs"]}
    conds = [c for c in CONDITIONS if (FRAMES_ROOT / c).is_dir()]
    # a run skipped as already-done by a relaunch is absent from the manifest; rebuild it from disk
    for c in conds:
        for s in seeds:
            if (c, s) not in runs:
                d = FRAMES_ROOT / c / f"seed_{s}"
                pngs = sorted(d.glob("step_*.png"))
                if len(pngs) != len(steps):
                    raise SystemExit(f"missing run {c} seed {s}: {len(pngs)} frames on disk")
                runs[(c, s)] = {"frames": [{"step": int(q.stem.split("_")[1]), "png": str(q)} for q in pngs]}

    # ---- embed every frame ----------------------------------------------------
    order = [(c, s, f["step"], Path(f["png"])) for c in conds for s in seeds for f in runs[(c, s)]["frames"]]
    embedder = DinoEmbedder(device=torch.device("cpu"))
    feats = np.concatenate([
        embedder.embed_decoded_batch(load_batch([o[3] for o in order[i:i + 16]]))
        for i in range(0, len(order), 16)
    ])
    idx = {(c, s, k): i for i, (c, s, k, _) in enumerate(order)}
    print(f"embedded {len(order)} frames", flush=True)

    # ---- axes from this run's final frames ------------------------------------
    cent = {c: feats[[idx[(c, s, 50)] for s in seeds]].mean(0) for c in CLOUDS}
    origin = 0.5 * (cent["solo_a"] + cent["solo_b"])
    u1 = cent["solo_b"] - cent["solo_a"]; u1 /= np.linalg.norm(u1)
    u2 = cent["joint"] - origin; u2 -= (u2 @ u1) * u1; u2 /= np.linalg.norm(u2)
    xy = np.stack([(feats - origin) @ u1, (feats - origin) @ u2], 1)

    tracks = {c: {str(s): [[round(float(xy[idx[(c, s, k)], 0]), 4), round(float(xy[idx[(c, s, k)], 1]), 4)]
                           for k in steps] for s in seeds} for c in conds}
    both_final = {c: round(float(np.mean([xy[idx[(c, s, 50)], 1] for s in seeds])), 4) for c in conds}

    # ---- thumbnails into the scene's public folder ----------------------------
    out_frames = SCENE / "public/frames"
    for c, s, k, p in order:
        q = out_frames / c / f"seed_{s}" / f"step_{k:03d}.png"
        q.parent.mkdir(parents=True, exist_ok=True)
        if not q.exists():
            Image.open(p).convert("RGB").resize((THUMB, THUMB), Image.LANCZOS).save(q, optimize=True)

    data = {
        "pair": man["pair_slug"],
        "seeds": seeds,
        "steps": steps,
        "num_inference_steps": man["num_inference_steps"],
        "conditions": {c: CONDITIONS[c] for c in conds},
        "tracks": tracks,
        "axes": {
            "x": "which animal: cat-alone centroid (left) to dog-alone centroid (right)",
            "y": "both-ness: solo midpoint (0) toward the joint-prompt centroid",
            "unit": "DINOv2 cosine units",
            "fit_on": "the final frames (step 50) of this run's cat-alone, dog-alone and joint-prompt renders",
            "both_ness_final_mean": both_final,
            "joint_centroid_both_ness": round(float((cent["joint"] - origin) @ u2), 4),
        },
        "frame": {"px": THUMB, "path": "frames/{condition}/seed_{seed}/step_{step:03d}.png",
                  "meaning": "the model's running estimate of the finished image at that step (Tweedie mean), decoded"},
        "correction": {"checkpoint": man["checkpoint"], "rank": man["lora_rank"], "step": 30050,
                       "window": "all 50 steps", "held_out_on_this_pair": True},
        "sampler": {"steps": man["num_inference_steps"], "guidance": man["guidance_scale"], "size": 1024,
                    "init_latents": "pinned per seed from the training cache, shared by all six conditions"},
    }
    (SCENE / "public/data.json").write_text(json.dumps(data))
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESULTS / "frames-dino-feats.npz", feats=feats,
                        order=np.array([f"{c}|{s}|{k}" for c, s, k, _ in order]), u1=u1, u2=u2, origin=origin)
    print("both-ness at step 50 (mean over seeds):", both_final)
    print(f"wrote {SCENE / 'public/data.json'} and {len(order)} thumbnails")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
