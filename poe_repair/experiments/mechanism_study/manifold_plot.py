"""Manifold density plot (plan 04 task 4): where does the chimera live?

Embeds five image sets with CLIP and lays them out on a cat↔dog axis:
  - pure cat / pure dog          → the two reference clouds
  - cat×dog plain PoE (λ=0)      → the broken compositions
  - cat×dog LoRA (λ=1)           → the fixed compositions
  - seed-9 training sweep        → the path from chimera to fixed

x-axis = projection onto (dog_centroid − cat_centroid); a chimera sits in the
middle, a clean two-animal image sits toward... both, so we also plot y = top
PCA direction orthogonal to the cat-dog axis. The seed-9 sweep is drawn as a
labelled path so you can watch it move as training proceeds.

Usage::

    python -m poe_repair.experiments.mechanism_study.manifold_plot --out <png>
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import torch

from poe_repair.experiments.residual_between_mono_and_poe.metrics import clip_image_embed

ATTN = Path("/datasets/mmolefe/poe_repair_min/outputs/attn_mechanism")
REF = ATTN / "manifold" / "references"
SWEEP = ATTN / "lora_train_sweep" / "a_cat__x__a_dog"


def _paths(pattern):
    return sorted(Path(p) for p in glob.glob(pattern))


def _embed(paths):
    if not paths:
        return np.zeros((0, 512))
    return clip_image_embed(paths).numpy()


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(prog="manifold_plot")
    ap.add_argument("--slug", default="a_cat__x__a_dog")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json-out", default=None,
                    help="also dump interactive-scatter JSON (points + thumbnails)")
    args = ap.parse_args(argv)

    sets = {
        "pure cat": _paths(str(REF / "pure_cat" / "seed_*.png")),
        "pure dog": _paths(str(REF / "pure_dog" / "seed_*.png")),
        "cat×dog λ=0 (broken)": _paths(
            str(ATTN / "plain_poe" / args.slug / "seed_*" / "sample.png")),
        "cat×dog λ=1 (fixed)": _paths(
            str(ATTN / "lora_lambda1" / args.slug / "seed_*" / "sample.png")),
    }
    # seed-9 training sweep, ordered by checkpoint step
    sweep_paths = []
    for d in _paths(str(SWEEP / "step_*")):
        m = re.match(r"step_(\d+)$", d.name)
        png = d / "seed_9" / "sample.png"
        if m and png.exists():
            sweep_paths.append((int(m.group(1)), png))
    sweep_paths.sort()

    emb = {k: _embed(v) for k, v in sets.items()}
    sweep_emb = _embed([p for _, p in sweep_paths])
    sweep_steps = [s for s, _ in sweep_paths]

    # cat↔dog axis from reference centroids
    cat_c = emb["pure cat"].mean(0)
    dog_c = emb["pure dog"].mean(0)
    axis = dog_c - cat_c
    axis = axis / (np.linalg.norm(axis) + 1e-9)

    # stack all points, remove the cat-dog axis, PCA the residual for y
    allpts = np.concatenate(
        [e for e in emb.values() if len(e)] + [sweep_emb], axis=0)
    mean = allpts.mean(0)
    centered = allpts - mean
    resid = centered - np.outer(centered @ axis, axis)
    # top PCA direction of the residual
    _, _, Vt = np.linalg.svd(resid, full_matrices=False)
    yaxis = Vt[0]

    def xy(e):
        c = e - mean
        return c @ axis, c @ yaxis

    COLORS = {
        "pure cat": "#3fb6c9", "pure dog": "#f2a03d",
        "cat×dog λ=0 (broken)": "#e8637a", "cat×dog λ=1 (fixed)": "#5bd6a4",
    }
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    for k, e in emb.items():
        if not len(e):
            continue
        x, y = xy(e)
        ax.scatter(x, y, s=48, alpha=.55, color=COLORS[k], label=k,
                   edgecolors="none")
        # centroid marker
        ax.scatter(x.mean(), y.mean(), s=220, marker="+",
                   color=COLORS[k], linewidths=2.2)

    # seed-9 sweep path
    if len(sweep_emb):
        sx, sy = xy(sweep_emb)
        ax.plot(sx, sy, "-", color="#c9b7f2", lw=1.6, alpha=.8, zorder=5)
        ax.scatter(sx, sy, s=30, color="#c9b7f2", zorder=6,
                   edgecolors="#2a2733")
        for i, st in enumerate(sweep_steps):
            if i == 0 or i == len(sweep_steps) - 1 or st in (17500, 20000):
                ax.annotate(f"{st//1000}k", (sx[i], sy[i]),
                            fontsize=8, color="#6c6579",
                            xytext=(4, 4), textcoords="offset points")
        ax.scatter([sx[0]], [sy[0]], s=140, marker="o", facecolors="none",
                   edgecolors="#e8637a", linewidths=2, zorder=7,
                   label="seed 9 start (chimera)")
        ax.scatter([sx[-1]], [sy[-1]], s=140, marker="o", facecolors="none",
                   edgecolors="#5bd6a4", linewidths=2, zorder=7,
                   label="seed 9 end (fixed)")

    ax.axhline(0, color="#2a2733", lw=.6)
    ax.set_xlabel("← more cat        cat↔dog CLIP axis        more dog →")
    ax.set_ylabel("orthogonal CLIP direction")
    ax.set_title("Where the chimera lives: cat×dog compositions on the cat–dog manifold")
    ax.legend(loc="best", fontsize=9, framealpha=.9)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"[manifold] wrote {out}")

    # a number to go with the picture: where does seed 9 sit on the cat-dog axis
    # across the sweep (should move toward the midpoint / dog as it separates)?
    if len(sweep_emb):
        sx, _ = xy(sweep_emb)
        print("\nseed-9 position on cat↔dog axis across training:")
        for st, x in zip(sweep_steps, sx):
            print(f"  step {st:>6}: x = {x:+.4f}")
        cat_x = xy(emb['pure cat'])[0].mean()
        dog_x = xy(emb['pure dog'])[0].mean()
        print(f"\n  (pure-cat centroid x={cat_x:+.4f}, pure-dog centroid x={dog_x:+.4f})")

    if args.json_out:
        import base64
        import io as _io
        import json as _json
        from PIL import Image as _Image

        def thumb(path, side=96):
            im = _Image.open(path).convert("RGB").resize((side, side), _Image.LANCZOS)
            b = _io.BytesIO(); im.save(b, format="JPEG", quality=70)
            return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()

        SETKEY = {"pure cat": "cat", "pure dog": "dog",
                  "cat×dog λ=0 (broken)": "broken", "cat×dog λ=1 (fixed)": "fixed"}
        pts = []
        for k, paths in sets.items():
            if not len(emb[k]):
                continue
            x, y = xy(emb[k])
            for i, p in enumerate(paths):
                pts.append({"set": SETKEY[k], "x": round(float(x[i]), 4),
                            "y": round(float(y[i]), 4), "img": thumb(p)})
        sweep = []
        if len(sweep_emb):
            sx, sy = xy(sweep_emb)
            for i, (stp, p) in enumerate(sweep_paths):
                sweep.append({"step": stp, "x": round(float(sx[i]), 4),
                              "y": round(float(sy[i]), 4), "img": thumb(p)})
        jout = {"points": pts, "sweep": sweep,
                "cat_x": round(float(xy(emb['pure cat'])[0].mean()), 4),
                "dog_x": round(float(xy(emb['pure dog'])[0].mean()), 4)}
        Path(args.json_out).write_text(_json.dumps(jout, separators=(",", ":")))
        print(f"[manifold] wrote json {args.json_out} "
              f"({Path(args.json_out).stat().st_size/1e6:.2f} MB, "
              f"{len(pts)} pts + {len(sweep)} sweep)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
