#!/usr/bin/env python
"""The trajectory-divergence read over all 19 dose pairs, plus the difficulty
correlation.

The dose cache saved 20-step trajectories for every pair at every λ, but only
the noisy latents ``x_t``, not the per-step estimates of the finished picture.
Under DDIM the estimate is recoverable in closed form from two consecutive
latents and the schedule: with ``A_i = sqrt(alpha_bar_i)`` and
``B_i = sqrt(1 - alpha_bar_i)``,

    x_{i+1} = (B_{i+1}/B_i) x_i + (A_{i+1} - A_i B_{i+1}/B_i) x0_i
    =>  x0_i = (x_{i+1} - (B_{i+1}/B_i) x_i) / (A_{i+1} - A_i B_{i+1}/B_i)

Before touching the dose cache, the formula is validated against ground truth:
the cross cache saved both ``trajectories`` and true ``x0_estimates`` for the
same sampler, so the recovered series must match the saved one to within
``MAX_RECOVERY_REL_ERR`` at every step past the first, or the script stops.

Then, per pair and seed, the two arms (λ=0, pure PoE, against λ=1, which
reproduces Mono) are read exactly as in ``trajectory_divergence.py``: decode
each recovered x̂₀ with the VAE, embed with CLIP and DINOv2, and place each
step on the axis from the cell's own PoE endpoint to its Mono endpoint.
Frames are embedded in memory and never written to disk.

The difficulty claim, with its bar written here before the run: a pair whose
arms separate more slowly early should be a harder pair. Predictor: mean
separation over the early window (timesteps >= EARLY_T, matching steps 0..9
of the 50-step runs). Outcome: the pair's area under its real-correction
compose-rate-against-λ curve from ``dose_curves.json``. Support needs
Spearman |rho| >= SUPPORT_ABS_RHO in CLIP space; below KILL_ABS_RHO the
predictor claim is dead. Sign is not pre-committed, only strength: either
direction of monotone association counts, and the sign is reported.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/dose_trajectory_divergence.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poe_repair import paths

DOSE_ROOT = paths.resolve(paths.HOW_MUCH_CORRECTION_IS_NEEDED) / "pairs"
DOSE_CURVES = paths.resolve(paths.HOW_MUCH_CORRECTION_IS_NEEDED) / "dose_curves.json"
CROSS_VALIDATE_CELL = (
    paths.resolve(paths.SAMPLES_AS_THE_WINDOW_MOVES_ONE_STEP_AT_A_TIME)
    / "pairs/a_cat__x__a_dog/seed_9/call__rall"
)
OUT_DIR = paths.resolve(paths.CACHE_ANALYSES) / "trajectory_divergence"

ARM_OFF = "teacher_residual_const_lam000"   # pure PoE
ARM_ON = "teacher_residual_const_lam100"    # reproduces Mono
# Seeds are discovered per pair rather than fixed: 8 pairs saved trajectories
# for seeds 9..12, the other 11 for seed 1 only. Up to this many are read per
# pair, lowest seed number first, and the seeds used are recorded per cell.
MAX_SEEDS = 2

# Bars, written before the first run of this script.
MAX_RECOVERY_REL_ERR = 0.05  # recovered x0 vs saved x0, steps 1.., median
EARLY_T = 750.0              # early window: timesteps >= this (first ~5 of 20)
SUPPORT_ABS_RHO = 0.5        # Spearman, CLIP space, per-pair
KILL_ABS_RHO = 0.3


def recover_x0(traj: torch.Tensor, timesteps: list[float],
               alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """``traj`` is [T+1, 1, C, H, W] noisy latents, pre-step; returns [T, ...]
    x̂₀ at each step. The final stored latent is x̂₀ of the last step already
    (the schedule ends at alpha_bar = 1), and is used as such."""
    T = len(timesteps)
    abar = torch.tensor(
        [float(alphas_cumprod[int(t)]) for t in timesteps] + [1.0])
    A, B = abar.sqrt(), (1 - abar).sqrt()
    out = []
    for i in range(T):
        if B[i + 1] < 1e-6:
            out.append(traj[i + 1].float())
            continue
        ratio = B[i + 1] / B[i]
        denom = A[i + 1] - A[i] * ratio
        out.append((traj[i + 1].float() - ratio * traj[i].float()) / denom)
    return torch.stack(out)


def validate_recovery(alphas_cumprod: torch.Tensor) -> float:
    blob = torch.load(CROSS_VALIDATE_CELL / "latent_trajectory.pt",
                      map_location="cpu")
    rec = recover_x0(blob["trajectories"].float(), blob["timesteps"],
                     alphas_cumprod)
    ref = blob["x0_estimates"].float()
    # Step 0's recovery divides by a near-degenerate denominator at the very
    # start of some schedules; the check runs on steps 1 onward, and step 0 is
    # excluded from the early window only if this validation says it is bad.
    errs = [float((rec[i] - ref[i]).norm() / ref[i].norm().clamp(min=1e-8))
            for i in range(rec.shape[0])]
    med = float(np.median(errs[1:]))
    print(f"x0 recovery vs saved ground truth (cross cat×dog seed_9): "
          f"median rel err steps 1..{len(errs)-1} = {med:.4f}, "
          f"step 0 = {errs[0]:.4f}")
    return med


def pair_auc_from_scores() -> dict[str, float]:
    d = json.loads(DOSE_CURVES.read_text())
    lambdas = d["lambdas"]
    by_pair: dict[str, dict[float, list[int]]] = {}
    for s in d["scores"]:
        if s["row"] != "oracle":
            continue
        by_pair.setdefault(s["pair"], {}).setdefault(
            float(s["lambda"]), []).append(int(s["compose"]))
    out = {}
    for pair, at in by_pair.items():
        if sorted(at) != sorted(float(l) for l in lambdas):
            print(f"  {pair}: missing λ points, dropped from correlation",
                  file=sys.stderr)
            continue
        rates = [float(np.mean(at[float(l)])) for l in lambdas]
        out[pair] = float(np.trapz(rates, lambdas))
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx * rx).sum() * (ry * ry).sum()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    from poe_repair._sdxl.runtime import decode_latents
    from poe_repair.experiments.residual_between_mono_and_poe.metrics import (
        clip_image_embed,
    )
    from poe_repair.run import make_ctx
    from scripts.decode_trajectory_frames import to_uint8
    from scripts.trajectory_divergence import dino_embed_paths

    ctx = make_ctx()
    abar = ctx.scheduler.alphas_cumprod

    med_err = validate_recovery(abar)
    if med_err > MAX_RECOVERY_REL_ERR:
        print(f"recovery error {med_err:.4f} > bar {MAX_RECOVERY_REL_ERR}; "
              f"stopping before the dose read", file=sys.stderr)
        return 1

    tmp = out_dir / "dose_frames_tmp"
    tmp.mkdir(exist_ok=True)

    def embed_frames(x0: torch.Tensor, tag: str):
        paths = []
        for i in range(x0.shape[0]):
            img = decode_latents(ctx.models, x0[i].to(ctx.device, ctx.dtype))
            p = tmp / f"{tag}_{i:02d}.png"
            Image.fromarray(to_uint8(img.cpu())).resize((512, 512)).save(p)
            paths.append(p)
        clip = torch.cat([clip_image_embed(paths[i:i + 32])
                          for i in range(0, len(paths), 32)]).numpy()
        dino = dino_embed_paths(paths)
        for p in paths:
            p.unlink()
        return clip, dino

    cells: list[dict] = []
    pairs = sorted(p for p in DOSE_ROOT.iterdir() if p.is_dir())
    for pair_dir in pairs:
        with_both = sorted(
            (s for s in pair_dir.iterdir() if s.name.startswith("seed_")
             and all((s / a / "latent_trajectory.pt").exists()
                     for a in (ARM_OFF, ARM_ON))),
            key=lambda s: int(s.name.split("_")[1]))
        if not with_both:
            print(f"  {pair_dir.name}: no seed with both arms' trajectories, "
                  f"skipped", file=sys.stderr)
            continue
        for cell in with_both[:MAX_SEEDS]:
            seed = cell.name
            arms = {}
            timesteps = None
            for key, arm in (("off", ARM_OFF), ("on", ARM_ON)):
                blob = torch.load(cell / arm / "latent_trajectory.pt",
                                  map_location="cpu")
                ts = [float(t) for t in blob["timesteps"]]
                timesteps = ts
                x0 = recover_x0(blob["trajectories"].float(), ts, abar)
                arms[key] = embed_frames(x0, f"{pair_dir.name}_{seed}_{key}")
            row = {"pair": pair_dir.name, "seed": seed,
                   "timesteps": timesteps}
            for si, space in enumerate(("clip", "dino")):
                a, b = arms["on"][si], arms["off"][si]
                e_off, e_on = b[-1], a[-1]
                axis = e_on - e_off
                axis_len2 = float((axis * axis).sum())
                sep = ((a - e_off) @ axis - (b - e_off) @ axis) / axis_len2
                early = [s for t, s in zip(timesteps, sep) if t >= EARLY_T]
                row[space] = {
                    "separation": [round(float(v), 4) for v in sep],
                    "early_sep_mean": round(float(np.mean(early)), 4),
                }
            cells.append(row)
            print(f"  {pair_dir.name}/{seed}: early sep "
                  f"clip {row['clip']['early_sep_mean']:+.3f} "
                  f"dino {row['dino']['early_sep_mean']:+.3f}", flush=True)
    tmp.rmdir()

    auc = pair_auc_from_scores()
    per_pair: dict[str, dict] = {}
    for row in cells:
        per_pair.setdefault(row["pair"], {"clip": [], "dino": []})
        for sp in ("clip", "dino"):
            per_pair[row["pair"]][sp].append(row[sp]["early_sep_mean"])

    corr = {}
    shared = sorted(set(per_pair) & set(auc))
    dropped = sorted(set(per_pair) ^ set(auc))
    if dropped:
        print(f"pairs without both a curve and an AUC, excluded: {dropped}")
    y = np.array([auc[p] for p in shared])
    for sp in ("clip", "dino"):
        x = np.array([float(np.mean(per_pair[p][sp])) for p in shared])
        rho = spearman(x, y)
        verdict = ("SUPPORT" if abs(rho) >= SUPPORT_ABS_RHO
                   else "DEAD" if abs(rho) < KILL_ABS_RHO
                   else "INCONCLUSIVE")
        corr[sp] = {"spearman_rho": round(rho, 4), "n_pairs": len(shared),
                    "verdict": verdict}
        print(f"{sp}: Spearman rho(early separation, pair AUC) = {rho:+.3f} "
              f"over {len(shared)} pairs -> {verdict} "
              f"(support >= {SUPPORT_ABS_RHO}, dead < {KILL_ABS_RHO})")

    blob = {
        "arms": {"on": ARM_ON, "off": ARM_OFF},
        "max_seeds_per_pair": MAX_SEEDS,
        "x0_recovery_median_rel_err": round(med_err, 5),
        "early_window_timestep_min": EARLY_T,
        "bars": {"support_abs_rho": SUPPORT_ABS_RHO,
                 "kill_abs_rho": KILL_ABS_RHO},
        "pair_auc": {p: round(v, 4) for p, v in auc.items()},
        "correlation": corr,
        "cells": cells,
    }
    out_json = out_dir / "dose_divergence.json"
    out_json.write_text(json.dumps(blob, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, space in zip(axes[:2], ("clip", "dino")):
        for row in cells:
            ax.plot(row["timesteps"], row[space]["separation"],
                    lw=0.9, alpha=0.6, color="C0")
        ax.invert_xaxis()
        ax.axvspan(1000, EARLY_T, color="0.9", zorder=0)
        ax.set_title(f"{space.upper()}: arm separation, 19 pairs x 2 seeds")
        ax.set_xlabel("timestep (noise → image, right to left)")
        ax.set_ylabel("pos(on) − pos(off), 1 = full axis")
    ax = axes[2]
    xs = [float(np.mean(per_pair[p]["clip"])) for p in shared]
    ax.scatter(xs, y, s=18)
    for p, xv, yv in zip(shared, xs, y):
        ax.annotate(p.replace("a_", "").replace("an_", ""), (xv, yv),
                    fontsize=5, alpha=0.7)
    ax.set_xlabel(f"early separation (CLIP), timesteps >= {EARLY_T:.0f}")
    ax.set_ylabel("pair AUC of real-correction dose curve")
    ax.set_title(f"rho = {corr['clip']['spearman_rho']:+.3f} "
                 f"({corr['clip']['verdict']})")
    fig.tight_layout()
    fig.savefig(out_dir / "dose_divergence_eyeball.png", dpi=150)
    print(f"wrote {out_json} and dose_divergence_eyeball.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
