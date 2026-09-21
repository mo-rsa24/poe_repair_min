#!/usr/bin/env python
"""The control energy of the true correction, read from the training cache (plan 19, scope 01).

Read the reverse run as a stochastic differential equation with the same marginals. The
correction r_t = eps_joint − eps_PoE is then a control drift, and Girsanov's theorem prices the
path-space KL between the joint-prompt run and the plain product-of-experts run as

    KL = sum_k 0.5 * gamma_k * ||r_k||^2,      gamma_k = beta_step_k / (1 − abar_k),

with beta_step_k = 1 − abar_k / abar_prev the diffusion integrated over one DDIM step. This
script computes that per step and per seed for cat × dog, held-out seeds 9 to 16, straight from
the cached predictions, on the CPU, and draws it. No render, no GPU.

Writes <out>/control-energy-over-steps.{png,json} and copies both into the results folder.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from correction_span_common import SEEDS, load_step, num_steps  # noqa: E402
from poe_repair.experiments.one_pair_one_seed.trainer import control_energy_weights  # noqa: E402

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
OUT_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/energy_penalty")
RESULTS = REPO_ROOT / "artifacts/results/does-charging-the-adapter-for-its-energy-sharpen-the-fix"
EARLY = range(0, 11)          # the window the composition is decided in (steps 0 to 10)
LATE = range(20, 50)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    from diffusers import DDIMScheduler
    ab = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler").alphas_cumprod

    n = num_steps(args.seeds[0])
    norm_sq = np.zeros((len(args.seeds), n))
    gamma = np.zeros(n)
    timesteps = np.zeros(n, dtype=int)
    for i, seed in enumerate(args.seeds):
        for k in range(n):
            s = load_step(seed, k)
            norm_sq[i, k] = float((s.r_t.double() ** 2).sum())
            if i == 0:
                timesteps[k] = int(s.timestep)
        print(f"[energy] seed {seed}: ||r||^2 step 0 {norm_sq[i, 0]:.1f}, step 10 {norm_sq[i, 10]:.1f}, "
              f"step 30 {norm_sq[i, 30]:.1f}, step 49 {norm_sq[i, 49]:.1f}", flush=True)
    gamma = control_energy_weights(timesteps.tolist(), ab, num_inference_steps=n).numpy().astype(np.float64)
    energy = 0.5 * gamma[None, :] * norm_sq                     # nats per step, per seed
    cumulative = np.cumsum(energy, axis=1)
    total = cumulative[:, -1]
    share_early = energy[:, list(EARLY)].sum(1) / total
    share_late = energy[:, list(LATE)].sum(1) / total

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    steps = np.arange(n)
    for row in np.sqrt(norm_sq):
        axes[0].plot(steps, row, color="#7570b3", lw=0.6, alpha=0.35)
    axes[0].plot(steps, np.sqrt(norm_sq).mean(0), color="#7570b3", lw=2.2, label="mean of 8 seeds")
    axes[0].set_title("size of the true correction ||r_t||\n(noise-prediction units, 4×128×128 latent)")
    axes[0].set_xlabel("DDIM step (0 = pure noise, 49 = last)"); axes[0].legend(fontsize=8)
    for row in energy:
        axes[1].plot(steps, row, color="#d7191c", lw=0.6, alpha=0.35)
    axes[1].plot(steps, energy.mean(0), color="#d7191c", lw=2.2, label="mean of 8 seeds")
    axes[1].plot(steps, 0.5 * gamma * norm_sq.mean(0) * 0 + gamma * energy.mean(0).max() / gamma.max(), color="grey",
                 lw=1.0, ls=":", label="Girsanov weight γ_k (rescaled)")
    axes[1].set_title("control energy per step  ½ γ_k ||r_k||²  (nats)\nγ_k = β_step / (1 − ᾱ_k)")
    axes[1].set_xlabel("DDIM step"); axes[1].legend(fontsize=8)
    for row in cumulative:
        axes[2].plot(steps, row, color="#111111", lw=0.6, alpha=0.35)
    axes[2].plot(steps, cumulative.mean(0), color="#111111", lw=2.2, label="mean of 8 seeds")
    axes[2].axvspan(EARLY.start, EARLY.stop - 1, color="#fee08b", alpha=0.35, lw=0,
                    label=f"steps 0 to 10: {100 * share_early.mean():.0f}% of the total")
    axes[2].set_title("cumulative energy = path-space KL(joint run ‖ PoE run)\nunder an SDE with the same marginals")
    axes[2].set_xlabel("DDIM step"); axes[2].legend(fontsize=8)
    fig.suptitle("cat × dog, held-out seeds 9 to 16, from the cached predictions on the plain PoE trajectory", fontsize=10)
    fig.tight_layout()
    png = out / "control-energy-over-steps.png"
    fig.savefig(png, dpi=140); plt.close(fig)

    payload = {
        "pair": "a_cat__x__a_dog", "seeds": list(args.seeds), "num_steps": n,
        "timesteps": timesteps.tolist(), "gamma": gamma.tolist(),
        "definition": "energy_k = 0.5 * gamma_k * ||r_k||^2 with gamma_k = (1 - abar_k/abar_prev) / (1 - abar_k); "
                      "cumulative sum is the Girsanov KL between the joint-prompt and plain-PoE reverse runs read as "
                      "SDEs with the same marginals; r_k = eps_joint - eps_PoE, both guided at 7.5, from the cache",
        "mean_over_seeds": {
            "r_norm_per_step": np.sqrt(norm_sq).mean(0).round(3).tolist(),
            "energy_per_step_nats": energy.mean(0).round(4).tolist(),
            "cumulative_nats": cumulative.mean(0).round(3).tolist(),
        },
        "per_seed": {str(s): {"total_nats": round(float(total[i]), 3),
                              "share_steps_0_to_10": round(float(share_early[i]), 4),
                              "share_steps_20_to_49": round(float(share_late[i]), 4)}
                     for i, s in enumerate(args.seeds)},
        "summary": {
            "total_nats_mean": round(float(total.mean()), 3),
            "total_nats_min": round(float(total.min()), 3), "total_nats_max": round(float(total.max()), 3),
            "share_steps_0_to_10_mean": round(float(share_early.mean()), 4),
            "share_steps_20_to_49_mean": round(float(share_late.mean()), 4),
            "peak_energy_step_mean_curve": int(np.argmax(energy.mean(0))),
        },
    }
    js = out / "control-energy-over-steps.json"
    js.write_text(json.dumps(payload, indent=1))
    RESULTS.mkdir(parents=True, exist_ok=True)
    for p in (png, js):
        shutil.copy2(p, RESULTS / p.name)
    print(json.dumps(payload["summary"], indent=1))
    print(f"[energy] wrote {png} and {js}; copied to {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
