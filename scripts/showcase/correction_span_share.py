#!/usr/bin/env python
"""Rung 2 of "what the correction is made of": how much of r_t lies inside the span of the
three predictions PoE already has, per step and seed, with the verdict read from the constants
in correction_span_common.py.

Writes into artifacts/results/what-the-correction-is-made-of/:
    orthogonal-share-over-steps.png   three panels against denoising step (0 = noise, 49 = last):
                                      (a) orthogonal share of ||r_t||^2, mean over seeds 9 to 16
                                          with the min-to-max band, steps 0 to 10 shaded, the two
                                          bar lines; (b) the same share for the joint guidance
                                          direction g (eps_j - eps_u) alone; (c) the norms of r_t,
                                          its orthogonal part, and eps_PoE
    orthogonal-share-over-steps.json  one row per (seed, step): the shares, the three
                                      least-squares coefficients, the norms; the early-window
                                      statistic; the verdict; the bar block
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import correction_span_common as C  # noqa: E402


def main() -> int:
    rows = []
    t0 = time.time()
    for seed in C.SEEDS:
        n = C.num_steps(seed)
        for k in range(n):
            s = C.load_step(seed, k)
            B = C.span_basis(s)
            r = C.project(B, s.r_t)
            j = C.project(B, C.GUIDANCE * (s.eps_j - s.eps_u))
            p = C.project(B, s.eps_poe)              # sanity: must be entirely in-span
            rows.append({
                "seed": seed, "step": k, "timestep": s.timestep,
                "ortho_share_r": r["ortho_share"], "in_share_r": r["in_share"],
                "coef_r_on_a_minus_u": float(r["coef"][0]), "coef_r_on_b_minus_u": float(r["coef"][1]),
                "coef_r_on_u": float(r["coef"][2]),
                "ortho_share_joint_dir": j["ortho_share"],
                "norm_r": r["norm"], "norm_r_out": float(r["out"].norm()), "norm_r_in": float(r["in"].norm()),
                "norm_joint_dir": j["norm"], "norm_eps_poe": p["norm"],
                "norm_eps_u": float(s.eps_u.norm()),
                "norm_a_minus_u": float((s.eps_a - s.eps_u).norm()), "norm_b_minus_u": float((s.eps_b - s.eps_u).norm()),
                "norm_j_minus_u": float((s.eps_j - s.eps_u).norm()),
                "cos_a_minus_u_vs_b_minus_u": C.cosine(s.eps_a - s.eps_u, s.eps_b - s.eps_u),
                "cos_j_minus_u_vs_a_minus_u": C.cosine(s.eps_j - s.eps_u, s.eps_a - s.eps_u),
                "cos_j_minus_u_vs_b_minus_u": C.cosine(s.eps_j - s.eps_u, s.eps_b - s.eps_u),
                "in_share_eps_poe_sanity": p["in_share"],
            })
        print(f"seed {seed}: {n} steps ({time.time() - t0:.0f}s)", flush=True)

    steps = sorted({r["step"] for r in rows})
    def mat(key):
        return np.array([[next(r[key] for r in rows if r["seed"] == sd and r["step"] == k) for k in steps]
                         for sd in C.SEEDS])
    ortho = mat("ortho_share_r")
    per_seed_early = ortho[:, [k for k in steps if k in C.EARLY_STEPS]].mean(1)
    early_stat = float(per_seed_early.mean())
    verdict = C.verdict(early_stat)
    sanity_min = min(r["in_share_eps_poe_sanity"] for r in rows)

    summary = {
        "early_window_orthogonal_share_mean_over_seeds": early_stat,
        "early_window_orthogonal_share_per_seed": {int(sd): float(v) for sd, v in zip(C.SEEDS, per_seed_early)},
        "orthogonal_share_r_mean_per_step": ortho.mean(0).round(4).tolist(),
        "orthogonal_share_r_min_per_step": ortho.min(0).round(4).tolist(),
        "orthogonal_share_r_max_per_step": ortho.max(0).round(4).tolist(),
        "orthogonal_share_joint_dir_mean_per_step": mat("ortho_share_joint_dir").mean(0).round(4).tolist(),
        "coef_r_on_a_minus_u_mean_per_step": mat("coef_r_on_a_minus_u").mean(0).round(3).tolist(),
        "coef_r_on_b_minus_u_mean_per_step": mat("coef_r_on_b_minus_u").mean(0).round(3).tolist(),
        "coef_r_on_u_mean_per_step": mat("coef_r_on_u").mean(0).round(3).tolist(),
        "norm_r_mean_per_step": mat("norm_r").mean(0).round(2).tolist(),
        "norm_r_out_mean_per_step": mat("norm_r_out").mean(0).round(2).tolist(),
        "norm_eps_poe_mean_per_step": mat("norm_eps_poe").mean(0).round(2).tolist(),
        "norm_a_minus_u_mean_per_step": mat("norm_a_minus_u").mean(0).round(2).tolist(),
        "norm_b_minus_u_mean_per_step": mat("norm_b_minus_u").mean(0).round(2).tolist(),
        "norm_j_minus_u_mean_per_step": mat("norm_j_minus_u").mean(0).round(2).tolist(),
        "cos_a_minus_u_vs_b_minus_u_mean_per_step": mat("cos_a_minus_u_vs_b_minus_u").mean(0).round(3).tolist(),
        "cos_j_minus_u_vs_a_minus_u_mean_per_step": mat("cos_j_minus_u_vs_a_minus_u").mean(0).round(3).tolist(),
        "cos_j_minus_u_vs_b_minus_u_mean_per_step": mat("cos_j_minus_u_vs_b_minus_u").mean(0).round(3).tolist(),
        "joint_guided_in_span_coef_on_a_minus_u_mean_per_step": (mat("coef_r_on_a_minus_u") + C.GUIDANCE).mean(0).round(3).tolist(),
        "joint_guided_in_span_coef_on_b_minus_u_mean_per_step": (mat("coef_r_on_b_minus_u") + C.GUIDANCE).mean(0).round(3).tolist(),
        "eps_poe_in_span_share_min_sanity": sanity_min,
        "verdict": verdict,
    }
    C.write_json(C.RESULTS / "orthogonal-share-over-steps.json", {
        "pair": C.PAIR, "seeds": list(C.SEEDS), "guidance": C.GUIDANCE, "steps": steps,
        "definitions": {
            "r_t": "guided joint prediction minus the PoE prediction at the same cached state, = g (eps_j - eps_a - eps_b + eps_u)",
            "span": "alpha eps_u + beta (eps_a - eps_u) + gamma (eps_b - eps_u); every per-step guidance re-weighting lies inside it",
            "ortho_share_r": "1 - ||proj_span r_t||^2 / ||r_t||^2",
            "ortho_share_joint_dir": "the same share for g (eps_j - eps_u) alone; its orthogonal part equals r_t's exactly",
            "coef_r_on_*": "least-squares coefficients of r_t on [eps_a - eps_u, eps_b - eps_u, eps_u]; r_t contains -g on the first two by construction",
            "early_window_statistic": "mean over seeds of the per-seed mean of ortho_share_r over EARLY_STEPS",
            "joint_guided_in_span_coef_on_*": "coef_r + g: the weight the guided joint prediction puts on each expert direction inside the span (g = 7.5 is what PoE puts there)",
        },
        "bar": C.bar_block(), "summary": summary, "rows": rows,
    })

    # ---- figure ------------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    x = np.array(steps)
    ax = axes[0]
    ax.fill_between(x, ortho.min(0), ortho.max(0), color="#e7298a", alpha=0.18, label="min to max over seeds 9 to 16")
    ax.plot(x, ortho.mean(0), color="#e7298a", lw=2.6, label="mean over the 8 seeds")
    ax.axvspan(min(C.EARLY_STEPS), max(C.EARLY_STEPS), color="#ffd92f", alpha=0.25, lw=0)
    ax.axhline(C.ORTHO_SHARE_REWEIGHT_IMPOSSIBLE, color="#444", ls="--", lw=1)
    ax.axhline(C.ORTHO_SHARE_REWEIGHT_CANDIDATE, color="#444", ls=":", lw=1)
    ax.text(49, C.ORTHO_SHARE_REWEIGHT_IMPOSSIBLE + 0.01, "above: no re-weighting could supply it (bar 0.5)", ha="right", fontsize=8, color="#444")
    ax.text(49, C.ORTHO_SHARE_REWEIGHT_CANDIDATE - 0.04, "below: re-weighting is a candidate fix (bar 0.25)", ha="right", fontsize=8, color="#444")
    ax.text(max(C.EARLY_STEPS) + 0.5, 0.02, f"steps 0 to 10, mean {early_stat:.3f}", fontsize=9)
    ax.set_ylim(0, 1); ax.set_ylabel("orthogonal share of ||r_t||^2\n(fraction outside the experts' span)")
    ax.set_title("cat x dog, held-out seeds 9 to 16: how much of the correction r_t no re-weighting of\n"
                 "{eps_u, eps_a - eps_u, eps_b - eps_u} could reach, per denoising step (from the training cache)", fontsize=11)
    ax.legend(loc="center right", fontsize=9)

    ax = axes[1]
    oj = mat("ortho_share_joint_dir")
    ax.fill_between(x, oj.min(0), oj.max(0), color="#7570b3", alpha=0.18)
    ax.plot(x, oj.mean(0), color="#7570b3", lw=2.6, label="joint guidance direction g (eps_j - eps_u) alone")
    ax.axvspan(min(C.EARLY_STEPS), max(C.EARLY_STEPS), color="#ffd92f", alpha=0.25, lw=0)
    ax.set_ylim(0, 1); ax.set_ylabel("orthogonal share of\n||g (eps_j - eps_u)||^2")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[2]
    for key, col, lab in (("norm_r", "#e7298a", "||r_t||"), ("norm_r_out", "#a50f15", "||r_t orthogonal part||"),
                          ("norm_eps_poe", "#111111", "||eps_PoE||"), ("norm_joint_dir", "#7570b3", "||g (eps_j - eps_u)||")):
        m = mat(key)
        ax.plot(x, m.mean(0), color=col, lw=2, label=lab)
        ax.fill_between(x, m.min(0), m.max(0), color=col, alpha=0.12)
    ax.axvspan(min(C.EARLY_STEPS), max(C.EARLY_STEPS), color="#ffd92f", alpha=0.25, lw=0)
    ax.set_ylabel("L2 norm over the 65,536 latent numbers"); ax.set_xlabel("denoising step (0 = pure noise, 49 = last cached step)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(C.RESULTS / "orthogonal-share-over-steps.png", dpi=160)
    print("early-window orthogonal share, mean over seeds:", round(early_stat, 4))
    print("per seed:", summary["early_window_orthogonal_share_per_seed"])
    print("eps_PoE in-span sanity (min over rows):", sanity_min)
    print("verdict:", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
