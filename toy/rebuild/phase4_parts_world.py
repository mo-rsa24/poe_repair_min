"""Phase 4 check: example 9's numbers.json recomputed, then every sampler on the parts world against exact draws.

    python -m rebuild.phase4_parts_world --out rebuild/out/phase4

Settings: (a) mosaic (cat tight on the head, dog tight on the body), (b) blend (both tight everywhere),
(c) co-occurrence (each animal in the left or the right half, 0.5 each). For each sampler, 2000 samples are
scored against 100,000 exact draws from the product:
  component-weight error: TV between the share of samples whose most responsible product component is k and
      the exact weight of k (0 when the mixture weights are right; only (c) has more than one component)
  projected TV: TV between the two half-means (head/body for a and b, left/right for c) on a 40 x 40 binning
  mean error: the largest gap between a half-mean of the samples and of the exact product
The perfect sampler is 2000 exact draws, so its numbers are the floor 2000 samples reach.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rebuild import parts_world as P, samplers as S
from rebuild.device import describe, pick_device
from rebuild.ideal import ABAR, BETAS, LEVELS

NUMBERS = "/home-mscluster/mmolefe/PhD/reduce-reuse-recycle/artifacts/ideas/poe-overlap-dial/example-9-parts-world/numbers.json"
NS = 2000
VAR_MIN = 0.005                                   # the tightest product variance (the blend)
V = np.array([1 - ABAR[t] + ABAR[t] * VAR_MIN for t in LEVELS])


def compare_numbers(ref, new, path=""):
    worst = 0.0
    for k, v in ref.items():
        if isinstance(v, dict):
            worst = max(worst, compare_numbers(v, new[k], path + k + "."))
        else:
            worst = max(worst, abs(v - new[k]) / max(1.0, abs(v)))
    return worst


def run_all(A, B, dev, rng, compose=lambda a, b: a + b):
    tgt = P.composed_target(A, B, ABAR, dev, compose)
    prod = P.mixture_product(A, B)
    out = {"perfect": prod.sample(NS, rng)[0]}
    out["reverse diffusion"] = S.reverse_diffusion(tgt, NS, P.D, BETAS, device=dev).cpu().numpy()
    out["ULA"] = S.ula(tgt, NS, P.D, 0.05 * V, device=dev).cpu().numpy()
    x, acc = S.mala(tgt, NS, P.D, 0.05 * V, device=dev); out["MALA"] = x.cpu().numpy()
    x, acc_t, steps = S.tuned_mala(tgt, NS, P.D, V, device=dev); out["MALA, tuned"] = x.cpu().numpy()
    out["UHA"] = S.uha(tgt, NS, P.D, 0.3 * np.sqrt(V), device=dev).cpu().numpy()
    accept = dict(mala=float(acc[-10:].mean()), mala_tuned=float(acc_t[-10:].mean()),
                  mala_tuned_sd_across_samples=float(acc_t[-10:].mean(0).std()))
    return prod, out, accept


def score(x, prod, ref, proj):
    x = x[np.isfinite(x).all(1)]
    we, share = P.weight_error(x, prod)
    hm = P.project(x, proj).mean(0); hr = P.project(ref, proj).mean(0)
    return dict(weight_error=float(we), share=share.tolist(), tv=float(P.tv_projected(x, ref, proj)),
                mean_error=float(np.abs(hm - hr).max()), half_means=hm.tolist(), kept=len(x))


def main(out):
    os.makedirs(out, exist_ok=True)
    dev = pick_device(); print(describe(dev))
    torch.manual_seed(0); rng = np.random.default_rng(0)

    ref = json.load(open(NUMBERS)); new = P.numbers()
    worst = compare_numbers(ref, new)
    print(f"numbers.json recomputed: largest relative difference {worst:.2e}  {'matches' if worst < 1e-9 else 'differs'}")
    json.dump(new, open(os.path.join(out, "numbers.json"), "w"), indent=1)

    report = dict(numbers_json=dict(largest_relative_difference=worst, mark="matches" if worst < 1e-9 else "differs"), settings={})
    cols = ["perfect", "reverse diffusion", "ULA", "MALA", "MALA, tuned", "UHA"]
    fig, ax = plt.subplots(3, 1 + len(cols), figsize=(2.3 * (1 + len(cols)), 7.5))
    for r, (name, (A, B, proj)) in enumerate(P.settings().items()):
        prod, smp, accept = run_all(A, B, dev, rng)
        exact = prod.sample(100000, np.random.default_rng(7))[0]
        rows = {c: score(smp[c], prod, exact, proj) for c in cols}
        report["settings"][name] = dict(weights=prod.w.tolist(), exact_half_means=P.project(exact, proj).mean(0).tolist(),
                                        accept=accept, samplers=rows)
        lab = ("head", "body") if name[0] in "ab" else ("left", "right")
        print(f"\n({name})  exact weights {np.round(prod.w, 3).tolist()}  exact {lab[0]}/{lab[1]} means "
              f"{np.round(P.project(exact, proj).mean(0), 3).tolist()}  MALA accept {accept['mala']:.2f}, tuned {accept['mala_tuned']:.2f}")
        for c in cols:
            x = rows[c]
            print(f"  {c:>17}: weight error {x['weight_error']:.3f}  projected TV {x['tv']:.3f}  mean error {x['mean_error']:.3f}  "
                  f"means {np.round(x['half_means'], 3).tolist()}  ({x['kept']} finite)")
        pe = P.project(exact[:4000], proj)
        ax[r, 0].scatter(*pe.T, s=1, c="tab:purple"); ax[r, 0].set_title("exact draws", fontsize=8)
        ax[r, 0].set_ylabel(f"({name})\n{lab[1]} mean", fontsize=8)
        for j, c in enumerate(cols, start=1):
            pc = P.project(smp[c][np.isfinite(smp[c]).all(1)], proj)
            ax[r, j].scatter(*pc.T, s=1, c="k", alpha=.4)
            ax[r, j].set_title(f"{c}\nTV {rows[c]['tv']:.2f} · weight err {rows[c]['weight_error']:.2f}", fontsize=8)
        for a in ax[r]:
            a.set_xlim(-1.5, 1.5); a.set_ylim(-1.5, 1.5); a.set_aspect("equal"); a.set_xlabel(f"{lab[0]} mean", fontsize=7)
    plt.suptitle("Parts world: each sample's two half-means, exact product against each sampler (2000 samples)", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(out, "parts-world-samplers-against-exact.png"), dpi=90); plt.close()
    json.dump(report, open(os.path.join(out, "phase4.json"), "w"), indent=1, default=float)


def outside_counts(out):
    """Count the projected points outside [-1.5, 1.5]^2, which P.tv_projected's histogram would drop.

    Replays main()'s sampling (same seeds and order) and, separately, the step file's step 7 build (1000 samples
    on the mosaic, run with fresh seeds 0 as that build's printed lines were). Adds the key
    "outside_projection_square" to phase4.json and changes nothing else in it; each run's TV is stored beside its
    count so the replay can be checked against the numbers it claims to reproduce."""
    dev = pick_device(); print(describe(dev))
    torch.manual_seed(0); rng = np.random.default_rng(0)
    path = os.path.join(out, "phase4.json"); report = json.load(open(path))

    def outside(x, proj):
        x = x[np.isfinite(x).all(1)]
        return int((np.abs(P.project(x, proj)) > 1.5).any(1).sum())

    runs = {}
    for name, (A, B, proj) in P.settings().items():
        prod, smp, _ = run_all(A, B, dev, rng)
        exact = prod.sample(100000, np.random.default_rng(7))[0]
        runs[name] = {"exact, 100000": dict(outside=outside(exact, proj), n=len(exact))}
        for c, x in smp.items():
            xs = x[np.isfinite(x).all(1)]
            runs[name][c] = dict(outside=outside(x, proj), n=len(x), tv_replay=float(P.tv_projected(xs, exact, proj)),
                                 tv_phase4=report["settings"][name]["samplers"][c]["tv"])
    torch.manual_seed(0); np.random.default_rng(0)
    D = P.D; HEAD, BODY = np.arange(0, 8), np.arange(8, 16)
    vcat = np.full(D, 1.0); vcat[HEAD] = 0.01; vdog = np.full(D, 1.0); vdog[BODY] = 0.01
    A_ = P.Mixture([1.0], np.ones(D), vcat); B_ = P.Mixture([1.0], -np.ones(D), vdog)
    tgt = P.composed_target(A_, B_, ABAR, dev)
    V7 = np.array([1 - ABAR[t] + ABAR[t] * 0.005 for t in range(len(ABAR) - 1, -1, -1)])
    exact = P.mixture_product(A_, B_).sample(100000, np.random.default_rng(7))[0]
    step7 = {}
    for name, s in [("UHA", S.uha(tgt, 1000, D, 0.3 * np.sqrt(V7), device=dev)),
                    ("tuned MALA", S.tuned_mala(tgt, 1000, D, V7, device=dev)[0]),
                    ("reverse diffusion", S.reverse_diffusion(tgt, 1000, D, BETAS, device=dev))]:
        s = s.cpu().numpy()
        step7[name] = dict(outside=outside(s, (HEAD, BODY)), n=len(s), head_mean=float(s[:, HEAD].mean()),
                           tv=float(P.tv_projected(s, exact, (HEAD, BODY))))
    step7["perfect, 1000"] = dict(outside=outside(exact, (HEAD, BODY)), n=len(exact),
                                  tv=float(P.tv_projected(exact[:1000], exact[1000:], (HEAD, BODY))))
    total = sum(v["outside"] for r in runs.values() for v in r.values()) + sum(v["outside"] for v in step7.values())
    gap = max(abs(v["tv_replay"] - v["tv_phase4"]) for r in runs.values() for c, v in r.items() if "tv_replay" in v)
    for name, r in runs.items():
        print(name, {c: v["outside"] for c, v in r.items()})
    print("step 7, 1000 samples:", {c: (v["outside"], round(v.get("head_mean", float("nan")), 3), round(v["tv"], 3)) for c, v in step7.items()})
    print(f"points outside in all runs: {total}; largest |TV replay - TV phase4.json| {gap:.2e}")
    report["outside_projection_square"] = dict(phase4_runs=runs, step7_1000_samples=step7, total_outside=total,
                                               max_tv_gap_replay_vs_phase4=gap)
    json.dump(report, open(path, "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rebuild/out/phase4")
    ap.add_argument("--outside", action="store_true",
                    help="only replay the sampling and add the outside-the-square counts to phase4.json")
    a = ap.parse_args()
    outside_counts(a.out) if a.outside else main(a.out)
