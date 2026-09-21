#!/usr/bin/env python
"""A per-pair interaction-strength number from the cache alone (use 1 of the Mitra route).

Mitra et al. 2026 (arXiv 2605.22596, Theorem 1) bound the score-addition error by
    || s_joint - s_comp || <= 2 sqrt(G M),
with G the bound on the interaction log-ratio g = log p(z|x) / prod_i p(z_i|x) (the pointwise
mutual information of the two concepts given the state) and M the bound on its Hessian.
This project's r_t = eps_J - eps_PoE is that error in eps units: s_joint - s_comp = -r_t / sigma_t.
Turned around, every cached (x_t, t) gives a lower bound on the interaction product:
    G M  >=  || r_t / sigma_t ||^2 / 4,      sigma_t = sqrt(1 - alpha_bar_t).

Reads every cached cell of the listed pairs, computes the bound per step, takes the median
over seeds, and draws two panels: the bound against denoising step (one curve per pair, log y),
and the early-window (steps 0-9) bound per pair as sorted bars, both coloured by the pair's
plain-PoE fail rate from fail_rate.md (8 seeds, instance-count scorer) where one exists.

Writes <out>/interaction_bound.json and <out>/interaction_bound.png.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from poe_repair.runtime import load_ddim_scheduler  # noqa: E402
from poe_repair.training_cache import DEFAULT_CACHE_ROOT  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
FAIL_MD = REPO / "artifacts/results/does-the-fix-reach-unseen-pairs/fail_rate.md"
EARLY = range(0, 10)


def fail_rates() -> dict[str, float]:
    out = {}
    for line in FAIL_MD.read_text().splitlines():
        m = re.match(r"\|\s*(\S+)\s*\|\s*\S+\s*\|\s*([0-9.]+)\s*\(", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--min-seeds", type=int, default=3)
    ap.add_argument("--extra-heldout", nargs="*", default=[
        "a_leopard__x__a_jaguar", "a_frog__x__a_toad", "an_eagle__x__a_hawk", "a_seal__x__a_walrus",
        "a_goose__x__a_swan", "a_cow__x__a_buffalo", "a_cat__x__a_dog", "an_elephant__x__a_penguin",
        "a_butterfly__x__a_flower_meadow"])
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(args.cache_root)
    gs = args.guidance_scale
    sched = load_ddim_scheduler("stabilityai/stable-diffusion-xl-base-1.0")
    ac = sched.alphas_cumprod
    fr = fail_rates()

    cells = []
    for pdir in sorted((root / "train").iterdir()):
        seeds = sorted(pdir.glob("seed_*"))
        if len(seeds) >= args.min_seeds:
            cells += [(pdir.name, "train", s) for s in seeds]
    for slug in args.extra_heldout:
        pdir = root / "heldout" / slug
        if pdir.exists():
            cells += [(slug, "heldout", s) for s in sorted(pdir.glob("seed_*"))]

    per_pair: dict[str, dict[int, list[float]]] = {}
    t0 = time.time()
    for slug, split, sdir in cells:
        for f in sorted((sdir / "residuals").glob("step_*.pt")):
            r = torch.load(f, map_location="cpu", weights_only=False)
            si, ts = int(r["step_index"]), int(r["timestep"])
            rt = gs * (r["eps_j_raw"].float() - r["eps_a_raw"].float() - r["eps_b_raw"].float() + r["eps_uncond"].float())
            sigma = float((1.0 - ac[ts]).sqrt())
            bound = float((rt / sigma).norm() ** 2 / 4.0)
            per_pair.setdefault(slug, {}).setdefault(si, []).append(bound)
        print(f"[bound] {split} {slug} {sdir.name} ({time.time()-t0:.0f}s)", flush=True)

    rows = {}
    for slug, by_step in per_pair.items():
        med = {s: st.median(v) for s, v in sorted(by_step.items())}
        rows[slug] = {"n_seeds": max(len(v) for v in by_step.values()), "fail_rate": fr.get(slug),
                      "median_bound_by_step": med,
                      "early_mean": st.mean([med[s] for s in EARLY if s in med])}
    (out / "interaction_bound.json").write_text(json.dumps({
        "definition": "GM >= ||r_t/sigma_t||^2/4 per cached state, median over seeds; r_t = gs*(eps_j - eps_a - eps_b + eps_uncond), sigma_t = sqrt(1-alpha_bar_t)",
        "guidance_scale": gs, "pairs": rows}, indent=1))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def colour(f):
        if f is None: return "#8a8a8a"
        return "#b22222" if f >= 0.99 else ("#e08a1e" if f >= 0.5 else "#2a7f2a")
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.35, 1]})
    for slug, r in rows.items():
        xs = sorted(r["median_bound_by_step"]); ys = [r["median_bound_by_step"][s] for s in xs]
        lw = 2.4 if slug in ("a_cat__x__a_dog", "a_butterfly__x__a_flower_meadow", "an_elephant__x__a_penguin") else 1.0
        a.plot(xs, ys, color=colour(r["fail_rate"]), lw=lw, alpha=0.9 if lw > 1 else 0.55)
        if lw > 1 or (r["fail_rate"] is not None and r["fail_rate"] < 0.99):
            a.annotate(slug.replace("__x__", " x ").replace("_", " "), (xs[-1], ys[-1]), fontsize=7.5, va="center", xytext=(3, 0), textcoords="offset points")
    a.set_yscale("log"); a.set_xlabel("denoising step (0 = pure noise)"); a.set_ylabel("lower bound on G·M   (‖r_t/σ_t‖² / 4, median over seeds)")
    a.set_title("interaction strength along the run, one curve per pair")
    a.axvspan(0, 9, color="#dddddd", alpha=0.5, lw=0); a.text(4.5, a.get_ylim()[1], "early window", ha="center", va="top", fontsize=8)
    order = sorted(rows, key=lambda s: rows[s]["early_mean"])
    b.barh([s.replace("__x__", " x ").replace("_", " ") for s in order], [rows[s]["early_mean"] for s in order],
           color=[colour(rows[s]["fail_rate"]) for s in order])
    for i, s in enumerate(order):
        f = rows[s]["fail_rate"]
        b.text(rows[s]["early_mean"], i, f"  fail {f:.2f}" if f is not None else "  fail rate not measured", va="center", fontsize=7.5)
    b.set_xscale("log"); b.set_xlabel("early-window (steps 0-9) mean of the bound"); b.tick_params(axis="y", labelsize=8)
    b.set_title("per pair, sorted; colour = plain-PoE fail rate (red 1.0, orange < 1, green < 0.5, grey unknown)")
    fig.suptitle("Mitra et al. Theorem 1 turned around: how strongly do the two concepts interact, per pair, from the cache", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "interaction_bound.png", dpi=130)
    print(f"[bound] wrote {out/'interaction_bound.png'}", flush=True)


if __name__ == "__main__":
    main()
