"""Which samplers reach the exact composed density on the four-quadrant toy with exact experts?

Samplers, all driven by the exact composed score at each noise level:
  reverse diffusion : DDPM ancestral steps with the summed score, levels 99 -> 0 -> clean
  annealed ULA      : the same ancestral step into each level, then ULA at that level
  annealed MALA     : the same, with MALA (fixed copy in mala_fixed.py), step tuned to ~0.6 acceptance
  annealed UHA      : the same, with unadjusted HMC
MCMC runs at all 100 noised levels and once more at the clean level, where its target is exact.
This is the loop shape of composable_diffusion/sampler_gd.py p_sample_loop_progressive.

Usage:  python run_samplers.py            # 3 seeds x 4 samplers + exact floor, figures
        python run_samplers.py --sweep    # also reverse diffusion with 1000 levels
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))
from anneal_samplers import AnnealedULASampler, AnnealedUHASampler  # noqa: E402  unchanged
from mala_fixed import AnnealedMALASampler  # noqa: E402
import exact_toy as T  # noqa: E402

N_SAMPLES = 2000
SEEDS = [0, 1, 2]
DT = T.DT

ULA = dict(steps_per_level=20, step_rule="ss = 0.05 * v_t")
MALA = dict(steps_per_level=20, step_init_rule="ss = 0.05 * v_t", target_accept=0.6,
            tuning="pilot run (seed + 1000), ss *= exp(acc - 0.6) after every step, capped at 2 v_t")
UHA = dict(steps_per_level=5, leapfrog_steps=3, damping=0.7, mass_diag_sqrt=1.0,
           step_rule="eps = 0.3 * sqrt(v_t)")


def level_var(sched, t):
    a = sched.abar_at(t)
    return a * T.SIGMA**2 + 1 - a


def reverse_step(x, t, sched, target):
    """DDPM ancestral step from level t to t-1 with the composed score; posterior variance."""
    beta = float(sched.betas[t])
    alpha = 1 - beta
    abar = sched.abar_at(t)
    abar_prev = sched.abar_at(t - 1)
    _, score = target.logp_score(x, t)
    mean = (x + beta * score) / math.sqrt(alpha)
    var = beta * (1 - abar_prev) / (1 - abar)
    return mean + math.sqrt(var) * torch.randn_like(x)


def run_chain(kind, seed, sched, steps=None, pilot=False, target_cls=T.ComposedTarget):
    """Run one sampler. Returns samples, composed-score calls, and extras (MALA acceptance)."""
    torch.manual_seed(seed)
    target = target_cls(sched)
    levels = list(range(sched.n - 1, -1, -1)) + [-1]          # 99, ..., 0, clean
    nl = len(levels)
    v = [level_var(sched, t) for t in levels]
    x = torch.randn(N_SAMPLES, 2, dtype=DT)
    extras = {}

    def grad_only(x, ts):
        return target.logp_score(x, ts)[1]

    def logp_and_grad(x, ts):
        return target.logp_score(x, ts)

    sampler = None
    if kind == "ULA":
        sampler = AnnealedULASampler(nl, ULA["steps_per_level"],
                                     [ULA_C * vi for vi in v], grad_only)
    elif kind == "UHA":
        sampler = AnnealedUHASampler(nl, UHA["steps_per_level"],
                                     [UHA_C * math.sqrt(vi) for vi in v], UHA["damping"],
                                     [UHA["mass_diag_sqrt"]] * nl, UHA["leapfrog_steps"], grad_only)
    elif kind == "MALA":
        ss = list(steps) if steps is not None else [MALA_C * vi for vi in v]
        k = MALA["steps_per_level"]
        sampler = AnnealedMALASampler(nl, 1 if pilot else k, ss, logp_and_grad)
        extras["accept_per_level"] = []

    for j, t in enumerate(levels):
        if sampler is not None:
            if kind == "MALA" and pilot:
                accs = []
                for _ in range(MALA["steps_per_level"]):
                    x = sampler.sample_step(x, j, t, {})
                    a = sampler.last_accept[0]
                    accs.append(a)
                    sampler._step_sizes[j] = min(sampler._step_sizes[j] * math.exp(a - 0.6), 2 * v[j])
                extras["accept_per_level"].append(float(np.mean(accs)))
            else:
                x = sampler.sample_step(x, j, t, {})
                if kind == "MALA":
                    extras["accept_per_level"].append(float(np.mean(sampler.last_accept)))
        if t >= 0:
            x = reverse_step(x, t, sched, target)
    if kind == "MALA":
        extras["step_sizes"] = [float(s) for s in sampler._step_sizes]
        extras["step_over_var"] = [float(s / vi) for s, vi in zip(sampler._step_sizes, v)]
    return x.numpy(), target.calls, extras


ULA_C, MALA_C, UHA_C = 0.05, 0.05, 0.3


def grid_shares(grid_mass, edges):
    """Exact region shares from the 60 x 60 cell masses (cell centres), for a cross-check."""
    c = 0.5 * (edges[1:] + edges[:-1])
    X1, X2 = np.meshgrid(c, c, indexing="ij")
    out = {}
    for q in T.QUADS:
        m = np.hypot(X1 - T.MU[q][0], X2 - T.MU[q][1]) < 3 * T.SIGMA
        out[f"in_{q}_blob"] = float(grid_mass[m].sum())
    out["at_origin"] = float(grid_mass[np.hypot(X1, X2) < 3 * T.SIGMA / math.sqrt(2)].sum())
    return out


def fmt(xs):
    return f"{np.mean(xs):.3f} ± {np.std(xs, ddof=1):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    sched = T.Schedule(100)
    grid_mass, edges = T.clean_density_grid()
    fine_mass, fine_edges = T.clean_density_grid(n=300, sub=4)

    print("== 1. Exact target, checked against the answer key")
    key = json.load(open(HERE.parent / "answer-key-quadrants" / "answer_key.json"))
    krow = [r for r in key if r["case"] == "construction 1, contains labels" and r["r_TR"] == 0.05][0]
    gs = grid_shares(fine_mass, fine_edges)
    for k in ["in_TR_blob", "in_BR_blob", "in_TL_blob", "in_BL_blob", "at_origin"]:
        print(f"   {k:11s} this grid {gs[k]:.4f}   answer key {krow[k]:.4f}")
    assert abs(gs["in_TR_blob"] - krow["in_TR_blob"]) < 0.003, "grid disagrees with answer key"
    print(f"   abar at level 0 = {sched.abar[0]:.6f}, level 99 = {sched.abar[99]:.2e}")
    print("   Takeaway: the target built here matches the answer key, so every sampler is scored"
          " against the right density.")

    rows = []
    common = dict(n_samples=N_SAMPLES, noise_levels=100, schedule="cosine, s=0.008, beta clip 0.999",
                  r_TR=T.R_TR, sigma=T.SIGMA, start="x ~ N(0, I) at level 99",
                  levels_with_mcmc="all 100 noised levels + clean level",
                  reverse_step="DDPM ancestral, posterior variance beta_t (1-abar_{t-1})/(1-abar_t)")

    def record(name, seed, xs, calls, settings, extra=None):
        sh = T.region_shares(xs)
        tv = T.total_variation(xs[np.isfinite(xs).all(1)], grid_mass, edges) if sh["nonfinite"] < 1 else 1.0
        fin = xs[np.isfinite(xs).all(1)]
        sh.update(mean_x1=float(fin[:, 0].mean()), mean_x2=float(fin[:, 1].mean()),
                  std_x1=float(fin[:, 0].std()), std_x2=float(fin[:, 1].std()))
        row = dict(sampler=name, seed=seed, settings=settings, **sh, tv=float(tv),
                   composed_score_evals=calls, expert_score_evals=3 * calls)
        if extra:
            row.update(extra)
        rows.append(row)
        np.save(HERE / f"samples_{name.replace(' ', '_')}_seed{seed}.npy", xs)
        return row

    print("\n== 2. Runs (2,000 samples each)")
    for seed in SEEDS:
        g = torch.Generator().manual_seed(seed)
        ts = time.time()
        xs, proposed = T.exact_samples(N_SAMPLES, g)
        r = record("exact samples", seed, xs.numpy(), 0,
                   dict(method="rejection from p(x), accept with P(A|x) P(B|x)", n_samples=N_SAMPLES),
                   dict(proposals=proposed, acceptance=N_SAMPLES / proposed, wall_s=time.time() - ts))
        print(f"   exact samples     seed {seed}  TR {r['in_TR_blob']:.3f}  TV {r['tv']:.3f}")

        for kind, name, settings in [
                (None, "reverse diffusion", dict(common, mcmc=None)),
                ("ULA", "annealed ULA", dict(common, mcmc="ULA (anneal_samplers.py, unchanged)", **ULA)),
                ("UHA", "annealed UHA", dict(common, mcmc="UHA (anneal_samplers.py, unchanged)", **UHA)),
                ("MALA", "annealed MALA", dict(common, mcmc="MALA (mala_fixed.py)", **MALA))]:
            ts = time.time()
            extra = {}
            steps = None
            if kind == "MALA":
                _, pcalls, pex = run_chain("MALA", seed + 1000, sched, pilot=True)
                steps = pex["step_sizes"]
                extra["tuning_composed_score_evals"] = pcalls
                extra["pilot_accept_per_level"] = pex["accept_per_level"]
            xs, calls, ex = run_chain(kind, seed, sched, steps=steps)
            extra.update(ex)
            if kind == "MALA":
                extra["accept_mean"] = float(np.mean(ex["accept_per_level"]))
            extra["wall_s"] = time.time() - ts
            r = record(name, seed, xs, calls, settings, extra)
            acc = f"  acceptance {r['accept_mean']:.3f}" if kind == "MALA" else ""
            print(f"   {name:17s} seed {seed}  TR {r['in_TR_blob']:.3f}  origin {r['at_origin']:.3f}"
                  f"  TV {r['tv']:.3f}  evals {calls}{acc}  nonfinite {r['nonfinite']:.3f}"
                  f"  {r['wall_s']:.1f}s")

    if args.sweep:
        print("\n== 2b. Sweep: reverse diffusion with 1000 cosine levels (is the miss a step-count error?)")
        s1000 = T.Schedule(1000)
        for seed in SEEDS:
            xs, calls, _ = run_chain(None, seed, s1000)
            r = record("reverse diffusion 1000 levels", seed, xs, calls,
                       dict(common, noise_levels=1000, mcmc=None))
            print(f"   reverse diffusion 1000 levels seed {seed}  TR {r['in_TR_blob']:.3f}"
                  f"  origin {r['at_origin']:.3f}  TV {r['tv']:.3f}  evals {calls}")

    print("\n== 2c. Control: reverse diffusion with the true noised score of the plain mixture p(x)")
    print("   Same reverse step, same 100 levels; target is p(x) itself, scored against p(x)'s own grid.")
    mix_mass, _ = T.clean_density_grid(target_cls=T.MixtureTarget)
    ctl = {"reverse diffusion on p(x)": [], "exact p(x) samples": []}
    for seed in SEEDS:
        xs, calls, _ = run_chain(None, seed, sched, target_cls=T.MixtureTarget)
        xe = T.mixture_samples(N_SAMPLES, torch.Generator().manual_seed(seed)).numpy()
        for name, x in [("reverse diffusion on p(x)", xs), ("exact p(x) samples", xe)]:
            tv = T.total_variation(x, mix_mass, edges)
            sh = T.region_shares(x)
            ctl[name].append(tv)
            rows.append(dict(sampler="control: " + name, seed=seed,
                             settings=dict(common, target="plain mixture p(x), not the composition", mcmc=None),
                             **sh, tv=float(tv), composed_score_evals=calls if "reverse" in name else 0))
            print(f"   {name:26s} seed {seed}  TV to p(x) grid {tv:.3f}  TR {sh['in_TR_blob']:.3f}"
                  f"  BR {sh['in_BR_blob']:.3f}")
    for name, v in ctl.items():
        print(f"   {name:26s} TV {fmt(v)}")
    print("   Takeaway: with a true noised score the reverse step lands on its floor, so the composed"
          " miss comes from the summed score, not from the reverse step or the 100-level schedule.")

    print("\n== 3. Summary, mean ± sd over seeds", SEEDS)
    order = ["exact samples", "reverse diffusion", "annealed ULA", "annealed MALA", "annealed UHA"]
    if args.sweep:
        order.append("reverse diffusion 1000 levels")
    print(f"   {'sampler':30s} {'top-right share':>17s} {'origin share':>15s} {'TV':>15s} {'score evals':>12s}"
          f"  {'sample centre (x1, x2)':>24s} {'sample std (x1, x2)':>20s}")
    print(f"   {'exact target (grid)':30s} {gs['in_TR_blob']:17.3f} {gs['at_origin']:15.3f} {'':>15s} {'':>12s}")
    summary = {}
    for name in order:
        rs = [r for r in rows if r["sampler"] == name]
        tr = [r["in_TR_blob"] for r in rs]
        og = [r["at_origin"] for r in rs]
        tv = [r["tv"] for r in rs]
        ev = rs[0]["composed_score_evals"]
        summary[name] = dict(tr=tr, origin=og, tv=tv, evals=ev)
        mc = (np.mean([r["mean_x1"] for r in rs]), np.mean([r["mean_x2"] for r in rs]))
        sdv = (np.mean([r["std_x1"] for r in rs]), np.mean([r["std_x2"] for r in rs]))
        print(f"   {name:30s} {fmt(tr):>17s} {fmt(og):>15s} {fmt(tv):>15s} {ev:12d}"
              f"  {f'({mc[0]:.3f}, {mc[1]:.3f})':>24s} {f'({sdv[0]:.3f}, {sdv[1]:.3f})':>20s}")
    print("   The origin disc (radius 3σ/√2) overlaps the top-right 3σ disc, so a sample can count in both.")
    mala = [r for r in rows if r["sampler"] == "annealed MALA"]
    print(f"\n   MALA acceptance (main run, mean over levels): "
          + ", ".join(f"seed {r['seed']} {r['accept_mean']:.3f}" for r in mala))
    acc = np.array([r["accept_per_level"] for r in mala])
    print(f"   MALA acceptance range over levels: min {acc.mean(0).min():.3f}, max {acc.mean(0).max():.3f};"
          f" clean level {acc[:, -1].mean():.3f}")
    print(f"   MALA tuned step / level variance: clean level {np.mean([r['step_over_var'][-1] for r in mala]):.3f},"
          f" level 99 {np.mean([r['step_over_var'][0] for r in mala]):.3f}")
    print(f"   MALA tuning pilot cost: {mala[0]['tuning_composed_score_evals']} composed-score evals per seed")

    json.dump(dict(query="On the four-quadrant toy with exact experts, which samplers reach the exact "
                         "composed density, and how far does reverse diffusion with summed scores miss it?",
                   exact_grid_shares=gs, answer_key_row=krow, rows=rows),
              open(HERE / "results.json", "w"), indent=1)
    import figures
    figures.draw(HERE, rows, grid_mass, edges, gs, order)
    print(f"\nwrote results.json and figures to {HERE}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
