"""Do sampler error and score error add up on the four-quadrant toy?

A 2 x 2 at each of two data shares r of the top-right blob (0.05 and 0.001):
  expert kind : exact (noised-mixture scores from exact_toy.py) or trained (three energy MLPs)
  sampler     : reverse diffusion with the summed composed score, or annealed MALA (fixed copy,
                step tuned per level toward 0.6 acceptance), both via run_samplers.run_chain
Measures per cell: TV to the exact clean grid density, with the exact-sample floor; top-right share;
sample centre. Effects on TV above the floor: sampler effect with exact experts, with trained
experts, and their difference (the interaction), paired by seed index.

Usage:
  python run_interaction.py train --r 0.05 --seed 0 --density A --device cuda:0   # one expert
  python run_interaction.py sample --r 0.05 --device cuda:0                        # all cells at r
  python run_interaction.py combine                                                # results.json, figures
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
PREV = HERE.parent / "samplers-on-exact-experts"
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(PREV))
import exact_toy as T  # noqa: E402  (previous run, unchanged)
import run_samplers as RS  # noqa: E402  (previous run, unchanged: run_chain, reverse_step, MALA settings)
import trained_experts as E  # noqa: E402

RS_LIST = [0.05, 0.001]
SEEDS = [0, 1, 2]
DENSITIES = ["all", "A", "B"]
SAMPLERS = ["reverse diffusion", "annealed MALA"]
KINDS = ["exact", "trained"]
MODELS = HERE / "models"
QUERY = ("Do sampler error and score error add up, or does a better sampler change how much the "
         "trained experts' error hurts, on the four-quadrant toy?")


def data_seed(seed, density):
    return 100 * seed + DENSITIES.index(density)


def model_path(r, seed, density):
    return MODELS / f"r{r}_seed{seed}_{density}.pt"


def cmd_train(r, seed, density, device):
    MODELS.mkdir(exist_ok=True)
    sched = T.Schedule(E.N_LEVELS)
    ds = data_seed(seed, density)
    x, counts = E.training_set(density, r, E.TRAIN["n_train"], torch.Generator().manual_seed(ds))
    in_disc = float(T.region_shares(x.numpy())["in_TR_blob"])
    t0 = time.time()
    net, trace = E.train_expert(x, ds, device, sched)
    wall = time.time() - t0
    torch.save(dict(state_dict={k: v.cpu() for k, v in net.state_dict().items()}, r_TR=r, seed=seed,
                    density=density, data_seed=ds, component_counts=counts,
                    tr_disc_count=int(round(in_disc * len(x))), loss_trace=trace,
                    train_settings=E.TRAIN, wall_s=wall), model_path(r, seed, density))
    print(f"trained r={r} seed={seed} density={density}: TR component count {counts.get('TR')}, "
          f"final loss {trace[-1]:.4f}, {wall:.0f}s", flush=True)


def load_nets(r, seed, device):
    nets, meta = {}, {}
    for d in DENSITIES:
        ck = torch.load(model_path(r, seed, d), map_location="cpu", weights_only=False)
        n = E.EnergyNet()
        n.load_state_dict(ck["state_dict"])
        nets[d] = n.to(device).eval()
        for p in nets[d].parameters():
            p.requires_grad_(False)
        meta[d] = {k: ck[k] for k in ("component_counts", "tr_disc_count", "data_seed", "wall_s")}
        meta[d]["final_loss"] = ck["loss_trace"][-1]
    return nets, meta


def run_cell(sampler, seed, sched, target_cls):
    """One sampler run through the previous run's run_chain. MALA gets its pilot tuning first."""
    extra = {}
    if sampler == "reverse diffusion":
        xs, calls, _ = RS.run_chain(None, seed, sched, target_cls=target_cls)
    else:
        _, pcalls, pex = RS.run_chain("MALA", seed + 1000, sched, pilot=True, target_cls=target_cls)
        xs, calls, ex = RS.run_chain("MALA", seed, sched, steps=pex["step_sizes"], target_cls=target_cls)
        extra.update(tuning_composed_score_evals=pcalls, accept_per_level=ex["accept_per_level"],
                     accept_mean=float(np.mean(ex["accept_per_level"])), step_over_var=ex["step_over_var"])
    return xs, calls, extra


def measure(xs, grid_mass, edges):
    sh = T.region_shares(xs)
    fin = xs[np.isfinite(xs).all(1)]
    tv = T.total_variation(fin, grid_mass, edges) + sh["nonfinite"] if len(fin) else 1.0
    sh.update(tv=float(tv), mean_x1=float(fin[:, 0].mean()), mean_x2=float(fin[:, 1].mean()),
              std_x1=float(fin[:, 0].std()), std_x2=float(fin[:, 1].std()))
    return sh


def score_error(nets, r, sched, levels=(0, 10, 30, 60, 90)):
    """Relative RMS error of the trained composed score against the exact one, at noised exact
    composed samples (x_t = sqrt(abar) x_0 + sqrt(1 - abar) eps, x_0 from the exact composition)."""
    g = torch.Generator().manual_seed(999)
    x0, _ = T.exact_samples(2000, g, r)
    ex, tr = T.ComposedTarget(sched, r), E.TrainedComposedTarget(sched, nets)
    out = {}
    for t in levels:
        a = sched.abar_at(t)
        xt = math.sqrt(a) * x0 + math.sqrt(1 - a) * torch.randn(x0.shape, dtype=T.DT, generator=g)
        _, se = ex.logp_score(xt, t)
        _, st = tr.logp_score(xt, t)
        out[str(t)] = float(((st - se) ** 2).sum(1).mean().sqrt() / (se**2).sum(1).mean().sqrt())
    return out


def cmd_sample(r, device):
    t0 = time.time()
    sched = T.Schedule(100)
    grid_mass, edges = T.clean_density_grid(r=r)
    fine_mass, fine_edges = T.clean_density_grid(n=300, sub=4, r=r)
    gs = RS.grid_shares(fine_mass, fine_edges)
    key = json.load(open(HERE.parent / "answer-key-quadrants" / "answer_key.json"))
    krow = [k for k in key if k["case"] == "construction 1, contains labels" and k["r_TR"] == r][0]
    print(f"== r = {r}: exact target, checked against the answer key")
    print(f"   top-right share: this grid {gs['in_TR_blob']:.4f}, answer key {krow['in_TR_blob']:.4f}")
    assert abs(gs["in_TR_blob"] - krow["in_TR_blob"]) < 0.003, "grid disagrees with answer key"

    common = dict(r_TR=r, sigma=T.SIGMA, n_samples=RS.N_SAMPLES, noise_levels=100,
                  schedule="cosine, s=0.008, beta clip 0.999", start="x ~ N(0, I) at level 99",
                  reverse_step="DDPM ancestral, posterior variance", tv_grid="60 x 60 over [-1.5, 1.5]^2")
    samp_settings = {"reverse diffusion": dict(mcmc=None),
                     "annealed MALA": dict(mcmc="MALA (samplers-on-exact-experts/mala_fixed.py)", **RS.MALA,
                                           levels_with_mcmc="all 100 noised levels + clean level")}
    rows = []

    def record(expert, sampler, seed, xs, calls, settings, extra):
        row = dict(expert=expert, sampler=sampler, r_TR=r, seed=seed, settings=settings,
                   **measure(xs, grid_mass, edges), composed_score_evals=calls, **extra)
        rows.append(row)
        np.save(HERE / f"samples_r{r}_{expert.replace(' ', '_')}_{sampler.replace(' ', '_')}_seed{seed}.npy", xs)
        return row

    def show(row):
        acc = f"  acceptance {row['accept_mean']:.3f}" if "accept_mean" in row else ""
        print(f"   {row['expert']:9s} {row['sampler']:17s} seed {row['seed']}  TV {row['tv']:.3f}"
              f"  TR {row['in_TR_blob']:.3f}  origin {row['at_origin']:.3f}"
              f"  centre ({row['mean_x1']:.3f}, {row['mean_x2']:.3f})  nonfinite {row['nonfinite']:.3f}"
              f"  evals {row['composed_score_evals']}{acc}  {row['wall_s']:.0f}s", flush=True)

    print("\n== exact-sample floor (rejection sampling, 2,000 samples)")
    for seed in SEEDS:
        ts = time.time()
        xs, proposed = T.exact_samples(RS.N_SAMPLES, torch.Generator().manual_seed(seed), r)
        row = record("exact samples", "none", seed, xs.numpy(), 0,
                     dict(common, method="rejection from p(x), accept with P(A|x) P(B|x)"),
                     dict(proposals=proposed, wall_s=time.time() - ts))
        print(f"   exact samples seed {seed}  TV {row['tv']:.3f}  TR {row['in_TR_blob']:.3f}")
    floor = float(np.mean([x["tv"] for x in rows]))
    print(f"   floor (mean over seeds) {floor:.3f}")

    print("\n== exact experts")
    for seed in SEEDS:
        for s in SAMPLERS:
            ts = time.time()
            xs, calls, extra = run_cell(s, seed, sched, lambda sc: T.ComposedTarget(sc, r))
            extra["wall_s"] = time.time() - ts
            show(record("exact", s, seed, xs, calls,
                        dict(common, experts="exact noised mixtures (exact_toy.py)", **samp_settings[s]), extra))

    print("\n== check: exact experts, clean step on the level-0 density (what trained experts must do)")
    for seed in SEEDS:
        ts = time.time()
        xs, calls, extra = run_cell("annealed MALA", seed, sched, lambda sc: E.ExactLevel0Target(sc, r))
        extra["wall_s"] = time.time() - ts
        show(record("exact, level-0 clean step", "annealed MALA", seed, xs, calls,
                    dict(common, experts="exact, clean step uses level 0", **samp_settings["annealed MALA"]),
                    extra))

    print("\n== trained experts")
    train_meta = {}
    for seed in SEEDS:
        nets, meta = load_nets(r, seed, device)
        train_meta[seed] = meta
        serr = score_error(nets, r, sched)
        print(f"   seed {seed}: TR component count in training sets "
              + ", ".join(f"{d} {meta[d]['component_counts'].get('TR')} / {E.TRAIN['n_train']}" for d in DENSITIES)
              + "; final loss " + ", ".join(f"{d} {meta[d]['final_loss']:.4f}" for d in DENSITIES))
        print("   relative RMS error of the composed score at noised composed samples, by level: "
              + ", ".join(f"t={k} {v:.3f}" for k, v in serr.items()))
        for s in SAMPLERS:
            ts = time.time()
            xs, calls, extra = run_cell(s, seed, sched, lambda sc: E.TrainedComposedTarget(sc, nets))
            extra.update(wall_s=time.time() - ts, training=meta, composed_score_rel_rmse=serr)
            show(record("trained", s, seed, xs, calls,
                        dict(common, experts="trained energy MLPs (trained_experts.py)", train=E.TRAIN,
                             clean_step="level-0 networks", model_files=[str(model_path(r, seed, d).name)
                                                                         for d in DENSITIES],
                             **samp_settings[s]), extra))

    json.dump(dict(query=QUERY, r_TR=r, floor_tv=floor, exact_grid_shares=gs, answer_key_row=krow,
                   rows=rows, wall_s=time.time() - t0), open(HERE / f"results_r{r}.json", "w"), indent=1)
    print(f"\nwrote results_r{r}.json  ({time.time() - t0:.0f}s)")


def cell_tv(rows, expert, sampler):
    """TV per seed, in seed order, for one cell."""
    rs = sorted([x for x in rows if x["expert"] == expert and x["sampler"] == sampler], key=lambda x: x["seed"])
    return np.array([x["tv"] for x in rs])


def effects(rows, floor):
    """Sampler effect (MALA minus reverse diffusion) on TV above the floor, per expert kind, and the
    interaction (trained effect minus exact effect), each per seed index. Negative = MALA closer."""
    above = {(k, s): cell_tv(rows, k, s) - floor for k in KINDS for s in SAMPLERS}
    se = above[("exact", "annealed MALA")] - above[("exact", "reverse diffusion")]
    st = above[("trained", "annealed MALA")] - above[("trained", "reverse diffusion")]
    ee_rev = above[("trained", "reverse diffusion")] - above[("exact", "reverse diffusion")]
    ee_mala = above[("trained", "annealed MALA")] - above[("exact", "annealed MALA")]
    return dict(above=above, sampler_effect_exact=se, sampler_effect_trained=st, interaction=st - se,
                expert_effect_reverse=ee_rev, expert_effect_mala=ee_mala)


def ms(v):
    v = np.asarray(v, dtype=float)
    return f"{v.mean():+.3f} ± {v.std(ddof=1):.3f}"


def cmd_combine():
    allrows, per_r = [], {}
    print("== Summary per cell, mean ± sd over seeds 0, 1, 2 (2,000 samples each)")
    for r in RS_LIST:
        res = json.load(open(HERE / f"results_r{r}.json"))
        per_r[r] = res
        rows, floor = res["rows"], res["floor_tv"]
        for x in rows:
            x["tv_above_floor"] = x["tv"] - floor
        allrows += rows
        print(f"\n-- r = {r}   answer key top-right share {res['answer_key_row']['in_TR_blob']:.4f}"
              f"   floor TV {floor:.3f}")
        print(f"   {'experts':26s} {'sampler':17s} {'TV':>15s} {'TV above floor':>15s} {'top-right share':>16s}"
              f" {'origin share':>15s} {'sample centre':>17s}")
        for k, s in [("exact samples", "none")] + [(k, s) for k in KINDS for s in SAMPLERS] + \
                    [("exact, level-0 clean step", "annealed MALA")]:
            rs = [x for x in rows if x["expert"] == k and x["sampler"] == s]
            f = lambda key: np.array([x[key] for x in rs])
            centre = f"({f('mean_x1').mean():.3f}, {f('mean_x2').mean():.3f})"
            print(f"   {k:26s} {s:17s} {ms(f('tv'))[1:]:>15s} {ms(f('tv_above_floor')):>15s}"
                  f" {ms(f('in_TR_blob'))[1:]:>16s} {ms(f('at_origin'))[1:]:>15s} {centre:>17s}")
        tr = [x for x in rows if x["expert"] == "trained" and x["sampler"] == "reverse diffusion"]
        print("   training-set samples drawn from the top-right blob, out of 10,000, per seed:")
        for d in DENSITIES:
            print(f"     {d:4s}: " + ", ".join(f"seed {x['seed']} {x['training'][d]['component_counts'].get('TR')}"
                                             f" (within 3σ: {x['training'][d]['tr_disc_count']})" for x in tr))
        print("   relative RMS composed-score error at noised composed samples (mean over seeds): "
              + ", ".join(f"t={t} {np.mean([x['composed_score_rel_rmse'][t] for x in tr]):.3f}"
                          for t in tr[0]["composed_score_rel_rmse"]))
        mala = [x for x in rows if x["sampler"] == "annealed MALA" and x["expert"] in KINDS]
        print("   MALA acceptance (mean over levels): " + ", ".join(
            f"{x['expert']} seed {x['seed']} {x['accept_mean']:.3f}" for x in mala))

    print("\n== Effects on TV above the floor, per seed index then mean ± sd over seeds")
    print("   sampler effect = TV(annealed MALA) - TV(reverse diffusion); negative means MALA is closer.")
    print("   interaction = sampler effect with trained experts - sampler effect with exact experts.")
    print("   Seeds pair by index only: trained seed k and exact sampling seed k share no randomness.")
    eff_out = {}
    for r in RS_LIST:
        ef = effects(per_r[r]["rows"], per_r[r]["floor_tv"])
        eff_out[str(r)] = {k: [float(v) for v in ef[k]] for k in ef if k != "above"}
        print(f"\n-- r = {r}")
        for k in ["sampler_effect_exact", "sampler_effect_trained", "interaction",
                  "expert_effect_reverse", "expert_effect_mala"]:
            v = ef[k]
            print(f"   {k:24s} {ms(v):>16s}   per seed " + ", ".join(f"{x:+.3f}" for x in v)
                  + f"   (standard error {v.std(ddof=1) / math.sqrt(len(v)):.3f})")
    json.dump(dict(query=QUERY, floor_tv={str(r): per_r[r]["floor_tv"] for r in RS_LIST},
                   answer_key_rows={str(r): per_r[r]["answer_key_row"] for r in RS_LIST},
                   effects=eff_out, rows=allrows), open(HERE / "results.json", "w"), indent=1)
    import interaction_figures
    interaction_figures.draw(HERE, per_r, effects)
    print(f"\nwrote results.json and figures to {HERE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "sample", "combine"])
    ap.add_argument("--r", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--density", choices=DENSITIES)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.cmd == "train":
        cmd_train(a.r, a.seed, a.density, a.device)
    elif a.cmd == "sample":
        cmd_sample(a.r, a.device)
    else:
        cmd_combine()


if __name__ == "__main__":
    main()
