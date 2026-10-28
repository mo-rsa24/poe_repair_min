"""Phase 2 check: the notebook's product, mixture and negation of trained models, sampled every way.

    python -m rebuild.phase2_compose --ckpt rebuild/out/phase1/ckpt --out rebuild/out/phase2

For each composition the exact target of the trained experts is computable on the grid in 2D: the composed
energy at t = 0, exponentiated and normalised. Each sampler's 2000 samples are scored against it (TV on a
40 x 40 binning) and by a measure that names the composition's success:
  product ring x bar: share within 0.1 of the two ring modes inside the bar, (0, 0.5) and (0, -0.5)
  mixture left + right: share with x1 > 0 (the right bar's share; 0.5 when both bars are kept)
  negation box(0.7) not box(0.3): share inside the small box
The notebook's own settings are used for its samplers (ULA step 0.001, Metropolis-adjusted UHA step 0.03,
damping 0.5, 3 leapfrog steps, 10 moves per level, 100 levels). MALA runs at the notebook's 0.001 and tuned to
0.6 acceptance, and its acceptance is reported per sample.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rebuild import ideal, samplers as S
from rebuild.compose import MixtureEBMDiffusionModel, NegationEBMDiffusionModel, ProductEBMDiffusionModel
from rebuild.datasets import PANELS
from rebuild.device import describe, pick_device
from rebuild.model import PortableDiffusionModel
from rebuild.phase1_check import load
from rebuild.train import slug

NS = 2000
COMPOSITIONS = [
    ("product: ring x bar", "toy_gmm(std=.03)", "bar(scale=0.2)", ProductEBMDiffusionModel,
     lambda s: float((np.minimum(np.hypot(s[:, 0], s[:, 1] - 0.5), np.hypot(s[:, 0], s[:, 1] + 0.5)) < 0.1).mean()),
     "share on the 2 in-bar modes", lambda a, b: a + b),
    ("mixture: left + right", "left_bar()", "right_bar()", MixtureEBMDiffusionModel,
     lambda s: float((s[:, 0] > 0).mean()), "right bar's share", lambda a, b: torch.logaddexp(a, b)),
    ("negation: box(0.7) not box(0.3)", "toy_box(0.7)", "toy_box(0.3)", NegationEBMDiffusionModel,
     lambda s: float(((np.abs(s[:, 0]) < 0.3) & (np.abs(s[:, 1]) < 0.3)).mean()), "share inside the small box",
     lambda a, b: 1.3 * a - 0.3 * b),
]
# where the mass sits between separated parts: top/bottom for the product, the four corners for the negation
SPLITS = {
    "product: ring x bar": lambda s: [float((s[:, 1] > 0).mean()), float((s[:, 1] < 0).mean())],
    "mixture: left + right": lambda s: [float((s[:, 0] < 0).mean()), float((s[:, 0] > 0).mean())],
    "negation: box(0.7) not box(0.3)": lambda s: [float(((s[:, 0] < 0) & (s[:, 1] > 0)).mean()), float(((s[:, 0] > 0) & (s[:, 1] > 0)).mean()),
                                                   float(((s[:, 0] < 0) & (s[:, 1] < 0)).mean()), float(((s[:, 0] > 0) & (s[:, 1] < 0)).mean())],
}
IDEAL_SIG, IDEAL_N = 0.01, 4000     # the data's own composition: 4000 fresh points per expert, smoothed by 0.01


def composed(ckpt, a, b, wrapper, dev):
    m1, m2 = load(os.path.join(ckpt, slug(a) + ".pt"), dev), load(os.path.join(ckpt, slug(b) + ".pt"), dev)
    return PortableDiffusionModel(2, 100, wrapper(m1.net, m2.net)).to(dev)


def trained_target(ddpm, dev):
    tv_ = lambda x, t: torch.full((x.shape[0],), t, dtype=torch.long, device=dev)
    return S.Target(lambda x, t: -ddpm.p_energy(x, tv_(x, t)), lambda x, t: -ddpm.p_gradient(x, tv_(x, t)))


def main(ckpt, out):
    dev = pick_device(); print(describe(dev))
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0); rng = np.random.default_rng(0)
    cols = ["perfect", "reverse diffusion", "ULA", "MALA (0.001)", "MALA, tuned", "UHA", "MUHA (notebook)"]
    report, samples, grids = [], {}, {}
    for name, a, b, wrapper, measure, mname, data_compose in COMPOSITIONS:
        ddpm = composed(ckpt, a, b, wrapper, dev)
        tgt = trained_target(ddpm, dev)
        p = ideal.grid_density(lambda g: tgt.logp(g, 0), dev)
        grids[name] = p
        tt = lambda d: torch.tensor(d, dtype=torch.float32, device=dev)
        da_, db_ = tt(PANELS[a](IDEAL_N, rng=rng)), tt(PANELS[b](IDEAL_N, rng=rng))
        p_data = ideal.grid_density(lambda g: data_compose(ideal.logq(g, da_, -1, IDEAL_SIG), ideal.logq(g, db_, -1, IDEAL_SIG)), dev)
        S_ = {"perfect": ideal.perfect_samples(p, NS, rng)}
        with torch.no_grad():
            S_["reverse diffusion"] = ddpm.sample(NS, dev).cpu().numpy()
        S_["ULA"] = S.ula(tgt, NS, 2, [0.001] * 100, device=dev).cpu().numpy()
        x, acc = S.mala(tgt, NS, 2, [0.001] * 100, device=dev); S_["MALA (0.001)"] = x.cpu().numpy(); acc_fixed = acc
        x, acc, steps = S.tuned_mala(tgt, NS, 2, ideal.V, device=dev); S_["MALA, tuned"] = x.cpu().numpy(); acc_tuned = acc
        S_["UHA"] = S.uha(tgt, NS, 2, [0.03] * 100, device=dev).cpu().numpy()
        x, acc = S.cha(tgt, NS, 2, [0.03] * 100, device=dev); S_["MUHA (notebook)"] = x.cpu().numpy(); acc_cha = acc
        samples[name] = S_
        exact = float(measure(np.c_[ideal.X1.ravel(), ideal.X2.ravel()][np.random.default_rng(5).choice(
            ideal.G * ideal.G, 200000, p=(p * ideal.CELL).ravel() / (p * ideal.CELL).sum())]))
        per_sample = lambda acc: acc[-10:].mean(0).cpu().numpy()     # each sample's acceptance over the last 10 levels
        row = dict(composition=name, measure_name=mname, exact_trained=exact, samplers={},
                   mala_fixed_accept=dict(mean=float(per_sample(acc_fixed).mean()), sd=float(per_sample(acc_fixed).std()),
                                          hist=np.histogram(per_sample(acc_fixed), bins=10, range=(0, 1))[0].tolist()),
                   mala_tuned_accept=dict(mean=float(per_sample(acc_tuned).mean()), sd=float(per_sample(acc_tuned).std()),
                                          hist=np.histogram(per_sample(acc_tuned), bins=10, range=(0, 1))[0].tolist(),
                                          final_step=steps[-1]),
                   muha_accept=dict(mean=float(per_sample(acc_cha).mean()), sd=float(per_sample(acc_cha).std())))
        split = SPLITS[name]
        draw = lambda q, n: np.c_[ideal.X1.ravel(), ideal.X2.ravel()][np.random.default_rng(5).choice(
            ideal.G * ideal.G, n, p=(q * ideal.CELL).ravel() / (q * ideal.CELL).sum())]
        row["split_trained_t0"], row["split_data"] = split(draw(p, 200000)), split(draw(p_data, 200000))
        row["measure_data"] = measure(draw(p_data, 200000))
        perfect_data = ideal.perfect_samples(p_data, NS, rng)
        row["floor_tv_data"] = float(ideal.tv(perfect_data, p_data))
        print(f"\n{name}  ({mname}; trained experts' composition at t = 0: {exact:.3f}; the data's own composition: {row['measure_data']:.3f})")
        print(f"  split of mass between parts: trained at t = 0 {np.round(row['split_trained_t0'], 2).tolist()}, data {np.round(row['split_data'], 2).tolist()}; "
              f"floor TV to the data's composition {row['floor_tv_data']:.3f}")
        for c in cols:
            s = S_[c]; s = s[np.isfinite(s).all(1)]
            row["samplers"][c] = dict(measure=measure(s), tv=float(ideal.tv(s, p)), tv_data=float(ideal.tv(s, p_data)),
                                      split=split(s), kept=len(s))
            print(f"  {c:>18}: {measure(s):.3f}  TV trained-t0 {ideal.tv(s, p):.3f}  TV data {ideal.tv(s, p_data):.3f}  "
                  f"split {np.round(split(s), 2).tolist()}  ({len(s)} finite)")
        for k in ("mala_fixed_accept", "mala_tuned_accept", "muha_accept"):
            print(f"  {k}: mean {row[k]['mean']:.3f} sd across samples {row[k]['sd']:.3f}"
                  + (f" histogram 0..1 {row[k]['hist']}" if "hist" in row[k] else ""))
        report.append(row)

    fig, ax = plt.subplots(len(COMPOSITIONS), 3 + len(cols), figsize=(2.3 * (3 + len(cols)), 2.6 * len(COMPOSITIONS)))
    for r, (name, a, b, *_rest) in enumerate(COMPOSITIONS):
        da, db = PANELS[a](1000, rng=rng), PANELS[b](1000, rng=rng)
        ax[r, 0].scatter(*da.T, s=1, c="tab:red"); ax[r, 0].set_title(a, fontsize=8); ax[r, 0].set_ylabel(name, fontsize=8)
        ax[r, 1].scatter(*db.T, s=1, c="tab:blue"); ax[r, 1].set_title(b, fontsize=8)
        ax[r, 2].imshow(grids[name], origin="lower", extent=[-ideal.L, ideal.L, -ideal.L, ideal.L], cmap="magma")
        ax[r, 2].set_title("exact, trained experts", fontsize=8)
        for j, c in enumerate(cols, start=3):
            s = samples[name][c]; rr = report[r]["samplers"][c]
            ax[r, j].scatter(*s.T, s=1, c="k", alpha=.4); ax[r, j].set_title(f"{c}\n{rr['measure']:.2f} · TV {rr['tv']:.2f}", fontsize=8)
        for x_ in ax[r]:
            x_.set_xlim(-ideal.L, ideal.L); x_.set_ylim(-ideal.L, ideal.L); x_.set_aspect("equal"); x_.set_xticks([]); x_.set_yticks([])
    plt.tight_layout(); plt.savefig(os.path.join(out, "product-mixture-negation-every-sampler.png"), dpi=90); plt.close()

    fig, ax = plt.subplots(1, len(COMPOSITIONS), figsize=(4 * len(COMPOSITIONS), 3))
    for a_, row in zip(ax, report):
        edges = np.linspace(0, 1, 11)
        a_.bar(edges[:-1] + 0.025, row["mala_fixed_accept"]["hist"], width=0.045, label="MALA, step 0.001")
        a_.bar(edges[:-1] + 0.075, row["mala_tuned_accept"]["hist"], width=0.045, label="MALA, tuned")
        a_.set_xlabel("one sample's acceptance over the last 10 levels (100 moves)"); a_.set_ylabel("samples")
        a_.set_title(row["composition"], fontsize=9); a_.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(out, "mala-acceptance-per-sample.png"), dpi=90); plt.close()
    json.dump(report, open(os.path.join(out, "phase2.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="rebuild/out/phase1/ckpt")
    ap.add_argument("--out", default="rebuild/out/phase2")
    a = ap.parse_args()
    main(a.ckpt, a.out)
