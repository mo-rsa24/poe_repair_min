"""how-7: trained energy-MLP experts against ideal (smoothed-data) experts, notebook sections 4b and 4c.

The data generators, grid, metrics and rng replay are copied from
what-an-expert-learns-and-kinds-of-composition.ipynb so the ideal numbers reproduce the notebook.
A trained expert is a scalar MLP E(x, log sigma); score = -grad_x E; trained by VE denoising score
matching. At the metric's sigma, log p = -E, so the product is exact on the grid with no sampler.

Usage: python how7_trained_vs_ideal.py [--steps 8000] [--seeds 3] [--workers 12]
"""
import argparse, json, os, time
import numpy as np, torch
from scipy.special import logsumexp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool

OUT = os.path.dirname(os.path.abspath(__file__))

# ---- copied from the notebook, cell 1 ----
L, G = 1.1, 241
xs = np.linspace(-L, L, G)
X1, X2 = np.meshgrid(xs, xs)
grid = np.stack([X1.ravel(), X2.ravel()], 1)
cell = (xs[1] - xs[0]) ** 2

def ideal_expert(data, sigma):
    d2 = ((grid[:, None, :] - np.asarray(data)[None]) ** 2).sum(-1)
    lp = logsumexp(-d2 / (2 * sigma**2), axis=1) - np.log(len(data)) - np.log(2 * np.pi * sigma**2)
    return lp.reshape(G, G)

def density(logp):
    p = np.exp(logp - logp.max()); return p / (p.sum() * cell)

def mass(p, region):
    return float((p * region).sum() * cell)

# ---- metric regions, copied from cells 11 and 13 ----
near = lambda Z, v: np.abs(Z - v) < 0.15
CO = (near(X1, -.5) & near(X2, .5)) | (near(X1, .5) & near(X2, -.5))
CHIMERA = ~(near(X1, -.5) | near(X1, .5) | near(X2, -.5) | near(X2, .5))
TOP = X2 > 0

def mean_x1_top(prod):
    return float((prod * TOP * X1).sum() / (prod * TOP).sum())


def make_data():
    """Replay the notebook's rng stream in cell order so the 4b and 4c sets are the notebook's sets."""
    rng = np.random.default_rng(0)
    # cell 7
    def make_sets(f, n=400):
        a = np.c_[rng.uniform(-1, 0, n), np.where(rng.random(n) < f, rng.uniform(0, 1, n), rng.uniform(-1, 0, n))]
        b = np.c_[np.where(rng.random(n) < f, rng.uniform(-1, 0, n), rng.uniform(0, 1, n)), rng.uniform(0, 1, n)]
        return a, b
    make_sets(0.2)
    # cell 9 (draws only; same call order as the notebook)
    n = 2000
    ring = lambda n: 0.5 * np.c_[np.cos(a := 2 * np.pi * rng.integers(0, 8, n) / 8), np.sin(a)] + 0.03 * rng.standard_normal((n, 2))
    bar = lambda n: np.c_[rng.uniform(-0.2, 0.2, n), rng.uniform(-1, 1, n)]
    left_half = lambda n: np.c_[rng.uniform(-1, 0, n), rng.uniform(-1, 1, n)]
    top_half = lambda n: np.c_[rng.uniform(-1, 1, n), rng.uniform(0, 1, n)]
    sub = lambda d: d[rng.choice(len(d), 600, replace=False)]
    for fa, fb in [(left_half, top_half), (ring, bar)]:
        sub(fa(n)); sub(fb(n))
    # cell 11
    def obj(v, n, varied):
        o = v + 0.05 * rng.standard_normal(n); other = rng.uniform(-1, 1, n)
        swap = (rng.random(n) < 0.5) if varied else np.zeros(n, bool)
        return np.where(swap[:, None], np.c_[other, o], np.c_[o, other])
    data = {}
    for varied in [False, True]:
        reg = "varied" if varied else "centred"
        data[("4b", reg, "cat")] = obj(-0.5, 600, varied)
        data[("4b", reg, "dog")] = obj(0.5, 600, varied)
    # cell 13
    def interaction_sets(f, n=600):
        top = rng.random(n) < f
        a = np.c_[np.where(top, -0.8, -0.5) + 0.06 * rng.standard_normal(n),
                  np.where(top, rng.uniform(0, 1, n), rng.uniform(-1, 0, n))]
        hid = rng.random(n) < f
        b = np.c_[np.where(hid, -0.8 + 0.06 * rng.standard_normal(n), rng.uniform(-1, 1, n)), rng.uniform(0, 1, n)]
        return a, b
    for f in [0.0, 0.2]:
        a, b = interaction_sets(f)
        data[("4c", f, "A")] = a; data[("4c", f, "B")] = b
    return data


def print_counts(data):
    print("== points per region, per data generator ==")
    for reg in ["centred", "varied"]:
        for name, v in [("cat", -0.5), ("dog", 0.5)]:
            d = data[("4b", reg, name)]
            s1 = int((np.abs(d[:, 0] - v) < 0.15).sum()); s2 = int((np.abs(d[:, 1] - v) < 0.15).sum())
            print(f"4b {reg:8s} {name}: n={len(d)}  object in slot1 (|x1-{v:+.1f}|<.15) = {s1}  in slot2 = {s2}")
    for f in [0.0, 0.2]:
        a, b = data[("4c", f, "A")], data[("4c", f, "B")]
        print(f"4c f={f}: A n={len(a)} top half={int((a[:,1]>0).sum())} bottom={int((a[:,1]<=0).sum())} "
              f"wet (x1<-0.65 & top)={int(((a[:,0]<-0.65)&(a[:,1]>0)).sum())} | "
              f"B n={len(b)} top half={int((b[:,1]>0).sum())} x1 within .2 of -0.8={int((np.abs(b[:,0]+0.8)<0.2).sum())}")


# ---- trained expert ----
def make_net():
    return torch.nn.Sequential(
        torch.nn.Linear(3, 128), torch.nn.SiLU(), torch.nn.Linear(128, 128), torch.nn.SiLU(),
        torch.nn.Linear(128, 128), torch.nn.SiLU(), torch.nn.Linear(128, 1))

def energy(net, x, logsig):
    return net(torch.cat([x, logsig], 1)).squeeze(1)

def train_one(job):
    key, d, seed, steps, eval_sigma = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    x0 = torch.tensor(d, dtype=torch.float32)
    lo, hi = np.log(0.01), np.log(1.0)
    losses = []
    for it in range(steps):
        ls = lo + (hi - lo) * torch.rand(len(x0), 1)
        sig = ls.exp()
        eps = torch.randn_like(x0)
        x = (x0 + sig * eps).requires_grad_(True)
        E = energy(net, x, ls)
        s = -torch.autograd.grad(E.sum(), x, create_graph=True)[0]
        loss = ((sig * s + eps) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    with torch.no_grad():
        g = torch.tensor(grid, dtype=torch.float32)
        lsg = torch.full((len(g), 1), float(np.log(eval_sigma)))
        logp = (-energy(net, g, lsg)).numpy().reshape(G, G).astype(np.float64)
    # fixed-noise loss for comparability across seeds
    gen = torch.Generator().manual_seed(1234)
    ls = lo + (hi - lo) * torch.rand(len(x0), 1, generator=gen); sig = ls.exp()
    eps = torch.randn(x0.shape, generator=gen)
    x = (x0 + sig * eps).requires_grad_(True)
    s = -torch.autograd.grad(energy(net, x, ls).sum(), x)[0]
    eval_loss = float(((sig * s + eps) ** 2).sum(1).mean())
    return key, seed, logp, np.array(losses), eval_loss


def kl(p, q):
    m = p > 0
    return float((p[m] * (np.log(p[m]) - np.log(np.maximum(q[m], 1e-300)))).sum() * cell)


def ms(v):
    v = np.asarray(v); return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, float(v.min()), float(v.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    t0 = time.time()
    data = make_data()
    print_counts(data)
    SIG = {"4b": 0.2, "4c": 0.08}

    # ideal experts and products
    ideal = {k: ideal_expert(v, SIG[k[0]]) for k, v in data.items()}
    rows = []
    def metrics(exp, lpa, lpb):
        prod = density(lpa + lpb)
        if exp == "4b":
            return prod, {"co_occurrence": mass(prod, CO), "chimera": mass(prod, CHIMERA)}
        return prod, {"mean_x1_top": mean_x1_top(prod)}
    ideal_prod = {}
    print("\n== ideal products (should match the notebook) ==")
    for exp, cond, a, b in [("4b", "centred", "cat", "dog"), ("4b", "varied", "cat", "dog"),
                            ("4c", 0.0, "A", "B"), ("4c", 0.2, "A", "B")]:
        prod, m = metrics(exp, ideal[(exp, cond, a)], ideal[(exp, cond, b)])
        ideal_prod[(exp, cond)] = prod
        print(exp, cond, {k: round(v, 3) for k, v in m.items()})
        rows.append({"experiment": exp, "regime" if exp == "4b" else "f": cond, "seed": None,
                     "expert_type": "ideal", "sigma": SIG[exp], **m})

    # train
    jobs = [(k, v, s, args.steps, SIG[k[0]]) for k, v in data.items() for s in range(args.seeds)]
    print(f"\ntraining {len(jobs)} experts, {args.steps} steps each, {args.workers} workers")
    with Pool(args.workers) as pool:
        res = pool.map(train_one, jobs)
    trained = {(k, s): (lp, losses, el) for k, s, lp, losses, el in res}
    t_train = time.time() - t0

    # fit check per single expert
    print("\n== fit check per single expert: KL(ideal || trained) on the grid, nats ==")
    fit = []
    for (k, s), (lp, losses, el) in sorted(trained.items(), key=lambda z: str(z[0])):
        klv = kl(density(ideal[k]), density(lp))
        w = 200
        sm = np.convolve(losses, np.ones(w) / w, mode="valid")
        r = {"experiment": k[0], "regime" if k[0] == "4b" else "f": k[1], "expert": k[2], "seed": s,
             "kl_ideal_trained": klv, "dsm_loss_fixed_noise": el,
             "loss_smoothed_at_25pct": float(sm[len(sm)//4]), "loss_smoothed_at_50pct": float(sm[len(sm)//2]),
             "loss_smoothed_at_75pct": float(sm[3*len(sm)//4]), "loss_smoothed_final": float(sm[-1])}
        fit.append(r)
        print(f"{k} seed {s}: KL={klv:.4f}  loss 25/50/75/100% = {r['loss_smoothed_at_25pct']:.4f} "
              f"{r['loss_smoothed_at_50pct']:.4f} {r['loss_smoothed_at_75pct']:.4f} {r['loss_smoothed_final']:.4f}")

    # trained products, per seed
    trained_prod = {}
    summary = []
    for exp, cond, a, b in [("4b", "centred", "cat", "dog"), ("4b", "varied", "cat", "dog"),
                            ("4c", 0.0, "A", "B"), ("4c", 0.2, "A", "B")]:
        per = []
        for s in range(args.seeds):
            prod, m = metrics(exp, trained[((exp, cond, a), s)][0], trained[((exp, cond, b), s)][0])
            trained_prod[(exp, cond, s)] = prod
            m["kl_ideal_trained_product"] = kl(ideal_prod[(exp, cond)], prod)
            per.append(m)
            rows.append({"experiment": exp, "regime" if exp == "4b" else "f": cond, "seed": s,
                         "expert_type": "trained", "sigma": SIG[exp], **m})
        ideal_m = [r for r in rows if r["experiment"] == exp and r.get("regime", r.get("f")) == cond and r["expert_type"] == "ideal"][0]
        for mk in per[0]:
            mean, sd, mn, mx = ms([p[mk] for p in per])
            summary.append({"experiment": exp, "regime" if exp == "4b" else "f": cond, "metric": mk,
                            "ideal": ideal_m.get(mk), "trained_mean": mean, "trained_sd": sd,
                            "trained_min": mn, "trained_max": mx, "n_seeds": args.seeds})
    print("\n== ideal against trained (trained: mean ± sd over seeds [min, max]) ==")
    for r in summary:
        cond = r.get("regime", r.get("f"))
        iv = "n/a" if r["ideal"] is None else f"{r['ideal']:.3f}"
        print(f"{r['experiment']} {str(cond):8s} {r['metric']:26s} ideal {iv:>6s}   trained {r['trained_mean']:.3f} ± {r['trained_sd']:.3f} [{r['trained_min']:.3f}, {r['trained_max']:.3f}]")

    wall = time.time() - t0
    import socket
    json.dump({"host": socket.gethostname(), "wall_time_s": wall, "train_time_s": t_train,
               "config": {"steps": args.steps, "seeds": args.seeds, "net": "MLP 3->128->128->128->1 SiLU, input (x1,x2,log sigma)",
                          "loss": "||sigma*s + eps||^2, sigma log-uniform [0.01,1], Adam 1e-3, full batch 600",
                          "sigma_eval": SIG, "grid": [G, L]},
               "products": rows, "summary": summary, "fit_check": fit},
              open(os.path.join(OUT, "results.json"), "w"), indent=1)

    # figures
    def show(ax, p, title):
        ax.imshow(p, origin="lower", extent=[-L, L, -L, L], cmap="magma")
        ax.axhline(0, color="w", lw=.4, alpha=.5); ax.axvline(0, color="w", lw=.4, alpha=.5)
        ax.set_title(title, fontsize=9); ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    def summ(exp, cond, mk):
        return [r for r in summary if r["experiment"] == exp and r.get("regime", r.get("f")) == cond and r["metric"] == mk][0]
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    for i, reg in enumerate(["centred", "varied"]):
        co, ch = summ("4b", reg, "co_occurrence"), summ("4b", reg, "chimera")
        show(ax[i, 0], ideal_prod[("4b", reg)], f"{reg}: ideal product\nco-occurrence {co['ideal']:.2f} · chimera {ch['ideal']:.2f}")
        show(ax[i, 1], trained_prod[("4b", reg, 0)],
             f"{reg}: trained product (seed 0 shown)\nco-occ {co['trained_mean']:.2f}±{co['trained_sd']:.2f} · chimera {ch['trained_mean']:.2f}±{ch['trained_sd']:.2f} (3 seeds)")
        for a in ax[i]: a.set_xlabel("slot 1 (x1)"); a.set_ylabel("slot 2 (x2)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "4b-contention-ideal-vs-trained.png"), dpi=120); plt.close()
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    for i, f in enumerate([0.0, 0.2]):
        m = summ("4c", f, "mean_x1_top")
        show(ax[i, 0], ideal_prod[("4c", f)], f"overlap {f:.0%}: ideal product\nmean x1 in top half {m['ideal']:.2f}")
        show(ax[i, 1], trained_prod[("4c", f, 0)], f"overlap {f:.0%}: trained product (seed 0 shown)\nmean x1 in top half {m['trained_mean']:.2f}±{m['trained_sd']:.2f} (3 seeds)")
        for a in ax[i]: a.set_xlabel("x1 (dog: -0.5 dry, -0.8 wet)"); a.set_ylabel("x2 (rain above 0)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "4c-interaction-ideal-vs-trained.png"), dpi=120); plt.close()
    fig, ax = plt.subplots(figsize=(7, 4))
    for (k, s), (lp, losses, el) in trained.items():
        ax.plot(np.convolve(losses, np.ones(200) / 200, mode="valid"), lw=.6)
    ax.set_xlabel("step"); ax.set_ylabel("DSM loss, 200-step moving average"); ax.set_title(f"all {len(trained)} trained experts")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "loss-curves.png"), dpi=120); plt.close()
    print(f"\nwall time {wall:.0f}s on {socket.gethostname()}")


if __name__ == "__main__":
    main()
