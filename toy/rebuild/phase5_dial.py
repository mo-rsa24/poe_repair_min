"""Phase 5, the overlap dial: the share of A-and-B examples in each expert's training set, from 0% to 100%.

    python -m rebuild.phase5_dial --out rebuild/out/phase5

Quadrant toy (2D, ideal experts, SIG = 0.05, 800 points each, the quadrant setup of the worked-examples
notebook): A is the left half, B the top half, A∩B the top-left quadrant. At overlap f, a share f of A's
points sit in the top-left (the rest bottom-left) and a share f of B's points sit in the top-left (the rest
top-right). Two A∩B numbers: mass in the top-left quadrant, and mass at least 3 SIG = 0.15 inside it, since
at 0% the two training squares touch at the origin and the product's mass sits in a sliver around that corner.

Parts world (16 pixels, Gaussian-mixture experts): A is "contains a cat", B "contains a dog". At overlap f,
A's set is cat-only images (cat on one half, the other half background) with weight 1 - f and cat-plus-dog
images (cat on one half, dog on the other) with weight f; B likewise. Background is 0 with variance 0.01
(an empty half is as definite as an object) or variance 1.0 (the expert says nothing about the other half,
as in example 9(c)). A sample is A∩B when one half's mean is above 0.75 (cat) and the other's below -0.75 (dog).

For each setting: the exact composition's A∩B mass, then tuned MALA, UHA and reverse diffusion (2000 samples,
3 data seeds for the quadrant toy), each with its A∩B mass and TV to the exact composition.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rebuild import ideal, parts_world as P, samplers as S
from rebuild.device import describe, pick_device

OVERLAPS_2D = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
OVERLAPS_PW = [0.0, 0.001, 0.01, 0.05, 0.2, 0.5, 1.0]
SEEDS = [0, 1, 2]
N, NS, MARGIN = 800, 2000, 0.15
SAMPLERS = ["MALA, tuned", "UHA", "reverse diffusion"]


def make_sets(rng, f, n=N):
    """The worked-examples notebook's quadrant sets: A in the left half, B in the top half, overlap f each."""
    a = np.c_[rng.uniform(-1, 0, n), np.where(rng.random(n) < f, rng.uniform(0, 1, n), rng.uniform(-1, 0, n))]
    b = np.c_[np.where(rng.random(n) < f, rng.uniform(-1, 0, n), rng.uniform(0, 1, n)), rng.uniform(0, 1, n)]
    return a, b


def quadrant(a, b, w):
    return float((w * ((a < 0) & (b > 0))).sum() / w.sum())


def interior(a, b, w):
    return float((w * ((a < -MARGIN) & (b > MARGIN))).sum() / w.sum())


def sample_2d(tgt, which, dev):
    if which == "MALA, tuned":
        return S.tuned_mala(tgt, NS, 2, ideal.V, device=dev)[0].cpu().numpy()
    if which == "UHA":
        return S.uha(tgt, NS, 2, 0.3 * np.sqrt(ideal.V), device=dev).cpu().numpy()
    return S.reverse_diffusion(tgt, NS, 2, ideal.BETAS, device=dev).cpu().numpy()


def quadrant_dial(dev):
    rows, pics = [], {}
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed); torch.manual_seed(100 + seed)
        for f in OVERLAPS_2D:
            A, B = (torch.tensor(d, dtype=torch.float32, device=dev) for d in make_sets(rng, f))
            p = ideal.exact_grid(A, B, lambda a, b: a + b, dev)
            w = p * ideal.CELL
            row = dict(toy="quadrant", seed=seed, overlap=f, exact_quadrant=quadrant(ideal.X1, ideal.X2, w),
                       exact_interior=interior(ideal.X1, ideal.X2, w))
            perfect = ideal.perfect_samples(p, NS, rng)
            row["floor_tv"] = float(ideal.tv(perfect, p))
            tgt = ideal.composed_target(A, B, lambda a, b: a + b)
            for c in SAMPLERS:
                s = sample_2d(tgt, c, dev); s = s[np.isfinite(s).all(1)]
                one = np.ones(len(s))
                row[c] = dict(quadrant=quadrant(s[:, 0], s[:, 1], one), interior=interior(s[:, 0], s[:, 1], one),
                              tv=float(ideal.tv(s, p)), kept=len(s))
                if seed == 0:
                    pics[(f, c)] = s
            if seed == 0:
                pics[(f, "exact")] = p; pics[(f, "data")] = (A.cpu().numpy(), B.cpu().numpy())
            rows.append(row)
            print(f"quadrant seed {seed} overlap {f:4.2f}: exact quadrant {row['exact_quadrant']:.3f} interior {row['exact_interior']:.3f} | "
                  + " | ".join(f"{c} {row[c]['interior']:.3f} TV {row[c]['tv']:.2f}" for c in SAMPLERS)
                  + f" | floor TV {row['floor_tv']:.2f}", flush=True)
    return rows, pics


def pw_experts(f, bg_var):
    cat_only = [P.placed(1.0, P.LEFT, var_out=bg_var), P.placed(1.0, P.RIGHT, var_out=bg_var)]
    dog_only = [P.placed(-1.0, P.LEFT, var_out=bg_var), P.placed(-1.0, P.RIGHT, var_out=bg_var)]
    both = []
    for cat_half, dog_half in [(P.LEFT, P.RIGHT), (P.RIGHT, P.LEFT)]:
        m = np.zeros(P.D); m[cat_half] = 1.0; m[dog_half] = -1.0
        both.append((m, np.full(P.D, 0.01)))
    comps = [(w, c) for w, c in zip([(1 - f) / 2] * 2 + [f / 2] * 2, cat_only + both) if w > 0]
    A = P.Mixture([w for w, _ in comps], [c[0] for _, c in comps], [c[1] for _, c in comps])
    comps = [(w, c) for w, c in zip([(1 - f) / 2] * 2 + [f / 2] * 2, dog_only + both) if w > 0]
    B = P.Mixture([w for w, _ in comps], [c[0] for _, c in comps], [c[1] for _, c in comps])
    return A, B


def pw_ab(x):
    l, r = x[:, P.LEFT].mean(1), x[:, P.RIGHT].mean(1)
    return ((l > 0.75) & (r < -0.75)) | ((l < -0.75) & (r > 0.75))


def parts_world_dial(dev):
    from rebuild.phase4_parts_world import V as PV
    rows = []
    for bg_var in [0.01, 1.0]:
        rng = np.random.default_rng(200); torch.manual_seed(200)
        for f in OVERLAPS_PW:
            A, B = pw_experts(f, bg_var)
            prod = P.mixture_product(A, B)
            exact_w = float(prod.w[pw_ab(prod.m)].sum())
            exact = prod.sample(100000, np.random.default_rng(7))[0]
            perfect = prod.sample(NS, rng)[0]
            row = dict(toy="parts world", background_variance=bg_var, overlap=f, exact_weight_ab=exact_w,
                       exact_draws_ab=float(pw_ab(exact).mean()),
                       floor_tv=float(P.tv_projected(perfect, exact, (P.LEFT, P.RIGHT))),
                       exact_left_right_means_of_main_component=P.project(prod.m[[int(np.argmax(prod.w))]], (P.LEFT, P.RIGHT)).tolist())
            tgt = P.composed_target(A, B, ideal.ABAR, dev)
            for c in SAMPLERS:
                if c == "MALA, tuned":
                    s = S.tuned_mala(tgt, NS, P.D, PV, device=dev)[0]
                elif c == "UHA":
                    s = S.uha(tgt, NS, P.D, 0.3 * np.sqrt(PV), device=dev)
                else:
                    s = S.reverse_diffusion(tgt, NS, P.D, ideal.BETAS, device=dev)
                s = s.cpu().numpy(); s = s[np.isfinite(s).all(1)]
                row[c] = dict(ab=float(pw_ab(s).mean()), tv=float(P.tv_projected(s, exact, (P.LEFT, P.RIGHT))), kept=len(s))
            rows.append(row)
            print(f"parts world bg var {bg_var} overlap {f:5.3f}: exact A∩B {exact_w:.3f} (main component left/right means "
                  f"{np.round(row['exact_left_right_means_of_main_component'][0], 2).tolist()}) | "
                  + " | ".join(f"{c} {row[c]['ab']:.3f} TV {row[c]['tv']:.2f}" for c in SAMPLERS)
                  + f" | floor TV {row['floor_tv']:.2f}", flush=True)
    return rows


def plot(q_rows, pw_rows, pics, out):
    plot_curves(q_rows, pw_rows, out)
    plot_pictures(pics, out)


def plot_curves(q_rows, pw_rows, out):
    colors = {"exact": "k", "MALA, tuned": "tab:orange", "UHA": "tab:green", "reverse diffusion": "tab:blue"}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    xs = np.arange(len(OVERLAPS_2D))
    for key, a, title in [("interior", ax[0], "quadrant toy: A∩B mass at least 0.15 inside the top-left quadrant"),
                          ("quadrant", ax[1], "quadrant toy: A∩B mass in the whole top-left quadrant")]:
        ex = np.array([[r[f"exact_{key}"] for r in q_rows if r["overlap"] == f] for f in OVERLAPS_2D])
        a.errorbar(xs, ex.mean(1), ex.std(1), c="k", marker="o", lw=2, label="exact composition", capsize=3)
        for c in SAMPLERS:
            v = np.array([[r[c][key] for r in q_rows if r["overlap"] == f] for f in OVERLAPS_2D])
            a.errorbar(xs + 0.08 * (SAMPLERS.index(c) + 1), v.mean(1), v.std(1), c=colors[c], marker=".", label=c, capsize=2)
        a.set_xticks(xs); a.set_xticklabels([f"{f:.0%}" for f in OVERLAPS_2D])
        a.set_xlabel("A-and-B share of each training set (3 seeds, bars 1 sd)")
        a.set_ylabel("share of the composition's mass"); a.set_title(title, fontsize=9); a.set_ylim(-0.03, 1.03); a.legend(fontsize=7)
    xs2 = np.arange(len(OVERLAPS_PW))
    for bg, ls in [(0.01, "-"), (1.0, "--")]:
        rr = [r for r in pw_rows if r["background_variance"] == bg]
        ax[2].plot(xs2, [r["exact_weight_ab"] for r in rr], c="k", ls=ls, marker="o", lw=2, label=f"exact, background variance {bg}")
        for c in SAMPLERS[:2]:
            ax[2].plot(xs2, [r[c]["ab"] for r in rr], c=colors[c], ls=ls, marker=".", label=f"{c}, background variance {bg}")
    ax[2].set_xticks(xs2); ax[2].set_xticklabels([f"{f:g}" for f in OVERLAPS_PW])
    ax[2].set_xlabel("share of cat-and-dog images in each expert's training set"); ax[2].set_ylabel("share of samples with a cat half and a dog half")
    ax[2].set_title("parts world: A∩B mass of the product", fontsize=9); ax[2].set_ylim(-0.03, 1.03); ax[2].legend(fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(out, "overlap-dial-curves.png"), dpi=100); plt.close()


def plot_pictures(pics, out):
    show = [0.0, 0.02, 0.2, 1.0]
    fig, ax = plt.subplots(len(show), 5, figsize=(12, 2.5 * len(show)))
    for r, f in enumerate(show):
        A, B = pics[(f, "data")]
        ax[r, 0].scatter(*A.T, s=1, c="tab:red", alpha=.5); ax[r, 0].scatter(*B.T, s=1, c="tab:blue", alpha=.5)
        ax[r, 0].set_title("A (red), B (blue)", fontsize=8); ax[r, 0].set_ylabel(f"overlap {f:.0%}")
        ax[r, 1].imshow(pics[(f, "exact")], origin="lower", extent=[-ideal.L, ideal.L, -ideal.L, ideal.L], cmap="magma"); ax[r, 1].set_title("exact", fontsize=8)
        for j, c in enumerate(SAMPLERS, start=2):
            ax[r, j].scatter(*pics[(f, c)].T, s=1, c="k", alpha=.4); ax[r, j].set_title(c, fontsize=8)
        for a in ax[r]:
            a.set_xlim(-ideal.L, ideal.L); a.set_ylim(-ideal.L, ideal.L); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
            a.axhline(0, c="grey", lw=.4); a.axvline(0, c="grey", lw=.4)
    plt.tight_layout(); plt.savefig(os.path.join(out, "overlap-dial-quadrant-pictures.png"), dpi=90); plt.close()


def main(out):
    os.makedirs(out, exist_ok=True)
    dev = pick_device(); print(describe(dev), flush=True)
    q_rows, pics = quadrant_dial(dev)
    pw_rows = parts_world_dial(dev)
    plot(q_rows, pw_rows, pics, out)
    json.dump(dict(quadrant=q_rows, parts_world=pw_rows), open(os.path.join(out, "overlap-dial.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rebuild/out/phase5")
    ap.add_argument("--replot", action="store_true", help="redraw the curves from overlap-dial.json without sampling")
    a = ap.parse_args()
    if a.replot:
        d = json.load(open(os.path.join(a.out, "overlap-dial.json")))
        plot_curves(d["quadrant"], d["parts_world"], a.out)
    else:
        main(a.out)
