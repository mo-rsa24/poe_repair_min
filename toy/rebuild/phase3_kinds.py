"""Phase 3 check: the kinds notebook's Figure 1, Figure 2, run 1 and run 2, from the rebuilt modules.

    python -m rebuild.phase3_kinds --out rebuild/out/phase3

Prints each table beside the numbers the kinds notebook printed, and a pass or differ mark per row.
Tolerance: exact-composition numbers to the notebook's printed precision (they are deterministic given the
data, and the data is drawn in the notebook's order from default_rng(0)); sampler numbers within 0.03,
since sampler randomness on a GPU need not repeat bit for bit.
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rebuild import ideal, kinds as K, samplers as S
from rebuild.device import describe, pick_device

# what the kinds notebook printed (artifacts/ideas/poe-overlap-dial/kinds-of-composition-on-the-papers-toys.ipynb)
REF_FIG1 = {  # kind: exact, then (measure, TV) for perfect, reverse diffusion, ULA, MALA, UHA
    "intersection": (0.71, [(0.71, .11), (0.62, .31), (0.51, .25), (0.50, .24), (0.66, .13)]),
    "disjunction": (0.50, [(0.51, .23), (0.50, .21), (0.52, .24), (0.50, .24), (0.49, .23)]),
    "co-occurrence": (0.81, [(0.81, .10), (0.92, .19), (0.72, .13), (0.71, .15), (0.80, .10)]),
    "contention": (0.67, [(0.67, .10), (0.57, .23), (0.41, .30), (0.42, .30), (0.62, .12)]),
    "dominance": (0.40, [(0.40, .11), (0.34, .58), (0.37, .41), (0.37, .38), (0.39, .29)]),
    "interaction": (-0.41, [(-0.41, .07), (-0.40, .30), (-0.40, .10), (-0.40, .09), (-0.40, .08)]),
    "negation": (0.10, [(0.09, .15), (0.20, .26), (0.13, .25), (0.11, .25), (0.12, .21)]),
}
REF_FIG2 = {
    "co-occurrence": [0.796, 0.818, 0.847, 0.718, 0.481, 0.116],
    "contention": [0.694, 0.664, 0.685, 0.684, 0.657, 0.002],
    "interaction": [-0.396, -0.394, -0.4, -0.39, -0.394, -0.215],
    "negation in-bar": [0.146, 0.121, 0.085, 0.037, 0.0, 0.0],
    "negation away": [0.387, 0.384, 0.391, 0.439, 0.863, 1.0],
}
REF_RUN1 = {  # kind: accept, TV tuned MALA, floor
    "intersection": (0.60, 0.11, 0.11), "disjunction": (0.61, 0.22, 0.23), "co-occurrence": (0.61, 0.11, 0.10),
    "contention": (0.60, 0.16, 0.10), "dominance": (0.60, 0.30, 0.11), "interaction": (0.61, 0.08, 0.07),
    "negation": (0.60, 0.21, 0.15)}
REF_RUN2 = {  # setting: exact, then (measure, TV) per sampler
    "co-occurrence, top edge 0.4": (0.44, {"perfect": (0.43, .11), "reverse diffusion": (0.38, .22), "ULA": (0.40, .19),
                                           "MALA": (0.42, .18), "MALA, tuned": (0.42, .13), "UHA": (0.43, .13)}),
    "contention, 5% rotated": (0.75, {"perfect": (0.73, .10), "reverse diffusion": (0.21, .69), "ULA": (0.26, .57),
                                      "MALA": (0.24, .57), "MALA, tuned": (0.53, .30), "UHA": (0.53, .27)}),
    "interaction, 2% overlap": (-0.38, {"perfect": (-0.38, .05), "reverse diffusion": (-0.35, .59), "ULA": (-0.37, .26),
                                        "MALA": (-0.37, .27), "MALA, tuned": (-0.38, .08), "UHA": (-0.38, .08)}),
}
SAMPLER_TOL = 0.03


def mark(ok):
    return "matches" if ok else "differs"


def run_samplers(kind, p, rng, dev, cols):
    A, B = kind["data_t"]
    tgt = ideal.composed_target(A, B, kind["compose"])
    out, extra = {}, {}
    for c in cols:
        if c == "perfect":
            out[c] = ideal.perfect_samples(p, K.NS, rng)
        elif c == "reverse diffusion":
            out[c] = S.reverse_diffusion(tgt, K.NS, 2, ideal.BETAS, device=dev).cpu().numpy()
        elif c == "ULA":
            out[c] = S.ula(tgt, K.NS, 2, 0.05 * ideal.V, device=dev).cpu().numpy()
        elif c == "UHA":
            out[c] = S.uha(tgt, K.NS, 2, 0.3 * np.sqrt(ideal.V), device=dev).cpu().numpy()
        elif c == "MALA":
            x, acc = S.mala(tgt, K.NS, 2, 0.05 * ideal.V, device=dev)
            out[c] = x.cpu().numpy(); extra["MALA accept"] = acc
        elif c == "MALA, tuned":
            x, acc, steps = S.tuned_mala(tgt, K.NS, 2, ideal.V, device=dev)
            out[c] = x.cpu().numpy(); extra["tuned accept"] = acc; extra["tuned steps"] = steps
    return out, extra


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dev = pick_device()
    print(describe(dev))
    rng = np.random.default_rng(0); torch.manual_seed(0)
    tt = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
    KINDS = K.build_kinds(rng)
    for k in KINDS:
        k["data_t"] = tuple(tt(d) for d in k["data"])
    report = dict(figure1=[], figure2={}, run1=[], run2=[], marks={})

    # Figure 1: every sampler on every kind; same call order as the notebook
    t0 = time.time()
    order = ["perfect", "reverse diffusion", "ULA", "UHA", "MALA"]
    for k in KINDS:
        p = ideal.exact_grid(*k["data_t"], k["compose"], dev); k["p"] = p
        k["exact"] = float(k["measure"](ideal.X1, ideal.X2, p * ideal.CELL))
        k["samples"], extra = run_samplers(k, p, rng, dev, order)
        k["accept"] = extra["MALA accept"]
    print(f"Figure 1 sampling: {time.time() - t0:.0f} s")
    cols = ["perfect", "reverse diffusion", "ULA", "MALA", "UHA"]
    print("\nFigure 1 (measure / TV), rebuilt  [notebook]  mark")
    fig1_ok = True
    for k in KINDS:
        ref_exact, ref_rows = REF_FIG1[k["name"]]
        row = dict(kind=k["name"], exact=k["exact"], mala_accept_mean=float(k["accept"][-10:].mean()),
                   mala_accept_sd_across_samples=float(k["accept"][-10:].mean(0).std()))
        ok_exact = abs(k["exact"] - ref_exact) <= 0.005
        cells = []
        for c, (rm, rt) in zip(cols, ref_rows):
            s = k["samples"][c]
            m, t = K.measure_samples(k, s), float(ideal.tv(s, k["p"]))
            ok = abs(m - rm) <= SAMPLER_TOL and abs(t - rt) <= SAMPLER_TOL
            row[c] = dict(measure=m, tv=t, ref_measure=rm, ref_tv=rt, mark=mark(ok))
            cells.append(f"{m:5.2f}/{t:.2f} [{rm:5.2f}/{rt:.2f}] {'ok' if ok else 'DIFF'}")
            fig1_ok &= ok if c in ("perfect",) else True
        fig1_ok &= ok_exact
        report["figure1"].append(row)
        print(f"{k['name']:>14} exact {k['exact']:5.2f} [{ref_exact:5.2f}] {'ok' if ok_exact else 'DIFF'} | " + " | ".join(cells))
    print("MALA (fixed library class, step 0.05 v_t) acceptance, last 10 levels:",
          {r["kind"]: round(r["mala_accept_mean"], 2) for r in report["figure1"]})

    # Figure 2: exact dials
    def exact_measure(A, B, compose, measure):
        p = ideal.exact_grid(tt(A), tt(B), compose, dev)
        return float(measure(ideal.X1, ideal.X2, p * ideal.CELL)), p
    dials = {
        "co-occurrence": ([1.0, 0.8, 0.6, 0.5, 0.4, 0.3], lambda v: (K.vbar(rng, K.N, 0.0, 0.2, -1.0, v), K.hbar(rng, K.N, 0.6)), KINDS[2]),
        "contention": ([0.5, 0.3, 0.2, 0.1, 0.05, 0.0], lambda v: (K.rotated(rng, -0.2, K.N, v), K.rotated(rng, 0.2, K.N, v)), KINDS[3]),
        "interaction": ([0.3, 0.2, 0.1, 0.05, 0.02, 0.0], lambda v: K.wet_dog(rng, v), KINDS[5]),
    }
    fig2_ok = True
    print("\nFigure 2 (exact dial curves), rebuilt [notebook]")
    for name, (vals, make, kind) in dials.items():
        ms = [exact_measure(*make(v), kind["compose"], kind["measure"])[0] for v in vals]
        ok = all(abs(a - b) <= 0.0005 + 1e-9 for a, b in zip(np.round(ms, 3), REF_FIG2[name]))
        fig2_ok &= ok
        report["figure2"][name] = dict(dial=vals, measure=ms, ref=REF_FIG2[name], mark=mark(ok))
        print(f"{name:>14}: {np.round(ms, 3).tolist()}  [{REF_FIG2[name]}]  {mark(ok)}")
    ws = [0.0, 0.02, 0.05, 0.1, 0.3, 1.0]
    ring, bar = K.toy_gmm(rng, K.N), K.vbar(rng, K.N, 0.0, 0.2)
    far = lambda a, b, w: (w * ~K.near(a, b, K.RING, 0.08)).sum()
    inbar, away = [], []
    for w in ws:
        m, p = exact_measure(ring, bar, K.negation(w), KINDS[6]["measure"])
        inbar.append(m); away.append(float(far(ideal.X1, ideal.X2, p * ideal.CELL)))
    for name, vals in [("negation in-bar", inbar), ("negation away", away)]:
        ok = all(abs(a - b) <= 0.0005 + 1e-9 for a, b in zip(np.round(vals, 3), REF_FIG2[name]))
        fig2_ok &= ok
        report["figure2"][name] = dict(dial=ws, measure=vals, ref=REF_FIG2[name], mark=mark(ok))
        print(f"{name:>14}: {np.round(vals, 3).tolist()}  [{REF_FIG2[name]}]  {mark(ok)}")

    # Run 1: tuned MALA against the floor
    print("\nRun 1: tuned MALA (accept, TV) [notebook]")
    for k in KINDS:
        A, B = k["data_t"]
        x, acc, steps = S.tuned_mala(ideal.composed_target(A, B, k["compose"]), K.NS, 2, ideal.V, device=dev)
        x = x.cpu().numpy(); t = float(ideal.tv(x, k["p"])); a = float(acc[-10:].mean())
        floor = next(r for r in report["figure1"] if r["kind"] == k["name"])["perfect"]["tv"]
        ra, rt, rf = REF_RUN1[k["name"]]
        ok = abs(a - ra) <= SAMPLER_TOL and abs(t - rt) <= SAMPLER_TOL
        report["run1"].append(dict(kind=k["name"], accept=a, tv=t, floor=floor, step_ratio=steps[-1] / (0.05 * ideal.V[-1]),
                                   accept_sd_across_samples=float(acc[-10:].mean(0).std()), ref=(ra, rt, rf), mark=mark(ok)))
        print(f"{k['name']:>14}: accept {a:.2f} TV {t:.2f} floor {floor:.2f} step x{steps[-1] / (0.05 * ideal.V[-1]):.1f}  [{ra:.2f} {rt:.2f} {rf:.2f}] {mark(ok)}")

    # Run 2: the steep points
    STEEP = [("co-occurrence, top edge 0.4", (K.vbar(rng, K.N, 0.0, 0.2, -1.0, 0.4), K.hbar(rng, K.N, 0.6)), KINDS[2]),
             ("contention, 5% rotated", (K.rotated(rng, -0.2, K.N, 0.05), K.rotated(rng, 0.2, K.N, 0.05)), KINDS[3]),
             ("interaction, 2% overlap", K.wet_dog(rng, 0.02), KINDS[5])]
    cols2 = ["perfect", "reverse diffusion", "ULA", "MALA", "MALA, tuned", "UHA"]
    print("\nRun 2: steep points (measure, TV) [notebook]")
    for name, data, base in STEEP:
        k = dict(base, name=name, data_t=tuple(tt(d) for d in data))
        p = ideal.exact_grid(*k["data_t"], k["compose"], dev)
        ex = float(k["measure"](ideal.X1, ideal.X2, p * ideal.CELL))
        smp, _ = run_samplers(k, p, rng, dev, cols2)
        ref_ex, ref = REF_RUN2[name]
        print(f"  {name}: exact {ex:.2f} [{ref_ex:.2f}]")
        for c in cols2:
            m, t = K.measure_samples(k, smp[c]), float(ideal.tv(smp[c], p))
            ok = abs(m - ref[c][0]) <= SAMPLER_TOL + 0.02 and abs(t - ref[c][1]) <= SAMPLER_TOL + 0.02
            report["run2"].append(dict(setting=name, sampler=c, exact=ex, measure=m, tv=t, ref=ref[c], mark=mark(ok)))
            print(f"    {c:>17}: {m:5.2f} TV {t:.2f}  [{ref[c][0]:5.2f} {ref[c][1]:.2f}] {mark(ok)}")

    # Figure 1 picture
    fig, ax = plt.subplots(len(KINDS), 8, figsize=(19, 2.5 * len(KINDS)))
    for r, k in enumerate(KINDS):
        A, B = k["data"]
        for j, (d, c, n) in enumerate([(A, "tab:red", "p1"), (B, "tab:blue", "p2")]):
            ax[r, j].scatter(*d.T, s=1, c=c, alpha=.5); ax[r, j].set_title(n)
        ax[r, 0].set_ylabel(k["name"], fontsize=9)
        ax[r, 2].imshow(k["p"], origin="lower", extent=[-ideal.L, ideal.L, -ideal.L, ideal.L], cmap="magma")
        ax[r, 2].set_title("exact")
        for j, c in enumerate(cols, start=3):
            s = k["samples"][c]
            ax[r, j].scatter(*s.T, s=1, c="k", alpha=.4)
            ax[r, j].set_title(f"{c}  TV {ideal.tv(s, k['p']):.2f}")
        for a in ax[r]:
            a.set_xlim(-ideal.L, ideal.L); a.set_ylim(-ideal.L, ideal.L); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "figure1-every-sampler-on-every-kind.png"), dpi=90); plt.close()

    report["marks"] = dict(figure1_exact_and_floor=mark(fig1_ok), figure2=mark(fig2_ok))
    json.dump(report, open(os.path.join(out_dir, "phase3.json"), "w"), indent=1, default=float)
    print("\nmarks:", report["marks"])


def figure1_off_canvas(out_dir):
    """Replay Figure 1's sampling as main() does (same seeds, kinds and call order) and record, per kind and
    sampler, the share of samples outside the [-L, L]^2 canvas. ideal.tv's histogram drops those samples, so
    its TV reads low by half that share. Adds the key "figure1_off_canvas" to phase3.json and changes nothing
    else in it. The replay's own TV is stored beside the share so it can be checked against figure1's."""
    dev = pick_device(); print(describe(dev))
    rng = np.random.default_rng(0); torch.manual_seed(0)
    tt = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
    KINDS = K.build_kinds(rng)
    for k in KINDS:
        k["data_t"] = tuple(tt(d) for d in k["data"])
    order = ["perfect", "reverse diffusion", "ULA", "UHA", "MALA"]
    path = os.path.join(out_dir, "phase3.json")
    report = json.load(open(path))
    fig1 = {r["kind"]: r for r in report["figure1"]}
    rows = {}
    for k in KINDS:
        p = ideal.exact_grid(*k["data_t"], k["compose"], dev)
        samples, _ = run_samplers(k, p, rng, dev, order)
        rows[k["name"]] = {}
        for c in order:
            s = samples[c]
            off = float((~np.isfinite(s).all(1) | (np.abs(s) > ideal.L).any(1)).mean())
            rows[k["name"]][c] = dict(off_canvas_share=off, tv_shift=off / 2, tv_replay=float(ideal.tv(s, p)),
                                      tv_figure1=fig1[k["name"]][c]["tv"])
        print(k["name"], {c: round(v["off_canvas_share"], 4) for c, v in rows[k["name"]].items()})
    worst = max(((n, c, v["off_canvas_share"]) for n, r in rows.items() for c, v in r.items()), key=lambda z: z[2])
    tv_gap = max(abs(v["tv_replay"] - v["tv_figure1"]) for r in rows.values() for v in r.values())
    print(f"largest off-canvas share {worst[2]:.4f} ({worst[1]} on {worst[0]}); largest |TV replay - TV figure1| {tv_gap:.4f}")
    report["figure1_off_canvas"] = dict(rows=rows, max_share=worst[2], max_kind=worst[0], max_sampler=worst[1],
                                        max_tv_gap_replay_vs_figure1=tv_gap)
    json.dump(report, open(path, "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rebuild/out/phase3")
    ap.add_argument("--off-canvas", action="store_true",
                    help="only replay Figure 1's sampling and add the off-canvas shares to phase3.json")
    a = ap.parse_args()
    figure1_off_canvas(a.out) if a.off_canvas else main(a.out)
