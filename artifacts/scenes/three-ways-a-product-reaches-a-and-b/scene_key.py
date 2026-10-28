"""Scene key for "Three ways a product reaches A-and-B".

Same toy, grid, regions and experts as the walk's answer key
(artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/answer_key.py), which this file
does not modify. It first reproduces every row of that key's answer_key.json and fails if any share
differs, then adds the values the scene's sliders need between the key's stops, plus three
quantities the scene explains its numbers with:

  circle_holds_data_blob   share of one std-0.15 blob inside its own 3-std circle (1 - e^-4.5)
  sd_x1                    the composed density's standard deviation along x1, per rescue-2b case
  tail_ratio_eps0          at eps = 0, the product's value at the origin over its value at the
                           top-right blob's centre

Run from this folder:  /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python scene_key.py
Writes scene_key.json beside itself.
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = os.path.join(HERE, "../../ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/answer_key.json")

S = 0.15
MU = {"TR": (0.5, 0.5), "BR": (0.5, -0.5), "TL": (-0.5, 0.5), "BL": (-0.5, -0.5)}
g = np.linspace(-1.1, 1.1, 441)
X1, X2 = np.meshgrid(g, g, indexing="xy")
dA = (g[1] - g[0]) ** 2


def blob(q):
    m = MU[q]
    return np.exp(-((X1 - m[0]) ** 2 + (X2 - m[1]) ** 2) / (2 * S**2)) / (2 * np.pi * S**2)


B = {q: blob(q) for q in MU}
near = {q: (X1 - MU[q][0]) ** 2 + (X2 - MU[q][1]) ** 2 < (3 * S) ** 2 for q in MU}
mid = np.hypot(X1, X2) < 3 * S / np.sqrt(2)


def mixture(w):
    return sum(w[q] * B[q] for q in MU)


def shares(dens):
    p = dens / (dens.sum() * dA)
    out = {f"in_{q}_blob": float((p * near[q]).sum() * dA) for q in MU}
    out["at_origin"] = float((p * mid).sum() * dA)
    out["Z"] = float(dens.sum() * dA)
    out["sd_x1"] = float(np.sqrt((p * (X1 - (p * X1).sum() * dA) ** 2).sum() * dA))
    return out


def leftover(own, eps, where):
    w = {q: 0.0 for q in MU}
    for q in where:
        w[q] = eps
    w[own] = 1 - eps * len(where)
    return w


def c2(eps, kind):
    where_a = ["TR", "TL", "BL"] if kind == "even" else ["TR", "BL"]
    where_b = ["TR", "BR", "BL"] if kind == "even" else ["TR", "BL"]
    return shares(mixture(leftover("BR", eps, where_a)) * mixture(leftover("TL", eps, where_b)))


def c1(rr):
    o = (1 - rr) / 3
    w = {"TR": rr, "BR": o, "TL": o, "BL": o}
    px = mixture(w)
    pxa = w["TR"] * B["TR"] + w["BR"] * B["BR"]
    pxb = w["TR"] * B["TR"] + w["TL"] * B["TL"]
    return shares(pxa * pxb / px)


def n1(v, m):
    return np.exp(-((v - m) ** 2) / (2 * S**2)) / np.sqrt(2 * np.pi * S**2)


a1, b2 = n1(X1, 0.5), n1(X2, 0.5)
bg1 = 0.5 * n1(X1, 0.5) + 0.5 * n1(X1, -0.5)
bg2 = 0.5 * n1(X2, 0.5) + 0.5 * n1(X2, -0.5)
pA, pB, pbg = a1 * bg2, bg1 * b2, bg1 * bg2

# 1. Reproduce the walk's key, row by row.
SH = ["in_TR_blob", "in_BR_blob", "in_TL_blob", "in_BL_blob", "at_origin", "Z"]
worst = 0.0
for row in json.load(open(KEY)):
    case = row["case"]
    if case.startswith("construction 2"):
        mine = c2(row["eps"], "even" if "even" in case else "adjacent")
    elif case.startswith("construction 1"):
        mine = c1(row["r_TR"])
    elif "plain" in case:
        mine = shares(pA * pB)
    else:
        mine = shares(pA * pB / pbg)
    for k in SH:
        worst = max(worst, abs(mine[k] - row[k]) / max(abs(row[k]), 1e-300))
assert worst < 1e-9, f"scene key disagrees with the walk's key: worst relative gap {worst}"
print(f"reproduced every row of answer_key.json, worst relative gap {worst:.1e}")

# 2. The scene's slider stops.
EPS = [0.25, 0.2, 0.15, 0.1, 0.075, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001, 0.0001, 1e-6, 0.0]
R = [0.25, 0.05, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-8, 0.0]
keep = lambda d: {k: d[k] for k in SH + ["sd_x1"]}
out = {
    "even": [dict(eps=e, formula_TR=(e / (2 - 4 * e)), **keep(c2(e, "even"))) for e in EPS],
    "adjacent": [dict(eps=e, **keep(c2(e, "adjacent"))) for e in EPS if e > 0],
    "c1": [dict(r=r, **keep(c1(r))) for r in R],
    "b_plain": keep(shares(pA * pB)),
    "b_divided": keep(shares(pA * pB / pbg)),
    "circle_holds_data_blob": float(1 - np.exp(-4.5)),
    "circle_holds_data_blob_grid": float((B["TR"] * near["TR"]).sum() * dA),
    # closed forms: the overlap of two blobs a distance d apart, relative to a blob with itself, is
    # exp(-d^2 / (4 s^2)); the product of two unit blobs sqrt(2) apart has total mass
    # exp(-2 / (4 s^2)) / (4 pi s^2); one blob at distance 1 from its centre is exp(-1 / (2 s^2)) of its peak
    "cross_adjacent": float(np.exp(-1 / (4 * S**2))),
    "cross_diagonal": float(np.exp(-2 / (4 * S**2))),
    "Z_eps0_closed": float(np.exp(-2 / (4 * S**2)) / (4 * np.pi * S**2)),
    "A_at_TR_over_peak": float(np.exp(-1 / (2 * S**2))),
    "tail_ratio_eps0": float(np.exp(-((0.5) + (0.5)) / (2 * S**2)) / np.exp(-((1.0) + (1.0)) / (2 * S**2))),
}
# 3. Expert A's blob times expert B's blob placed on each corner: how much the two overlap sets Z.
# Must agree with the same pins in "Where the Product Lands" (when-a-product-reaches-a-and-b).
out["support"] = {q: keep(shares(B["BR"] * B[q])) for q in ["BR", "TR", "TL"]}
PINS = json.load(open(os.path.join(HERE, "../when-a-product-reaches-a-and-b/checks/scene_key.json")))["support_pins"]
for q, s in out["support"].items():
    gap = max(abs(s[k] - PINS[q][k]) / max(abs(PINS[q][k]), 1e-300) for k in PINS[q])
    assert gap < 1e-9, f"support pin {q} disagrees with the other scene's key: {gap}"
json.dump(out, open(os.path.join(HERE, "scene_key.json"), "w"), indent=1)
for q, s in out["support"].items():
    print(f"support   B at {q}: Z={s['Z']:.4g} BR={s['in_BR_blob']:.4f} TR={s['in_TR_blob']:.4f} origin={s['at_origin']:.4f}")
for name in ["even", "adjacent"]:
    for r in out[name]:
        print(f"{name:9s} eps={r['eps']:<7g} TR={r['in_TR_blob']:.4f} BR={r['in_BR_blob']:.4f} "
              f"TL={r['in_TL_blob']:.4f} BL={r['in_BL_blob']:.4f} origin={r['at_origin']:.4f} Z={r['Z']:.3g}")
for r in out["c1"]:
    print(f"c1        r={r['r']:<7g} TR={r['in_TR_blob']:.4f} origin={r['at_origin']:.4f} Z={r['Z']:.4g}")
for k in ["b_plain", "b_divided"]:
    print(f"{k:9s} TR={out[k]['in_TR_blob']:.4f} sd_x1={out[k]['sd_x1']:.4f}")
print(f"cross adjacent {out['cross_adjacent']:.3g}, diagonal {out['cross_diagonal']:.3g}, "
      f"Z(eps=0) closed form {out['Z_eps0_closed']:.3g}, A at TR centre / A peak {out['A_at_TR_over_peak']:.3g}")
print(f"circle holds {out['circle_holds_data_blob']:.4f} of a data blob (grid {out['circle_holds_data_blob_grid']:.4f}); "
      f"eps=0 origin/TR-centre ratio {out['tail_ratio_eps0']:.3g}")
