"""Exact answer key for the four-quadrant toy: where the composed density sits.

Ideal experts are blob mixtures with known weights, so every number is computed on a grid,
with no training and no sampler. Region "in the A-and-B blob" = within 3 std of (0.5, 0.5).
"""
import json
import numpy as np

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
mid = np.hypot(X1, X2) < 3 * S / np.sqrt(2)  # where the product of two far-apart blobs lands


def mixture(w):
    return sum(w[q] * B[q] for q in MU)


def shares(dens):
    p = dens / (dens.sum() * dA)
    out = {f"in_{q}_blob": float((p * near[q]).sum() * dA) for q in MU}
    out["at_origin"] = float((p * mid).sum() * dA)
    out["Z"] = float(dens.sum() * dA)
    return out


def leftover(own, eps, where):
    w = {q: 0.0 for q in MU}
    for q in where:
        w[q] = eps
    w[own] = 1 - eps * len(where)
    return w


rows = []
# Separately fine-tuned experts (construction 2), leftover spread evenly over the other 3 quadrants.
for eps in [0.25, 0.1, 0.05, 0.01, 0.001, 1e-4, 1e-6, 0.0]:
    a = mixture(leftover("BR", eps, ["TR", "TL", "BL"]))
    b = mixture(leftover("TL", eps, ["TR", "BR", "BL"]))
    r = shares(a * b)
    r.update(case="construction 2, leftover even", eps=eps,
             formula_TR=(eps / (2 - 4 * eps)) if eps > 0 else 0.0)
    rows.append(r)
# Leftover only on the two quadrants next to each expert's own.
for eps in [0.1, 0.01, 0.001]:
    a = mixture(leftover("BR", eps, ["TR", "BL"]))
    b = mixture(leftover("TL", eps, ["TR", "BL"]))
    r = shares(a * b)
    r.update(case="construction 2, leftover adjacent only", eps=eps, formula_TR=0.5)
    rows.append(r)
# One conditional model (construction 1), "contains" labels, A-and-B share r of the data.
for rr in [0.25, 0.05, 0.01, 0.001, 0.0]:
    o = (1 - rr) / 3
    w = {"TR": rr, "BR": o, "TL": o, "BL": o}
    px = mixture(w)
    pxa = w["TR"] * B["TR"] + w["BR"] * B["BR"]  # p(x, A)
    pxb = w["TR"] * B["TR"] + w["TL"] * B["TL"]  # p(x, B)
    r = shares(pxa * pxb / px)  # proportional to p(x) p(A|x) p(B|x)
    r.update(case="construction 1, contains labels", r_TR=rr)
    rows.append(r)

# Rescue 2b: experts that model only their own coordinate (x1 is the "lung" part, x2 the "chest wall"
# part) and take the other coordinate from the background marginal of all data. No A-and-B data at all.
def n1(v, m):
    return np.exp(-((v - m) ** 2) / (2 * S**2)) / np.sqrt(2 * np.pi * S**2)


a1 = n1(X1, 0.5)                       # expert A's own part: A data (bottom-right only) seen along x1
b2 = n1(X2, 0.5)                       # expert B's own part: B data (top-left only) seen along x2
bg1 = 0.5 * n1(X1, 0.5) + 0.5 * n1(X1, -0.5)   # background marginals, from all four quadrants
bg2 = 0.5 * n1(X2, 0.5) + 0.5 * n1(X2, -0.5)
pA, pB, pbg = a1 * bg2, bg1 * b2, bg1 * bg2
for name, dens in [("plain product pA*pB", pA * pB), ("background-divided pA*pB/pbg", pA * pB / pbg)]:
    r = shares(dens)
    r.update(case="2b factorised experts, " + name, r_TR=0.0)
    rows.append(r)

for r in rows:
    keys = ["in_TR_blob", "in_BR_blob", "in_TL_blob", "in_BL_blob", "at_origin", "Z"]
    tag = r["case"] + (f" eps={r['eps']}" if "eps" in r else f" r={r['r_TR']}")
    print(f"{tag:48s} " + " ".join(f"{k[3:5] if k.startswith('in_') else k[:6]}={r[k]:.4g}" for k in keys)
          + (f"  formula={r['formula_TR']:.4g}" if "formula_TR" in r else ""))
json.dump(rows, open("answer_key.json", "w"), indent=1)
