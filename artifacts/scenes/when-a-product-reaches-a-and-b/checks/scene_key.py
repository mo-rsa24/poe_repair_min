"""Every number the scene shows, computed exactly as the walk's answer key computes it.

The grid, the blobs, the regions and the four constructions are copied unchanged from
artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/answer_key.py. This script adds
only the slider stops that key does not print, plus the drag pins of the support press, and first
reproduces every row of the key's answer_key.json to prove it is the same computation.
Writes scene_key.json beside itself; prints one PASS or FAIL line per reproduced key row.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = os.path.join(HERE, "../../../ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/answer_key.json")

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
    return out

def leftover(own, eps, where):
    w = {q: 0.0 for q in MU}
    for q in where:
        w[q] = eps
    w[own] = 1 - eps * len(where)
    return w

def c2(eps, mode):
    where_a = ["TR", "TL", "BL"] if mode == "even" else ["TR", "BL"]
    where_b = ["TR", "BR", "BL"] if mode == "even" else ["TR", "BL"]
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
pA2b, pB2b, pbg = a1 * bg2, bg1 * b2, bg1 * bg2

# 1. Reproduce every row of the walk's key.
fails = 0
for row in json.load(open(KEY)):
    case = row["case"]
    if case.startswith("construction 2"):
        got = c2(row["eps"], "even" if "even" in case else "adjacent")
        tag = f"{case} eps={row['eps']}"
    elif case.startswith("construction 1"):
        got = c1(row["r_TR"]); tag = f"{case} r={row['r_TR']}"
    elif "plain" in case:
        got = shares(pA2b * pB2b); tag = case
    else:
        got = shares(pA2b * pB2b / pbg); tag = case
    for k in ["in_TR_blob", "in_BR_blob", "in_TL_blob", "in_BL_blob", "at_origin", "Z"]:
        ok = abs(got[k] - row[k]) <= 1e-12 * max(1.0, abs(row[k]))
        fails += not ok
        if not ok:
            print(f"FAIL  {tag} {k}: got {got[k]!r}, key {row[k]!r}")
    print(f"{'PASS' if all(abs(got[k]-row[k]) <= 1e-12*max(1,abs(row[k])) for k in got) else 'FAIL'}  {tag}")

# 2. The scene's slider stops.
EPS = [0.25, 0.2, 0.15, 0.1, 0.075, 0.05, 0.025, 0.01, 0.005, 0.001, 0.0]
R = [0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0]
out = {
    "source_key": "artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/answer_key.json",
    "c2_even": {"%g" % e: dict(c2(e, "even"), formula_TR=(e / (2 - 4 * e))) for e in EPS},
    "c2_adjacent": {"%g" % e: c2(e, "adjacent") for e in EPS},
    "c1": {"%g" % r: c1(r) for r in R},
    "b2_plain": shares(pA2b * pB2b),
    "b2_divided": shares(pA2b * pB2b / pbg),
}
# 3. Support press: expert A is the bottom-right blob alone; expert B is one blob dragged to each quadrant.
out["support_pins"] = {q: shares(B["BR"] * B[q]) for q in MU}
# The share one data blob keeps inside its own 3-std circle, 1 - exp(-9/2), for the disclosure.
out["one_blob_inside_3std"] = {"grid": float((B["TR"] * near["TR"]).sum() * dA), "closed_form": float(1 - np.exp(-4.5))}
json.dump(out, open(os.path.join(HERE, "scene_key.json"), "w"), indent=1)
print(f"{'FAIL' if fails else 'PASS'}  all key rows reproduced ({fails} mismatches)")
for e in EPS:
    r = out["c2_even"]["%g" % e]; a = out["c2_adjacent"]["%g" % e]
    print(f"eps={e:<6} even TR={r['in_TR_blob']:.4f} BR={r['in_BR_blob']:.4f} origin={r['at_origin']:.4f} Z={r['Z']:.3g} formula={r['formula_TR']:.4f} | adjacent TR={a['in_TR_blob']:.4f} BL={a['in_BL_blob']:.4f} origin={a['at_origin']:.4f}")
for r in R:
    c = out["c1"]["%g" % r]; print(f"r={r:<6} c1 TR={c['in_TR_blob']:.4f} origin={c['at_origin']:.4f} Z={c['Z']:.3g}")
for q, s in out["support_pins"].items():
    print(f"support B at {q}: Z={s['Z']:.4g} BR={s['in_BR_blob']:.4f} TR={s['in_TR_blob']:.4f} BL={s['in_BL_blob']:.4f} origin={s['at_origin']:.4f}")
print("one blob inside 3 std:", out["one_blob_inside_3std"])
sys.exit(1 if fails else 0)
