"""Scene key for the parts-world page.

First reproduces every number in the walk's answer key
(artifacts/ideas/poe-overlap-dial/example-9-parts-world/numbers.json, written by parts_world.py)
with the same two functions, copied unchanged. Then adds the values the page shows that the walk's
key does not print: the slider stops, the tightened-dog-head case, the hand check, the per-pixel
pieces of each log Z, and the noise sweep. Writes scene_key.json beside this file.
"""
import itertools, json, os
import numpy as np
from scipy.stats import multivariate_normal as mvn

HERE = os.path.dirname(os.path.abspath(__file__))
WALK = os.path.join(HERE, "../../../ideas/poe-overlap-dial/example-9-parts-world/numbers.json")

D = 16
HEAD, BODY = np.arange(0, 8), np.arange(8, 16)
LEFT = np.array([r * 4 + c for r in range(4) for c in (0, 1)]); RIGHT = np.setdiff1d(np.arange(D), LEFT)

# ---- copied unchanged from parts_world.py ----
def gauss_product(ma, va, mb, vb):
    v = 1 / (1 / va + 1 / vb); m = v * (ma / va + mb / vb)
    logZ = mvn.logpdf(ma, mb, np.diag(va + vb))
    return m, v, logZ

def mixture_product(A, B):
    comps = []
    for wa, ma, va in A:
        for wb, mb, vb in B:
            m, v, logZ = gauss_product(ma, va, mb, vb)
            comps.append([np.log(wa) + np.log(wb) + logZ, m, v])
    lw = np.array([c[0] for c in comps]); w = np.exp(lw - lw.max()); w /= w.sum()
    return [(float(wi), c[1], c[2], float(c[0])) for wi, c in zip(w, comps)]
# ----------------------------------------------

def experts(cat_head, cat_body, dog_head, dog_body):
    vc = np.empty(D); vc[HEAD] = cat_head; vc[BODY] = cat_body
    vd = np.empty(D); vd[HEAD] = dog_head; vd[BODY] = dog_body
    return np.ones(D), vc, -np.ones(D), vd

def single(cat_head, cat_body, dog_head, dog_body):
    ma, va, mb, vb = experts(cat_head, cat_body, dog_head, dog_body)
    m, v, logZ = gauss_product(ma, va, mb, vb)
    return dict(head_mean=float(m[HEAD].mean()), body_mean=float(m[BODY].mean()),
                head_var=float(v[HEAD][0]), body_var=float(v[BODY][0]), logZ=float(logZ))

def noised(cat_head, cat_body, dog_head, dog_body, abar):
    ma, va, mb, vb = experts(cat_head, cat_body, dog_head, dog_body)
    m, v, _ = gauss_product(ma, va, mb, vb)
    s = np.sqrt(abar)
    nm, nv, _ = gauss_product(s * ma, abar * va + (1 - abar), s * mb, abar * vb + (1 - abar))
    return dict(head_summed=float(nm[HEAD].mean()), body_summed=float(nm[BODY].mean()),
                head_target=float(s * m[HEAD].mean()), body_target=float(s * m[BODY].mean()))

out = {}

# 1. reproduce the walk's key
walk = json.load(open(WALK))
mine = {}
a = single(0.01, 1.0, 1.0, 0.01)
mine["a"] = dict(head_mean=a["head_mean"], body_mean=a["body_mean"], logZ=a["logZ"])
b = single(0.01, 0.01, 0.01, 0.01)
mine["b"] = dict(mean=(b["head_mean"] + b["body_mean"]) / 2, product_sd=float(np.sqrt(b["head_var"])), logZ=b["logZ"],
                 sds_from_each_expert=float((1 - b["head_mean"]) / np.sqrt(0.01)))
def placed(value, half):
    m = np.zeros(D); m[half] = value; v = np.full(D, 1.0); v[half] = 0.01; return m, v
cat = [(0.5, *placed(1.0, LEFT)), (0.5, *placed(1.0, RIGHT))]
dog = [(0.5, *placed(-1.0, LEFT)), (0.5, *placed(-1.0, RIGHT))]
prod = mixture_product(cat, dog)
names = ["cat-left x dog-left", "cat-left x dog-right", "cat-right x dog-left", "cat-right x dog-right"]
mine["c"] = {n: dict(weight=w, logw_unnorm=lw) for n, (w, m, v, lw) in zip(names, prod)}
d = noised(0.01, 1.0, 1.0, 0.01, 0.5)
mine["d"] = dict(abar=0.5, head_mean_product_of_noised=d["head_summed"], head_mean_noised_product=d["head_target"])

def flat(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items(): yield from flat(v, p + "/" + k)
    else: yield p, x
W, M = dict(flat(walk)), dict(flat(mine))
assert W.keys() == M.keys(), (W.keys() ^ M.keys())
worst = 0.0
for k in W:
    gap = abs(W[k] - M[k]) / max(1.0, abs(W[k])); worst = max(worst, gap)
    print(f"walk key {k:40s} {W[k]!r:>24}  this script {M[k]!r:>24}  rel gap {gap:.1e}")
assert worst < 1e-12, worst
print(f"all {len(W)} walk-key numbers reproduced, worst relative gap {worst:.1e}")
out["walk"] = walk

# 2. extras the page shows
out["a_dog_head_tight"] = single(0.01, 1.0, 0.01, 0.01)
out["hand_check"] = single(0.02, 1.0, 0.5, 0.01)
out["pixel_logZ"] = {
    "sure_vs_loose_opposite": float(mvn.logpdf(1, -1, 1.01)),     # +1 sure against -1 loose
    "sure_vs_sure_opposite": float(mvn.logpdf(1, -1, 0.02)),      # +1 sure against -1 sure
    "sure_vs_loose_zero": float(mvn.logpdf(1, 0, 1.01)),          # +1 sure against a loose 0
    "loose_zero_vs_loose_zero": float(mvn.logpdf(0, 0, 2.0))}
out["c_component_means"] = {n: dict(left=float(m[LEFT].mean()), right=float(m[RIGHT].mean()), var_left=float(v[LEFT][0]), var_right=float(v[RIGHT][0]))
                            for n, (w, m, v, lw) in zip(names, prod)}
out["c_gap"] = float(prod[1][3] - prod[0][3])
out["d_noised_head"] = dict(cat_var=0.5 * 0.01 + 0.5, dog_var=0.5 * 1.0 + 0.5, precision_ratio=(1 / 0.505) / (1 / 1.0),
                            cat_mean=float(np.sqrt(0.5)), dog_mean=float(-np.sqrt(0.5)))

VSTOPS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
ASTOPS = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
out["vstops"], out["astops"] = VSTOPS, ASTOPS
grid = {}
for ch, cb, dh, db in itertools.product(VSTOPS, repeat=4):
    grid[f"{ch},{cb},{dh},{db}"] = single(ch, cb, dh, db)
out["grid"] = grid
rng = np.random.default_rng(0)
combos = [(0.01, 1.0, 1.0, 0.01), (0.01, 0.01, 0.01, 0.01)] + [tuple(rng.choice(VSTOPS, 4)) for _ in range(18)]
out["noise"] = {f"{c[0]},{c[1]},{c[2]},{c[3]}|{ab}": noised(*c, ab) for c in combos for ab in ASTOPS}
json.dump(out, open(os.path.join(HERE, "scene_key.json"), "w"), indent=1)

print("tight dog head:", out["a_dog_head_tight"])
print("hand check (cat head 0.02, dog head 0.5):", out["hand_check"])
print("pixel log Z:", out["pixel_logZ"])
print("(c) component means:", out["c_component_means"])
print("(c) gap:", out["c_gap"])
print("(d) noised head:", out["d_noised_head"])
print(f"grid states {len(grid)}, noise states {len(out['noise'])}")
