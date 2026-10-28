"""Example 9: a 4x4 parts world where cat x dog products are exact Gaussian(-mixture) products."""
import numpy as np, matplotlib.pyplot as plt, json
from scipy.stats import multivariate_normal as mvn

D = 16
HEAD, BODY = np.arange(0, 8), np.arange(8, 16)           # top two rows, bottom two rows
LEFT = np.array([r * 4 + c for r in range(4) for c in (0, 1)]); RIGHT = np.setdiff1d(np.arange(D), LEFT)

def gauss_product(ma, va, mb, vb):
    """Diagonal Gaussians N(ma, diag va) * N(mb, diag vb) = Z * N(m, diag v). Returns m, v, log Z."""
    v = 1 / (1 / va + 1 / vb); m = v * (ma / va + mb / vb)
    logZ = mvn.logpdf(ma, mb, np.diag(va + vb))
    return m, v, logZ

def mixture_product(A, B):
    """A, B: lists of (weight, mean, var). Returns the product's components with normalised weights."""
    comps = []
    for wa, ma, va in A:
        for wb, mb, vb in B:
            m, v, logZ = gauss_product(ma, va, mb, vb)
            comps.append([np.log(wa) + np.log(wb) + logZ, m, v])
    lw = np.array([c[0] for c in comps]); w = np.exp(lw - lw.max()); w /= w.sum()
    return [(float(wi), c[1], c[2], float(c[0])) for wi, c in zip(w, comps)]

res = {}
# (a) complementary tightness: cat is consistent on the head, dog on the body -> mosaic
vcat = np.full(D, 1.0); vcat[HEAD] = 0.01
vdog = np.full(D, 1.0); vdog[BODY] = 0.01
ma, a_v, a_logZ = gauss_product(np.ones(D), vcat, -np.ones(D), vdog)
res["a"] = dict(head_mean=float(ma[HEAD].mean()), body_mean=float(ma[BODY].mean()), logZ=float(a_logZ))
# (b) both consistent everywhere -> midpoint blend
mb_, b_v, b_logZ = gauss_product(np.ones(D), np.full(D, 0.01), -np.ones(D), np.full(D, 0.01))
dist_sd = (1 - mb_[0]) / np.sqrt(0.01)
res["b"] = dict(mean=float(mb_.mean()), product_sd=float(np.sqrt(b_v[0])), logZ=float(b_logZ), sds_from_each_expert=float(dist_sd))
# (c) position varied: each animal sits in the left or the right half (0.5 each), tight there, loose elsewhere
def placed(value, half):
    m = np.zeros(D); m[half] = value; v = np.full(D, 1.0); v[half] = 0.01; return m, v
cat = [(0.5, *placed(1.0, LEFT)), (0.5, *placed(1.0, RIGHT))]
dog = [(0.5, *placed(-1.0, LEFT)), (0.5, *placed(-1.0, RIGHT))]
prod = mixture_product(cat, dog)
names = ["cat-left x dog-left", "cat-left x dog-right", "cat-right x dog-left", "cat-right x dog-right"]
res["c"] = {n: dict(weight=w, logw_unnorm=lw) for n, (w, m, v, lw) in zip(names, prod)}
# (d) the reverse-diffusion error on (a) at abar = 0.5: product of noised experts vs noised product
abar = 0.5
nm, nv, _ = gauss_product(np.sqrt(abar) * np.ones(D), abar * vcat + (1 - abar), -np.sqrt(abar) * np.ones(D), abar * vdog + (1 - abar))
res["d"] = dict(abar=abar, head_mean_product_of_noised=float(nm[HEAD].mean()), head_mean_noised_product=float(np.sqrt(abar) * ma[HEAD].mean()))
json.dump(res, open("numbers.json", "w"), indent=1)
print(json.dumps(res, indent=1))

# figure: cat, dog, and the product mean for (a), (b), (c) top two components
cm = plt.get_cmap("coolwarm_r")
def tile(ax, x, title):
    ax.imshow(np.asarray(x).reshape(4, 4), cmap=cm, vmin=-1, vmax=1)
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
pc = sorted(prod, key=lambda c: -c[0])
fig, ax = plt.subplots(1, 6, figsize=(10, 2.2))
tile(ax[0], np.ones(D), "cat"); tile(ax[1], -np.ones(D), "dog")
tile(ax[2], ma, "(a) mosaic"); tile(ax[3], mb_, "(b) blend")
tile(ax[4], pc[0][1], f"(c) w={pc[0][0]:.2f}"); tile(ax[5], pc[1][1], f"(c) w={pc[1][0]:.2f}")
plt.tight_layout(); plt.savefig("parts-world-tiles.png", dpi=200)
