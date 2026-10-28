"""The 16-pixel parts world: diagonal Gaussian-mixture experts whose noised densities and products are exact.

A sample is 16 numbers drawn as a 4 x 4 image. Pixels 0 to 7 are the head (top two rows) and 8 to 15 the
body; LEFT and RIGHT are the two column halves. An expert is a mixture of diagonal Gaussians. Noising a
mixture at step t keeps it a mixture (means sqrt(abar) m, variances abar v + 1 - abar), and the product of two
mixtures is a mixture over every pair of components, so every target here is exact in 16 dimensions.
"""
import numpy as np
import torch

D = 16
HEAD, BODY = np.arange(0, 8), np.arange(8, 16)
LEFT = np.array([r * 4 + c for r in range(4) for c in (0, 1)])
RIGHT = np.setdiff1d(np.arange(D), LEFT)
LOG2PI = np.log(2 * np.pi)


class Mixture:
    """Weights (K,), means (K, D), variances (K, D), float64 NumPy."""

    def __init__(self, w, m, v):
        self.w = np.asarray(w, dtype=np.float64)
        self.m = np.atleast_2d(np.asarray(m, dtype=np.float64))
        self.v = np.atleast_2d(np.asarray(v, dtype=np.float64))
        self.w = self.w / self.w.sum()

    def noised(self, abar):
        return Mixture(self.w, np.sqrt(abar) * self.m, abar * self.v + (1 - abar))

    def sample(self, n, rng):
        k = rng.choice(len(self.w), n, p=self.w)
        return self.m[k] + np.sqrt(self.v[k]) * rng.standard_normal((n, self.m.shape[1])), k

    def torch(self, device):
        f = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
        return f(np.log(self.w)), f(self.m), f(self.v)

    def logp_torch(self, x, params):
        logw, m, v = params
        comp = -0.5 * (((x[:, None, :] - m) ** 2) / v + torch.log(v) + LOG2PI).sum(-1)
        return torch.logsumexp(comp + logw, 1)

    def responsibilities(self, x):
        """Each sample's posterior over components, (n, K), float64."""
        with np.errstate(divide="ignore"):   # a component of weight 0 gets log-weight -inf, so responsibility 0
            comp = -0.5 * (((x[:, None, :] - self.m) ** 2) / self.v + np.log(self.v) + LOG2PI).sum(-1) + np.log(self.w)
        comp -= comp.max(1, keepdims=True)
        r = np.exp(comp); return r / r.sum(1, keepdims=True)


def gauss_product(ma, va, mb, vb):
    """N(ma, diag va) N(mb, diag vb) = Z N(m, diag v). Returns m, v, log Z, with Z = N(ma; mb, diag(va + vb))."""
    v = 1 / (1 / va + 1 / vb)
    m = v * (ma / va + mb / vb)
    s = va + vb
    logZ = -0.5 * (((ma - mb) ** 2) / s + np.log(s) + LOG2PI).sum(-1)
    return m, v, logZ


def mixture_product(A, B):
    """The product A x B as a Mixture over every pair (i, j), pair index i * len(B.w) + j, with unnormalised log weights."""
    ma, va = A.m[:, None, :], A.v[:, None, :]
    mb, vb = B.m[None, :, :], B.v[None, :, :]
    m, v, logZ = gauss_product(ma, va, mb, vb)
    lw = np.log(A.w)[:, None] + np.log(B.w)[None, :] + logZ
    lw = lw.reshape(-1)
    w = np.exp(lw - lw.max())
    out = Mixture(w, m.reshape(-1, A.m.shape[1]), v.reshape(-1, A.m.shape[1]))
    out.logw_unnorm = lw
    return out


def placed(value, half, var_in=0.01, var_out=1.0, bg=0.0):
    """An object of a given value on one half, tight there; the other half at bg with variance var_out."""
    m = np.full(D, bg); m[half] = value
    v = np.full(D, var_out); v[half] = var_in
    return m, v


def numbers():
    """The four results of example 9 (numbers.json), recomputed."""
    res = {}
    vcat = np.full(D, 1.0); vcat[HEAD] = 0.01
    vdog = np.full(D, 1.0); vdog[BODY] = 0.01
    ma, _, a_logZ = gauss_product(np.ones(D), vcat, -np.ones(D), vdog)
    res["a"] = dict(head_mean=float(ma[HEAD].mean()), body_mean=float(ma[BODY].mean()), logZ=float(a_logZ))
    mb_, b_v, b_logZ = gauss_product(np.ones(D), np.full(D, 0.01), -np.ones(D), np.full(D, 0.01))
    res["b"] = dict(mean=float(mb_.mean()), product_sd=float(np.sqrt(b_v[0])), logZ=float(b_logZ),
                    sds_from_each_expert=float((1 - mb_[0]) / np.sqrt(0.01)))
    cat = Mixture([0.5, 0.5], *zip(placed(1.0, LEFT), placed(1.0, RIGHT)))
    dog = Mixture([0.5, 0.5], *zip(placed(-1.0, LEFT), placed(-1.0, RIGHT)))
    prod = mixture_product(cat, dog)
    names = ["cat-left x dog-left", "cat-left x dog-right", "cat-right x dog-left", "cat-right x dog-right"]
    res["c"] = {n: dict(weight=float(w), logw_unnorm=float(lw)) for n, w, lw in zip(names, prod.w, prod.logw_unnorm)}
    abar = 0.5
    nm, _, _ = gauss_product(np.sqrt(abar) * np.ones(D), abar * vcat + (1 - abar), -np.sqrt(abar) * np.ones(D), abar * vdog + (1 - abar))
    res["d"] = dict(abar=abar, head_mean_product_of_noised=float(nm[HEAD].mean()),
                    head_mean_noised_product=float(np.sqrt(abar) * ma[HEAD].mean()))
    return res


def settings():
    """The three sampler settings: (a) mosaic, (b) blend, (c) co-occurrence, each as (cat, dog, projection)."""
    vcat = np.full(D, 1.0); vcat[HEAD] = 0.01
    vdog = np.full(D, 1.0); vdog[BODY] = 0.01
    return {
        "a mosaic": (Mixture([1.0], np.ones(D), vcat), Mixture([1.0], -np.ones(D), vdog), (HEAD, BODY)),
        "b blend": (Mixture([1.0], np.ones(D), np.full(D, 0.01)), Mixture([1.0], -np.ones(D), np.full(D, 0.01)), (HEAD, BODY)),
        "c co-occurrence": (Mixture([0.5, 0.5], *zip(placed(1.0, LEFT), placed(1.0, RIGHT))),
                            Mixture([0.5, 0.5], *zip(placed(-1.0, LEFT), placed(-1.0, RIGHT))), (LEFT, RIGHT)),
    }


EDGES = np.linspace(-1.5, 1.5, 41)


def project(x, proj):
    return np.c_[x[:, proj[0]].mean(1), x[:, proj[1]].mean(1)]


def tv_projected(x, ref, proj):
    """TV between two sample sets on a 40 x 40 binning of the two half-means in [-1.5, 1.5]^2."""
    a, b = project(x, proj), project(ref, proj)
    ha = np.histogram2d(a[:, 0], a[:, 1], [EDGES, EDGES])[0] / len(a)
    hb = np.histogram2d(b[:, 0], b[:, 1], [EDGES, EDGES])[0] / len(b)
    return 0.5 * np.abs(ha - hb).sum()


def weight_error(x, prod):
    """TV between the share of samples assigned to each product component (largest responsibility) and its exact weight."""
    k = prod.responsibilities(x).argmax(1)
    share = np.bincount(k, minlength=len(prod.w)) / len(x)
    return 0.5 * np.abs(share - prod.w).sum(), share


def composed_target(A, B, abar, device, compose=lambda a, b: a + b):
    """Annealing target at step t: compose(log q^A_t, log q^B_t), exact for mixtures."""
    from rebuild.samplers import Target
    cache = {}

    def params(t):
        if t not in cache:
            a = 1.0 if t < 0 else abar[t]
            cache[t] = (A.noised(a).torch(device), B.noised(a).torch(device))
        return cache[t]

    def logp(x, t):
        pa, pb = params(t)
        return compose(A.logp_torch(x, pa), B.logp_torch(x, pb))
    return Target(logp)
