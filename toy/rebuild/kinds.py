"""The seven kinds of composition on the paper's toy data, one dial each (the kinds notebook, rebuilt as a module).

Data builders take the Generator explicitly and draw in the kinds notebook's order, so a run seeded with
default_rng(0) sees the same training points as the notebook did.
"""
import numpy as np
import torch

from rebuild.datasets import ring_means

N, NS = 800, 2000                       # points per expert, samples per sampler
RING = ring_means(8, 0.5)


def toy_gmm(rng, n, std=0.03):
    ang = 2 * np.pi * rng.integers(0, 8, n) / 8
    return 0.5 * np.c_[np.cos(ang), np.sin(ang)] + std * rng.standard_normal((n, 2))


def vbar(rng, n, c=0.0, h=0.2, lo=-1.0, hi=1.0):
    return np.c_[c + rng.uniform(-h, h, n), rng.uniform(lo, hi, n)]


def hbar(rng, n, c=0.0, h=0.2, lo=-1.0, hi=1.0):
    return np.c_[rng.uniform(lo, hi, n), c + rng.uniform(-h, h, n)]


def rotated(rng, c, n, p):
    rot = rng.random(n) < p
    return np.where(rot[:, None], hbar(rng, n, c, 0.1), vbar(rng, n, c, 0.1))


def wet_dog(rng, f):
    """Interaction: B is rain (a horizontal band); inside it, A (a dog) sits at x1 = -0.4 instead of -0.2."""
    k = int(f * N)
    A = np.vstack([vbar(rng, N - k, -0.2, 0.1, -1.0, 0.0), vbar(rng, k, -0.4, 0.1, 0.4, 0.8)])
    B = np.vstack([hbar(rng, N - k, 0.6), np.c_[-0.4 + rng.uniform(-0.1, 0.1, k), rng.uniform(0.4, 0.8, k)]])
    return A, B


def near(a, b, pts, r):
    return np.any(np.hypot(a[..., None] - pts[:, 0], b[..., None] - pts[:, 1]) < r, -1)


PRODUCT = lambda a, b: a + b
MIXTURE = lambda a, b: torch.logaddexp(a, b)
negation = lambda w: (lambda a, b: (1 + w) * a - w * b)

# a measure takes coordinates (a, b) and weights w, so it works on the grid (w = p * cell) and on samples (w = 1/n)
MEASURES = {
    "intersection": (lambda a, b, w: (w * near(a, b, RING, 0.08)).sum(), "mass on ring modes"),
    "disjunction": (lambda a, b, w: (w * (a > 0)).sum(), "B's share"),
    "co-occurrence": (lambda a, b, w: (w * ((np.abs(a) < 0.2) & (b > 0.4) & (b < 0.8))).sum(), "mass in crossing"),
    "contention": (lambda a, b, w: (w * near(a, b, np.array([[-0.2, 0.2], [0.2, -0.2]]), 0.12)).sum(), "co-occurrence mass"),
    "dominance": (lambda a, b, w: (w * np.hypot(a, b)).sum() / w.sum(), "mean radius"),
    "interaction": (lambda a, b, w: (w * ((b > 0.3) & (b < 0.9)) * a).sum() / (w * ((b > 0.3) & (b < 0.9))).sum(),
                    "mean x1 in band"),
    "negation": (lambda a, b, w: (w * near(a, b, RING[[2, 6]], 0.08)).sum(), "mass on the 2 in-bar modes"),
}


def build_kinds(rng):
    """The seven kinds at their base settings, drawn in the notebook's order."""
    spec = [
        ("intersection", lambda: (toy_gmm(rng, N), vbar(rng, N, 0.0, 0.05)), PRODUCT),
        ("disjunction", lambda: (vbar(rng, N, -0.4, 0.1), vbar(rng, N, 0.4, 0.2)), MIXTURE),
        ("co-occurrence", lambda: (vbar(rng, N, 0.0, 0.2), hbar(rng, N, 0.6)), PRODUCT),
        ("contention", lambda: (rotated(rng, -0.2, N, 0.5), rotated(rng, 0.2, N, 0.5)), PRODUCT),
        ("dominance", lambda: (toy_gmm(rng, N), rng.standard_normal((N, 2)) * 0.1), PRODUCT),
        ("interaction", lambda: wet_dog(rng, 0.3), PRODUCT),
        ("negation", lambda: (toy_gmm(rng, N), vbar(rng, N, 0.0, 0.2)), negation(0.05)),
    ]
    kinds = []
    for name, make, compose in spec:
        A, B = make()
        m, mname = MEASURES[name]
        kinds.append(dict(name=name, data=(A, B), compose=compose, measure=m, mname=mname))
    return kinds


def measure_samples(kind, s):
    return float(kind["measure"](s[:, 0], s[:, 1], np.full(len(s), 1 / len(s))))
