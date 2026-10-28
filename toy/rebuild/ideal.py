"""Ideal experts and exact targets on a 2D grid.

An ideal expert is its training points smoothed by N(0, SIG^2 I), the optimum a diffusion model reaches on a
finite set. Its noised density at step t is exact:

    q_t(x) = 1/N sum_i N(x; sqrt(abar_t) x_i, (1 - abar_t + abar_t SIG^2) I).
"""
import numpy as np
import torch

from rebuild.model import cosine_beta_schedule
from rebuild.samplers import Target

L, G, SIG = 1.1, 241, 0.05          # canvas half-width, grid points per side, smoothing
T = 100
BETAS = cosine_beta_schedule(T)
ABAR = np.cumprod(1 - BETAS)
LEVELS = list(range(T - 1, -1, -1))
V = np.array([1 - ABAR[t] + ABAR[t] * SIG ** 2 for t in LEVELS])     # noise variance per level, high to low

xs = np.linspace(-L, L, G)
X1, X2 = np.meshgrid(xs, xs)
CELL = (xs[1] - xs[0]) ** 2
EDGES = np.linspace(-L, L, 41)


def logq(x, data, t, sig=SIG):
    """Exact log-density of the ideal expert at diffusion step t (t = -1 means no diffusion)."""
    a = 1.0 if t < 0 else ABAR[t]
    v = (1 - a) + a * sig ** 2
    d2 = torch.cdist(x, np.sqrt(a) * data) ** 2
    return torch.logsumexp(-d2 / (2 * v), 1) - np.log(len(data)) - np.log(2 * np.pi * v)


def grid(device):
    return torch.tensor(np.stack([X1.ravel(), X2.ravel()], 1), dtype=torch.float32, device=device)


def exact_grid(A, B, compose, device):
    """The composed density at t = 0 on the grid, normalised to integrate to 1."""
    g = grid(device)
    with torch.no_grad():
        lp = compose(logq(g, A, -1), logq(g, B, -1)).reshape(G, G).cpu().numpy().astype(np.float64)
    p = np.exp(lp - lp.max())
    return p / (p.sum() * CELL)


def grid_density(logp_fn, device, chunk=20000):
    """Normalise any log-density function of (B, 2) points on the grid."""
    g = grid(device)
    with torch.no_grad():
        lp = torch.cat([logp_fn(g[i:i + chunk]) for i in range(0, len(g), chunk)]).reshape(G, G)
    lp = lp.cpu().numpy().astype(np.float64)
    p = np.exp(lp - lp.max())
    return p / (p.sum() * CELL)


def perfect_samples(p, n, rng):
    """n points drawn straight from the grid density: no sampler error, only finite-sample error."""
    w = (p * CELL).ravel()
    idx = rng.choice(len(w), n, p=w / w.sum())
    pts = np.c_[X1.ravel()[idx], X2.ravel()[idx]]
    return pts + rng.uniform(-0.5, 0.5, pts.shape) * (xs[1] - xs[0])


def tv(samples, p):
    """Total variation between samples and a grid density, on a 40 x 40 binning of the canvas (0 same, 1 disjoint)."""
    pb = np.histogram2d(X2.ravel(), X1.ravel(), [EDGES, EDGES], weights=(p * CELL).ravel())[0]
    sb = np.histogram2d(samples[:, 1], samples[:, 0], [EDGES, EDGES])[0] / len(samples)
    return 0.5 * np.abs(pb / pb.sum() - sb).sum()


def composed_target(A, B, compose):
    """The annealing target at step t: the composition of the two experts' noised densities."""
    return Target(lambda x, t: compose(logq(x, A, t), logq(x, B, t)))
