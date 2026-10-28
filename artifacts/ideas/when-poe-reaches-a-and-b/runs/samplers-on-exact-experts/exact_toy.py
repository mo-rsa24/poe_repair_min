"""Exact four-quadrant toy: noised mixture densities, scores, composed energy, metrics.

Construction (1): one model with "contains" labels. Blobs at (+-0.5, +-0.5), std 0.15.
Base weights w_TR = r, the other three (1 - r) / 3. p(x, A) keeps the two right blobs,
p(x, B) the two top blobs. Composed target at noise level t is
    log q_t(x, A) + log q_t(x, B) - log q_t(x)
and its score is the signed sum of the three exact scores.

Under variance-preserving noising a mixture stays a mixture: means sqrt(abar) * mu_q,
variance abar * sigma^2 + 1 - abar. Everything here is in float64.
"""
import math
import numpy as np
import torch

SIGMA = 0.15
R_TR = 0.05
QUADS = ["TR", "BR", "TL", "BL"]
MU = {"TR": (0.5, 0.5), "BR": (0.5, -0.5), "TL": (-0.5, 0.5), "BL": (-0.5, -0.5)}
DT = torch.float64


def base_weights(r=R_TR):
    o = (1 - r) / 3
    return {"TR": r, "BR": o, "TL": o, "BL": o}


def expert_sets():
    """Which blobs each density keeps: p(x), p(x, A) right blobs, p(x, B) top blobs."""
    return {"all": QUADS, "A": ["TR", "BR"], "B": ["TR", "TL"]}


def cosine_beta_schedule(timesteps, s=0.008):
    # Copied from notebooks/simple_distributions.ipynb, cell 10.
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps, dtype=np.float64)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


class Schedule:
    """100 noise levels t = 0..99 plus the clean level (abar = 1), indexed here as t = -1."""

    def __init__(self, n=100):
        self.n = n
        self.betas = cosine_beta_schedule(n)
        self.alphas = 1 - self.betas
        self.abar = np.cumprod(self.alphas)

    def abar_at(self, t):
        return 1.0 if t < 0 else float(self.abar[t])


def mixture_logp_score(x, abar, keep, w, sigma=SIGMA):
    """Unnormalised-weight mixture sum_{q in keep} w_q N(x; sqrt(abar) mu_q, s2 I).

    Returns (log density per row, score per row). Weights are NOT renormalised, so this is
    the joint q_t(x, A) when keep is A's blobs; constants cancel in every score and ratio.
    """
    s2 = abar * sigma**2 + 1.0 - abar
    means = torch.tensor([MU[q] for q in keep], dtype=DT, device=x.device) * math.sqrt(abar)
    logw = torch.tensor([math.log(w[q]) for q in keep], dtype=DT, device=x.device)
    diff = means[None, :, :] - x[:, None, :]                      # (N, K, 2)
    logc = logw[None, :] - (diff**2).sum(-1) / (2 * s2) - math.log(2 * math.pi * s2)
    lp = torch.logsumexp(logc, dim=1)
    resp = torch.softmax(logc, dim=1)
    score = (resp[:, :, None] * diff).sum(1) / s2
    return lp, score


class ComposedTarget:
    """Composed density at each level, with a counter of composed-score evaluations.

    One evaluation = one call on the whole batch = three expert scores (A, B, unconditional).
    """

    def __init__(self, sched, r=R_TR):
        self.sched = sched
        self.w = base_weights(r)
        self.sets = expert_sets()
        self.calls = 0

    def logp_score(self, x, t):
        self.calls += 1
        abar = self.sched.abar_at(t)
        la, sa = mixture_logp_score(x, abar, self.sets["A"], self.w)
        lb, sb = mixture_logp_score(x, abar, self.sets["B"], self.w)
        l0, s0 = mixture_logp_score(x, abar, self.sets["all"], self.w)
        return la + lb - l0, sa + sb - s0


class MixtureTarget(ComposedTarget):
    """Control: the plain data mixture p(x) noised to each level. Its score is the true score of
    the noised distribution, so reverse diffusion with it should reproduce p(x) exactly."""

    def logp_score(self, x, t):
        self.calls += 1
        return mixture_logp_score(x, self.sched.abar_at(t), self.sets["all"], self.w)


def clean_density_grid(n=60, lo=-1.5, hi=1.5, sub=12, r=R_TR, target_cls=ComposedTarget):
    """Exact probability mass of the clean composed target in each cell of an n x n grid.

    Integrated with sub x sub midpoints per cell, normalised over the box (mass outside
    [-1.5, 1.5]^2 is below 1e-12 for this target).
    """
    edges = np.linspace(lo, hi, n + 1)
    fine = np.linspace(lo, hi, n * sub + 1)
    mids = 0.5 * (fine[1:] + fine[:-1])
    X1, X2 = np.meshgrid(mids, mids, indexing="ij")
    pts = torch.tensor(np.stack([X1.ravel(), X2.ravel()], 1), dtype=DT)
    tgt = target_cls(Schedule(), r)
    lp, _ = tgt.logp_score(pts, -1)
    d = torch.exp(lp - lp.max()).numpy().reshape(n * sub, n * sub)
    mass = d.reshape(n, sub, n, sub).sum((1, 3))
    return mass / mass.sum(), edges


def hist_mass(samples, edges):
    """Share of samples in each grid cell, plus the share that fell outside the box."""
    s = np.asarray(samples)
    h, _, _ = np.histogram2d(s[:, 0], s[:, 1], bins=[edges, edges])
    n = len(s)
    return h / n, 1.0 - h.sum() / n


def total_variation(samples, grid_mass, edges):
    """0.5 * sum |empirical cell share - exact cell mass|, counting outside-box samples as error."""
    h, outside = hist_mass(samples, edges)
    return 0.5 * (np.abs(h - grid_mass).sum() + outside)


def region_shares(samples, sigma=SIGMA):
    """Share within 3 sigma of each blob centre, and within 3 sigma / sqrt(2) of the origin.

    The origin radius matches runs/answer-key-quadrants/answer_key.py so the shares compare.
    """
    s = np.asarray(samples)
    out = {}
    for q in QUADS:
        d = np.hypot(s[:, 0] - MU[q][0], s[:, 1] - MU[q][1])
        out[f"in_{q}_blob"] = float((d < 3 * sigma).mean())
    out["at_origin"] = float((np.hypot(s[:, 0], s[:, 1]) < 3 * sigma / math.sqrt(2)).mean())
    nonfinite = ~np.isfinite(s).all(1)
    out["nonfinite"] = float(nonfinite.mean())
    return out


def mixture_samples(n, gen, r=R_TR):
    """Exact samples from the plain data mixture p(x)."""
    w = base_weights(r)
    probs = torch.tensor([w[q] for q in QUADS], dtype=DT)
    means = torch.tensor([MU[q] for q in QUADS], dtype=DT)
    idx = torch.multinomial(probs, n, replacement=True, generator=gen)
    return means[idx] + SIGMA * torch.randn(n, 2, dtype=DT, generator=gen)


def exact_samples(n, gen, r=R_TR):
    """Rejection sampling: propose from p(x), accept with P(A|x) P(B|x) <= 1.

    target = p(x,A) p(x,B) / p(x) = p(x) * [p(x,A)/p(x)] * [p(x,B)/p(x)], so this is exact.
    Expected acceptance equals the normaliser, r for construction (1).
    """
    w = base_weights(r)
    probs = torch.tensor([w[q] for q in QUADS], dtype=DT)
    means = torch.tensor([MU[q] for q in QUADS], dtype=DT)
    sets = expert_sets()
    out, proposed = [], 0
    while sum(len(o) for o in out) < n:
        m = 20 * n
        idx = torch.multinomial(probs, m, replacement=True, generator=gen)
        x = means[idx] + SIGMA * torch.randn(m, 2, dtype=DT, generator=gen)
        la, _ = mixture_logp_score(x, 1.0, sets["A"], w)
        lb, _ = mixture_logp_score(x, 1.0, sets["B"], w)
        l0, _ = mixture_logp_score(x, 1.0, sets["all"], w)
        acc = torch.rand(m, dtype=DT, generator=gen) < torch.exp(la + lb - 2 * l0)
        out.append(x[acc])
        proposed += m
    allx = torch.cat(out)[:n]
    return allx, proposed
