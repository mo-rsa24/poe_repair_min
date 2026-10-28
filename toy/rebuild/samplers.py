"""Annealed samplers over a sequence of targets pi_t, t = T-1 down to 0, built on the classes in anneal_samplers.py.

A target is any object with logp(x, t) -> (B,) unnormalised log-density of the step-t target, and
grad(x, t) -> (B, D) its gradient. ULA and UHA are the repository's classes unchanged; MALA and the
Metropolis-adjusted UHA (CHA) are the repository's classes with the per-sample fix.
"""
import numpy as np
import torch

from anneal_samplers import AnnealedCHASampler, AnnealedMALASampler, AnnealedULASampler, AnnealedUHASampler


def autograd_grad(logp):
    def g(x, t):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            return torch.autograd.grad(logp(x, t).sum(), x)[0].detach()
    return g


class Target:
    def __init__(self, logp, grad=None):
        self.logp = logp
        self.grad = grad or autograd_grad(logp)

    def both(self, x, t):
        return self.logp(x, t).detach(), self.grad(x, t)


def _levels(T):
    return list(range(T - 1, -1, -1))


def _start(n, dim, device, seed):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(n, dim, device=device)


def ula(target, n, dim, step_sizes, k=10, device="cpu", seed=None):
    T = len(step_sizes)
    s = AnnealedULASampler(T, k, list(step_sizes), lambda x, t: target.grad(x, t))
    x = _start(n, dim, device, seed)
    for i, t in enumerate(_levels(T)):
        x = s.sample_step(x, i, t, {})
    return x.detach()


def uha(target, n, dim, step_sizes, k=10, damping=0.5, mass=1.0, leapfrog=3, device="cpu", seed=None):
    T = len(step_sizes)
    s = AnnealedUHASampler(T, k, list(step_sizes), damping, [torch.full((1,), mass, device=device)] * T, leapfrog,
                           lambda x, t: target.grad(x, t))
    x = _start(n, dim, device, seed)
    for i, t in enumerate(_levels(T)):
        x = s.sample_step(x, i, t, {})
    return x.detach()


def cha(target, n, dim, step_sizes, k=10, damping=0.5, mass=1.0, leapfrog=3, device="cpu", seed=None):
    """Metropolis-adjusted UHA: the notebook's AnnealedMUHASampler, the repository's AnnealedCHASampler."""
    T = len(step_sizes)
    s = AnnealedCHASampler(T, k, list(step_sizes), damping, [mass] * T, leapfrog, target.both)
    x = _start(n, dim, device, seed)
    acc = []
    for i, t in enumerate(_levels(T)):
        x = s.sample_step(x, i, t, {})
        acc.append(s.last_accept)
    return x.detach(), torch.stack(acc)          # acceptance: (levels, samples)


def mala(target, n, dim, step_sizes, k=10, device="cpu", seed=None):
    """Fixed-step annealed MALA. Returns samples and per-sample acceptance, shape (levels, samples)."""
    T = len(step_sizes)
    s = AnnealedMALASampler(T, k, list(step_sizes), target.both)
    x = _start(n, dim, device, seed)
    acc = []
    for i, t in enumerate(_levels(T)):
        x = s.sample_step(x, i, t, {})
        acc.append(s.last_accept)
    return x.detach(), torch.stack(acc)


def tuned_mala(target, n, dim, variances, k=10, start=0.05, accept_target=0.6, device="cpu", seed=None):
    """MALA whose step adapts toward accept_target at every level, as run 1 of the kinds notebook.

    The step is carried across levels scaled by the noise variance, and after each move it is multiplied by
    exp(acceptance - target). Each move is one sample_step of the repository's AnnealedMALASampler.
    """
    T = len(variances)
    V = np.asarray(variances, dtype=float)
    x = _start(n, dim, device, seed)
    ss, steps, acc = start * V[0], [], []
    for i, t in enumerate(_levels(T)):
        ss = ss if i == 0 else ss * V[i] / V[i - 1]
        lvl = []
        for _ in range(k):
            s = AnnealedMALASampler(1, 1, [ss], target.both)
            x = s.sample_step(x, 0, t, {})
            r = float(s.last_accept.mean())
            lvl.append(s.last_accept)
            ss *= float(np.exp(r - accept_target))
        steps.append(ss)
        acc.append(torch.stack(lvl).mean(0))
    return x.detach(), torch.stack(acc), steps


def reverse_diffusion(target, n, dim, betas, device="cpu", seed=None):
    """DDPM step with the composed score: x_{t-1} = (x_t + beta_t s_t) / sqrt(alpha_t) + sqrt(posterior var) z."""
    betas = np.asarray(betas, dtype=float)
    alphas, abar = 1 - betas, np.cumprod(1 - betas)
    x = _start(n, dim, device, seed)
    for t in _levels(len(betas)):
        s = target.grad(x, t)
        var = betas[t] * (1 - (abar[t - 1] if t > 0 else 1.0)) / (1 - abar[t])
        x = (x + betas[t] * s) / np.sqrt(alphas[t])
        if t > 0:
            x = x + np.sqrt(var) * torch.randn_like(x)
    return x.detach()
