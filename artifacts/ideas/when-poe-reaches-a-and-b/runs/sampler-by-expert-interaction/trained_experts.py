"""Trained energy-parameterised diffusion experts for the four-quadrant toy, in PyTorch.

The network mirrors notebooks/simple_distributions.ipynb cells 4 and 5:
  ResnetDiffusionModel: Linear(2 -> 128), 4 residual blocks (LayerNorm, swish, Linear 128 -> 256,
  + Linear(time embedding 32 -> 256), swish, Linear 256 -> 256, swish, Linear 256 -> 128 zero-init),
  Linear(128 -> 2) zero-init, learned time embedding over 100 levels.
  EBMDiffusionModel: scalar energy E(x, t) = ||net(x, t) - x||^2, noise prediction eps_hat = grad_x E.
Training is the notebook's simple loss (denoising score matching in noise-prediction form):
  x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps,  loss = mean ||grad_x E(x_t, t) - eps||^2,
Adam 1e-3, batch 1000, 15,000 steps, EMA 0.999 (the EMA weights are the ones used).

The density the energy stands for at level t is log q_t(x) = -E(x, t) / sqrt(1 - abar_t) + const,
so score = -grad_x E / sqrt(1 - abar_t). This is the notebook's p_energy / p_gradient scaling.
"""
import copy
import math

import numpy as np
import torch
from torch import nn

import exact_toy as T

N_LEVELS = 100
TRAIN = dict(n_train=10000, steps=15000, batch=1000, lr=1e-3, ema=0.999, optimizer="Adam",
             loss="noise prediction, mean ||grad_x E(x_t, t) - eps||^2, t uniform on 0..99",
             net="resnet MLP, 4 blocks, hidden 128 (inner 256), learned time embedding 32",
             energy="E(x, t) = ||net(x, t) - x||^2; log q_t = -E / sqrt(1 - abar_t)",
             schedule="cosine, s=0.008, 100 levels (exact_toy.Schedule)")


class ResnetDiffusionModel(nn.Module):
    def __init__(self, n_steps=N_LEVELS, n_layers=4, x_dim=2, h_dim=128, emb_dim=32, widen=2):
        super().__init__()
        self.emb = nn.Embedding(n_steps, emb_dim)
        self.inp = nn.Linear(x_dim, h_dim)
        w = h_dim * widen
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            out = nn.Linear(w, h_dim)
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
            self.blocks.append(nn.ModuleDict(dict(
                norm=nn.LayerNorm(h_dim), lin_h=nn.Linear(h_dim, w), lin_emb=nn.Linear(emb_dim, w),
                lin_int=nn.Linear(w, w), lin_out=out)))
        self.out = nn.Linear(h_dim, x_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.act = nn.SiLU()

    def forward(self, x, t):
        emb = self.emb(t)
        h0 = self.inp(x)
        for b in self.blocks:
            h = self.act(b["norm"](h0))
            h = b["lin_h"](h) + b["lin_emb"](emb)
            h = self.act(h)
            h = self.act(b["lin_int"](h))
            h0 = h0 + b["lin_out"](h)
        return self.out(h0)


class EnergyNet(nn.Module):
    """Scalar energy per row: E(x, t) = ||net(x, t) - x||^2 (notebook cell 5)."""

    def __init__(self):
        super().__init__()
        self.net = ResnetDiffusionModel()

    def energy(self, x, t):
        return ((self.net(x, t) - x) ** 2).sum(-1)


def training_set(density, r, n, gen):
    """n samples of p(x) ('all'), p(x | A) ('A', right blobs) or p(x | B) ('B', top blobs).

    The kept blobs keep their mixture proportions. Returns samples (float64) and the number of
    samples drawn from each blob, so the top-right count is exact rather than estimated.
    """
    w = T.base_weights(r)
    keep = T.expert_sets()[density]
    probs = torch.tensor([w[q] for q in keep], dtype=T.DT)
    probs = probs / probs.sum()
    means = torch.tensor([T.MU[q] for q in keep], dtype=T.DT)
    idx = torch.multinomial(probs, n, replacement=True, generator=gen)
    x = means[idx] + T.SIGMA * torch.randn(n, 2, dtype=T.DT, generator=gen)
    counts = {q: int((idx == i).sum()) for i, q in enumerate(keep)}
    return x, counts


def train_expert(x_train, seed, device, sched, steps=TRAIN["steps"], batch=TRAIN["batch"],
                 lr=TRAIN["lr"], ema=TRAIN["ema"], log_every=0):
    """Train one EnergyNet on x_train. Returns the EMA network and the loss trace (every 500 steps)."""
    torch.manual_seed(seed)
    net = EnergyNet().to(device)
    ema_net = copy.deepcopy(net)
    for p in ema_net.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    abar = torch.tensor(sched.abar, dtype=torch.float32, device=device)
    X = x_train.to(device=device, dtype=torch.float32)
    g = torch.Generator(device=device).manual_seed(seed)
    trace, run = [], []
    params, ema_params = list(net.parameters()), list(ema_net.parameters())
    for i in range(steps):
        idx = torch.randint(0, len(X), (batch,), device=device, generator=g)
        t = torch.randint(0, sched.n, (batch,), device=device, generator=g)
        eps = torch.randn(batch, 2, device=device, generator=g)
        a = abar[t][:, None]
        xt = (a.sqrt() * X[idx] + (1 - a).sqrt() * eps).requires_grad_(True)
        e = net.energy(xt, t).sum()
        (eps_hat,) = torch.autograd.grad(e, xt, create_graph=True)
        loss = ((eps_hat - eps) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            torch._foreach_mul_(ema_params, ema)
            torch._foreach_add_(ema_params, params, alpha=1 - ema)
        run.append(loss.item() if (i % 50 == 0) else None)
        if (i + 1) % 500 == 0:
            vals = [v for v in run if v is not None]
            trace.append(float(np.mean(vals)))
            run = []
            if log_every and (i + 1) % log_every == 0:
                print(f"      step {i + 1:6d}  loss {trace[-1]:.4f}", flush=True)
    ema_net.eval()
    return ema_net, trace


def expert_logp_score(net, x, t, sched):
    """log q_t(x) up to a constant, and its score, for one trained expert. t = -1 uses level 0."""
    tt = max(t, 0)
    scale = 1.0 / math.sqrt(1.0 - float(sched.abar[tt]))
    dev = next(net.parameters()).device
    xx = x.detach().to(device=dev, dtype=torch.float32).requires_grad_(True)
    tv = torch.full((len(xx),), tt, dtype=torch.long, device=dev)
    u = net.energy(xx, tv) * scale
    (g,) = torch.autograd.grad(u.sum(), xx)
    return (-u).detach().to(device=x.device, dtype=x.dtype), (-g).to(device=x.device, dtype=x.dtype)


class TrainedComposedTarget:
    """The composed density from three trained experts: log q(x|A) + log q(x|B) - log q(x).

    Same interface and evaluation counter as exact_toy.ComposedTarget. The trained models have no
    clean level, so the clean step (t = -1) uses the level-0 networks (abar_0 = 0.99937).
    """

    def __init__(self, sched, nets):
        self.sched = sched
        self.nets = nets          # dict with keys 'A', 'B', 'all'
        self.calls = 0

    def logp_score(self, x, t):
        self.calls += 1
        la, sa = expert_logp_score(self.nets["A"], x, t, self.sched)
        lb, sb = expert_logp_score(self.nets["B"], x, t, self.sched)
        l0, s0 = expert_logp_score(self.nets["all"], x, t, self.sched)
        return la + lb - l0, sa + sb - s0


class ExactLevel0Target(T.ComposedTarget):
    """Exact experts, but the clean step uses the level-0 density, as the trained experts must.
    Used only for the check that this substitution does not by itself move TV."""

    def logp_score(self, x, t):
        return super().logp_score(x, max(t, 0))
