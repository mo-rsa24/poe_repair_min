"""The notebook's score network, EBM wrapper, cosine schedule and diffusion model (cells 4, 5, 10, 12), in PyTorch.

Shapes: x is (B, D) float32, t is (B,) int64 step indices in [0, n_steps).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps, dtype=np.float64)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


def haiku_linear(n_in, n_out, zero=False):
    """nn.Linear with Haiku's default init: truncated normal, std 1/sqrt(fan_in), cut at 2 std; zero bias."""
    lin = nn.Linear(n_in, n_out)
    if zero:
        nn.init.zeros_(lin.weight)
    else:
        std = 1 / math.sqrt(n_in)
        nn.init.trunc_normal_(lin.weight, std=std, a=-2 * std, b=2 * std)
    nn.init.zeros_(lin.bias)
    return lin


class ResnetDiffusionModel(nn.Module):
    """Resnet score model. Adds a learned embedding of the step after each block's first linear layer."""

    def __init__(self, n_steps, n_layers, x_dim, h_dim, emb_dim, widen=2):
        super().__init__()
        self.emb = nn.Embedding(n_steps, emb_dim)
        nn.init.trunc_normal_(self.emb.weight, std=1.0, a=-2.0, b=2.0)
        self.inp = haiku_linear(x_dim, h_dim)
        w = h_dim * widen
        self.blocks = nn.ModuleList([nn.ModuleDict(dict(
            norm=nn.LayerNorm(h_dim, eps=1e-5),
            layer_h=haiku_linear(h_dim, w),
            layer_emb=haiku_linear(emb_dim, w),
            layer_int=haiku_linear(w, w),
            layer_out=haiku_linear(w, h_dim, zero=True),
        )) for _ in range(n_layers)])
        self.out = haiku_linear(h_dim, x_dim, zero=True)

    def forward(self, x, t):
        emb = self.emb(t)
        x = self.inp(x)
        for b in self.blocks:
            h = F.silu(b["norm"](x))
            h = b["layer_h"](h) + b["layer_emb"](emb)
            h = F.silu(b["layer_int"](F.silu(h)))
            x = x + b["layer_out"](h)
        return self.out(x)


class EBMDiffusionModel(nn.Module):
    """Energy parameterisation: E(x, t) = |net(x, t) - x|^2, and the noise prediction is grad_x E."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def neg_logp_unnorm(self, x, t):
        return ((self.net(x, t) - x) ** 2).sum(-1)

    def forward(self, x, t, create_graph=None):
        create_graph = torch.is_grad_enabled() if create_graph is None else create_graph
        with torch.enable_grad():
            x = x if x.requires_grad else x.detach().requires_grad_(True)
            e = self.neg_logp_unnorm(x, t)
            g, = torch.autograd.grad(e.sum(), x, create_graph=create_graph)
        return g


class PortableDiffusionModel(nn.Module):
    """DDPM with the notebook's 'simple' loss and 'beta_forward' reverse variance."""

    def __init__(self, dim, n_steps, net):
        super().__init__()
        self.net, self.dim, self.n_steps = net, dim, n_steps
        betas = cosine_beta_schedule(n_steps)
        alphas = 1.0 - betas
        ac = np.cumprod(alphas)
        ac_prev = np.append(1.0, ac[:-1])
        f32 = lambda a: torch.tensor(a, dtype=torch.float32)
        self.register_buffer("betas", f32(betas))
        self.register_buffer("alphas_cumprod", f32(ac))
        self.register_buffer("sqrt_ac", f32(np.sqrt(ac)))
        self.register_buffer("sqrt_1m_ac", f32(np.sqrt(1 - ac)))
        self.register_buffer("sqrt_recip_ac", f32(1 / np.sqrt(ac)))
        self.register_buffer("sqrt_recipm1_ac", f32(np.sqrt(1 / ac - 1)))
        self.register_buffer("score_scale", f32(np.sqrt(1.0 / (1 - ac))))   # notebook: _sqrt_recipm1_alphas_cumprod_custom
        self.register_buffer("post_c1", f32(betas * np.sqrt(ac_prev) / (1 - ac)))
        self.register_buffer("post_c2", f32((1 - ac_prev) * np.sqrt(alphas) / (1 - ac)))

    @staticmethod
    def _ex(a, t):
        return a[t][:, None]

    def q_sample(self, x0, t, noise):
        return self._ex(self.sqrt_ac, t) * x0 + self._ex(self.sqrt_1m_ac, t) * noise

    def loss(self, x0):
        t = torch.randint(0, self.n_steps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        eps = self.net(self.q_sample(x0, t, noise), t, create_graph=True)
        return ((eps - noise) ** 2).mean(1)

    def p_sample(self, x, t):
        eps = self.net(x, t, create_graph=False)
        x0 = self._ex(self.sqrt_recip_ac, t) * x - self._ex(self.sqrt_recipm1_ac, t) * eps
        mean = self._ex(self.post_c1, t) * x0 + self._ex(self.post_c2, t) * x
        return (mean + self._ex(self.betas, t).sqrt() * torch.randn_like(x)).detach()   # noise at t = 0 too, as the notebook

    def sample(self, n, device):
        x = torch.randn(n, self.dim, device=device)
        for j in range(self.n_steps - 1, -1, -1):
            x = self.p_sample(x, torch.full((n,), j, dtype=torch.long, device=device))
        return x

    def p_gradient(self, x, t):
        """Noise prediction scaled by 1/sqrt(1 - abar_t). Minus this is the score of the noised data."""
        return (self.net(x, t, create_graph=False) * self._ex(self.score_scale, t)).detach()

    def p_energy(self, x, t):
        """Energy scaled the same way. Minus this is an unnormalised log-density whose gradient is the score."""
        return (self.net.neg_logp_unnorm(x, t) * self.score_scale[t]).detach()


def make_model(data_dim=2, n_steps=100, n_layers=4, h_dim=128, emb_dim=32):
    return PortableDiffusionModel(data_dim, n_steps,
                                  EBMDiffusionModel(ResnetDiffusionModel(n_steps, n_layers, data_dim, h_dim, emb_dim)))
