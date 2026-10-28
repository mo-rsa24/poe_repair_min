"""The notebook's composition wrappers (cells 6 to 8), on top of two EBMDiffusionModel nets.

Each wrapper has the same interface as EBMDiffusionModel: neg_logp_unnorm(x, t) and forward(x, t) -> noise
prediction, so PortableDiffusionModel can wrap it and sample it by reverse diffusion.
"""
import torch
import torch.nn as nn


class _Composed(nn.Module):
    def __init__(self, net, net2):
        super().__init__()
        self.net, self.net2 = net, net2

    def forward(self, x, t, create_graph=False):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            g, = torch.autograd.grad(self.neg_logp_unnorm(x, t).sum(), x, create_graph=create_graph)
        return g


class ProductEBMDiffusionModel(_Composed):
    """Product: add the energies, and the noise predictions."""

    def neg_logp_unnorm(self, x, t):
        return self.net.neg_logp_unnorm(x, t) + self.net2.neg_logp_unnorm(x, t)

    def forward(self, x, t, create_graph=False):
        return self.net(x, t, create_graph=create_graph) + self.net2(x, t, create_graph=create_graph)


class MixtureEBMDiffusionModel(_Composed):
    """Mixture: a soft minimum of the energies, -logsumexp(-3.5 E), with the notebook's temperature 3.5."""

    def neg_logp_unnorm(self, x, t):
        e = torch.stack([self.net.neg_logp_unnorm(x, t), self.net2.neg_logp_unnorm(x, t)], -1)
        return -torch.logsumexp(-e * 3.5, -1)


class NegationEBMDiffusionModel(_Composed):
    """Negation: 1.3 E1 - 0.3 E2, the notebook's weights."""

    def __init__(self, net, net2, w=0.3):
        super().__init__(net, net2)
        self.w = w

    def neg_logp_unnorm(self, x, t):
        return (1 + self.w) * self.net.neg_logp_unnorm(x, t) - self.w * self.net2.neg_logp_unnorm(x, t)
