"""Checks for the MALA and CHA fixes in anneal_samplers.py.

    /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m pytest tests/test_anneal_samplers.py -s
"""
import numpy as np
import torch

from anneal_samplers import AnnealedCHASampler, AnnealedMALASampler


def gauss(x, ts=None):
    """Standard normal target, one log-density per sample (independent chains)."""
    return -0.5 * (x ** 2).flatten(1).sum(1), -x


def run_mala(n=4000, moves=300, ss=0.9, seed=0, dim=1):
    torch.manual_seed(seed)
    s = AnnealedMALASampler(1, moves, [ss], gauss)
    x = 3 * torch.randn(n, dim)
    x = s.sample_step(x, 0, None, {})
    return x, s.last_accept


def test_proposal_logdensity_is_the_normal_with_variance_2ss():
    torch.manual_seed(0)
    ss, x = 0.3, torch.randn(500, 2)
    _, g = gauss(x)
    xp = x + ss * g + (2 * ss) ** .5 * torch.randn_like(x)
    ours = -((xp - x - ss * g) ** 2).sum(1) / (4 * ss)
    ref = torch.distributions.Normal(x + ss * g, (2 * ss) ** .5).log_prob(xp).sum(1)
    gap = ref - ours                       # the Normal's constant, the same for every sample
    assert torch.allclose(gap, gap[0].expand_as(gap), atol=1e-5)


def test_each_sample_is_accepted_on_its_own():
    x, acc = run_mala(n=4000, moves=10, ss=1.5)
    print("\nper-sample acceptance over 10 moves: mean %.3f, sd %.3f, share at 0 %.3f, share at 1 %.3f"
          % (acc.mean(), acc.std(), (acc == 0).float().mean(), (acc == 1).float().mean()))
    print("histogram over 0.0, 0.1, ..., 1.0:", torch.bincount((acc * 10).round().long(), minlength=11).tolist())
    assert acc.shape == (4000,)
    assert acc.std() > 0.1                 # one shared uniform per move would make every entry equal
    assert 0.2 < acc.mean() < 0.95


def test_mala_targets_the_standard_normal():
    x, acc = run_mala()
    print("\nfixed MALA, ss 0.9: mean %.3f var %.3f acceptance %.2f" % (x.mean(), x.var(), acc.mean()))
    assert abs(float(x.mean())) < 0.06 and abs(float(x.var()) - 1) < 0.08


def test_old_divisor_misses_the_standard_normal():
    """The same chain with the old 2*ss**2 divisor, per sample: the check above fails on it."""
    torch.manual_seed(0)
    ss, x = 0.9, 3 * torch.randn(4000, 1)
    e, g = gauss(x)
    for _ in range(300):
        xp = x + ss * g + (2 * ss) ** .5 * torch.randn_like(x)
        ep, gp = gauss(xp)
        fwd = -((xp - x - ss * g) ** 2).sum(1) / (2 * ss ** 2)
        bwd = -((x - xp - ss * gp) ** 2).sum(1) / (2 * ss ** 2)
        a = torch.log(torch.rand(len(x))) < ep - e + bwd - fwd
        x, e, g = torch.where(a[:, None], xp, x), torch.where(a, ep, e), torch.where(a[:, None], gp, g)
    print("\nold divisor, ss 0.9: var %.3f (target 1)" % x.var())
    assert abs(float(x.var()) - 1) > 0.08


def test_joint_scalar_energy_as_inf_sample_uses_it():
    """inf_sample.py's gradient_cha returns one energy for the whole batch and the same gradient on every row."""
    torch.manual_seed(0)
    betas = np.linspace(1e-4, 0.02, 100)

    def gradient_cha(x_t, ts, **kwargs):
        half = x_t[:1]
        energy = -0.5 * (half ** 2).sum() * x_t.shape[0]
        eps = torch.cat([half] * x_t.shape[0], 0)
        return energy, -eps

    s = AnnealedMALASampler(100, 20, betas * 0.035, gradient_cha)
    x = torch.randn(3, 3, 8, 8).expand(3, 3, 8, 8).clone()
    x[1:] = x[:1]
    out = s.sample_step(x, 50, torch.tensor([50, 50, 50]), {"y": None})
    assert out.shape == x.shape
    assert torch.all(s.last_accept == s.last_accept[0])   # one decision per move for the joint chain


def test_cha_targets_the_standard_normal_and_accepts_per_sample():
    torch.manual_seed(0)
    s = AnnealedCHASampler(1, 200, [0.6], 0.5, [1.0], 3, gauss)
    x = s.sample_step(3 * torch.randn(4000, 1), 0, None, {})
    print("\nfixed CHA: mean %.3f var %.3f acceptance %.2f (sd across samples %.2f)"
          % (x.mean(), x.var(), s.last_accept.mean(), s.last_accept.std()))
    assert abs(float(x.mean())) < 0.06 and abs(float(x.var()) - 1) < 0.08
    assert s.last_accept.std() > 0.0
