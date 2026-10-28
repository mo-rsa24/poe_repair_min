"""Each check the run relies on, with a case that makes it fail.

Run:  python -m pytest -q test_samplers.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4]))
sys.path.insert(0, str(HERE))
import exact_toy as T  # noqa: E402
from mala_fixed import AnnealedMALASampler as FixedMALA  # noqa: E402
from anneal_samplers import AnnealedMALASampler as OriginalMALA  # noqa: E402

SCHED = T.Schedule(100)


@pytest.mark.parametrize("t", [-1, 0, 30, 70, 99])
def test_composed_score_is_gradient_of_composed_logp(t):
    tgt = T.ComposedTarget(SCHED)
    x = torch.randn(64, 2, dtype=T.DT, requires_grad=True)
    lp, score = tgt.logp_score(x, t)
    (g,) = torch.autograd.grad(lp.sum(), x)
    assert torch.allclose(g, score, atol=1e-8)
    # A sign error on the unconditional term must be caught by the same comparison.
    abar = SCHED.abar_at(t)
    _, s0 = T.mixture_logp_score(x.detach(), abar, T.QUADS, tgt.w)
    assert not torch.allclose(g, score + 2 * s0, atol=1e-3)


def stein_residual(t, var_override=None):
    """E[x . score] = -d under q_t. Samples are drawn by actually noising clean data."""
    g = torch.Generator().manual_seed(0)
    x0 = T.mixture_samples(200000, g)
    a = SCHED.abar_at(t)
    xt = math.sqrt(a) * x0 + math.sqrt(1 - a) * torch.randn(x0.shape, dtype=T.DT, generator=g)
    sig = T.SIGMA if var_override is None else var_override
    _, s = T.mixture_logp_score(xt, a, T.QUADS, T.base_weights(), sigma=sig)
    return float((xt * s).sum(1).mean() + 2)


@pytest.mark.parametrize("t", [0, 20, 50, 90])
def test_noised_mixture_formula_matches_actually_noised_data(t):
    assert abs(stein_residual(t)) < 0.03
    # A wrong blob std must break the identity (skip the level where noise swamps sigma).
    if SCHED.abar_at(t) > 0.5:
        assert abs(stein_residual(t, var_override=0.25)) > 0.1


def test_total_variation_bounds():
    gm, edges = T.clean_density_grid()
    g = torch.Generator().manual_seed(1)
    big = T.exact_samples(200000, g)[0].numpy()
    assert T.total_variation(big, gm, edges) < 0.05
    corner = np.full((1000, 2), -1.4)
    assert T.total_variation(corner, gm, edges) > 0.99
    outside = np.full((1000, 2), 5.0)
    assert T.total_variation(outside, gm, edges) > 0.99


def test_region_shares_on_known_points():
    pts = np.array([[0.5, 0.5], [0.5, -0.5], [0.0, 0.0], [1.4, -1.4]])
    sh = T.region_shares(pts)
    assert sh["in_TR_blob"] == 0.25 and sh["in_BR_blob"] == 0.25 and sh["at_origin"] == 0.25
    assert sh["in_TL_blob"] == 0.0


def test_rejection_sampler_matches_answer_key():
    g = torch.Generator().manual_seed(2)
    xs, proposed = T.exact_samples(40000, g)
    sh = T.region_shares(xs.numpy())
    assert abs(sh["in_TR_blob"] - 0.9889) < 0.004
    assert abs(40000 / proposed - T.R_TR) < 0.01      # acceptance equals the normaliser r
    # Accepting with P(A|x) alone samples p(x, A), which splits between the two right blobs.
    w = T.base_weights()
    x = T.mixture_samples(40000, g)
    la, _ = T.mixture_logp_score(x, 1.0, ["TR", "BR"], w)
    l0, _ = T.mixture_logp_score(x, 1.0, T.QUADS, w)
    keep = x[torch.rand(40000, dtype=T.DT, generator=g) < torch.exp(la - l0)]
    assert T.region_shares(keep.numpy())["in_TR_blob"] < 0.5


def gaussian_logp_grad_per_sample(x, ts):
    return -0.5 * (x**2).sum(1), -x


def gaussian_logp_grad_summed(x, ts):
    return -0.5 * (x**2).sum(), -x


def test_fixed_mala_reaches_unit_gaussian_from_wrong_start():
    torch.manual_seed(3)
    x = 2.0 * torch.randn(4000, 2, dtype=T.DT)
    s = FixedMALA(1, 200, [0.8], gaussian_logp_grad_per_sample)
    x = s.sample_step(x, 0, 0, {})
    assert abs(float(x.var(0).mean()) - 1.0) < 0.06
    acc = np.mean(s.last_accept)
    assert 0.05 < acc < 0.95                           # per-sample: neither all nor none
    # The original class (one accept for the whole batch, wrong denominator) stays far off.
    torch.manual_seed(3)
    x = 2.0 * torch.randn(4000, 2, dtype=T.DT)
    o = OriginalMALA(1, 200, [0.8], gaussian_logp_grad_summed)
    x = o.sample_step(x, 0, 0, {})
    assert abs(float(x.var(0).mean()) - 1.0) > 0.3


def test_fixed_mala_keeps_target_invariant_with_large_step():
    # Start exactly at the target; a correct MH kernel leaves the variance at 1 even with a big step.
    torch.manual_seed(4)
    x = torch.randn(20000, 2, dtype=T.DT)
    x = FixedMALA(1, 50, [1.5], gaussian_logp_grad_per_sample).sample_step(x, 0, 0, {})
    assert abs(float(x.var(0).mean()) - 1.0) < 0.04


def test_denominator_fix_matters_on_its_own():
    # Rebuild mala_fixed.py with only the old denominator 2*ss**2 restored (per-sample accept kept).
    src = (HERE / "mala_fixed.py").read_text()
    assert src.count("/ (4 * ss)") == 2
    ns = {}
    exec(compile(src.replace("/ (4 * ss)", "/ (2 * ss**2)"), "mala_denominator_bug", "exec"), ns)
    torch.manual_seed(4)
    x = torch.randn(20000, 2, dtype=T.DT)
    x = ns["AnnealedMALASampler"](1, 50, [1.5], gaussian_logp_grad_per_sample).sample_step(x, 0, 0, {})
    assert abs(float(x.var(0).mean()) - 1.0) > 0.08
