"""Each check this run relies on, with a case that makes it fail.

Run:  python -m pytest -q test_interaction.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / "samplers-on-exact-experts"))
import exact_toy as T  # noqa: E402
import trained_experts as E  # noqa: E402
import run_interaction as RI  # noqa: E402

SCHED = T.Schedule(100)


class ExactEnergy(nn.Module):
    """A stand-in 'trained' expert whose energy is exactly -sqrt(1 - abar_t) log q_t(x)."""

    def __init__(self, density, r, scale_ok=True):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.keep, self.w, self.scale_ok = T.expert_sets()[density], T.base_weights(r), scale_ok

    def energy(self, x, t):
        tt = int(t[0])
        a = SCHED.abar_at(tt)
        lp, _ = T.mixture_logp_score(x.double(), a, self.keep, self.w)
        return -(math.sqrt(1 - a) if self.scale_ok else 1.0) * lp + 0 * self.dummy


def exact_nets(r, scale_ok=True):
    return {d: ExactEnergy(d, r, scale_ok) for d in ["all", "A", "B"]}


@pytest.mark.parametrize("r", [0.05, 0.001])
def test_training_set_holds_the_stated_top_right_share(r):
    x, c = E.training_set("A", r, 200000, torch.Generator().manual_seed(0))
    p = r / (r + (1 - r) / 3)
    assert abs(c["TR"] / 200000 - p) < 4 * math.sqrt(p * (1 - p) / 200000)
    assert set(c) == {"TR", "BR"} and sum(c.values()) == 200000
    # The same draw at the other r gives a top-right share far outside that band.
    _, c2 = E.training_set("A", 0.05 if r == 0.001 else 0.001, 200000, torch.Generator().manual_seed(0))
    assert abs(c2["TR"] / 200000 - p) > 10 * math.sqrt(p * (1 - p) / 200000)
    # p(x | B) keeps the top blobs only.
    xb, cb = E.training_set("B", r, 1000, torch.Generator().manual_seed(1))
    assert set(cb) == {"TR", "TL"} and float(xb[:, 1].mean()) > 0.4


@pytest.mark.parametrize("t", [0, 30, 90])
def test_trained_target_wrapper_reproduces_exact_composition(t):
    r = 0.05
    x = torch.randn(256, 2, dtype=T.DT)
    le, se = T.ComposedTarget(SCHED, r).logp_score(x, t)
    lt, st = E.TrainedComposedTarget(SCHED, exact_nets(r)).logp_score(x, t)
    assert torch.allclose(st, se, rtol=1e-3, atol=1e-3 * float(se.abs().max()))
    d1, d2 = lt - lt[0], le - le[0]                                   # log-densities up to a constant
    assert torch.allclose(d1, d2, rtol=1e-3, atol=1e-2)
    # Dropping the 1 / sqrt(1 - abar) energy scale must break the agreement (where the scale is not ~1).
    if math.sqrt(1 - SCHED.abar_at(t)) > 0.9:
        return
    _, sb = E.TrainedComposedTarget(SCHED, exact_nets(r, scale_ok=False)).logp_score(x, t)
    assert not torch.allclose(sb, se, rtol=1e-2, atol=1e-2 * float(se.abs().max()))


def test_clean_step_uses_level_zero():
    x = torch.randn(128, 2, dtype=T.DT)
    _, s_clean = T.ComposedTarget(SCHED, 0.05).logp_score(x, -1)
    _, s0 = T.ComposedTarget(SCHED, 0.05).logp_score(x, 0)
    _, s_l0 = E.ExactLevel0Target(SCHED, 0.05).logp_score(x, -1)
    _, s_tr = E.TrainedComposedTarget(SCHED, exact_nets(0.05)).logp_score(x, -1)
    assert torch.allclose(s_l0, s0) and torch.allclose(s_tr, s0, rtol=1e-3, atol=1e-2)
    assert not torch.allclose(s_clean, s0, rtol=1e-3)                 # the substitution is real


def test_score_error_is_zero_for_exact_stand_ins_and_not_for_a_wrong_one():
    e = RI.score_error(exact_nets(0.05), 0.05, SCHED, levels=(0, 30))
    assert max(e.values()) < 1e-3
    wrong = exact_nets(0.05)
    wrong["all"] = ExactEnergy("A", 0.05)                             # unconditional replaced by p(x, A)
    e2 = RI.score_error(wrong, 0.05, SCHED, levels=(0, 30))
    assert min(e2.values()) > 0.05


def _rows(tv):
    """tv[(kind, sampler)] = list of 3 per-seed TVs -> rows in the runner's format."""
    return [dict(expert=k, sampler=s, seed=i, tv=v) for (k, s), vs in tv.items() for i, v in enumerate(vs)]


def test_effects_are_zero_interaction_when_errors_add_and_not_otherwise():
    base = {("exact", "reverse diffusion"): [0.20, 0.21, 0.19], ("exact", "annealed MALA"): [0.13, 0.14, 0.12]}
    add = dict(base)
    add[("trained", "reverse diffusion")] = [v + 0.10 for v in base[("exact", "reverse diffusion")]]
    add[("trained", "annealed MALA")] = [v + 0.10 for v in base[("exact", "annealed MALA")]]
    ef = RI.effects(_rows(add), floor=0.13)
    assert np.allclose(ef["interaction"], 0) and np.allclose(ef["sampler_effect_exact"], -0.07)
    assert np.allclose(ef["expert_effect_reverse"], 0.10) and np.allclose(ef["expert_effect_mala"], 0.10)
    # MALA removes nothing once the experts are trained: interaction = +0.07.
    sub = dict(add)
    sub[("trained", "annealed MALA")] = list(add[("trained", "reverse diffusion")])
    ef2 = RI.effects(_rows(sub), floor=0.13)
    assert np.allclose(ef2["interaction"], 0.07)


def test_measure_counts_nonfinite_and_outside_as_error():
    gm, edges = T.clean_density_grid()
    xs, _ = T.exact_samples(2000, torch.Generator().manual_seed(0))
    xs = xs.numpy()
    ok = RI.measure(xs, gm, edges)["tv"]
    bad = xs.copy()
    bad[:500] = np.nan
    m = RI.measure(bad, gm, edges)
    assert m["nonfinite"] == 0.25 and m["tv"] > ok + 0.2


def test_short_training_learns_the_score_and_an_untrained_net_does_not():
    torch.manual_seed(0)
    x, _ = E.training_set("A", 0.05, 10000, torch.Generator().manual_seed(5))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, trace = E.train_expert(x, 5, dev, SCHED, steps=1500)
    fresh = E.EnergyNet().to(dev)
    xt = math.sqrt(SCHED.abar_at(30)) * x[:2000] + math.sqrt(1 - SCHED.abar_at(30)) * torch.randn(2000, 2, dtype=T.DT)
    _, s_ex = T.mixture_logp_score(xt, SCHED.abar_at(30), ["TR", "BR"], T.base_weights(0.05))

    def rel(n):
        _, s = E.expert_logp_score(n, xt, 30, SCHED)
        return float(((s - s_ex) ** 2).sum(1).mean().sqrt() / (s_ex**2).sum(1).mean().sqrt())

    assert trace[-1] < trace[0]
    assert rel(net) < 0.25
    assert rel(fresh) > 0.5
