"""The batched sampler must agree with the samplers it replaces.

A batched rewrite is the kind of change that silently shifts a baseline: a
branch ordered wrongly, a mask off by one, a λ endpoint rounded differently, and
every curve drawn from it is measured against something other than plain PoE.
These tests pin it down.

  1. Prompt on everywhere, λ=0 everywhere, must equal run_teacher_residual at
     λ=0, which is plain PoE.
  2. Prompt on everywhere, correction inside a window, must equal
     run_teacher_residual with that same correction window.
  3. Three cells batched together must each equal the same cell sampled alone,
     so batching cannot leak one sample's schedule into another's trajectory.
  4. The mask grammar itself: schedule_masks must zero λ wherever the prompt is
     off, and must invert the conditioning window when asked.

1 to 3 need the model and a GPU, so they are skipped without CUDA. 4 is pure
arithmetic and always runs.
"""

from __future__ import annotations

import pytest
import torch

from poe_repair.methods._window_batch import run_window_batch, schedule_masks

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
TOL = 1e-3          # fp16 sampling: compare on decoded latents, not bitwise


# --------------------------------------------------------------------------
# 4. The mask grammar, no model needed.
# --------------------------------------------------------------------------

def test_lambda_is_zero_wherever_the_prompt_is_off():
    cond_on, lam = schedule_masks(
        num_steps=10, cond_window=(0, 5), corr_window=None, lambda_max=1.0,
    )
    assert cond_on == [True] * 5 + [False] * 5
    # Injecting a correction where no prompt acts is not a weaker experiment,
    # it is a meaningless one, so those steps must be exactly zero.
    assert lam == [1.0] * 5 + [0.0] * 5


def test_correction_window_clips_lambda():
    _, lam = schedule_masks(
        num_steps=10, cond_window=None, corr_window=(3, 6), lambda_max=1.0,
    )
    assert lam == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_cond_outside_inverts_the_window():
    cond_on, _ = schedule_masks(
        num_steps=8, cond_window=(0, 3), corr_window=None, cond_outside=True,
    )
    assert cond_on == [False, False, False, True, True, True, True, True]


def test_lambda_max_zero_injects_nothing_anywhere():
    _, lam = schedule_masks(
        num_steps=6, cond_window=None, corr_window=None, lambda_max=0.0,
    )
    assert lam == [0.0] * 6


def test_prompt_on_everywhere_by_default():
    cond_on, lam = schedule_masks(
        num_steps=6, cond_window=None, corr_window=None, lambda_max=1.0,
    )
    assert cond_on == [True] * 6
    assert lam == [1.0] * 6


# --------------------------------------------------------------------------
# 1 to 3. Against the real sampler.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rig():
    from poe_repair.composers._helpers import (
        encode_pair, get_joint_embeds, init_latents_for_cell,
    )
    from poe_repair.experiments.interaction_term.cell import cell_from_slug
    from poe_repair.run import make_ctx

    steps = 6          # enough for trajectories to diverge, cheap to run
    cell = cell_from_slug("a_cat__x__a_dog", 9)
    ctx = make_ctx(num_inference_steps=steps)
    init, sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)
    seq_j, pool_j = get_joint_embeds(cell, ctx)
    return dict(cell=cell, ctx=ctx, init=init, sigma=sigma, emb=emb,
                seq_j=seq_j, pool_j=pool_j, steps=steps)


def _single(rig, *, lambda_max, correction_window):
    from poe_repair.methods._sampling import run_teacher_residual
    r = rig
    return run_teacher_residual(
        init_latents=r["init"], models=r["ctx"].models, scheduler=r["ctx"].scheduler,
        seq_a=r["emb"]["seq_a"], pool_a=r["emb"]["pool_a"],
        seq_b=r["emb"]["seq_b"], pool_b=r["emb"]["pool_b"],
        seq_j=r["seq_j"], pool_j=r["pool_j"],
        seq_e=r["emb"]["seq_e"], pool_e=r["emb"]["pool_e"],
        guidance_scale=r["ctx"].guidance_scale,
        num_inference_steps=r["steps"],
        height=r["cell"].height, width=r["cell"].width,
        euler_init_noise_sigma=r["sigma"],
        device=r["ctx"].device, dtype=r["ctx"].dtype,
        lambda_schedule="constant", lambda_max=lambda_max,
        correction_window=correction_window,
    ).latents


def _batch(rig, specs):
    """specs: list of (cond_window, corr_window, lambda_max, cond_outside)."""
    r = rig
    n = len(specs)
    masks = [schedule_masks(num_steps=r["steps"], cond_window=cw,
                            corr_window=rw, lambda_max=lm, cond_outside=co)
             for cw, rw, lm, co in specs]
    cond_on = torch.tensor([m[0] for m in masks], dtype=torch.bool)
    lam = torch.tensor([m[1] for m in masks], dtype=torch.float32)
    rep = lambda t: t.repeat(n, *([1] * (t.dim() - 1)))
    return run_window_batch(
        init_latents=rep(r["init"]), models=r["ctx"].models,
        scheduler=r["ctx"].scheduler,
        seq_a=rep(r["emb"]["seq_a"]), pool_a=rep(r["emb"]["pool_a"]),
        seq_b=rep(r["emb"]["seq_b"]), pool_b=rep(r["emb"]["pool_b"]),
        seq_j=rep(r["seq_j"]), pool_j=rep(r["pool_j"]),
        seq_e=rep(r["emb"]["seq_e"]), pool_e=rep(r["emb"]["pool_e"]),
        cond_on=cond_on, lam=lam,
        guidance_scale=r["ctx"].guidance_scale,
        num_inference_steps=r["steps"],
        height=r["cell"].height, width=r["cell"].width,
        euler_init_noise_sigma=r["sigma"],
        device=r["ctx"].device, dtype=r["ctx"].dtype,
    )


@CUDA
def test_nothing_injected_reproduces_plain_poe(rig):
    want = _single(rig, lambda_max=0.0, correction_window=None)
    got = _batch(rig, [(None, None, 0.0, False)]).latents[0:1]
    assert (got.float() - want.float()).abs().max().item() < TOL


@CUDA
def test_windowed_correction_matches_the_single_cell_sampler(rig):
    want = _single(rig, lambda_max=1.0, correction_window=(2, 4))
    got = _batch(rig, [(None, (2, 4), 1.0, False)]).latents[0:1]
    assert (got.float() - want.float()).abs().max().item() < TOL


@CUDA
def test_a_sample_follows_its_schedule_not_its_position(rig):
    """Reorder the batch; each cell must come out the same.

    This is the leak test. It holds the batch SIZE fixed and permutes the
    order, because the same UNet returns slightly different numbers at
    different batch shapes (the reason the single-cell identity check compares
    four branches against four branches rather than against the three-branch
    PoE sampler). Comparing a batch of three against a batch of one would be
    testing that property, not testing leakage.
    """
    a = (None, None, 0.0, False)
    b = (None, (2, 4), 1.0, False)
    c = (None, None, 1.0, False)
    fwd = _batch(rig, [a, b, c]).latents
    rev = _batch(rig, [c, b, a]).latents
    for i, j in ((0, 2), (1, 1), (2, 0)):
        assert (fwd[i].float() - rev[j].float()).abs().max().item() < TOL


@CUDA
def test_batch_shape_changes_the_numbers_so_the_grid_must_not_mix_shapes(rig):
    """Documents the constraint the runner has to honour.

    The same cell sampled in a batch of one and in a batch of three does not
    come out identical. The differences are mostly fp16-sized but amplify
    along the trajectory, so a grid whose final partial batch is smaller than
    the rest would carry a systematic difference that looks like a result.
    The runner therefore pads every batch to a fixed size.
    """
    spec = (None, (2, 4), 1.0, False)
    alone = _batch(rig, [spec]).latents[0]
    in_three = _batch(rig, [spec, (None, None, 0.0, False),
                            (None, None, 1.0, False)]).latents[0]
    drift = (alone.float() - in_three.float()).abs().max().item()
    assert drift > TOL, (
        "batch shape no longer changes the numbers; the padding the runner "
        "does to keep batch size fixed may no longer be needed"
    )


@CUDA
def test_prompt_off_everywhere_is_not_the_guided_run(rig):
    """The conditioning gate must actually do something.

    A gate that silently never fires would leave every crossed cell equal to
    its always-guided row, and the whole conditioning axis would be a no-op
    that still produced plausible pictures.
    """
    guided = _batch(rig, [(None, None, 0.0, False)]).latents[0]
    uncond = _batch(rig, [((0, 0), None, 0.0, False)]).latents[0]
    assert (guided.float() - uncond.float()).abs().max().item() > 0.05
