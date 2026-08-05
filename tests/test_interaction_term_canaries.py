"""The checks the causal claim rests on.

Plan 03 measures what happens as the dose lambda rises. Plan 04 measures what
happens as the injection window moves. Both readings are only meaningful if the
injection machinery does *nothing* when it is told to do nothing. These tests
assert that.

## Why the canary is not "bit-exact against run_cfg_poe"

The obvious test is: run ``run_teacher_residual`` at lambda=0, compare against
``run_cfg_poe``, demand equality. That test fails, and not because of a bug.

``run_cfg_poe`` batches three UNet branches (A, B, uncond).
``run_teacher_residual`` batches four (A, B, J, uncond). The same UNet, given
identical inputs, returns different numbers at batch 3 and batch 4: cuBLAS
selects different kernels by shape, and in fp16 that shows up at ~2e-3 per
step. Measured on this repo (RTX 3090, torch 2.x):

    eps_a      batch-3 vs batch-4   max|diff| = 1.953e-03
    eps_b                           max|diff| = 1.953e-03
    eps_uncond                      max|diff| = 1.953e-03
    same batch shape, run twice     bit-identical

Over 50 denoising steps that compounds to ~0.6 in the final latent. So a
cross-sampler comparison cannot be bit-exact no matter how correct the
injection code is, and a tolerance loose enough to pass (>0.6) would be far too
loose to catch a real leak.

The fix is to compare like with like. Every test below holds the batch shape
fixed at four branches and varies only lambda, which is the thing under test.
Within one batch shape the sampler is deterministic, so these assertions are
bit-exact and meaningful.

These need a GPU. ~2 minutes at 8 steps.

Run:
    python -m pytest tests/test_interaction_term_canaries.py -v
"""

from __future__ import annotations

import gc

import pytest
import torch

from poe_repair._sdxl.metrics import guided_eps, poe_eps
from poe_repair.composers._helpers import (
    encode_pair,
    get_joint_embeds,
    init_latents_for_cell,
)
from poe_repair.experiments.interaction_term.cell import cell_from_slug
from poe_repair.methods._sampling import run_teacher_residual
from poe_repair.run import make_ctx

PAIR = "a_cat__x__a_dog"
SEED = 9
STEPS = 8  # enough for divergence to compound; keeps the test near two minutes


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="canaries run the UNet; need a GPU"
)


@pytest.fixture(scope="module")
def rig():
    """Shared sampler inputs, built once: loading the UNet dominates runtime."""
    cell = cell_from_slug(PAIR, SEED)
    ctx = make_ctx(num_inference_steps=STEPS)
    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)
    seq_j, pool_j = get_joint_embeds(cell, ctx)
    return dict(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_a=emb["seq_a"], pool_a=emb["pool_a"],
        seq_b=emb["seq_b"], pool_b=emb["pool_b"],
        seq_j=seq_j, pool_j=pool_j,
        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        guidance_scale=ctx.guidance_scale,
        num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width,
        euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
    )


def _run(rig, **kw):
    """One sampler run, freeing GPU memory first: two 4-branch runs won't fit."""
    gc.collect()
    torch.cuda.empty_cache()
    return run_teacher_residual(**rig, lambda_schedule="constant", **kw)


@pytest.fixture(scope="module")
def baseline(rig):
    """lambda=0: the sampler's own PoE trajectory, at the 4-branch batch shape."""
    return _run(rig, lambda_max=0.0).latents.clone()


def test_lambda_zero_is_reproducible(rig, baseline):
    """The reference itself must be stable, or nothing below means anything."""
    again = _run(rig, lambda_max=0.0).latents
    assert torch.equal(again, baseline), (
        "lambda=0 is not reproducible run-to-run: max |diff| = "
        f"{(again.float() - baseline.float()).abs().max().item():.3e}. "
        "No canary can be trusted until this is fixed."
    )


def test_lambda_zero_injects_nothing(rig, baseline):
    """Dose 0 must leave every step's lambda at zero.

    The sampler branches on ``lam == 0.0`` and steps with eps_poe itself
    rather than eps_poe + 0.0*delta, so this is exact by construction.
    """
    out = _run(rig, lambda_max=0.0)
    assert all(lam == 0.0 for lam in out.extras["lambda_per_step"])
    assert torch.equal(out.latents, baseline)


def test_window_off_equals_lambda_zero(rig, baseline):
    """Injecting nothing anywhere must equal dose 0.

    Note this is lambda_max=0, NOT correction_window=None. In the sampler,
    correction_window=None means "apply lambda at every step", so passing None
    as 'off' would silently test full injection instead of none. That trap is
    why this test exists separately from the one above.
    """
    out = _run(rig, lambda_max=0.0, correction_window=None)
    assert torch.equal(out.latents, baseline)


def test_window_outside_range_contributes_nothing(rig, baseline):
    """A window past the last step must equal window-off.

    This isolates the window bounds from the lambda=0 shortcut: here
    lambda_max is 1.0, so only the window suppresses the injection. An
    off-by-one in the bounds check shows up here and nowhere else.
    """
    out = _run(rig, lambda_max=1.0, correction_window=(STEPS + 10, STEPS + 20))
    assert all(lam == 0.0 for lam in out.extras["lambda_per_step"])
    assert torch.equal(out.latents, baseline)


def test_canaries_can_fail(rig, baseline):
    """Proof the assertions above are load-bearing.

    A real dose must move the trajectory, and by far more than the numerical
    noise the tests above tolerate (which is zero). If this fails, the other
    tests would pass against inert injection code and prove nothing.
    """
    out = _run(rig, lambda_max=1.0)
    assert not torch.equal(out.latents, baseline), (
        "lambda=1 reproduced the lambda=0 trajectory exactly: the injection is "
        "inert, so the canary tests above prove nothing"
    )
    moved = (out.latents.float() - baseline.float()).abs().max().item()
    assert moved > 1e-2, f"lambda=1 barely moved the trajectory ({moved:.3e})"


def test_window_inside_range_does_inject(rig, baseline):
    """The mirror of the window canary: a window over real steps must act."""
    out = _run(rig, lambda_max=1.0, correction_window=(2, 5))
    lam = out.extras["lambda_per_step"]
    assert [i for i, v in enumerate(lam) if v > 0] == [2, 3, 4], lam
    assert not torch.equal(out.latents, baseline)


def test_lambda_zero_steps_with_exactly_eps_poe(rig, tmp_path):
    """The one canary that does not use a sampler run as its own reference.

    Every other test compares one sampler run against another. That shares a
    blind spot: if lambda=0 leaked a little of delta, *both* sides would shift
    together and every assertion would still pass. Verified by mutation, a
    0.1%% leak at lambda=0 (``eps_t = eps_poe + 1e-3 * delta``) passes all six
    of the tests above.

    This one closes that hole using the sampler's own saved ``eps_poe``, which
    is written to disk *before* the injection branch runs and so cannot move
    with it, and comparing it against the eps actually stepped with (the
    tracker's recorded velocity). At lambda=0 those must be the same tensor.

    Two details that decide whether this test works at all, both measured:

    - The error is normalised by ``||delta||``, not ``||eps_poe||``. A leak of
      k*delta is the thing being hunted and ``||delta||`` is ~40x smaller, so
      normalising by eps_poe buries a 0.1%% leak below the noise.
    - The reference is the *saved* eps_poe, not eps_poe recomputed from the
      saved raw outputs. Recomputing carries 0.4-1.5%% fp16 cancellation error,
      which is itself larger than the leak being looked for.

    Verified by mutation: with ``eps_t = eps_poe + 1e-3 * delta`` at lambda=0,
    this reads 0.09%% and the six tests above all still pass.
    """
    res_dir = tmp_path / "residuals"
    out = _run(rig, lambda_max=0.0, save_residuals_dir=res_dir,
               save_x0_estimates=True)

    step_files = sorted(res_dir.glob("step_*.pt"))
    assert step_files, "sampler wrote no residual files"

    worst = 0.0
    for f in step_files:
        sd = torch.load(f, map_location="cpu", weights_only=True)
        assert "eps_poe" in sd, "need save_x0_estimates=True for the reference"
        expected = sd["eps_poe"].float()
        stepped = out.tracker.velocities[int(sd["step_index"])].float()
        delta_norm = sd["delta"].float().norm().item()
        worst = max(
            worst, (stepped - expected).norm().item() / max(delta_norm, 1e-12)
        )

    # Clean code round-trips through fp16 storage only, which lands near 0.
    # The mutation lands at 0.09%. 0.02% sits between them with margin.
    assert worst < 2e-4, (
        f"at lambda=0 the sampler did not step with eps_PoE: worst error is "
        f"{worst:.4%} of ||delta|| across {len(step_files)} steps. Something "
        "is being added when nothing should be."
    )


def test_pmi_identity_holds(rig):
    """r_t is the interaction term, checked numerically by the sampler.

    The sampler records the relative error of
    Delta_t == w * (eps_J + eps_uncond - eps_A - eps_B) at every step. That
    identity is the scope's central claim about what r_t *is*, so it is worth
    asserting rather than leaving in an extras dict nobody reads.

    The bound comes from a measured 50-step run on this pair, not a guess:

        step  0   residual 11.8%   ||delta||  8.8    <- smallest delta
        step  5           3.5%              29.7
        step 25           3.4%              28.9
        step 49           3.1%              26.5
        median            3.4%

    The residual is largest exactly where ``||delta||`` is smallest, which is
    the signature of fp16 cancellation in a difference of large quantities, not
    of a broken identity. So this test allows 8% on the median and treats only
    a gross departure as a failure. It guards the identity, not the noise
    floor. Plan 05 is where the curve gets read as a result.
    """
    out = _run(rig, lambda_max=1.0)
    residuals = out.extras["pmi_identity_residual_per_step"]
    assert residuals, "sampler recorded no PMI identity residuals"
    median = sorted(residuals)[len(residuals) // 2]
    assert median < 0.08, (
        f"PMI identity broke: median relative residual {median:.3e} across "
        f"{len(residuals)} steps (expected ~3%). r_t is not the guided "
        "interaction term claimed."
    )
