# When does a run's outcome stop changing, and does that step move with the pair?

**EXP-01 of [EXPERIMENTS.md](../../../EXPERIMENTS.md).** Run 2026-08-12, cache only, no GPU.
`python scripts/commitment_step.py` → `result.json`, `commitment-step-per-pair.png`.

**Why it was asked.** If the correction's schedule has to follow the pair, the adapter's single
learned schedule will miss pairs whose outcome locks unusually early, and that is a failure mode
worth testing. It is only worth testing if the locking step varies at all.

## The measure

For each of 162 cells (19 pairs, 8 seeds each, 50 steps), form the model's own estimate of the
finished image at every step by Tweedie's formula in latent space,
`x0(t) = (x_t - sqrt(1 - abar_t) * eps_PoE(t)) / sqrt(abar_t)`, and track
`cos(x0(t), x0(final))`. The commitment step is the first step after which that cosine stays at or
above 0.90 for the rest of the run.

## The registered answer: it varies

Per-pair medians span **18 steps**, from dolphin × porpoise at 18 to cat × dog at 36, against a
pre-registered bar of 5. The between-pair spread is 1.90 times the across-seed spread, against a
bar of 1.5. The verdict survives both sensitivity thresholds: the range is 21 steps at 0.80 and 13
steps at 0.95.

## The unregistered answer, which matters more

**Every pair settles at step 18 or later, and the correction only works over steps 0 to 10.** Not
one pair's median lands inside the window. The correction stops working between 8 and 26 steps
before the picture settles.

A simple reading of commitment would say the correction should keep working right up to the step
where the outcome locks. It does not. So one of two things is true, and they are not the same
paper:

- the decision that matters happens well before the estimate settles, and settling is a later,
  downstream event, or
- this latent measure is not tracking the decision at all.

## What is owed before this is used

The perceptual version. The registered definition asked for decoded frames, and this ran in latent
space because the decoded reading exists for only 3 pairs where this covers 19. Latent distance is
not perceptual distance, so until the same settling definition is applied to decoded frames, this
number is a stand-in.

The substitute check does not settle it. Against the fork step, which is independent and already
trusted, the correlation is +0.08 across 19 cells and +0.20 at pair level. The fork read carries
one seed per pair, so that comparison is too weak to confirm or condemn. It is not evidence of
agreement and should not be quoted as such.

## What it does to the plan

Failure mode (c) of EXP-05, the adapter's window sitting in the wrong place for a pair, stays in
the running: the gate asked whether commitment varies and it does. But EXP-04 is now the decisive
experiment rather than a follow-up, because it asks the question this one cannot: does the window
where the correction actually works move with the pair? If it moves with this settling step, the
measure means something. If the window sits at steps 0 to 10 for every pair regardless, then this
measure is tracking the wrong event and the honest claim is that composition is decided far earlier
than anything visible in the trajectory.

## Limits

- One arm only, the uncorrected PoE path. Nothing here observes what a corrected run does.
- Descriptive, not causal. It records when the estimate stops changing, not when the outcome stops
  being changeable. Only the handover sweep can tell the difference.
- The composing window itself (steps 0 to 10) is measured on cat × dog and elephant × penguin, not
  on all 19 pairs, so the comparison against it is per-pair on one side and pooled on the other.
- Latent space, fp16 cache upcast to fp32 before any arithmetic.
