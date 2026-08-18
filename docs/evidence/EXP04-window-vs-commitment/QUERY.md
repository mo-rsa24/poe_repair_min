# Does the window where the correction works move with the pair?

**EXP-04 of [EXPERIMENTS.md](../../../EXPERIMENTS.md).** Run 2026-08-12, no GPU: the window sweep
already existed. `python scripts/window_vs_commitment.py` → `result.json`,
`window-vs-commitment.png`.

**Why it was asked.** EXP-01 found that the step where a run's picture settles varies from pair to
pair, 18 to 36 across 19 pairs. It also found that every pair settles at step 18 or later while
the correction only works over steps 0 to 10, which no simple commitment story predicts. This is
the experiment that separates the two readings. If settling is the event that matters, a pair that
settles late should keep responding to the correction later.

## The answer: no, and not by a little

Every one of the 8 pairs peaks at the same window, centre 5, meaning steps 0 to 10. The span of
best window centres is **0 steps**, against a pre-registered bar of 5 for "moves" and 2 for "does
not move". Meanwhile the settling step across these same 8 pairs spans 13 steps, from 23 to 36.

| pair | settles at | best window centre | latest window that still works |
|---|---|---|---|
| frog × toad | 23 | 5 | 45 (one seed of four, at the grid's noise level) |
| seal × walrus | 23 | 5 | 10 |
| eagle × hawk | 23 | 5 | 5 |
| goose × swan | 25 | 5 | 10 |
| leopard × jaguar | 26 | 5 | 5 |
| cow × buffalo | 27 | 5 | 15 |
| elephant × penguin | 33 | 5 | 15 |
| cat × dog | 36 | 5 | 20 |

The correlation between settling step and best window is undefined, because one of the two
variables never changes. That is the finding rather than a missing number.

Two secondary summaries were declared beside the registered one and both come in weak: the
compose-weighted centre of each pair's profile correlates with settling at rho +0.16, and the
latest window that still works at rho +0.26. The right-hand column above does show cat × dog, the
latest settler, keeping a nonzero rate furthest out. That is a hint at a tail relationship, well
below the bar, on 8 pairs at 4 seeds. It is not a result.

## What this costs

**Failure mode (c) is dead.** The adapter's correction schedule cannot be "in the wrong place for
this pair" when the right place is the same place for every pair. It is removed from EXP-05's
classifier rather than reported as an empty bucket.

**EXP-01's measure is not tracking the decision.** The step where the picture settles varies by 13
steps across these pairs and predicts nothing about when the correction can still act. Whatever
locks composition happens earlier, and is not visible in when the trajectory's own estimate stops
moving.

**The speciation framing does not survive in its simple form.** Composition is not decided at a
transition that moves with the content. It is decided inside the first fifth of the run, in the
same place for pairs as different as eagle × hawk and elephant × penguin.

## The censoring, which cuts the other way

Every pair peaks at centre 5, which is the earliest window the grid holds. So this bounds the
window rather than locating it: the true best window may be earlier still, or narrower. The claim
the data supports is "at or before steps 0 to 10 for every pair", and the honest next measurement
is a finer grid inside the first ten steps rather than a wider one.

That censoring strengthens rather than weakens the reading. The decision is at least as early as
the earliest thing measured, and possibly earlier.

## Limits

- 8 pairs, 4 seeds per window, so a per-pair rate moves in steps of 0.25 and a single seed is
  visible as a bump. The frog × toad points at centres 40 and 45 are one seed each.
- Fixed window width of 10 steps and a fixed dose. A pair could in principle need a wider window
  rather than a differently placed one, which this design cannot see.
- The settling step comes from the latent measure in EXP-01, whose perceptual validation is still
  owed. If that measure is wrong, this experiment shows only that the window does not move with a
  quantity that may itself be meaningless. What survives either way is the left-hand panel: every
  pair's correction works early and only early.
