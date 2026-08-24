# What is F6's spectrum actually measuring?

**Asked:** while designing F6, whether the figure's headline (the stacked corrections
carry 11 to 14 times more energy in their top directions than a same-shape random
stack) means the corrections share a small set of directions.

**Date:** 2026-08-12 · Cache-only, no GPU, read-only.
`python evidence/f6-what-the-spectrum-measures/control.py` → `result.json`.

## The short answer

It does not. The excess over F6's floor is two things the figure was not built to
claim, and neither is a shared subspace:

| what it is | how big |
|---|---|
| the spread of ‖r_t‖ across the run, which is F3's subject | most of it |
| each single run's own smoothness over its steps, which is D1's subject | the rest |
| directions shared across pairs and seeds | none detectable |

## The floor was the problem

F6's floor is a Gaussian of the same shape: same row count, same dimension, every row
the same expected norm. Real rows do not have the same norm. ‖r_t‖ tracks the noise
level, and across the 440 rows it runs from 8.7 to 107.6, a factor of 4.5 against the
median. A stack of rows pointing in unrelated directions still concentrates its energy
in the top few singular directions when some rows are far bigger than the rest, so
that floor cannot separate shared directions from uneven sizes.

Give the floor the real norms and the claim goes away. Energy in the top k directions
of the pooled stack, against random directions carrying the real ‖r_t‖ values:

| k | corrections | equal-norm floor | ratio | norm-matched floor | ratio |
|---|---|---|---|---|---|
| 1 | 3.8% | 0.3% | 14.1x | 2.4% | **1.5x** |
| 8 | 22.6% | 2.1% | 10.7x | 16.1% | **1.4x** |
| 64 | 63.0% | 16.4% | 3.8x | 56.3% | **1.1x** |

## What is left once size is removed

Scale every row to unit norm and the spectrum reads directions only. Then the pooled
stack does beat the floor: 23x at k=1, 7.8x at k=8. But that structure is entirely
inside single runs. Two stacks of identical shape (50 rows by 65536), one floor:

| k | 50 steps of ONE cell | one step from each of 50 DIFFERENT cells |
|---|---|---|
| 1 | 11.2x | 1.4x |
| 8 | 4.8x | 1.2x |
| 32 | 1.5x | 1.0x |

Within one run the correction turns smoothly through its steps, which is what D1
already says and what the +0.81 cosine between adjacent steps already measures. Across
runs and pairs there is nothing: 1.2x at k=8 is a random stack. That agrees with D3
(different pairs share no direction at any step) and with the near-orthogonality
measured in `artifacts/results/which-way-the-correction-points/does-the-subspace-test-predict-transfer/QUERY.md` (cosine about 0.00 even
train-to-train).

## What this costs the paper

**F6 cannot argue that a shared low-dimensional structure is what makes the correction
learnable.** The pooled spectrum, read against the only floor that controls for size,
does not distinguish the corrections from random directions. The sentence "rank 8 is
not a lucky hyperparameter, it is the right size read off the data" has nothing under
it, and it was about to be the figure's whole point.

What still stands, untouched by this:

- The adapter works. Rank 8, 11 training pairs, 96.9% compose on six unseen pairs
  where plain PoE composes 0%. That is behavioural and this control does not touch it.
- Each run's correction is smooth over its own steps (D1).
- Different pairs' corrections share no direction (D3), now confirmed from a second
  direction: not merely low pairwise cosine, but no shared subspace at any k.

The learnability claim rests on the adapter's measured behaviour, not on the geometry
of the cached vectors. Every attempt to find a cache-only geometric proxy for it has
now failed twice, in the subspace-overlap test and here.

## Limits

- One split, 11 training pairs at 8 seeds, every 10th step. 440 rows.
- The norm-matched floor randomises direction while keeping the real norms. It does
  not preserve the within-run correlation between consecutive steps, which is why the
  within-versus-across comparison is reported separately rather than folded into it.
- "No detectable shared structure across cells" is a statement at these k values and
  this sample size, not a proof that none exists.
- The pooled numbers come from `scripts/spectrum.py --pool --stride 10 --max-seeds 8`,
  which now prints both floors.
