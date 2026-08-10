# The measure decides the answer: weight vs content, done two ways

**2026-08-05.** One cell (`an_eagle__x__a_hawk`, seed 9, steps 10/25/40),
adapter `lora_step_100000.pt`. Cache-free, in-session on the 3090.

## Why this was checked

Plan 02's hypothesis: the LoRA changes **what** a word paints, not **where** it
looks. So the content maps should move more than the weight maps.

The first measure I ran said the opposite, by 1.7x. Before spending 64 GPU-hours
on a sweep built to confirm or deny that, it was worth asking whether the
measure was fair. It was not.

## The two answers

| measure | weight | content | reading |
|---|---|---|---|
| relative norm, ‖on − off‖ / ‖off‖ | 0.236 | 0.139 | weight moves **1.70x more** |
| shape distance, after unit-normalising both | 0.034 | 0.054 | content moves **1.56x more** |

Same tensors, opposite conclusions.

## Why they disagree

The weight map's change is almost entirely a **uniform dimming**, not a change
in pattern. Fit the single best rescale `alpha` of the OFF map onto the ON map
and split the change in two:

| | how much a single rescale explains (\|α−1\|) | what it cannot explain (pattern) |
|---|---|---|
| weight | 0.235 | **0.034** |
| content | 0.130 | **0.054** |

The adapter scales attention weights down about 25% across the board. That
dominates the raw norm and says nothing about where the word looks. Once the
rescale is allowed for, the weight map's spatial pattern barely moves, and the
content map's pattern moves 1.57x more.

Raw totals confirm the dimming is real and not an aggregation artifact:

| | sum off | sum on | change |
|---|---|---|---|
| weight | 287.33 | 215.43 | −25.0% |
| content | 378.87 | 333.39 | −12.0% |

## Three guards

**Is the pattern residual just noise?** Shuffle the ON map's pixels and redo
the fit. Noise would give a similar residual; structure gives a much worse one.

| | real residual | shuffled | ratio |
|---|---|---|---|
| weight | 0.0369 | 0.2084 | 5.7x |
| content | 0.0588 | 0.2682 | 4.6x |

Structure, not noise.

**Does the ranking depend on the denominator?** Dividing by ‖on‖ or by ‖off‖:

| | / ‖on‖ | / ‖off‖ |
|---|---|---|
| weight | 0.0369 | 0.0275 |
| content | 0.0588 | 0.0513 |

Content is larger either way.

**Is the dimming an artifact of aggregation?** No, the raw sums drop (table
above).

## What this changes

**The relative-norm measure must not be used for this comparison.** Weight maps
are row-stochastic and content maps are not, so their norms are not on the same
footing, and a uniform gain change swamps the pattern change that the
hypothesis is actually about.

The sweep should record, per cell:

- `alpha` (the best single rescale) and `|alpha - 1|`, the gain change
- the residual after that rescale, the pattern change
- the shuffled-map residual, as a per-cell noise floor

and the verdict should compare the **pattern** terms, not the raw norms.

## Status of the finding

On this one cell, the pattern change is **1.57x larger for content than for
weight**, which supports the value-channel hypothesis. My first reading said
the reverse and was wrong.

One cell, one seed, one pair. Not a verdict, and the direction of a single cell
is not evidence of replication. The sweep is what decides Goal 6.
