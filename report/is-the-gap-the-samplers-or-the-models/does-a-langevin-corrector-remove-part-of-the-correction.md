# Does a Langevin corrector remove part of the correction and leave part of it?   ❓ inconclusive · verified 2026-09-06

**The claim**

A Langevin corrector was built on the product-of-experts score and run to 200 corrector steps at
every one of the 50 noise levels, at three step sizes, on the pair that blends and the pair that
composes. It cannot say how much of the correction belongs to the sampler.

The reason is the instrument, not the science. One seed gives one trajectory per corrector count,
and that trajectory's read-out scatters by more than the bars it is judged against: the same
uncorrected cell reads 0.153, 0.169 and 0.180 on three different GPUs, and neighbouring corrector
counts differ by 30% to 70%. The bars are 5%.

The numbers that did arrive have the shape the hypothesis predicted, a fall of 18% to 36% with
two thirds still there, at every step size. None of it is licensed, because the same curve rises
again at the next corrector count.

## Table of contents

- [What would have counted](#what-would-have-counted)
- [1. The curve at the step size the scope proceeds at](#1-the-curve-at-the-step-size-the-scope-proceeds-at)
- [2. The number the bars were compared against moves more than the bars](#2-the-number-the-bars-were-compared-against-moves-more-than-the-bars)
- [3. The step size the search picked destroys the sample while every numeric guard passes](#3-the-step-size-the-search-picked-destroys-the-sample-while-every-numeric-guard-passes)
- [4. The pair that composes by default says what the rising curve is reading](#4-the-pair-that-composes-by-default-says-what-the-rising-curve-is-reading)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Where this came from](#where-this-came-from)
- [Depends on](#depends-on)
- [Still open](#still-open)

## What would have counted

Navigation: 📋 [TOC](#table-of-contents) | [Next](#1-the-curve-at-the-step-size-the-scope-proceeds-at) ➡️

Three branches, with their thresholds written as module-level constants in
`scripts/corrector_residual_curve.py` before any corrector existed, so moving one after the answer
arrived shows up in a diff. Support if, over the last five denoising steps, the correction's size
relative to the product-of-experts prediction falls by at least `MIN_DROP_FOR_SPLIT = 0.20` of its
uncorrected value with at least `MIN_REMAINDER_FOR_SPLIT = 0.20` still there. Null if the change
is under `MAX_DRIFT_FOR_NULL = 0.05` at every step while the chain provably moved. Inconclusive if
the 100-step and 200-step curves differ by more than `MAX_K_INSTABILITY = 0.05`, if the chain moved
less than `MIN_CHAIN_DISPLACEMENT = 0.05`, or if the pair that composes by default behaves like the
pair that blends.

Pinned in
[what is left once the chain settles](../../plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis/03-what-is-left-once-the-chain-settles.md)
and repeated in [its review file](../../plans/06-is-the-gap-the-samplers-or-the-models/review/03-what-is-left-once-the-chain-settles.md),
both written before the composer was built.

## 1. The curve at the step size the scope proceeds at

Navigation: ⬅️ [Previous](#what-would-have-counted) | 📋 [TOC](#table-of-contents) | [Next](#2-the-number-the-bars-were-compared-against-moves-more-than-the-bars) ➡️

![The correction's size against denoising step, one curve per corrector count, one panel per pair, with the two norms on a second row](../../artifacts/results/is-the-gap-the-samplers-or-the-models/corrector-residual-curves-c3.png)
*y is the correction's size relative to the product-of-experts prediction at the point the chain
settled to, x is the denoising step, one curve per corrector count, cat × dog left and butterfly ×
meadow right. What to notice: on the left panel the 100-step and 200-step curves cross each other
all the way along, and every curve collapses onto every other inside the grey band at steps 0 to 10.*

**The number.** Read-zone mean, the last five denoising steps, cat × dog seed 9 at `c = 3`, by
corrector count 0, 1, 5, 20, 100, 200: 0.153, 0.138, 0.261, 0.203, 0.103, 0.125, unitless (the
correction's Euclidean norm divided by the product-of-experts prediction's, float32 upcast from
fp16). The 100-step and 200-step values differ by 21% of the smaller, against the 5% bar, so the
inconclusive branch fires. From
`/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/residual_curves.json`, rows
with `c == 3`, and `verdict_c3.txt`.

## 2. The number the bars were compared against moves more than the bars

Navigation: ⬅️ [Previous](#1-the-curve-at-the-step-size-the-scope-proceeds-at) | 📋 [TOC](#table-of-contents) | [Next](#3-the-step-size-the-search-picked-destroys-the-sample-while-every-numeric-guard-passes) ➡️

![The same figure at the step size the search picked, where both pairs fail the norm bound and the composing pair's curve rises](../../artifacts/results/is-the-gap-the-samplers-or-the-models/corrector-residual-curves-c30.png)
*Same axes, step size ten times larger. What to notice: the right panel's curves separate upward
with corrector count, which is the composing pair's control failing, and the read zone on the left
is the one place the six curves come back together.*

**The number.** The same read-zone mean for the uncorrected run alone, cat × dog, measured on three
GPUs during this scope's runs: 0.153 (RTX 3090), 0.169 (RTX PRO 6000 Blackwell), 0.180 (Quadro RTX
8000), a spread of 15% from fp16 arithmetic over 50 steps with no corrector running at all. Across
corrector counts at one step size the same pair swings by 30% to 70% between neighbours. The
instability bar is 5% and the drift bar is 5%. From the `k = 0` cells of
`residual_curves.json` at `c = 0.3`, `3` and `30`, each rendered on the device its own grid ran on.

## 3. The step size the search picked destroys the sample while every numeric guard passes

Navigation: ⬅️ [Previous](#2-the-number-the-bars-were-compared-against-moves-more-than-the-bars) | 📋 [TOC](#table-of-contents) | [Next](#4-the-pair-that-composes-by-default-says-what-the-rising-curve-is-reading) ➡️

![Cat × dog at 20 corrector steps across every searched step size, from a photograph at 0.035 to texture noise at 100](../../artifacts/results/is-the-gap-the-samplers-or-the-models/corrector-cat-dog-across-step-sizes.png)
*One tile per step-size multiplier at 20 corrector steps, cat × dog seed 9. What to notice: real
photographs up to 0.3, a different scene with a leash at 3, paint at 10, and texture noise from 30,
which is the value the pre-registered search rule picked.*

**The number.** The search's own columns at 20 corrector steps, cat × dog seed 9: the multiplier
0.01 stalls (median relative chain displacement 0.049, under the 0.05 floor); 0.035 through 30 all
pass both divergence guards (maximum latent norm 1.001 to 1.113 times the chain's start, against
the 1.5 bound, and the residual ratio rises within a level at 2% to 18% of levels with a median
change between −0.006 and +0.008); 100 and 300 diverge at 3.09 and 5.24 times. The rule picks the
largest usable value, 30, whose renders are the noise above. From
`/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/step_size_search.json`, fields
`rows[].median_chain_disp_rel`, `max_latent_norm_rel` and `picked_c`.

## 4. The pair that composes by default says what the rising curve is reading

Navigation: ⬅️ [Previous](#3-the-step-size-the-search-picked-destroys-the-sample-while-every-numeric-guard-passes) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

![Butterfly and meadow across corrector counts at the working step size: a photograph through 20 steps, a flat graphic at 100 and 200](../../artifacts/results/is-the-gap-the-samplers-or-the-models/corrector-butterfly-meadow-across-k-c3.png)
*One tile per corrector count, butterfly × meadow seed 9 at the working step size, each labelled
with its read-zone value and its latent norm. What to notice: the scene survives to 20 steps and
becomes a flat graphic at 100, while the latent norm never passes 1.02 times its start.*

**The number.** Composing pair, read-zone mean by corrector count 0 to 200: at the working step size
0.097, 0.095, 0.099, 0.105, 0.090, 0.086, a fall of 10% and the one control that passes; at the
smaller step size 0.092, 0.094, 0.092, 0.093, 0.105, 0.104, a rise of 13%; at the search's pick
0.090, 0.107, 0.134, 0.169, 0.134, 0.111, a rise of 24%. The drift bar is 5%, so the control fails
at two of three step sizes. From `residual_curves.json` and the three `verdict_*.txt` files.

## What this cannot tell you

Navigation: ⬅️ [Previous](#4-the-pair-that-composes-by-default-says-what-the-rising-curve-is-reading) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**Nothing about the early window, by construction.** At high noise what remains after the chain
settles is the gap from noising and multiplying not commuting, plus the model's own gap, and no
number of corrector steps separates those two. The compose-decisive window is steps 0 to 10, which
is exactly that region, so this measurement bounds the timing result rather than explaining it.

**Nothing about the split, at any step size.** The support-shaped numbers exist (cat × dog's
read-zone value sits 36%, 18% and 33% under its uncorrected value at the largest corrector count,
with 64%, 82% and 67% still there) and are not licensed, because the same pair's 100-step value at
the smallest step size is 0.186, above its uncorrected 0.180.

**The residual norm is a proxy, not the distributional gap.** The corrector does not change the
function; it changes where the function is evaluated. A falling curve says the settled latents sit
where the two networks agree better.

**Two corrector counts are two trajectories.** They separate at the first noise level and never
meet, so no axis here reads as one path measured twice.

**One seed, one pair each.** Every number above is a single trajectory.

## Where this came from

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#depends-on) ➡️

| What | Source | Mark |
|---|---|---|
| The three curves and their branches | `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/residual_curves.json` (1800 rows, 600 per step size), `verdict_c30.txt`, `verdict_c3.txt`, `verdict_c0p3.txt`, read 2026-09-06 | verified |
| The step-size search | the same folder's `step_size_search.json`, ten multipliers at 20 corrector steps, read 2026-09-06 | verified |
| The corrector is inert when switched off | `corrector/logs/leak_checks.log`, 2026-09-05: 0 corrector steps and 200 corrector steps with the window past the last step both byte-identical to the reference sampler | verified |
| The renders behind rungs 3 and 4 | `corrector/pairs/<pair>/seed_9/poe_langevin_k<count>_c<multiplier>/`, one per measured chain | verified |
| The runs | steps 25 and 26 of the running order, on mscluster108 device 1, mscluster110 device 0 and mscluster85 device 0, 2026-09-05 to 06, commit 0150704; the tables are in [the step-size review](../../plans/06-is-the-gap-the-samplers-or-the-models/review/02-the-corrector-and-the-step-size-it-runs-at.md) and [the chain review](../../plans/06-is-the-gap-the-samplers-or-the-models/review/03-what-is-left-once-the-chain-settles.md) | verified |
| Regenerate with | `scripts/mechanism_study/run_corrector_curve.sh` with `STAGE=search`, `STAGE=grid`, then `scripts/corrector_residual_curve.py --verdict --c <multiplier>`, per [decide where a run goes](../../runbook/running-things-on-the-cluster/launching-and-harvesting-a-run.md#1-decide-where-a-run-goes) | |

## Depends on

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- What the correction is, and why its size is read against the product-of-experts prediction:
  [the interaction term](../../context/world/interaction-term.md)
- What product-of-experts composition is and what the joint prompt is the target of:
  [PoE composition](../../context/world/poe-composition.md)
- Which python build runs on which card, and why a comparison across corrector counts stays on one
  device: [the cluster nodes](../../environment/hpc/nodes.md)
- Why the grid ran outside Slurm and was harvested with `pgrep`:
  [the execution protocol](../../environment/hpc/execution-protocol.md)

## Still open

Navigation: ⬅️ [Previous](#depends-on) | 📋 [TOC](#table-of-contents)

- [ ] The named next action: seeds 10, 11 and 12 at the working step size and corrector counts 0,
      20, 100 and 200, both pairs, about two hours on the Blackwell card, then the same three-way
      threshold on the seed-mean curve with the bars unchanged. Until then the split has no
      licensed number.
- [ ] A divergence guard that sees what the eye sees. The latent-norm bound cannot trip below the
      point where the Euler step stops contracting, around a multiplier of 170, so it passed every
      step size whose renders are noise. Every measured chain now saves its final image, which is
      the check that caught it.
- [ ] The free bound on the model's share (step 24 of the running order) has not run, so the
      remainder measured here has no second route to be checked against.
