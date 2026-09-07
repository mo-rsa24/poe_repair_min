# Does steering the plain product's particles on the compose scorer find a composing proposal?   ⚪ null · verified 2026-09-06

**The claim**

Feynman-Kac steering (Singhal et al., arXiv 2501.06848) run over the plain product-of-experts
sampler, with the validated compose scorer itself as the reward, did not produce a two-animal
image on any of the 8 held-out seeds of cat × dog, at 4 particles or at 16.

The reason is in the proposals. All 128 unweighted 16-particle draws per seed set were single
animals, so there was nothing for selection to find. The reward read the same verdict as the
final judge on 99% of particles from step 10 on, so it was not blind.

Steering changed which single animal appeared and never reached two.

**What would have counted**

Support if steering at K 16 composed at least 0.25 more often than the same particles left
unweighted. Null if the two were within 0.10 at both K 4 and K 16. Inconclusive if the reward read
on the step-10 estimate of the finished image disagreed with the final verdict on more than half
the particles. Pinned in [the plan](../../plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/10-feynman-kac-steering-on-a-detector-reward.md#the-claim)
and as `PASS_MARGIN`, `NULL_MARGIN`, `MAX_REWARD_DISAGREEMENT` and `REWARD_BLIND_STEP` in
`poe_repair/experiments/fk_steering/steer.py`, before the run.

## Table of contents

- [1. Every particle is one animal, weighted or not](#1-every-particle-is-one-animal-weighted-or-not)
- [2. The control pair shows what the scorer measures there](#2-the-control-pair-shows-what-the-scorer-measures-there)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Where this came from](#where-this-came-from)
- [Depends on](#depends-on)
- [Still open](#still-open)

## 1. Every particle is one animal, weighted or not

Navigation: 📋 [TOC](#table-of-contents) | [Next](#2-the-control-pair-shows-what-the-scorer-measures-there) ➡️

![Cat × dog, seeds 9 to 16, by Mono, plain PoE, the unweighted control, best of 16, and Feynman-Kac steering at K 4 and K 16](../../artifacts/results/is-the-gap-the-samplers-or-the-models/fk-steering-cat-dog-sheet.png)
*What to notice: the Mono column is two animals in every row and every other column is one, a
cat-dog blend in most rows. The steered tiles are different single animals from the control's
(seed 14 turns into a black cat, seed 9 goes monochrome), so the weights did move the population.
The number on each tile is the detector's animal count.*

**The number.** Compose rate over the 8 seeds, the share whose finished image the validated
detector counts two or more animals in:

| Column | Cat × dog |
|---|---|
| Mono, the joint prompt, DDIM `eta` 0 | 1.0 |
| Plain PoE, `eta` 0 | 0.0 |
| Unweighted control, particle 0 of 16, `eta` 1 | 0.0 |
| Best of the 16 unweighted, by final count | 0.0 |
| Feynman-Kac steering, K 4 | 0.0 |
| Feynman-Kac steering, K 16 | 0.0 |

Over every particle rather than the shown one: 0 of 128 unweighted and 0 of 128 steered at K 16,
0 of 32 and 0 of 32 at K 4. Agreement between the reward read on the decoded step-s estimate and
the detector's verdict on the finished image, K 16, over all particles: 0.38 at step 0, 0.99 at
step 10, 0.99 at 20, 0.98 at 30, 1.0 at 40. The reward separated particles at most once per run
(0.625 informative resamples of 5, mean over seeds), because nearly every particle scored 1 at
every read.

From `summary.json` and `verdict.json` in
`/datasets/mmolefe/poe_repair_min/outputs/interaction_term/fk_steering/fk_K4-16_s50_20260905-165958/`,
copied to [fk-steering-summary.json](../../artifacts/results/is-the-gap-the-samplers-or-the-models/fk-steering-summary.json).
Mark: verified, read 2026-09-06.

## 2. The control pair shows what the scorer measures there

Navigation: ⬅️ [Previous](#1-every-particle-is-one-animal-weighted-or-not) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

![Butterfly × meadow, same layout](../../artifacts/results/is-the-gap-the-samplers-or-the-models/fk-steering-butterfly-meadow-sheet.png)
*What to notice: a butterfly over a meadow counts as one animal, so Mono itself is red in 7 of 8
rows. The green steered tiles hold two butterflies, or a butterfly drawn out of flowers. No
steered tile lost the butterfly or the meadow.*

**The number.** Same columns: Mono 0.125, plain PoE 0.0, control 0.0, best of 16 0.625, steering
K 4 0.125, steering K 16 0.75. These are butterfly counts, not composition. The pair was on the run
as the composing control, and what it shows is that the instance-count scorer does not measure
composition for an animal-plus-scene pair; that it did not break the pair is the one thing it
can say here.

## What this cannot tell you

Navigation: ⬅️ [Previous](#2-the-control-pair-shows-what-the-scorer-measures-there) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**Nothing about a larger K.** 16 particles per seed found no composing state; whether 64 or 256
would is not measured.

**Nothing about selection on top of a corrected sampler.** With the rank-32 correction in the
proposal the span is different; that is
[plan 11](../../plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/11-feynman-kac-steering-on-top-of-the-rank-32-correction.md).

**The guide and the judge are one instrument.** A detector error would be selected for. None
occurred here because no cat × dog particle ever counted 2, but the design carries the risk.

**One departure from the paper.** Resampling is systematic rather than multinomial, so equal
weights leave the population unchanged; the paper's multinomial resampling would have duplicated
and dropped particles at random when all rewards tied.

## Where this came from

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#depends-on) ➡️

| What | Source | Mark |
|---|---|---|
| The compose rates, fractions and agreement | `summary.json` in the run directory above, read 2026-09-06 | verified |
| The renders | `a_cat__x__a_dog/seed_*/` and `a_butterfly__x__a_flower_meadow/seed_*/` in the run directory, rendered 2026-09-05 on mscluster109 device 1 | verified |
| The run | task **3.1** in [the plan](../../plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/10-feynman-kac-steering-on-a-detector-reward.md); verdict in [its review file](../../plans/06-is-the-gap-the-samplers-or-the-models/review/10-feynman-kac-steering-on-a-detector-reward.md#runs); W&B `prime_lab/poe-repair-animals-compose/runs/czim1n0w` | verified |
| The method | [the register row for 2501.06848](../../plans/standing/literature/reading-register.md) | |
| Regenerate with | `GPU=<idx> bash scripts/fk_steering/run_fk_steering.sh full` on a free `biggpu` device, about 3 h on an A6000 | |

## Depends on

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- What a compose rate is and what the detector can and cannot count:
  [context: compose rate](../../context/world/compose-rate.md)
- What PoE composition is and why the product's proposals blend:
  [context: PoE composition](../../context/world/poe-composition.md)
- When the run commits to one animal, which is why the reward is not blind at step 10:
  [where does each condition land](../when-does-the-outcome-lock-in/where-does-each-condition-land.md)
- The node and the environment: [environment: nodes](../../environment/hpc/nodes.md)

## Still open

Navigation: ⬅️ [Depends on](#depends-on) | 📋 [TOC](#table-of-contents)

- [ ] The same steering with the rank-32 correction in the proposal, at λ 0.5 and 1.2, with an
      ImageReward tie-breaker so the cleanest two-animal particle is kept: plan 11, in flight.
- [ ] How this null reads against the in-span share from
      [what the correction is made of](../../plans/05-when-does-the-outcome-lock-in/review/07-what-the-correction-is-made-of.md),
      once its rung 2 runs. Selection can only reach what the proposal spans; this run measured
      an empty span at K 16.
- [ ] A scorer that can read the butterfly × meadow pair, so the composing control means
      composition there.
