# Does selecting among the corrected sampler's own draws produce a clean cat and dog?   ✅ support on the pre-registered bar · ⚪ null on what the resampling itself adds · verified 2026-09-07

**The claim**

With the rank-32 correction at half dose inside the proposal, running 16 particles and steering
them on the compose scorer produces two separate animals on all 8 held-out seeds of cat × dog.
The corrected sampler on its own produces two animals on 3 of those 8.

The gap comes from drawing 16 particles rather than from the resampling: the best of the same 16
unweighted particles is also 8 of 8.

What the resampling changes is which of the composing particles survives. The steered pick scores
higher on a human-preference reward than the best unweighted particle, and higher than the joint
prompt's own render.

At the full dose the corrected sampler is already at 8 of 8, so there is no compose headroom left
and only the picture quality separates the columns.

**What would have counted**

Support if steering at K 16 composes at least 0.25 more often than the unweighted control at
λ 0.5.

Null if the two are within 0.10 at both λ 0.5 and λ 1.2.

Inconclusive if the reward read on the running estimate at step 10 disagrees with the final
scorer verdict on more than half the particles.

Pinned as `PASS_MARGIN`, `NULL_MARGIN`, `MAX_REWARD_DISAGREEMENT`, `REWARD_BLIND_STEP`,
`ADAPTER_LAMBDAS` and `ADAPTER_JUDGED_LAMBDA` in
`poe_repair/experiments/fk_steering/steer.py`, and in
[the plan](../../plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/13-feynman-kac-steering-on-top-of-the-rank-32-correction.md#the-claim),
before the run.

## Table of contents

- [1. Half dose plus selection reaches what the full dose reaches](#1-half-dose-plus-selection-reaches-what-the-full-dose-reaches)
- [2. Sixteen draws do the composing; the weights choose the picture](#2-sixteen-draws-do-the-composing-the-weights-choose-the-picture)
- [3. At the full dose only the picture quality separates the columns](#3-at-the-full-dose-only-the-picture-quality-separates-the-columns)
- [4. What the count cannot see](#4-what-the-count-cannot-see)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Where this came from](#where-this-came-from)
- [Depends on](#depends-on)
- [Still open](#still-open)

## 1. Half dose plus selection reaches what the full dose reaches

Navigation: 📋 [TOC](#table-of-contents) | [Next](#2-sixteen-draws-do-the-composing-the-weights-choose-the-picture) ➡️

![Cat by dog, eight seeds by six conditions, rank-32 correction at half dose](../../artifacts/results/is-the-gap-the-samplers-or-the-models/fk-steering-on-adapter-cat-dog-lambda-0.5-sheet.png)

**What you are looking at.** One row per held-out seed, 9 to 16. The first three columns are
single deterministic renders: the joint prompt, the plain product of experts, and the product plus
half of the rank-32 correction. The last three run 16 particles from the same starting noise with
that same half correction: the first particle left alone, the best of the 16 left alone, and the
particle that survives steering. Each tile carries the detector's animal count, green at two or
more, and its human-preference score.

**The number.** Share of the 8 seeds the validated instance-count scorer reads as two or more
animals:

| Condition, cat × dog at λ 0.5 | Seeds of 8 |
|---|---|
| the joint prompt | 8 |
| plain product of experts | 0 |
| the correction alone, half dose | 3 |
| 16 particles, first one, no weights | 2 |
| 16 particles, best of them, no weights | 8 |
| 16 particles, steered | 8 |

The steered column beats the first-particle control by 0.75, against a bar of 0.25.

From `verdict.json` and `summary.json` in
`/datasets/mmolefe/poe_repair_min/outputs/interaction_term/fk_steering/fka_K16_s50_20260906-025826/`,
copied to
[the summary beside the sheet](../../artifacts/results/is-the-gap-the-samplers-or-the-models/fk-steering-on-adapter-summary.json).
Mark: verified, read 2026-09-07.

## 2. Sixteen draws do the composing; the weights choose the picture

Navigation: ⬅️ [Previous](#1-half-dose-plus-selection-reaches-what-the-full-dose-reaches) | 📋 [TOC](#table-of-contents) | [Next](#3-at-the-full-dose-only-the-picture-quality-separates-the-columns) ➡️

**What you are looking at.** The fifth and sixth columns of the same sheet. Both start from the
same 16 particles. The fifth keeps the best one after the fact; the sixth resamples toward the
promising ones five times during the run.

**The number.** Both reach 8 of 8 on the count, so the resampling adds nothing there. They differ
on the human-preference score, ImageReward against the prompt "a cat and a dog", averaged over the
8 seeds:

| Condition, cat × dog at λ 0.5 | Mean ImageReward |
|---|---|
| steered pick | +0.75 |
| best of 16, no weights | +0.45 |
| the joint prompt | +0.24 |
| the correction alone, half dose | −0.77 |
| plain product of experts | −1.51 |

The scale runs about −2 to +2 and rewards images a person would pick for the prompt. Steering
also raised the share of all 16 final particles that compose, from 0.23 unweighted to 0.94.

Resampling separated the particles on 4.9 of its 5 opportunities per run, so the weights were
doing work throughout, unlike the plain-product run where the reward was flat.

Same source files as rung 1. Mark: verified, read 2026-09-07.

## 3. At the full dose only the picture quality separates the columns

Navigation: ⬅️ [Previous](#2-sixteen-draws-do-the-composing-the-weights-choose-the-picture) | 📋 [TOC](#table-of-contents) | [Next](#4-what-the-count-cannot-see) ➡️

![Cat by dog, eight seeds by six conditions, rank-32 correction at full dose](../../artifacts/results/is-the-gap-the-samplers-or-the-models/fk-steering-on-adapter-cat-dog-lambda-1.2-sheet.png)

**What you are looking at.** The same six columns with the correction at λ 1.2, the dose the
showcase renders ship at.

**The number.** The correction alone reads as two animals on 7 of 8 seeds and the unweighted
16-particle control on 8 of 8, so the steered column's 8 of 8 has no room to show a gap: the
compose difference is 0.00, which is the null branch at this dose. Mean ImageReward still
separates the columns: +1.16 steered, +0.97 for best of 16, +0.05 for the correction alone,
+0.24 for the joint prompt.

This is the condition that produces the cleanest pairs in the whole run. Both animals are sharp
and separate, and no tile carries the fused face that the plain product produces.

Same source files as rung 1. Mark: verified, read 2026-09-07.

## 4. What the count cannot see

Navigation: ⬅️ [Previous](#3-at-the-full-dose-only-the-picture-quality-separates-the-columns) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**What you are looking at.** The scorer counts instances of a generic "animal" query, so it reads
two separate bodies and never checks that one is a cat and the other a dog. Three steered tiles at
λ 0.5 were opened at full resolution to test that.

**The number.** By eye, over the 8 steered tiles at λ 0.5: two separate animals on 8, one clear cat
beside one clear dog on 5 (seeds 9, 10, 12, 15, 16). Seed 14 is a large black dog beside a small
white terrier, two dogs. Seeds 11 and 13 place a small animal beside one that keeps cat ears and a
dog muzzle, a chimera that the count reads as a second animal. At λ 1.2 the same read gives 7 of 8
as one cat beside one dog, with seed 12 showing two cats.

A per-concept detector pass does not settle this. Querying "a cat" and "a dog" separately returns
both on every tile of every column, including the plain-product tiles that are single fused
animals, because the detector fires both labels on one chimera. The read stands on the eye, and
the per-concept output is filed only as the evidence that it cannot be used here:
`fk_steering_a_cat__x__a_dog_lam0.5_sheet_identity.json` in the run directory, written by
`scripts/fk_steering/identity_check.py`. Mark: verified, read 2026-09-07.

## What this cannot tell you

Navigation: ⬅️ [Previous](#4-what-the-count-cannot-see) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**The guide and the judge are the same scorer.** The particle shown is chosen by the instrument
that then scores it, so the steered column can only win ties in its own favour. The best-of-16
column carries the same advantage without any resampling, which is why the two are read together.

**Sixteen draws are not free.** The steered column costs 16 trajectories with two network passes
each, about 25 minutes per seed on an RTX 3090, against seconds for the single corrected render.
Nothing here says the compose rate per unit of compute improved.

**The composing control pair is unreadable by this scorer.** On butterfly × meadow the joint
prompt itself scores 1 of 8, because a butterfly over a meadow is one animal. Its steered column
reads 8 of 8 by counting several butterflies. No steered tile lost the butterfly or the meadow, so
the method did not break the pair, and no compose number from that pair means composition.

**One pair, one adapter, one checkpoint.** Cat × dog at rank 32, step 30050. Whether selection
helps on a pair the adapter transfers to less well is untested.

**Two doses, not a curve.** λ 0.5 and λ 1.2 only, so where between them the headroom closes is not
measured.

## Where this came from

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#depends-on) ➡️

| What | Source | Mark |
|---|---|---|
| The compose rates, the fidelity means, the agreement and resample counts | `summary.json` and `verdict.json` in the run directory above, read 2026-09-07 | verified |
| The three sheets and their sidecars | the same run, `fk_steering_*_sheet.png`; copied into [the results folder](../../artifacts/results/is-the-gap-the-samplers-or-the-models/README.md) | verified |
| The eye read of identity | five steered tiles opened at full resolution, 2026-09-07 | stated |
| The per-concept detector pass | `scripts/fk_steering/identity_check.py` on mscluster85 device 0, 2026-09-07 | verified |
| The run | task **2.1** in [the plan](../../plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/13-feynman-kac-steering-on-top-of-the-rank-32-correction.md); verdict in [its review file](../../plans/06-is-the-gap-the-samplers-or-the-models/review/13-feynman-kac-steering-on-top-of-the-rank-32-correction.md#runs); W&B `prime_lab/poe-repair-animals-compose/runs/dfrceppl`, Slurm job 50332 | verified |
| The method | Singhal et al., [arXiv 2501.06848](https://arxiv.org/abs/2501.06848), registered in [the reading register](../../plans/standing/literature/reading-register.md) | verified |
| Regenerate with | `sbatch --nodelist=<idle bigbatch node> scripts/fk_steering/fk_steering.sbatch adapter`; no runbook recipe yet | |

## Depends on

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- The same steering over the plain product, which found nothing to select:
  [does steering on the scorer find a composing proposal](does-steering-on-the-scorer-find-a-composing-proposal.md)
- What the correction is and which checkpoint this one is:
  [context: the LoRA corrector](../../context/world/lora-corrector.md)
- What the count can and cannot tell you: [context: compose rate](../../context/world/compose-rate.md)
- Which dose the adapter alone composes at:
  [which checkpoint composes best](../does-training-longer-help-the-pooled-lora/which-checkpoint-composes-best-and-does-more-correction-help.md)
- The node and environment the run needed: [environment: nodes](../../environment/hpc/nodes.md)

## Still open

Navigation: ⬅️ [Previous](#depends-on) | 📋 [TOC](#table-of-contents)

- [ ] A scorer that separates a cat beside a dog from two dogs. The instance count reads both as
      two animals, and a per-concept query fires on a chimera, so identity currently rests on the
      eye.
- [ ] Whether the same selection helps at ranks 8 and 16, and on a pair the adapter transfers to
      less well.
- [ ] The compose rate per unit of compute, against the single corrected render.
- [ ] A runbook recipe for launching and harvesting a steering run.
