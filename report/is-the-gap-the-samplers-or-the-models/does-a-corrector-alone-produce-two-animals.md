# Does a Langevin corrector, adding nothing to the score, produce two animals?   ⚪ null · verified 2026-09-06

**The claim**

No. Twenty corrector steps at every one of the 50 noise levels leave cat × dog at exactly plain
[product-of-experts](../../context/world/poe-composition.md)' [compose rate](../../context/world/compose-rate.md), zero of eight held-out seeds, and every one of the forty
renders from a sliding ten-step corrector window is one animal too.

What the corrector does change is the kind of picture. Five of the eight [settled](../../../../../goal-setting/learning/deep-learning/diffusion-models/sampler-correctors-for-composition/plans/11-stationary-and-detailed-balance.md) renders leave
photography for a cartoon or a line drawing, and the detector's confidence on the control pair's
butterfly falls while its verdict stays green.

This is the behavioural answer for the window where the size measurement cannot attribute. A
sampler that settles the latent into the distribution the summed score actually describes does not
do what adding the missing term does.

## Table of contents

- [What would have counted](#what-would-have-counted)
- [1. No window composes, and the injected correction's window does](#1-no-window-composes-and-the-injected-corrections-window-does)
- [2. On the held-out seeds the corrector is plain product-of-experts](#2-on-the-held-out-seeds-the-corrector-is-plain-product-of-experts)
- [3. What the corrector changes instead](#3-what-the-corrector-changes-instead)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Where this came from](#where-this-came-from)
- [Depends on](#depends-on)
- [Still open](#still-open)

## What would have counted

Navigation: 📋 [TOC](#table-of-contents) | [Next](#1-no-window-composes-and-the-injected-corrections-window-does) ➡️

The window question was pre-registered as a comparison recorded either way, with no pass or fail:
the same pair, the same four seeds and the same nine ten-step window positions the injected
correction used, and the question of whether the compose rate peaks in the same window. What would
have made it worthless is a layout that does not match the figure it is compared against, since the
comparison is the whole read.

The sheet question carried one bar: the control pair's corrector column may not lose more than
`FIDELITY_MAX_COMPOSE_LOSS_SEEDS = 1` composed seed against its own plain-PoE column, the constant
sitting in `scripts/corrector_window_sweep.py`.

Both are pinned in
[does the corrector compose in the same window](../../plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis/04-does-the-corrector-compose-in-the-same-window.md)
and [its review file](../../plans/06-is-the-gap-the-samplers-or-the-models/review/04-does-the-corrector-compose-in-the-same-window.md),
written before the composer existed.

## 1. No window composes, and the injected correction's window does

Navigation: ⬅️ [Previous](#what-would-have-counted) | 📋 [TOC](#table-of-contents) | [Next](#2-on-the-held-out-seeds-the-corrector-is-plain-product-of-experts) ➡️

![Four seeds by ten columns: the corrector switched on inside one ten-step window, every frame red](../../artifacts/results/is-the-gap-the-samplers-or-the-models/corrector-window-slides-cat-dog.png)
*Rows are seeds 9 to 12, columns the nine ten-step window positions plus the corrector on all 50
steps; the green bar under each column shows where the corrector acts, and the frame is the
detector's verdict. What to notice: no frame is green, and the leftmost columns change the drawing
style while the columns from step 20 on leave the picture almost as plain product-of-experts made it.*

**The number.** Seeds composed of 4, by window position 0-10, 5-15, 10-20, 15-25, 20-30, 25-35,
30-40, 35-45, 40-50 and all 50: 0 in every column, by the validated instance count (GroundingDINO
"animal", confidence at or above 0.30, non-max suppression below 0.5 intersection over union, two
or more instances). Every one of the 40 renders counts exactly one animal. The injected correction
on the same pair and the same seeds composes 0.656 of its runs at steps 0 to 10 and 0.250 at 5 to
15, from [the timing verdict](../../plans/03-does-the-correction-cause-composition/review/05-when-in-the-run-it-matters.md).
From `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/window_curves_mcmc.json`,
fields `per_column[].composed_of_4` and `cells[].n_instances`.

## 2. On the held-out seeds the corrector is plain product-of-experts

Navigation: ⬅️ [Previous](#1-no-window-composes-and-the-injected-corrections-window-does) | 📋 [TOC](#table-of-contents) | [Next](#3-what-the-corrector-changes-instead) ➡️

![Eight seeds by three columns: the joint prompt, plain product-of-experts, and the corrector on all 50 steps](../../artifacts/results/is-the-gap-the-samplers-or-the-models/corrector-cat-dog-eight-seed-sheet.png)
*Rows are the held-out seeds 9 to 16. What to notice: the first column is two animals in every row
and the second and third are one animal in every row, differing from each other in style rather
than in count.*

**The number.** Compose rate over the eight held-out seeds of cat × dog, validated instance count:
joint prompt 8 of 8 (1.00), plain product-of-experts 0 of 8 (0.00), the corrector at 20 steps per
level 0 of 8 (0.00). The control pair butterfly × meadow, read as both concepts detected above
confidence 0.30 because the validated animal-count rule does not apply to a meadow: 8, 8, 8 of 8,
so the control bar holds. From
`/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/sheet_scores.json`, field
`summary`.

## 3. What the corrector changes instead

Navigation: ⬅️ [Previous](#2-on-the-held-out-seeds-the-corrector-is-plain-product-of-experts) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

![Cat and dog at the smaller step size across corrector counts: a photograph at 0 and 1, a stylised cat by 20, an illustration at 100 and 200](../../artifacts/results/is-the-gap-the-samplers-or-the-models/corrector-cat-dog-across-k-c0p3.png)
*One tile per corrector count at the smaller step size, cat × dog seed 9. What to notice: one
animal in every tile, and the drift from a photograph to a drawing as the count rises, with the
latent norm never above 1.01 times its start.*

**The number.** The control pair's detector confidence for "butterfly" on the eight-seed sheet,
joint prompt against the corrector: 0.94 to 0.73 (seed 9), 0.50 to 0.33 (seed 13) and 0.95 to 0.56
(seed 15), while every frame stays green. Same `sheet_scores.json`, field
`rows[].corrector_score.concept_conf`. The style drift is what those falling confidences read.

## What this cannot tell you

Navigation: ⬅️ [Previous](#3-what-the-corrector-changes-instead) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**One compute budget.** Twenty corrector steps per noise level, which is 60 extra network
evaluations inside a window. The corrector count was chosen because it is the largest at which the
settled samples of both pairs are still photographs, not because the residual curve licensed a flat
part; the curve licensed none.

**One step size.** The multiplier the composing pair's control survives. Larger destroys the
sample, smaller leaves the control rising.

**Behaviour, not attribution.** In the window where this composes or does not, steps 0 to 10, the
sampler's error and the model's error cannot be told apart, so a null here bounds what a
training-free corrector does and says nothing about which share is larger.

**The detector, on a pair it is known to argue with.** The eye read agrees with the count on all 40
window renders and all 16 sheet renders here, which is not always true on this pair.

**One pair for the failing case.** Cat × dog only.

## Where this came from

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#depends-on) ➡️

| What | Source | Mark |
|---|---|---|
| The per-column counts and every window render's instance count | `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/window_curves_mcmc.json`, 40 cells, read 2026-09-06 | verified |
| The eight-seed compose rates and the concept confidences | the same folder's `sheet_scores.json`, 16 rows, read 2026-09-06 | verified |
| The renders | `corrector/window/pairs/a_cat__x__a_dog/seed_<n>/`, `corrector/sheet/` and `corrector/pairs/`, all under `/datasets` | verified |
| The injected correction's own window numbers | [the timing verdict](../../plans/03-does-the-correction-cause-composition/review/05-when-in-the-run-it-matters.md), quoted not re-run | verified |
| The runs | step 27 of the running order: the window sweep on mscluster108 device 1 (1.6 h) and the sheet on the same device (2.3 h), 2026-09-05 to 06, commit 0150704; W&B `prime_lab/poe-repair-animals-compose/s61hldbc`; tables in [the review file](../../plans/06-is-the-gap-the-samplers-or-the-models/review/04-does-the-corrector-compose-in-the-same-window.md) | verified |
| Regenerate with | recipes 4 and 5 of [running the Langevin corrector](../../runbook/running-things-on-the-cluster/running-the-langevin-corrector.md): the sliding corrector window, then the eight-seed sheets | |

## Depends on

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- What a compose rate is and what the detector can and cannot see:
  [compose rate](../../context/world/compose-rate.md)
- What the correction is, the thing the corrector adds nothing of:
  [the interaction term](../../context/world/interaction-term.md)
- What the pair pool is and why cat × dog is the failing case:
  [the animal pair](../../context/world/animal-pair.md)
- Why the runs are invisible to the scheduler:
  [the execution protocol](../../environment/hpc/execution-protocol.md)

## Still open

Navigation: ⬅️ [Previous](#depends-on) | 📋 [TOC](#table-of-contents)

- [ ] Whether a larger budget composes: the same ten columns at 100 corrector steps, or at the
      smaller step size, about 40 minutes on the Blackwell card. The renders at 100 steps are flat
      graphics on both pairs, so the expectation is a change of style and not of count.
- [ ] The style drift has no bar written for it. Every step size and both pairs agree that the
      settled sample moves from a photograph toward an illustration, which is a statement about
      what the product of the two diffused marginals looks like, and no plan asks it as a question
      yet.
