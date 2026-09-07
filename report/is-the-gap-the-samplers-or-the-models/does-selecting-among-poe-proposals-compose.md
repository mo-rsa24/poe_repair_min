# Does choosing among the product's own proposals produce two animals?   ❓ inconclusive · verified 2026-09-05

**The claim**

A particle sampler that adds no correction to the score, and only reweights and resamples the
plain product-of-experts sampler, did not produce two animals on any of three animal pairs.

The weights came from a classifier trained to tell joint-prompt latents from product-of-experts
latents at the same noise level. That classifier memorised its 120 training images instead of
learning composition.

So by the bar written before the run, the answer is inconclusive. The direction points at null:
every selected state was still one animal, and so was every unselected one. All 12 particles per
checkpoint (4 per cell, 3 cells) have now been scored: the control's 12 are 0 of 12, the final
checkpoint's 12 are 0 of 12, and the detector's 10 hits at the four mid-run checkpoints are each
one fused animal to the eye.

**What would have counted**

Pass if the resampled particle composes at least 0.25 more often than the same particles left
unweighted.

Null if the two compose fractions are within 0.10 of each other while the classifier's
validation accuracy is at or above 0.60.

Inconclusive if validation accuracy never reaches 0.60.

Pinned in
[the plan](../../plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md#the-claim)
and as the constants `PASS_MARGIN`, `NULL_MARGIN` and `MIN_VAL_ACC` in
`poe_repair/experiments/twisted_smc/train.py`, before the run.

## Table of contents

- [1. The choice moves, the composition does not](#1-the-choice-moves-the-composition-does-not)
- [2. The classifier memorised rather than learned](#2-the-classifier-memorised-rather-than-learned)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Where this came from](#where-this-came-from)
- [Depends on](#depends-on)
- [Still open](#still-open)

## 1. The choice moves, the composition does not

Navigation: 📋 [TOC](#table-of-contents) | [Next](#2-the-classifier-memorised-rather-than-learned) ➡️

![Three cells by Mono, PoE and eleven SMC checkpoints](../../artifacts/results/is-the-gap-the-samplers-or-the-models/twisted-smc-checkpoint-timeline.png)
*What to notice: every Mono tile has two animals, every PoE tile one, and every SMC tile one, a
different one from the PoE tile in its row. Rows are wolf × husky and lion × tiger (training
pairs) and cat × dog (held out); columns after the first two are the same four noise draws under
the twist at that training step.*
📊 Drawn in [Figure 1 of the figure explainer](../../artifacts/results/is-the-gap-the-samplers-or-the-models/figure-explainer.md#figure-1-the-checkpoint-timeline-three-cells-by-eleven-checkpoints).

![Mono, product-of-experts, and twisted SMC for cat × dog seed 1 at training step 60k](../../artifacts/results/is-the-gap-the-samplers-or-the-models/twisted-smc-cat-x-dog-step-060000-strip.png)
*The held-out row at one checkpoint, full size. The middle and right panels are each one animal;
the right one was chosen after seven resamples from the noise that gave the middle one.*
📊 Drawn in [Figure 2 of the figure explainer](../../artifacts/results/is-the-gap-the-samplers-or-the-models/figure-explainer.md#figure-2-cat--dog-at-step-60k-the-three-panels-at-full-size).

**The number.** Compose fraction, the share of images the validated detector counts two or more
animal instances in. Two reads: over the shown particle per cell (3 images per checkpoint, what
the run itself scored), and over every particle (4 per cell, 12 images per checkpoint):

| Panel | Shown particle, step 100k | All 12 particles, step 100k | All 12 particles, other checkpoints |
|---|---|---|---|
| Mono (joint prompt) | 1.0 (3 of 3) | 1.0 (12 of 12) | same images at every checkpoint |
| PoE control (same noise, no weights) | 0.0 (0 of 3) | 0.0 (0 of 12) | same images at every checkpoint |
| Twisted SMC | 0.0 (0 of 3) | 0.0 (0 of 12) | 0 of 12 at 0, 10k, 20k, 30k, 50k, 90k; 3 of 12 at 40k and 80k; 2 of 12 at 60k and 70k |

The detector's 10 hits across the four mid-run checkpoints are 9 cat × dog particles and 1
wolf × husky particle. Each was opened and read by eye on 2026-09-05: every one is a single
animal with one head and one body (the cat × dog ones a fused cat-dog with a collar tag, the
wolf × husky one a husky), and the second box the detector draws sits on the tag or a patch of
coat. The shown particle's 0.33 at step 40k is one of these. So the detector read is 10 of 132
SMC particles and the eye read is 0 of 132; the detector is the pinned rule, and the eye read is
recorded beside it rather than used to overrule it.

Resampling fired 5 to 8 times per 20-step run at every checkpoint, so the weights were not flat.

From `history.json` keys `eval/compose_mono_mean`, `eval/compose_poe_mean`,
`eval/compose_smc_mean` and `eval/n_resample_mean`, from `verdict.json`, and from
`all_particles_scores.json` (every particle's count, written by
`scripts/twisted_smc/score_all_particles.py` on 2026-09-05), all in
`/datasets/mmolefe/poe_repair_min/outputs/interaction_term/twisted_smc/twist_w64_b16_lr1e-04_s100000_20260905-072331/`.
A copy of the per-particle scores sits at
[twisted-smc-all-particles-scores.json](../../artifacts/results/is-the-gap-the-samplers-or-the-models/twisted-smc-all-particles-scores.json).
Mark: verified, read 2026-09-05.

## 2. The classifier memorised rather than learned

Navigation: ⬅️ [1. The choice moves](#1-the-choice-moves-the-composition-does-not) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

![Accuracy against step, and training loss per noise bucket against step](../../artifacts/results/is-the-gap-the-samplers-or-the-models/twist-training-curves.png)
*What to notice: the grey training line sits at 1.0 while the blue held-out line wanders around
the 0.60 bar and ends below it. On the right, the green high-noise bucket falls twenty times
under chance, where the two classes are near-identical noise and content cannot separate them.*
📊 Drawn in [Figure 3 of the figure explainer](../../artifacts/results/is-the-gap-the-samplers-or-the-models/figure-explainer.md#figure-3-what-the-twist-head-learned-accuracy-and-loss-by-noise-level).

**The number.** Validation accuracy on 16 held-out cells from 8 pairs never in training: 0.60 at
step 1k, a best of 0.66 at step 31k, 0.53 at step 61k, 0.56 at step 100k. Validation loss rose
from 2.2 to 5.1 over the run. Training accuracy reached 1.0 by step 30k. Training loss by noise
bucket at step 100k: 0.00 for timesteps under 300, 0.00 for 300 to 700, 0.03 for 700 and above,
against 0.69 at step 1. At the highest noise the two classes are statistically the same
distribution, so a loss far below chance there can only come from recognising which of the 120
training latents a sample was made from. From `history.json` keys `val/acc`, `val/loss`,
`train/acc` and `train/loss_bucket/*` in the run directory above; the drawn values are copied
into `twisted-smc-report-figures.json` beside the figure. Mark: verified, read 2026-09-05.

## What this cannot tell you

Navigation: ⬅️ [2. The classifier](#2-the-classifier-memorised-rather-than-learned) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

Whether a twist that reads composition rather than pair identity would change the result. That
is the missing piece behind the inconclusive verdict, and no run has tested it.

Whether the detector's 10 mid-run hits are composition. The eye says no on all 10, and a compose
scorer that fires on a collar tag is a scorer error the pinned rule does not correct for. The
number that reaches the paper is the detector's, with this sentence beside it.

Nothing about more particles. K was 4; the product's proposals at K 16 or 64 were not sampled.

The Mono and control panels use DDIM with `eta` 1.0, the sampler the particles need, and the
repo's other figures use `eta` 0. The three panels here are comparable with each other and not
directly with those.

## Where this came from

Navigation: ⬅️ [What this cannot tell you](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#depends-on) ➡️

| What | Source | Mark |
|---|---|---|
| The compose fractions and resample counts | `history.json` and `verdict.json` in the run directory above, read 2026-09-05 | verified |
| The accuracy and loss curves | the same `history.json`; also W&B `prime_lab/poe-repair-animals-compose/runs/3cwrxlw0` | verified |
| The renders | `references/` and `samples/step_*/` in the run directory, rendered 2026-09-05 on mscluster110 | verified |
| The all-particle scores | `all_particles_scores.json` in the run directory, written by `scripts/twisted_smc/score_all_particles.py` on mscluster85 device 0, 2026-09-05; the 10 detector hits opened and read by eye the same day | verified |
| The run | task **2.1** in [the plan](../../plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md); verdict in [its review file](../../plans/06-is-the-gap-the-samplers-or-the-models/review/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md#runs) | verified |
| How the thread got here | [the note](../../artifacts/notes/from-an-x-post-to-a-twisted-smc-baseline/note.md) | |
| Regenerate with | no runbook recipe yet; the launcher is `scripts/twisted_smc/train_twist.sbatch full`, the figures `scripts/twisted_smc/report_figures.py` | |

## Depends on

Navigation: ⬅️ [Where this came from](#where-this-came-from) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- What the interaction term is, and why the twist is its potential:
  [context: the interaction term](../../context/world/interaction-term.md)
- What a compose rate is and what the detector can and cannot tell you:
  [context: compose rate](../../context/world/compose-rate.md)
- The node and the python environment the run needed:
  [environment: nodes](../../environment/hpc/nodes.md)
- The idea the twist borrows, and the two posts that led to it:
  [source 5, CDM](../../context/sources.md#5-cdm-contrastive-distribution-matching-for-discrete-diffusion-and-the-two-x-posts-that-led-to-it)

## Still open

Navigation: ⬅️ [Depends on](#depends-on) | 📋 [TOC](#table-of-contents)

- [ ] A runbook recipe for launching and harvesting the twisted-SMC run.
- [ ] A twist that cannot memorise: more positives and a head that does not see pair identity.
      The design is in [the review file's next step](../../plans/06-is-the-gap-the-samplers-or-the-models/review/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md#next-step).
