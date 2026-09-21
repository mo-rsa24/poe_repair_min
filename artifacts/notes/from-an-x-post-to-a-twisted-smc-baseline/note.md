# From an X post to a twisted-SMC baseline: the record of one day's thread

One conversation on 2026-09-05 went from a question about a browser visualiser, through two posts
about reward-guided sampling, to a built and finished experiment in scope 06. This is the record
of that thread for a reader who was not in it: what was asked, what was borrowed, what was
hypothesised, what was built, what ran, what the pictures show, and what the numbers say. The
verdict itself lives in [the finding](../../../report/is-the-gap-the-samplers-or-the-models/does-selecting-among-poe-proposals-compose.md);
this note is the path to it.

## Table of contents

- [Words this note uses](#words-this-note-uses)
- [1. The first question: can Diffusion Explorer show our sampler](#1-the-first-question-can-diffusion-explorer-show-our-sampler)
- [2. The two posts, and the connection](#2-the-two-posts-and-the-connection)
- [3. The hypothesis and the bar](#3-the-hypothesis-and-the-bar)
- [4. What was built](#4-what-was-built)
- [5. The runs, in order](#5-the-runs-in-order)
- [6. What the pictures show](#6-what-the-pictures-show)
- [7. What the numbers say](#7-what-the-numbers-say)
- [8. What it means, and what it does not](#8-what-it-means-and-what-it-does-not)
- [9. Where everything is](#9-where-everything-is)
- [Still open](#still-open)

## Words this note uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#1-the-first-question-can-diffusion-explorer-show-our-sampler) ➡️

- **Mono**: the joint prompt ("a cat and a dog") sampled as one prompt. The target. Owned by
  [PoE composition](../../../context/world/poe-composition.md).
- **PoE**: product of experts, the two single-animal predictions added and the null subtracted.
  The default that fails. Same owner.
- **The interaction term**, `r_t`: the per-step gap between the joint prompt's prediction and
  the product's. Owned by [the interaction term](../../../context/world/interaction-term.md).
- **The twist**: a learned scalar, `log psi_t(x_t)`, estimating how much more likely a noisy
  latent is under the joint prompt than under the product. Its gradient is the interaction term.
- **Twisted SMC**: sequential Monte Carlo, K particles run in parallel, each weighted by the twist
  and the population resampled when a few particles carry most of the weight.
- **Compose fraction**: the share of images the detector counts two or more animal instances in.
  Owned by [compose rate](../../../context/world/compose-rate.md).

## 1. The first question: can Diffusion Explorer show our sampler

Navigation: ⬅️ [Words](#words-this-note-uses) | 📋 [TOC](#table-of-contents) | [Next](#2-the-two-posts-and-the-connection) ➡️

**The ask**

Could SDXL inference for a joint prompt be ported into
[Diffusion Explorer](../../../context/sources.md#2-diffusion-explorer-interactive-exploration-of-diffusion-models),
so that joint, concept A, concept B and PoE could be watched side by side?

**The answer**

Not as a port. The tool trains small two-dimensional models in the browser and every renderer
assumes a sample is two numbers. What transfers is the design: precomputed trajectories loaded
from a file, one time slider driving every panel, and the arrows-on-the-current-sample idiom of
its classifier-free-guidance page, which draws the conditional and unconditional predictions on a
moving point. Our training cache already stores, per step, the state and the four raw predictions
(A, B, joint, null), so the four arrows and the gap between the last two are recoverable with no
GPU. That figure is still owed and is listed in the landing-figures plan; the thread moved on to
the second question.

## 2. The two posts, and the connection

Navigation: ⬅️ [1. The first question](#1-the-first-question-can-diffusion-explorer-show-our-sampler) | 📋 [TOC](#table-of-contents) | [Next](#3-the-hypothesis-and-the-bar) ➡️

**What the posts said**

Two posts, read through an API mirror because X blocks direct fetches. Minhyuk Sung, 2026-09-01:
"SMC is a powerful approach for diffusion reward guidance, but it can be extremely costly for
discrete diffusion due to repeated rollouts from every particle. CDM learns a twist function in
advance with a contrastive objective, amortizing this rollout cost while avoiding the train-test
mismatch of previous objectives." The quoted post by Jaihoon Kim: "Steering diffusion models
toward a reward is slow. CDM makes it up to 50× faster, with less than 5% compute overhead." Both
point at [CDM, arXiv 2605.23346](../../../context/sources.md#5-cdm-contrastive-distribution-matching-for-discrete-diffusion-and-the-two-x-posts-that-led-to-it),
a discrete-diffusion paper with no images and no composition in it.

**The question that followed**

Could diffusion be steered toward joint prompting the same way?

**The connection**

Set the base to PoE and the target to the joint prompt. The optimal twist is the ratio of the two
diffused densities, `p_J,t / p_PoE,t`, and its gradient is the interaction term this project has
measured for months. So the trained LoRA corrector is the gradient of CDM's twist, learned by
regression and added to the score. A twisted-SMC sampler would learn the potential itself and add
nothing: it would only choose among states the product already proposes. That makes it the one
sampler-side baseline whose success would mean the product's proposals already contain composing
states, which is the question scope 06 exists to ask. The 50× speed-up does not transfer, since
continuous latents already get a cheap twist from the Tweedie estimate, as the project page says.

## 3. The hypothesis and the bar

Navigation: ⬅️ [2. The two posts](#2-the-two-posts-and-the-connection) | 📋 [TOC](#table-of-contents) | [Next](#4-what-was-built) ➡️

**The claim under test**

Reweighting and resampling the plain PoE sampler on a learned joint-versus-PoE twist raises the
compose fraction over the same particles left unweighted.

**The bar, written before the run**

Pass if the SMC compose fraction beats the unweighted control's by 0.25 or more. Null if the two
are within 0.10 while the twist's validation accuracy on unseen pairs is at or above 0.60.
Inconclusive if that accuracy never reaches 0.60, because then the sampler was never really
tested. The three constants sit in the training script as `PASS_MARGIN`, `NULL_MARGIN` and
`MIN_VAL_ACC`, and the run writes `verdict.json` from them, so the bar could not move after the
result. Pinned in [the plan's claim section](../../../plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md#the-claim).

**What each outcome would have meant**

A pass would put a real share of the gap on the sampler's side and make twisted SMC a row in the
paper's baseline comparison. A null would be the model-side reading: the composing states are not
among what the product proposes, so a correction that adds a direction is required. Inconclusive
means the twist is the weak part and the question is still open from this side.

## 4. What was built

Navigation: ⬅️ [3. The hypothesis](#3-the-hypothesis-and-the-bar) | 📋 [TOC](#table-of-contents) | [Next](#5-the-runs-in-order) ➡️

Four modules under `poe_repair/experiments/twisted_smc/` and one launcher, all new that day.

**The bank** (`data.py`)

Every training-cache cell already holds `mono.png` and `poe.png`, the finished images of the joint
prompt and of the product for one pair and seed. Both are VAE-encoded once into latents and
stored. Positives are the Mono latents, negatives the PoE latents. Each batch re-noises both
classes to one shared random timestep with the forward kernel, which is CDM's amortisation trick
in continuous form and what keeps the timestep from being a giveaway. 120 training cells over 18
pairs; 16 validation cells from 8 held-out pairs, none of which appear in training. Horizontal
flips as the only image augmentation.

**The head** (`twist.py`)

A 4.7M-parameter convolutional classifier on the 4×128×128 latent: a stem, four downsampling
blocks each modulated by a sinusoidal timestep embedding plus a projection of the joint prompt's
pooled text embedding, a global pool, and one logit. Trained with binary cross-entropy. At the
optimum the logit is `log p_J,t − log p_PoE,t`.

**The sampler** (`sampler.py`)

K particles through DDIM with `eta` 1.0, because deterministic DDIM would make duplicated
particles identical forever and resampling could only lose diversity. Per step, each particle's
log-weight gains `log psi_{t'}(x_{t'}) − log psi_t(x_t)`; when the effective sample size drops
under K/2 the population is systematically resampled. With the twist off it is K independent
draws of the base sampler, from the same noise, which is the control.

**The trainer** (`train.py`)

AdamW on the classifier, validation every 1k steps, a checkpoint every 10k, and every 10k steps a
render pass: for each render cell, Mono and PoE references (rendered once at the start) and the
twisted-SMC pick at the current checkpoint, pasted into one strip, logged to W&B as an image and
bundled into an artifact, with the detector reading the three panels. The verdict is written at
the last render.

**The launcher** (`scripts/twisted_smc/train_twist.sbatch`)

Smoke and full modes, the disk guard on the output root, and the python environment chosen by
hostname, since the Blackwell nodes need `co3_bw`.

## 5. The runs, in order

Navigation: ⬅️ [4. What was built](#4-what-was-built) | 📋 [TOC](#table-of-contents) | [Next](#6-what-the-pictures-show) ➡️

| Run | What it was for | What happened |
|---|---|---|
| Smoke, Slurm job 49849, W&B `5isfz35a`, 06:38 | prove the whole path at toy settings: 40 steps, K 2, 512², 4 render steps | clean in 3 minutes: references, three strips, a checkpoint, the artifact, a verdict file. The images are noise at those settings, which is the plumbing test |
| First full launch, job 49850, W&B `cyhd80oz`, 06:42 | the real run | cancelled after 5 minutes: the default render cells included butterfly × flower meadow and barn × pencil drawing, which an animal-instance detector cannot score. Also found that the held-out split repeats six training pairs, so the validation bank now excludes any pair present in train |
| Full, job 49853, W&B `3cwrxlw0`, 06:51 to 07:36 | 100k steps at batch 16 positives plus 16 negatives, renders at K 4, 20 steps, 1024² on wolf × husky and lion × tiger (training pairs) and cat × dog (held out) | finished in 45 minutes at about 37 steps per second. Eleven strips per cell, the final checkpoint, `verdict.json` |

All three on mscluster110, an RTX PRO 6000 Blackwell with 96 GB, under `co3_bw`. Full detail with
the intermediate reads is in [the review file's Runs table](../../../plans/06-is-the-gap-the-samplers-or-the-models/review/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md#runs).

## 6. What the pictures show

Navigation: ⬅️ [5. The runs](#5-the-runs-in-order) | 📋 [TOC](#table-of-contents) | [Next](#7-what-the-numbers-say) ➡️

![Three cells by Mono, PoE and eleven SMC checkpoints](../../results/is-the-gap-the-samplers-or-the-models/twisted-smc-checkpoint-timeline.png)
*One row per cell. Mono and PoE are fixed; each SMC column is the same four noise draws under the
head at that training step, showing the largest-weight particle.*
📊 Drawn in [Figure 1 of the figure explainer](../../results/is-the-gap-the-samplers-or-the-models/figure-explainer.md#figure-1-the-checkpoint-timeline-three-cells-by-eleven-checkpoints).

**What to notice**

Every Mono tile has two animals. Every PoE tile is one animal with parts of both. Every SMC tile
is also one animal, and a different one from the PoE tile beside it. The weights moved the choice
and never reached a two-animal state.

![Mono, PoE and twisted SMC for cat × dog at step 60k](../../results/is-the-gap-the-samplers-or-the-models/twisted-smc-cat-x-dog-step-060000-strip.png)
*The held-out cell at one checkpoint, full size.*
📊 Drawn in [Figure 2 of the figure explainer](../../results/is-the-gap-the-samplers-or-the-models/figure-explainer.md#figure-2-cat--dog-at-step-60k-the-three-panels-at-full-size).

**What to notice**

Cat × dog was never in the twist's training set. The right panel is a black-and-white animal with
a dog's face on a cat's body, chosen after seven resamples from the same noise that gave the
middle panel's grey chimera.

## 7. What the numbers say

Navigation: ⬅️ [6. What the pictures show](#6-what-the-pictures-show) | 📋 [TOC](#table-of-contents) | [Next](#8-what-it-means-and-what-it-does-not) ➡️

![Accuracy against step, and training loss per noise bucket against step](../../results/is-the-gap-the-samplers-or-the-models/twist-training-curves.png)
*Left: accuracy at telling a joint-prompt latent from a PoE latent, training batch in grey and 16
held-out cells in blue, against the 0.60 bar. Right: training loss per noise bucket on a log axis,
with chance at 0.69.*
📊 Drawn in [Figure 3 of the figure explainer](../../results/is-the-gap-the-samplers-or-the-models/figure-explainer.md#figure-3-what-the-twist-head-learned-accuracy-and-loss-by-noise-level).

| Quantity | Value | Source |
|---|---|---|
| Compose fraction at step 100k, over the 3 shown particles and over all 12 particles | Mono 1.0, PoE control 0.0, twisted SMC 0.0 on both reads | `verdict.json`, `all_particles_scores.json` |
| The only non-zero SMC read | 0.33 at step 40k, the cat × dog particle | `history.json`, `eval/compose/smc/heldout/a_cat__x__a_dog/seed_01` |
| Resamples per 20-step run, mean over cells | 5 to 8 at every checkpoint | `history.json`, `eval/n_resample_mean` |
| Validation accuracy | 0.60 at 1k, best 0.66 at 31k, 0.53 at 61k, 0.56 at 100k | `history.json`, `val/acc` |
| Training accuracy | 1.0 from step 30k | `history.json`, `train/acc` |
| Training loss at step 100k, by timestep bucket | under 300: 0.00; 300 to 700: 0.00; 700 and above: 0.03, against 0.69 chance | `history.json`, `train/loss_bucket/*` |

All in `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/twisted_smc/twist_w64_b16_lr1e-04_s100000_20260905-072331/`,
read 2026-09-05, and copied into the sidecar `twisted-smc-report-figures.json` beside the figures.

## 8. What it means, and what it does not

Navigation: ⬅️ [7. What the numbers say](#7-what-the-numbers-say) | 📋 [TOC](#table-of-contents) | [Next](#9-where-everything-is) ➡️

**The verdict**

Inconclusive, by the bar. Validation accuracy ended under 0.60, so the head did not learn a ratio
that carries to unseen pairs, and the sampler was not tested with a trustworthy twist.

**Why the head failed**

It memorised. At timestep 700 and above the two classes are near-identical noise and no
content-based classifier can beat chance there, yet the training loss in that bucket sits twenty
times below chance. The only feature that separates the classes at that noise is which of the 120
training images the sample was made from. So the weights that moved the choice in the pictures
encode pair identity rather than composition.

**What the pictures still say**

With the caveat above, choosing among the product's proposals at K 4 gave a different chimera
every time and never two animals, on a training pair and on a held-out one alike. That is the
direction the model-side reading predicts. It is a null for this twist and not for twisted SMC in
general.

**What was not measured**

K above 4, and any twist that reads composition rather than identity. The three unshown particles
per cell were scored afterwards: 0 of 12 at the final checkpoint, 10 detector hits of 132 at
steps 40k to 80k, each one fused animal by eye. The Mono and control panels use `eta` 1.0, so they compare with each other and not with
the repo's `eta` 0 renders.

## 9. Where everything is

Navigation: ⬅️ [8. What it means](#8-what-it-means-and-what-it-does-not) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Where |
|---|---|
| The verdict, with its evidence pairs | [the finding](../../../report/is-the-gap-the-samplers-or-the-models/does-selecting-among-poe-proposals-compose.md) |
| The pictures read one by one | [the figure explainer](../../results/is-the-gap-the-samplers-or-the-models/figure-explainer.md) |
| The design and the bar | [the plan](../../../plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md) |
| The runs and the pre-registered questions, answered | [the review file](../../../plans/06-is-the-gap-the-samplers-or-the-models/review/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md) |
| The code | `poe_repair/experiments/twisted_smc/` and `scripts/twisted_smc/` |
| The run outputs: references, every particle at every checkpoint, checkpoints, `history.json`, `verdict.json` | `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/twisted_smc/twist_w64_b16_lr1e-04_s100000_20260905-072331/` |
| The W&B run, with strips under `samples/` and artifacts `strips-step<N>` | `prime_lab/poe-repair-animals-compose`, group `twisted-smc`, run `3cwrxlw0` |
| The paper and posts the design borrows from | [source 5 in the context sources](../../../context/sources.md#5-cdm-contrastive-distribution-matching-for-discrete-diffusion-and-the-two-x-posts-that-led-to-it) |
| The tool the first question was about | [source 2](../../../context/sources.md#2-diffusion-explorer-interactive-exploration-of-diffusion-models) |

## Still open

Navigation: ⬅️ [9. Where everything is](#9-where-everything-is) | 📋 [TOC](#table-of-contents)

- [ ] A twist that cannot memorise: positives from every Mono final outside the validation pairs
      plus scorer-labelled composing renders, and a head with a frozen or random-projection front
      end so pair identity is not a cheap feature. Same bars. The design is in
      [the review file's next step](../../../plans/06-is-the-gap-the-samplers-or-the-models/review/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md#next-step).
- [ ] The four-arrow figure from section 1: the state and the A, B, PoE and joint predictions at
      the current step under a time slider, from the cache alone. Owed to the landing-figures plan
      in scope 05.
- [ ] A runbook recipe for launching and harvesting this run.
