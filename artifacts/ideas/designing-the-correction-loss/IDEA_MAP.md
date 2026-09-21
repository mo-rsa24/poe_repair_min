# 💡 Designing the correction adapter's loss and its outputs

## Position in the idea

| Claim | Mark | Settled by |
|---|---|---|
| 1. the unguided PoE prediction is the two experts minus the null once | holds | `poe_repair/_sdxl/metrics.py:16`, the score of the product density divided once by the unconditional |
| 2. with CFG, inference computes `null + w(e1-null) + w(e2-null)` | holds | `poe_repair/experiments/one_pair_one_seed/trainer.py:331-333` and `poe_repair/_sdxl/metrics.py:12`, composed of two `guided_eps` calls minus one raw null |
| 3. the adapter has outputs that can be designed | wrong as stated | the adapter emits three modified branch predictions, not a correction; `trainer.py:443-447` |
| 4. the composition can be trained to match a data score | needs a check: train at w=1, render at w=1 and w=7.5 | the additive floor at step 0 is 0.031 of the correction (seed 1) and 0.028 (seed 2), against a shuffled ceiling of 0.217 and 0.233, so a per-concept adapter can reach 97% of it; `additive_floor_step0.json` |
| 5. the guidance weight is inside the training target, not outside it | holds | inference is `w·T − (w−1)·ε_∅`, so a perturbation the loss cannot see moves the sampled output by (w−1) times its size. The exact fix is implemented as `--null-anchor` in `train_pooled.py:136`, weighting `‖ε_θ(∅) − ε_frozen(∅)‖²`, and defaults to 0.0, meaning off in every run so far |
| 6 | dead, and it does not matter | photographs are reachable but unusable: the accepted set is watermarked stock and product shots, judged by eye. V5 is dropped for V6, whose corpus is the eye-filtered render set that already exists |


Load-bearing: claim 6, now settled. V5 is dropped and V6 takes its place, so the idea no longer
depends on an outside corpus. What it depends on instead is whether the existing eye-filtered render
set is large enough.

Claim 4, the proposal itself: train the three branches so the fixed
product-of-experts arithmetic outputs the true score of images where both concepts co-occur, by
denoising score matching against real data rather than by distilling the model's own joint-prompt
branch.

## Table of contents

- [Position in the idea](#position-in-the-idea)
- [Quick context: where you are](#quick-context-where-you-are)
- [The idea, as it stands](#the-idea-as-it-stands)
- [The claims](#the-claims)
- [What the words are](#what-the-words-are)
- [Held claims](#held-claims)
- [Dead ends](#dead-ends)
- [Checks outstanding](#checks-outstanding)
- [Runs](#runs)
- [Routes](#routes)
- [Sources](#sources)
- [Next step](#next-step)

## Quick context: where you are

Navigation: ⬅️ [Position](#position-in-the-idea) | 📋 [TOC](#table-of-contents) | [Next](#the-idea-as-it-stands) ➡️

**What the idea is**

The adapter that repairs product-of-experts composition is currently trained to make the composed
prediction match the joint-prompt prediction, under one fixed arithmetic. The idea is that both the
loss and what the adapter is asked to produce are design choices, and that choosing them
differently could buy something the current choice does not.

**Where the walk is**

All six claims carry a mark. Claim 6 is the only one still open, on the corpus question.

**What compile would produce today**

Destination A. The objective survived its check, the guidance degeneracy has an exact fix,
and what remains open is where the images come from.

## The idea, as it stands

Navigation: ⬅️ [Quick context](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claims) ➡️

Product-of-experts composition predicts noise as the two single-prompt predictions added with the
unconditional one subtracted once, and classifier-free guidance turns that into the unconditional
prediction plus each expert's guidance term scaled by the guidance weight. A LoRA is trained so that
this composed prediction lands on the joint-prompt prediction instead. Both halves of how that LoRA
is trained, the quantity the loss compares and the arithmetic the adapter's three branch outputs are
fed through, are choices rather than facts, and the idea is to choose them for what the project needs
rather than for the plain match.

## The claims

Navigation: ⬅️ [The idea](#the-idea-as-it-stands) | 📋 [TOC](#table-of-contents) | [Next](#what-the-words-are) ➡️

### 1. ✅ The unguided product-of-experts prediction is the two experts minus the null once

Mark: holds. `poe_repair/_sdxl/metrics.py:16` computes exactly this.

### 2. ✅ With CFG, inference computes the null plus each expert's guidance term scaled by w

Mark: holds. Composed in code as two `guided_eps` calls with one raw null subtracted, at
`poe_repair/experiments/one_pair_one_seed/trainer.py:331-333`.

### 3. ❌ The adapter has outputs that can be designed

Mark: wrong as stated. The adapter is LoRA on the UNet's cross-attention projections, so what it
emits is three modified branch predictions for prompt A, prompt B and the null. The composition
arithmetic downstream is fixed, and the correction is what that arithmetic induces.

### 4. ✅ The composition can be trained to match a data score instead of the joint branch

Mark: needs a check. The proposal replaces a model-derived target (the guided joint-prompt
prediction, which the model gets wrong on some seeds) with a data-derived one: denoising score
matching on real images where both concepts co-occur, with the loss applied to the composed
prediction rather than to any branch. The direction is sound and it sidesteps the known obstruction
that a product of noised marginals is not the noised marginal of the product, because nothing is
asked to hold by algebra; the constrained function class is fitted to the true target instead.

The named check: the loss pins only the combination `ε_1 + ε_2 − ε_∅`, while inference computes
`w(ε_1+ε_2) + (1−2w)ε_∅`, which equals `w·T − (w−1)·ε_∅`. Adding the same perturbation to `ε_1` and
`ε_∅` leaves the loss untouched and moves the guided output by `(w−1)` times that perturbation, so
at w=7.5 an invisible change is amplified 6.5-fold. The check is whether a composition trained at
w=1 still composes when sampled at w=7.5.

### 6. ⚰️ Real co-occurrence images exist per pair, and enough of them to estimate a score

Mark: dead, and the idea does not need it. V5 is dropped in favour of V6.

**The structural objection was false.** The worry was that a pair has co-occurrence photographs
because the two things genuinely turn up together, which is also why the model already composes
them. `a_turtle__x__a_tortoise` is one of the 11 Open Images co-occurring pairs and fails 8 of 8 in
[the fail-rate table](../../results/does-the-fix-reach-unseen-pairs/fail_rate.md);
`an_elephant__x__a_penguin` never co-occurs and also fails 8 of 8. All 18 animal pairs measured
under plain PoE fail, so there is no set of pairs the model already composes.

**Retrieval reaches every pair, and the images are not usable.** Over 6,259,883 DataComp-1B
captions, a 1-in-224 sample, all 18 pairs have candidates (cat×dog 3,035, turtle×tortoise 99,
cheetah×cougar 2), and URLs resolve at 77% to 88%. What comes back is the problem. Filtered by a
per-species GroundingDINO query, 7 of 16 cat×dog passes are real photographs of both animals and
0 of 11 turtle×tortoise passes are. The real ones are watermarked stock plates and studio product
shots. Judged by eye against what V5 would train on, the accepted set was rejected.

**The filter fails on exactly this pool.** Fed images whose caption names one species only, it
wrongly reports both at 0.070 for cat×dog (under the 0.10 bar) and 0.323 for turtle×tortoise (over
it). It cannot separate look-alike species, and the training pool is look-alike species throughout.

**The demonstration that worked cannot be used.** cat×dog appears in no training cell pool, is
marked `reference` in the fail-rate table and is the tracking set's out-pair. It is the held-out
test.

The evidence, including all 276 downloaded images sorted by what the filter accepted, is filed under
[can real photographs train the composition](../../results/can-real-photographs-train-the-composition/README.md).

**What replaces it.** V6, whose corpus is model renders that passed a composition filter. That set
already exists: [cells_v56.json](../../_shared/cross_pair_pool_configs/cells_v56.json) at 72 cells
over 30 pairs and [cells_v57.json](../../_shared/cross_pair_pool_configs/cells_v57.json) at 43 over
29, both selected by eye for showing both concepts, seven of the pairs being objects rather than
animals. Selection by eye is what makes it trustworthy: the detector weakness measured above never
touches it.

### 5. ✅ The guidance weight is inside the training target, not outside it

Mark: holds. Inference reads the null branch a second time at `−(w−1)`, so a perturbation added to
both a branch and the null leaves the loss untouched and moves the sample by 6.5 times its size at
w=7.5. The exact fix is already in the trainer as `--null-anchor`
(`poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py:136`), a weight on
`‖ε_θ(∅) − ε_frozen(∅)‖²` that requires `--branch-prompt-style plain`. It defaults to 0.0, so no run
so far has used it.

## What the words are

Navigation: ⬅️ [The claims](#the-claims) | 📋 [TOC](#table-of-contents) | [Next](#held-claims) ➡️

| My phrase | The field's name | What it means | Confidence |
|---|---|---|---|
| teach it to match the score with the right data | denoising score matching | at noise level σ the regression target for the ε-network is the noise `z` that was added, and its expectation is the true score | confident |
| the product is not the thing we want | the noised product is not the product of the noised | noising is a convolution, and a convolution does not distribute over a product, so composing score estimates is wrong at every σ>0 even with perfect experts | confident on the statement, verify the attribution to Du et al. 2023 |
| two concepts in one image rather than a chimera | co-occurrence versus intersection | the product concentrates on things that are both concepts at once, while the wanted distribution has both concepts present as separate objects | confident, the project's own reframe |
| the two experts added with the null taken off | the score of a product of experts | the log of a product of densities is a sum of logs, so its gradient is a sum of scores | confident |
| attaching CFG to the product | composable diffusion's guided composition | each expert's own guidance term is scaled by the weight and summed on top of the unconditional prediction | confident on the form, verify the attribution |
| the residual between mono and PoE | the interaction term | what a product of independent experts drops | confident, the project's own term |

## Held claims

Navigation: ⬅️ [What the words are](#what-the-words-are) | 📋 [TOC](#table-of-contents) | [Next](#dead-ends) ➡️

| Claim | What is unresolved | What would settle it |
|---|---|---|

## Dead ends

Navigation: ⬅️ [Held claims](#held-claims) | 📋 [TOC](#table-of-contents) | [Next](#checks-outstanding) ➡️

| Claim | The workaround | Why it failed |
|---|---|---|
| 6 | Open Images as the corpus | its 601-class vocabulary covers 83 of 317 pool concepts, and only 11 of 1146 pairs have any co-occurring validation image |
| 6 | caption retrieval filtered by a per-species GroundingDINO query, on the look-alike pool | the filter passes merchandise and misreads one species as two: 0 of 11 turtle×tortoise passes are real animals, and its false-positive rate on single-species photographs is 0.323 against a 0.10 bar |

## Checks outstanding

Navigation: ⬅️ [Dead ends](#dead-ends) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

| Claim | The check | What each outcome means |
|---|---|---|
| 6 (V6) | the data-scaling curve: train V6 at 10, 20, 40 and all 72 eye-filtered cells and read compose rate against cell count | saturates below 72: the existing set is enough and no more filtering is needed. Still rising at 72: the set must grow, and growing it needs an automatic filter, which the probes above show does not yet exist for look-alike pairs |
| 5 | the one-step-off-trajectory probe: take one adapted DDIM step from each cached state, then compare the fit loss at the displaced state against at the cached successor, bar at a factor of 2 written into the script first | within 2: the fit survives its own drift and off-policy shift is not the binding problem. Above 2 anywhere in steps 0 to 25: the cache must be rebuilt with the adapter attached, and more training on the frozen cache cannot fix it |
| 5 | log the unguided residual ‖T − eps_J‖² beside `loss_fit` (one line under no_grad at `trainer.py:450-453`, no extra forward), and record the per-concept factor norms `additive_floor_step0.py` already solves for and discards | the first turns the anchored arm's bracket of [0.0007, 3.9] into an equality and says whether the guided and unguided objectives actually agree; the second gives the size of the distortion each concept branch takes, which is what "the correction is small" has to mean |
| 4 (run) | train the composed prediction against the denoising target `z` on a small co-occurrence set at w=1, then render at w=1 and at w=7.5 from the same seeds | composes at both: the degeneracy is benign in practice. Composes at w=1 only: the null branch drifted and the target must carry the sampling w, as the current trainer's does. Composes at neither: the objective is underdetermined at this capacity |

## Runs

Navigation: ⬅️ [Checks outstanding](#checks-outstanding) | 📋 [TOC](#table-of-contents) | [Next](#routes) ➡️

| # | Anchor | What it executed | State | Finding |
|---|---|---|---|---|
| 7 | claim 6 | `scripts/idea_probes/yield_false_positives.py`, the filter against single-species photographs | done, 6 minutes | false-positive rate 0.070 for cat×dog (under the 0.10 bar) and 0.323 for turtle×tortoise (over it). The detector cannot separate look-alike species |
| 6 | claim 6 | `scripts/idea_probes/cooccurrence_yield.py`, download candidates and filter with a per-species GroundingDINO query | done, 25 minutes | cat×dog 58 of 213 pass (7 of 16 real by eye); turtle×tortoise 11 of 46 pass (0 real by eye); wolf×husky 1 of 17 |
| 5 | claim 6 | `scripts/idea_probes/caption_url_liveness.py`, HEAD requests against matched URLs | done, 2 minutes | 77% to 88% of URLs still serve an image |
| 4 | claim 6 | `scripts/idea_probes/caption_cooccurrence.py`, caption co-occurrence over 6.26M DataComp-1B captions | done, 12 minutes | all 18 failing pairs have candidates, from 447 projected (cheetah×cougar) to 678,767 (cat×dog) |
| 3 | claim 6 | `scripts/idea_probes/additive_transfer_step0.py`, five-fold split over pairs: fit the per-concept functions on 80% of pairs, predict the held-out 20% | done, 3 minutes | held-out residual 0.088 (seed 1) and 0.077 (seed 2) of the correction, against in-sample 0.028 and 0.025 and a shuffled ceiling of 0.217 and 0.233. Per-concept functions transfer: about 91% of the correction on a pair never fitted |
| 2 | claim 6 | `scripts/idea_probes/openimages_coverage.py`, the 601-class match and validation co-occurrence count | done, 2 minutes | 83 of 317 concepts exact, 168 of 1146 pairs both-matched, 11 pairs with any co-occurring image, all of them look-alikes |
| 1 | claim 4 | `scripts/idea_probes/additive_floor_step0.py`, the best per-concept additive fit against the joint prediction at step 0, seeds 1 and 2, 2013 cached cells, no GPU | done, 4 minutes | floor 0.031 and 0.028 of the correction; validator 0.0002 and 0.0001; shuffled ceiling 0.217 and 0.233; restricting to recurring concepts changes nothing (0.032, 0.028) |

## Routes

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#sources) ➡️

| # | Anchor | The ask | Return line | State |
|---|---|---|---|---|
| 4 | claim 6, V6 | turn V6 into an ordered experiment with its falsification criterion written before any run, the first question being whether the 72 eye-filtered cells are enough | bring back: the falsification criterion with its bar and where the bar lives in code, the ordered run list with the data-scaling curve first, and what each outcome does to the scope | emitted |
| 3 | claims 4, 5, 6 | write the comparison plainly: what the repository trains, what the board wrote, what differs, and every route out with its cost and what would kill it | bring back: one side-by-side table of the three objectives, and a routes menu in the order to attempt them | returned, written to [the comparison and routes menu](maths/what-we-train-and-the-routes-out.tex) |
| 2 | claims 4, 5, 6 | reconcile the guided objective the trainer runs at w=7.5 with the unguided objective the board writes; settle whether the unguided-to-mono target is well posed, what the null anchor does to the two, which single-concept protection to add, the off-policy gap, and whether the board's data-target objective is reachable here | bring back: the verdict on the target, the verdict on the anchor, the protection term to add, and one cached-data experiment for the off-policy gap | returned, written to [the guided-and-unguided reconciliation](maths/guided-and-unguided-objectives.tex) |
| 1 | claim 4 | enumerate the flavours of the denoising-score-matching objective (plurality-conditioned base measure, null-only adapter, the "and" connective adapter, the λ(σ) weighting) and say what each can and cannot represent | bring back: one table of variants, what each one's free functions are, which are indistinguishable at step 0 under the additive floor, and which single variant to train first | emitted |

## Sources

Navigation: ⬅️ [Routes](#routes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Source | What it gives the idea | Confidence |
|---|---|---|
| Open Images V7 boxable classes, 601 entries, 12 KB | the annotated vocabulary this pool would have to live inside; ruled out by measurement | confident, downloaded and counted |
| Open Images V5 validation boxes, 37,306 images, 25 MB | per-pair co-occurrence counts; the train split is 2.1 GB and about 45 times larger | confident, downloaded and counted |
| Vincent 2011, denoising score matching | why the regression target is the added noise and its minimiser is the score | confident on the result, verify the citation before it reaches the paper |
| Du et al. 2023, composition with energy-based diffusion and MCMC | the statement that a product of noised marginals is not the noised product | likely, verify the attribution |

## Next step

Navigation: ⬅️ [Sources](#sources) | 📋 [TOC](#table-of-contents)

`compile` produces destination A with V6 as the route. The open question is no longer where images
come from but how many are needed: whether 43 to 72 eye-filtered cells is enough, which the
data-scaling curve answers.
