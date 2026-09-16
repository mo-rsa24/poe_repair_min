# 💡 Designing the correction adapter's loss and its outputs

## Position in the idea

| Claim | Mark | Settled by |
|---|---|---|
| 1. the unguided PoE prediction is the two experts minus the null once | holds | `poe_repair/_sdxl/metrics.py:16`, the score of the product density divided once by the unconditional |
| 2. with CFG, inference computes `null + w(e1-null) + w(e2-null)` | holds | `poe_repair/experiments/one_pair_one_seed/trainer.py:331-333` and `poe_repair/_sdxl/metrics.py:12`, composed of two `guided_eps` calls minus one raw null |
| 3. the adapter has outputs that can be designed | wrong as stated | the adapter emits three modified branch predictions, not a correction; `trainer.py:443-447` |
| 4. the composition can be trained to match a data score | holds, given a guidance fix | the additive floor at step 0 is 0.031 of the correction (seed 1) and 0.028 (seed 2), against a shuffled ceiling of 0.217 and 0.233, so a per-concept adapter can reach 97% of it; `additive_floor_step0.json` |
| 5. the guidance weight is inside the training target, not outside it | holds | inference is `w·T − (w−1)·ε_∅`, so a perturbation the loss cannot see moves the sampled output by (w−1) times its size; three fixes named, the frozen null in the guidance term being exact |
| **6 (current)** | **needs a check** | **Open Images is ruled out by measurement (83 of 317 concepts, 11 of 1146 pairs with any co-occurring image, and those 11 are look-alikes). The named check is whether caption retrieval plus the project's GroundingDINO filter reaches the pairs that actually fail** |


Load-bearing: claim 6. Claim 4 survived its check, so the idea now dies only on data.

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

### 6. 🔍 Real co-occurrence images exist per pair, and enough of them to estimate a score  ← current

Mark: needs a check, with Open Images eliminated. This repo holds no real image data at all:
`data/pilot/` is SDXL output and no dataset loader appears anywhere in `poe_repair/` or
`scripts/`. Against Open Images' 601 boxable classes, 83 of 317 pool concepts match exactly,
30 match only by a variant that is mostly wrong on inspection, and 204 miss. Of 1146 pairs,
168 have both concepts matched and 11 have at least one co-occurring validation image.

Those 11 are the break, not the shortage. They are whale with dolphin, swan with duck, owl
with falcon, turtle with tortoise: same habitat, look alike, and already solved by the
additive fit to twelve decimal places. None of the hard pairs appears. A pair has
co-occurrence photographs because the two things genuinely turn up together, which is also
why the model already composes them, so the data is abundant where it is least needed.

The named check: whether caption retrieval over a web-scale corpus, filtered by this
project's validated instance-count scorer, reaches the pairs that fail. It needs no fixed
class vocabulary, which is what eliminated Open Images.

The reframing that softens the requirement holds, measured. Under the additive model the
correction for a pair is `f_i + f_j`, so each concept needs to appear in some co-occurring
pair rather than in the failing pair specifically. Fitting the per-concept functions on 80%
of pairs and predicting the held-out 20% leaves 0.088 of the correction at seed 1 and 0.077
at seed 2, against a shuffled ceiling of 0.217 and 0.233. So a pair never fitted is still
about 91% corrected from its two concepts alone, and the data requirement is per concept
rather than per pair.

### 5. ⬜ The guidance weight is inside the training target, not outside it

Mark: open.

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

## Checks outstanding

Navigation: ⬅️ [Dead ends](#dead-ends) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

| Claim | The check | What each outcome means |
|---|---|---|
| 6 | how many of the 294 pool concepts appear as annotated classes in Open Images (600 classes with boxes) and how many pairs have at least 50 images containing both | most pairs covered: the objective can be fed from an existing corpus. Few covered: retrieval by caption over a web corpus, with a detector filter. None: the target must be generated and filtered, which reintroduces the model into its own target |
| 5 | the one-step-off-trajectory probe: take one adapted DDIM step from each cached state, then compare the fit loss at the displaced state against at the cached successor, bar at a factor of 2 written into the script first | within 2: the fit survives its own drift and off-policy shift is not the binding problem. Above 2 anywhere in steps 0 to 25: the cache must be rebuilt with the adapter attached, and more training on the frozen cache cannot fix it |
| 5 | log the unguided residual ‖T − eps_J‖² beside `loss_fit` (one line under no_grad at `trainer.py:450-453`, no extra forward), and record the per-concept factor norms `additive_floor_step0.py` already solves for and discards | the first turns the anchored arm's bracket of [0.0007, 3.9] into an equality and says whether the guided and unguided objectives actually agree; the second gives the size of the distortion each concept branch takes, which is what "the correction is small" has to mean |
| 4 (run) | train the composed prediction against the denoising target `z` on a small co-occurrence set at w=1, then render at w=1 and at w=7.5 from the same seeds | composes at both: the degeneracy is benign in practice. Composes at w=1 only: the null branch drifted and the target must carry the sampling w, as the current trainer's does. Composes at neither: the objective is underdetermined at this capacity |

## Runs

Navigation: ⬅️ [Checks outstanding](#checks-outstanding) | 📋 [TOC](#table-of-contents) | [Next](#routes) ➡️

| # | Anchor | What it executed | State | Finding |
|---|---|---|---|---|
| 3 | claim 6 | `scripts/idea_probes/additive_transfer_step0.py`, five-fold split over pairs: fit the per-concept functions on 80% of pairs, predict the held-out 20% | done, 3 minutes | held-out residual 0.088 (seed 1) and 0.077 (seed 2) of the correction, against in-sample 0.028 and 0.025 and a shuffled ceiling of 0.217 and 0.233. Per-concept functions transfer: about 91% of the correction on a pair never fitted |
| 2 | claim 6 | `scripts/idea_probes/openimages_coverage.py`, the 601-class match and validation co-occurrence count | done, 2 minutes | 83 of 317 concepts exact, 168 of 1146 pairs both-matched, 11 pairs with any co-occurring image, all of them look-alikes |
| 1 | claim 4 | `scripts/idea_probes/additive_floor_step0.py`, the best per-concept additive fit against the joint prediction at step 0, seeds 1 and 2, 2013 cached cells, no GPU | done, 4 minutes | floor 0.031 and 0.028 of the correction; validator 0.0002 and 0.0001; shuffled ceiling 0.217 and 0.233; restricting to recurring concepts changes nothing (0.032, 0.028) |

## Routes

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#sources) ➡️

| # | Anchor | The ask | Return line | State |
|---|---|---|---|---|
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

`compile` produces destination A. The one open question is the corpus: caption retrieval with
the project's instance-count filter, sized per concept rather than per pair.
