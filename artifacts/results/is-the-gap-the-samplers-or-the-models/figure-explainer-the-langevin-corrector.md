# The Langevin corrector: what these pictures mean

Twelve pictures from the corrector runs of 2026-09-05 and 06, root steps 25 to 27: a Markov chain
run on the product-of-experts score at every noise level, measured as a curve and rendered as
images, then put on top of the trained adapter twice to see whether it cleans up what the adapter
softens. They are rendered from results rather than from a diagram map, so no prompt sits behind
any of them; the script that drew each is on its provenance line. 7 figure sections · 12 files ·
0 waiting on a render.

The three findings they belong to are
[does a Langevin corrector remove part of the correction](../../../report/is-the-gap-the-samplers-or-the-models/does-a-langevin-corrector-remove-part-of-the-correction.md),
[does a corrector alone produce two animals](../../../report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md)
and [can a corrector or a clean tail sharpen the adapter's renders](../../../report/is-the-gap-the-samplers-or-the-models/can-a-corrector-or-a-clean-tail-sharpen-the-adapters-renders.md).
The twisted-SMC pictures in this same folder have
[their own explainer](figure-explainer.md).

## Table of contents

- [Where this sits](#where-this-sits)
- [The cast](#the-cast)
- [Figure 1: the residual curve, at two step sizes](#figure-1-the-residual-curve-at-two-step-sizes)
- [Figure 2: what each step size does to the picture](#figure-2-what-each-step-size-does-to-the-picture)
- [Figure 3: what more corrector steps do to the picture](#figure-3-what-more-corrector-steps-do-to-the-picture)
- [Figure 4: the corrector in one window at a time](#figure-4-the-corrector-in-one-window-at-a-time)
- [Figure 5: the corrector against plain product-of-experts on eight seeds](#figure-5-the-corrector-against-plain-product-of-experts-on-eight-seeds)
- [Figure 6: the corrector on the adapter's own tail](#figure-6-the-corrector-on-the-adapters-own-tail)
- [Figure 7: the adapter early, the frozen model after](#figure-7-the-adapter-early-the-frozen-model-after)
- [How the figures connect](#how-the-figures-connect)
- [Where this touches the repo](#where-this-touches-the-repo)
- [Still open](#still-open)

## Where this sits

Navigation: 📋 [TOC](#table-of-contents) | [Next](#the-cast) ➡️

**The system drawn.** SDXL sampling at 50 DDIM steps, guidance 7.5, 1024 square, under
product-of-experts. Before each reverse step the latent is nudged `k` times along the
product-of-experts score with a little fresh noise each time, then the ordinary reverse step is
taken from wherever that lands. Everything here is either a measurement taken at that settled
point or the image the run finished with.

**What the pictures were read off.** One output root,
`/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/`, holding the per-cell
measurement files under `curves/`, the three verdicts, the four scored grids, and the renders under
`pairs/`, `window/`, `sheet/`, `tail/` and `clean_tail/`. The runs are logged as W&B run
`prime_lab/poe-repair-animals-compose/s61hldbc`.

**Which plans opened it.** Root steps 25 to 27 of
[the sampler-against-model scope](../../../plans/06-is-the-gap-the-samplers-or-the-models/MASTER_PLAN.md):
[the corrector and the step size it runs at](../../../plans/06-is-the-gap-the-samplers-or-the-models/plans/tools/02-the-corrector-and-the-step-size-it-runs-at.md),
[what is left once the chain settles](../../../plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis/03-what-is-left-once-the-chain-settles.md)
and [does the corrector compose in the same window](../../../plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis/04-does-the-corrector-compose-in-the-same-window.md).

**What the set was meant to achieve.** To split the correction into the part a better sampler could
remove and the part only the model can account for. Summing two diffused scores gives the score of
the product of two diffused marginals, which is not the forward diffusion of anything, so reverse
diffusion on it does not sample what it was asked for; a Markov chain needs no valid forward
process and can sample the product exactly. Whatever survives that at the low-noise end is the
model's. Du et al. and Soiffer et al. bracket the expected answer, and both are quoted in the
scope's direction. The theory behind the chain is built from the floor in
[the sampler-correctors journey](../../../../goal-setting/learning/deep-learning/diffusion-models/sampler-correctors-for-composition/MASTER_PLAN.md),
whose objective 7 is this measurement at toy scale.

## The cast

Navigation: ⬅️ [Where this sits](#where-this-sits) | 📋 [TOC](#table-of-contents) | [Next](#figure-1-the-residual-curve-at-two-step-sizes) ➡️

| Drawn as | It is | Owned by |
|---|---|---|
| the corrector count, `k` | how many Langevin steps run at each of the 50 noise levels before the reverse step; `k = 0` is plain product-of-experts, byte for byte | [Langevin dynamics](../../../../goal-setting/learning/deep-learning/diffusion-models/sampler-correctors-for-composition/plans/21-langevin-dynamics.md) |
| the step-size multiplier, `c` | the chain's one free parameter: the step at noise level `t` is `c` times the scheduler's beta there | the step-size review, [its search table](../../../plans/06-is-the-gap-the-samplers-or-the-models/review/02-the-corrector-and-the-step-size-it-runs-at.md) |
| the ratio on the y axis | the correction's size relative to the prediction it corrects, `‖eps_J − eps_PoE‖ / ‖eps_PoE‖`, evaluated at the settled point | [the interaction term](../../../context/world/interaction-term.md) |
| the read zone, shaded green | the last five denoising steps, where the sampler's share has vanished by construction and what is left is the model's | the chain plan's own section on where the two errors separate |
| the grey band, steps 0 to 10 | the window where the compose rate is decided and where the two errors cannot be told apart | the same section |
| "Mono", the joint prompt column | "a cat and a dog" sampled as one prompt, the target the correction is defined from | [PoE composition](../../../context/world/poe-composition.md) |
| the green or red frame | the validated instance count, two or more animals or fewer | [compose rate](../../../context/world/compose-rate.md) |
| the adapter | the pooled rank-32 LoRA at step 30050, applied at strength 1.2 | [the LoRA corrector](../../../context/world/lora-corrector.md) |
| d(joint) on a tile | that render's cosine distance to its own seed's joint-prompt render, in the compose scorer's DINOv2 embedding; lower is nearer | the clean-tail finding, rung 2 |
| sharp on a tile | Laplacian variance of the greyscale render, higher being more edge energy | the same finding, rung 1 |

## Figure 1: the residual curve, at two step sizes

Navigation: ⬅️ [The cast](#the-cast) | 📋 [TOC](#table-of-contents) | [Next](#figure-2-what-each-step-size-does-to-the-picture) ➡️

[observed] · Explains: `corrector-residual-curves-c3.png` and `corrector-residual-curves-c30.png`, drawn 2026-09-06 by `scripts/corrector_residual_curve.py --plot` · Read off `corrector/residual_curves.json` and the three `verdict_c*.json`

![The correction's size against denoising step, one curve per corrector count, two panels, the two norms beneath](corrector-residual-curves-c3.png)

![The same figure at the step size the search picked](corrector-residual-curves-c30.png)

**What you are looking at**

The measurement the whole scope turns on. Two panels: the pair that blends on the left, the pair
that composes on the right. Six curves each, one per corrector count from 0 to 200. Under them,
the same runs' two norms drawn separately, so a ratio that falls because its denominator grew is
visible rather than hidden. The first file is the step size the scope proceeds at; the second is
the larger one the search rule picked.

**The components**

| In the picture | In the data | Read more |
|---|---|---|
| one curve | one `(pair, c, k)` cell, 50 rows | `corrector/curves/<pair>__seed9__c<c>__k<kkk>.json` |
| the green band at the right | the last five steps, averaged into the read-zone value every bar is judged on | the finding's rung 1 |
| the grey band at the left | steps 0 to 10, where no corrector count separates the two errors | the finding's What this cannot tell you |
| the second row's solid and dashed lines | the numerator and the denominator of the ratio above | the finding's rung 1 |

**The flows**

Left to right is the denoising run, noise to image. Each curve is its own trajectory: two corrector
counts separate at the first noise level and never meet again, so nothing here is one path
measured twice.

**What it teaches**

At the working step size the six curves lie on top of each other through the grey band, fan apart
by a factor of three in the middle of the run, and come back together in the read zone, where the
100-step and 200-step curves cross rather than converge. That crossing is the inconclusive branch
firing: the rule needs them within 5% of each other and they are 21% apart. At the larger step
size the right panel's curves separate upward with count, which is the composing pair's control
failing.

**What it does not show**

Whether the sampler's share is large or small. The bars could not be met at one seed, so no
reading of the fall is licensed. It also shows nothing about the early window by construction.

**Read next**

[Figure 2](#figure-2-what-each-step-size-does-to-the-picture), which is why the larger step size is
not used, and the finding's rung 1 for every number on these axes.

## Figure 2: what each step size does to the picture

Navigation: ⬅️ [Figure 1](#figure-1-the-residual-curve-at-two-step-sizes) | 📋 [TOC](#table-of-contents) | [Next](#figure-3-what-more-corrector-steps-do-to-the-picture) ➡️

[observed] · Explains: `corrector-cat-dog-across-step-sizes.png`, assembled 2026-09-05 from the composer's own renders · Read off `corrector/pairs/a_cat__x__a_dog/seed_9/poe_langevin_k020_c*/`

![Cat and dog, seed 9, at 20 corrector steps, one tile per step-size multiplier from 0.035 to 300](corrector-cat-dog-across-step-sizes.png)

**What you are looking at**

One finished image per step size, all at the same corrector count and the same seed. This is the
step-size search's own rows, rendered. The multiplier 1.0 tile is absent because that search row
ran before the composer started saving images.

**The flows**

Larger step size to the right. The chain moves further per level as it grows, from a median
relative displacement of 0.049 at the smallest to 1.36 at the largest.

**What it teaches**

The picture is the instrument finding. Real photographs up to multiplier 0.3, a different scene
with a leash at 3, paint at 10, and texture noise from 30, which is the value the pre-registered
rule picked as the largest that neither stalls nor diverges. Both numeric guards passed every one
of those tiles: an unadjusted Langevin step contracts toward the score's mean whatever its size
until the step passes 2, so the latent-norm bound cannot trip while the sample is being wrecked,
and the rise-with-count guard reads the joint branch at a point the chain has already left the
data manifold at. The eye sees it and the numbers do not, which is why the scope proceeds at 3.

**What it does not show**

Where between 3 and 30 the picture breaks. The tested multipliers jump by roughly threefold, so
the boundary is bracketed rather than located.

**Read next**

[The step-size search table](../../../plans/06-is-the-gap-the-samplers-or-the-models/review/02-the-corrector-and-the-step-size-it-runs-at.md#the-step-size-search)
for the columns each tile passed.

## Figure 3: what more corrector steps do to the picture

Navigation: ⬅️ [Figure 2](#figure-2-what-each-step-size-does-to-the-picture) | 📋 [TOC](#table-of-contents) | [Next](#figure-4-the-corrector-in-one-window-at-a-time) ➡️

[observed] · Explains: `corrector-butterfly-meadow-across-k-c3.png` and `corrector-cat-dog-across-k-c0p3.png`, assembled 2026-09-05 and 06 · Read off the same per-cell render folders, with each tile's read-zone ratio taken from `corrector/curves/`

![Butterfly and flower meadow across corrector counts at the working step size](corrector-butterfly-meadow-across-k-c3.png)

![Cat and dog across corrector counts at the smaller step size](corrector-cat-dog-across-k-c0p3.png)

**What you are looking at**

The same seed at one step size, one tile per corrector count from 0 to 200, with the tile's
read-zone ratio printed above it. The first is the pair that composes by default, the second the
pair that blends.

**What it teaches**

Both pairs keep a coherent photograph through 20 corrector steps and drift toward an illustration
by 100, with the latent norm never above 1.02 times its start. So the chain does not blow up; it
settles somewhere whose typical image is a different kind of picture. On cat × dog the count never
becomes two animals at any step. This is the one observation every step size and both pairs agree
on, and no bar in the scope was written for it.

**What it does not show**

Why the drift happens. That the product of two diffused marginals has a more illustrated typical
sample than the joint prompt's is what the pictures suggest and nothing here measures.

**Read next**

The finding's rung 3, which reads the composing pair's rising ratio against these tiles, and
[Figure 5](#figure-5-the-corrector-against-plain-product-of-experts-on-eight-seeds) for the same
drift over eight seeds.

## Figure 4: the corrector in one window at a time

Navigation: ⬅️ [Figure 3](#figure-3-what-more-corrector-steps-do-to-the-picture) | 📋 [TOC](#table-of-contents) | [Next](#figure-5-the-corrector-against-plain-product-of-experts-on-eight-seeds) ➡️

[observed] · Explains: `corrector-window-slides-cat-dog.png`, rendered and drawn 2026-09-06 by `scripts/corrector_window_sweep.py` · Read off `corrector/window_curves_mcmc.json`

![Four seeds by ten columns: the corrector switched on inside one ten-step window, then on all 50 steps](corrector-window-slides-cat-dog.png)

**What you are looking at**

Rows are seeds 9 to 12 of cat × dog; columns are the nine ten-step windows in which the corrector
is allowed to act, then a tenth with it on for all 50 steps. The green bar under each column draws
where the window sits on the run. The frame is the detector's verdict.

**The components**

| In the picture | In the data | Read more |
|---|---|---|
| the frame colour | `cells[].compose`, the validated instance count reaching two | [compose rate](../../../context/world/compose-rate.md) |
| the green bar | `cells[].window`, a half-open range of step indices | the plan's own window definition |
| the layout itself | matched to `paper/iclr/figures/when-the-correction-arrives/poe/samples-as-a-ten-step-window-slides.png` on pair, seeds, positions and border rule | the finding's rung 1 |

**What it teaches**

No frame is green. The early columns change the drawing style, into a cartoon on seed 9 and a line
drawing on seed 11, and from the fifth column on the picture barely changes as the window slides,
which is the same late-window inertness the injected correction showed. The figure it is matched
to has its peak at the leftmost column, 0.656 of its runs. Same layout, same seeds, different
mechanism, and only one of the two has a peak.

**What it does not show**

Anything at a larger corrector count. It ran at 20 steps per level, which is a compute budget
rather than a property of the problem, and the residual curve licensed no flat part to choose it
from.

**Read next**

The finding's rung 1, and the timing verdict it is compared against.

## Figure 5: the corrector against plain product-of-experts on eight seeds

Navigation: ⬅️ [Figure 4](#figure-4-the-corrector-in-one-window-at-a-time) | 📋 [TOC](#table-of-contents) | [Next](#figure-6-the-corrector-on-the-adapters-own-tail) ➡️

[observed] · Explains: `corrector-cat-dog-eight-seed-sheet.png` and `corrector-butterfly-meadow-eight-seed-sheet.png`, rendered 2026-09-06 on mscluster108 · Read off `corrector/sheet_scores.json`

![Eight seeds by three columns: the joint prompt, plain product-of-experts, and the corrector on all 50 steps](corrector-cat-dog-eight-seed-sheet.png)

![The same three columns for the pair that already composes](corrector-butterfly-meadow-eight-seed-sheet.png)

**What you are looking at**

The read-out every parallel session in this thread reports on: the held-out seeds 9 to 16, the
joint prompt first, plain product-of-experts second, and the session's own solution third. Every
render starts from the same cached initial noise for its seed.

**What it teaches**

On cat × dog the first column is two animals in every row and the second and third are one animal
in every row, differing from each other in style rather than in count: the corrector alone scores
exactly what plain product-of-experts scores. On the control pair every tile keeps its butterfly,
so the corrector does not break what already works, and the detector's confidence in the butterfly
falls on three seeds while the frame stays green, which is the style drift read as a number.

**What it does not show**

The control pair's column is not the validated rule. A flower meadow is not an animal, so its read
is a per-concept detection and is marked unvalidated wherever it is used.

**Read next**

[Figure 6](#figure-6-the-corrector-on-the-adapters-own-tail), which puts the corrector on top of
something that does compose.

## Figure 6: the corrector on the adapter's own tail

Navigation: ⬅️ [Figure 5](#figure-5-the-corrector-against-plain-product-of-experts-on-eight-seeds) | 📋 [TOC](#table-of-contents) | [Next](#figure-7-the-adapter-early-the-frozen-model-after) ➡️

[observed] · Explains: `corrector-on-adapter-tail-cat-dog-eight-seed-sheet.png` and `corrector-on-adapter-tail-butterfly-meadow-eight-seed-sheet.png`, rendered 2026-09-06 on mscluster85 · Read off `corrector/tail_fidelity.json`

![Eight seeds by five columns: the joint prompt, plain PoE, the adapter alone, and the adapter plus 5 and 20 corrector steps on its last fifteen](corrector-on-adapter-tail-cat-dog-eight-seed-sheet.png)

![The same five columns for the control pair](corrector-on-adapter-tail-butterfly-meadow-eight-seed-sheet.png)

**What you are looking at**

The adapter is now doing the composing, and the corrector is asked to clean up after it. Columns
four and five add 5 and 20 Langevin steps per level on the adapter's own corrected prediction,
inside steps 35 to 49 only. Each tile carries its instance count and its Laplacian variance.

**What it teaches**

The last three columns are the same picture with fur and edges redrawn, and none reaches the
crispness of the first column. Mean sharpness moves 57.1 to 61.7, a rise of 8% inside the 10%
band, so the null branch fires, and per seed the change goes both ways. The control pair's
sharpness rises much further, 506 to 726, which is the meadow's fine texture returning as the
chain settles.

**What it does not show**

Whether a different checkpoint's tail would behave differently. One checkpoint, one strength, one
step size.

**Read next**

[Figure 7](#figure-7-the-adapter-early-the-frozen-model-after), which changes what the tail is
rather than what runs on it.

## Figure 7: the adapter early, the frozen model after

Navigation: ⬅️ [Figure 6](#figure-6-the-corrector-on-the-adapters-own-tail) | 📋 [TOC](#table-of-contents) | [Next](#how-the-figures-connect) ➡️

[observed] · Explains: `corrector-clean-tail-cat-dog-eight-seed-sheet.png` and `corrector-clean-tail-butterfly-meadow-eight-seed-sheet.png`, rendered 2026-09-06 on mscluster85 · Read off `corrector/clean_tail.json`

![Eight seeds by nine columns: the joint prompt, plain PoE, the adapter alone, then the adapter switched off after step 19 or 29 with 0, 5 or 20 corrector steps on the frozen score](corrector-clean-tail-cat-dog-eight-seed-sheet.png)

![The same nine columns for the control pair](corrector-clean-tail-butterfly-meadow-eight-seed-sheet.png)

**What you are looking at**

Six conditions beside the three references. The adapter sets the composition on the early steps
only, where the timing result says the outcome is decided, and then the frozen model's own
product-of-experts step draws the rest. The corrector, where it acts, now runs on the frozen
score. Each tile carries its count, its distance to that seed's joint-prompt render, and its
sharpness.

**What it teaches**

The composition survives the hand-off on every row. The sixth column, the adapter switched off
after step 29 with no corrector, is the crispest of the adapter columns on seeds 9, 12, 13 and 16
and reads closest to the first column, moving the 8-seed mean distance from 0.472 to 0.434 against
a 0.05 bar. The corrector columns beside it move back away and add grain. So the hand-off is the
lever that points the right way and the chain is not.

**What it does not show**

Whether a strength schedule across the hand-off closes the remaining 0.012. That is the next lever
and it lives in another scope's plan.

**Read next**

The clean-tail finding's rung 2 for every number, and
[correct early, then clean up](../../../plans/01-showcase-the-trained-lora/plans/experiments/14-correct-early-then-clean-up.md)
for the lever that follows.

## How the figures connect

Navigation: ⬅️ [Figure 7](#figure-7-the-adapter-early-the-frozen-model-after) | 📋 [TOC](#table-of-contents) | [Next](#where-this-touches-the-repo) ➡️

Figures 1 to 3 are one argument about the instrument: the curve is what was measured, Figure 2 is
why its step size is not the one the rule picked, and Figure 3 is what the chain does to a picture
as it settles. Figure 4 asks the behavioural question the curve cannot answer, and Figure 5 puts
the same corrector on the seeds every session reports on. Figures 6 and 7 change the question from
composition to fidelity: 6 keeps the adapter's own score and 7 hands the tail to the frozen model.

Read 1, 2, 3 for whether the measurement can be believed, then 4 and 5 for what the corrector does,
then 6 and 7 for what it is worth on top of something that already works. A reader who only wants
the verdicts stops after Figures 1, 4 and 7.

## Where this touches the repo

Navigation: ⬅️ [How the figures connect](#how-the-figures-connect) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| Figure | Points at | Why | Backlink |
|---|---|---|---|
| 1, 2, 3 | [the residual-curve finding](../../../report/is-the-gap-the-samplers-or-the-models/does-a-langevin-corrector-remove-part-of-the-correction.md) | the numbers and the three verdicts | ✅ written |
| 4, 5 | [the compose finding](../../../report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md) | the per-column and per-seed counts | ✅ written |
| 6, 7 | [the fidelity finding](../../../report/is-the-gap-the-samplers-or-the-models/can-a-corrector-or-a-clean-tail-sharpen-the-adapters-renders.md) | the sharpness and distance numbers with their bars | ✅ written |
| 2 | [the step-size review](../../../plans/06-is-the-gap-the-samplers-or-the-models/review/02-the-corrector-and-the-step-size-it-runs-at.md) | the search table each tile is a row of | ✅ written |
| 1 | [the chain review](../../../plans/06-is-the-gap-the-samplers-or-the-models/review/03-what-is-left-once-the-chain-settles.md) | the three-way threshold and which branch fired | ✅ written |
| 4, 5, 6, 7 | [the window review](../../../plans/06-is-the-gap-the-samplers-or-the-models/review/04-does-the-corrector-compose-in-the-same-window.md) | the questions written before these ran | ✅ written |
| 4 | [the timing verdict](../../../plans/03-does-the-correction-cause-composition/review/05-when-in-the-run-it-matters.md) | the injected-correction figure this one is matched to | ✅ written |
| all | [running the Langevin corrector](../../../runbook/running-things-on-the-cluster/running-the-langevin-corrector.md) | the commands that regenerate every one of them | ✅ written |
| cast | [the interaction term](../../../context/world/interaction-term.md), [PoE composition](../../../context/world/poe-composition.md), [compose rate](../../../context/world/compose-rate.md), [the LoRA corrector](../../../context/world/lora-corrector.md) | what each drawn quantity is | ✅ written |
| Where this sits | [the sampler-correctors journey](../../../../goal-setting/learning/deep-learning/diffusion-models/sampler-correctors-for-composition/MASTER_PLAN.md) | the theory these runs are the full-scale version of | ✅ written |
| all | `scripts/corrector_residual_curve.py`, `scripts/corrector_window_sweep.py` | the scripts that drew them | n/a |

## Still open

Navigation: ⬅️ [Where this touches the repo](#where-this-touches-the-repo) | 📋 [TOC](#table-of-contents)

- Figure 1 at seeds 10 to 12, so the curve is a mean over four trajectories rather than one. Until
  then no reading of its fall is licensed, and this is the single thing that would change that.
- A picture for the photograph-to-illustration drift as a measurement rather than an observation.
  Figures 3 and 5 both show it and nothing quantifies it.
- This folder now holds four findings' figures and two explainers, this one and
  [the twisted-SMC one](figure-explainer.md), while the SuperDiff and noise-search figures beside
  them have none. Whether the folder splits by question or the explainers merge is a filing
  question for a tidy pass, not for either explainer to decide.
