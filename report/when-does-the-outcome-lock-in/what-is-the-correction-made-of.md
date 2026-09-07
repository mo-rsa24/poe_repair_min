# What is the correction made of, and could PoE have supplied it by re-weighting?   ❓ inconclusive on the pre-registered bar · verified 2026-09-05

**The claim**

Over the first ten steps of the plain product-of-experts run on cat × dog, 0.37 of the
correction's energy lies outside anything a re-weighting of the two experts could reach. The
bar asked for above 0.5 or below 0.25, so the pre-registered question is inconclusive.

What the reachable part is made of is the result. The joint prompt's prediction, written in the
experts' own directions, weights each expert at 1 to 3 where PoE weights each at 7.5, and the two
experts pull against each other from step 5 on. So most of what re-weighting could supply is
"turn both experts down", and the rest is a direction of its own that peaks at step 5.

The rank-32 adapter reproduces the re-weightable part almost exactly and about half of the new
direction, with less of its own energy in that direction than the target has.

From step 10 on, runs on the product keep moving about three and a half times as much as the
run on the joint prompt, corrected or not, and five of eight plain runs change animal after
step 10. These three reads carry no bar and are reported as reads.

## Table of contents

- [Where the idea came from](#where-the-idea-came-from)
- [What we set out to see](#what-we-set-out-to-see)
- [What would have counted](#what-would-have-counted)
- [What was tried, in order](#what-was-tried-in-order)
- [1. A third of the early correction is out of the experts' reach, and the rest is mostly damping](#1-a-third-of-the-early-correction-is-out-of-the-experts-reach-and-the-rest-is-mostly-damping)
- [2. The two experts draw different animals in the same place](#2-the-two-experts-draw-different-animals-in-the-same-place)
- [3. The adapter learned the damping and half of the new direction](#3-the-adapter-learned-the-damping-and-half-of-the-new-direction)
- [4. Runs on the product keep moving, and the plain one keeps changing animal](#4-runs-on-the-product-keep-moving-and-the-plain-one-keeps-changing-animal)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Where this came from](#where-this-came-from)
- [Depends on](#depends-on)
- [Still open](#still-open)

## Where the idea came from

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-we-set-out-to-see) ➡️

**The split.** Claim 2 of [which variable explains what PoE is missing](../../artifacts/ideas/which-variable-explains-what-poe-is-missing/IDEA_MAP.md)
records, after Du et al., that the correction is a sampler error and a model error added
together and that nothing in this repository separates them.

**The parallel sessions.** Three sessions are testing fixes that select among or re-weight the
product's own proposals. Whether any of them can work depends on how much of the correction
lies inside what the product already spans, which nothing had measured.

**The connection to this project.** The mechanism claims so far are the correction's size over
the run and its cosine with itself across seeds. This finding says what it is made of relative
to the predictions PoE already has, shows where the experts disagree in the picture, and reads
which part the adapter learned.

## What we set out to see

Navigation: ⬅️ [Previous](#where-the-idea-came-from) | 📋 [TOC](#table-of-contents) | [Next](#what-would-have-counted) ➡️

Four reads of one cached run, cat × dog, held-out seeds 9 to 16, 50 DDIM steps at guidance 7.5.
At every step the cache holds the state and the four raw predictions (cat, dog, joint prompt,
empty prompt) made at it, so the guided predictions, the PoE prediction and the correction are
closed forms; the rule is written in
[the plan](../../plans/05-when-does-the-outcome-lock-in/plans/tests/07-what-the-correction-is-made-of.md#the-rule-the-cache-used).
Every quantity is projected onto the span of {ε_∅, ε_a − ε_∅, ε_b − ε_∅}, which is exactly the set
of predictions a per-step re-weighting of the two experts can produce. The adapter is evaluated at
the same states with and without the adapter in one process. The tracks are the frames already
decoded for [where each condition lands](where-does-each-condition-land.md), embedded with the
scorer's DINOv2 encoder. Nothing was rendered.

**The hypothesis.** Most of the correction in steps 0 to 10 points outside the span, so no
re-weighting could have supplied it, and the adapter's output points there too.

## What would have counted

Navigation: ⬅️ [Previous](#what-we-set-out-to-see) | 📋 [TOC](#table-of-contents) | [Next](#what-was-tried-in-order) ➡️

The plan pre-registered one statistic: the mean over seeds of each seed's mean, over steps 0 to
10, of the orthogonal share of the correction's squared norm. Support above
`ORTHO_SHARE_REWEIGHT_IMPOSSIBLE = 0.5`, null below `ORTHO_SHARE_REWEIGHT_CANDIDATE = 0.25`,
inconclusive between, with the constants in `scripts/showcase/correction_span_common.py`. It is
pinned in the **What has to pass before this runs** section of
[the plan](../../plans/05-when-does-the-outcome-lock-in/plans/tests/07-what-the-correction-is-made-of.md#what-has-to-pass-before-this-runs),
written before the script ran.

**The statistic came out at 0.374, between the two constants: inconclusive.** Rungs 2, 3 and 4
below had questions written before the run and no bar; they are reads.

## What was tried, in order

Navigation: ⬅️ [Previous](#what-would-have-counted) | 📋 [TOC](#table-of-contents) | [Next](#1-a-third-of-the-early-correction-is-out-of-the-experts-reach-and-the-rest-is-mostly-damping) ➡️

| When (2026-09-05) | What | Where it landed | Outcome |
|---|---|---|---|
| 16:30 | rung 2, the projection over 400 cached steps, CPU on mscluster85 | `artifacts/results/what-the-correction-is-made-of/orthogonal-share-over-steps.{png,json}` | 9 seconds; verdict inconclusive |
| 16:35 | rung 4, the tracks from the stored frame embeddings, CPU | `.../track-kinetic-energy.png`, `.../which-animal-over-steps.png`, `.../track-energy-and-which-animal.json` | first figure drew energy against a floor thirty times smaller, unreadable; redrawn per condition with the floor as a tick, and a from-step-10 panel added |
| 16:45 | rungs 1 and 3 chained with `nohup`, first launch | nothing | the foreground CUDA check stalled the launching shell past its timeout; the chain script was never written |
| 16:50 to 16:51 | rung 1, seed 15 decoded on mscluster85 device 0 (RTX 3090, shared with another session's process at 7.5 GB), PID 257033 | `/datasets/mmolefe/poe_repair_min/outputs/showcase/what_the_correction_is_made_of/tweedie/seed_15/`, `.../seed-15-experts-tweedie-strip.png`, `.../same-place-share.json` | 35 decodes in 51 seconds |
| 16:51 to 17:01 | rung 3, the adapter at 400 cached states, same device, PID 257603 | `.../adapter-span-share-over-steps.png`, `.../adapter-span-share.json`, `.../seed-15-adapter-vs-correction-norm-maps.png` | sanity cosine live against cached 0.9996 |
| 17:02 | W&B log | run `yb933cr6` in `prime_lab/poe-repair-animals-compose` | six images, one artifact of four sidecars |

## 1. A third of the early correction is out of the experts' reach, and the rest is mostly damping

Navigation: ⬅️ [Previous](#what-was-tried-in-order) | 📋 [TOC](#table-of-contents) | [Next](#2-the-two-experts-draw-different-animals-in-the-same-place) ➡️

![Orthogonal share of the correction's squared norm against denoising step, mean and min-to-max band over seeds 9 to 16, with the joint direction's share and the norms below](../../artifacts/results/what-the-correction-is-made-of/orthogonal-share-over-steps.png)
*What to notice: the pink line sits between the two bars through the shaded window, peaks at
step 5, and drops under 0.1 by step 20; the purple line in the middle panel climbs to 0.7 and
stays; in the bottom panel the correction's norm grows sixfold while its orthogonal part grows
threefold.*
📊 Drawn in [Figure 1 of the figure explainer](../../artifacts/results/what-the-correction-is-made-of/figure-explainer.md#figure-1-the-orthogonal-share-of-the-correction-over-the-run).

**The number.** Orthogonal share of ‖r_t‖², the fraction of the correction's squared norm outside
the span, mean over seeds 9 to 16 of the per-seed mean over steps 0 to 10: 0.374, per seed 0.16
(seed 9) to 0.65 (seed 16). Per step, mean over seeds: 0.30 at step 0, 0.51 at step 5, 0.29 at
step 10, 0.09 at step 20, 0.08 at step 30, 0.27 at step 49. The joint guidance direction's own
orthogonal share: 0.31 at step 0, 0.64 at step 10, 0.71 at step 30. The weight the joint prompt's
guided prediction puts on the cat expert's direction inside the span, against PoE's 7.5: 2.5 at
step 0, 3.3 at step 5, 2.9 at step 10, 1.3 at step 30; on the dog expert's direction 5.3, 3.8,
1.1, 0.9 at the same steps. Cosine between the two experts' directions ε_a − ε_∅ and ε_b − ε_∅:
0.67 at step 0, −0.13 at step 5, −0.40 at step 10, −0.59 at step 40. Norms, mean over seeds:
‖r_t‖ 11 at step 0, 29 at step 10, 71 at step 30; its orthogonal part 4.8, 12.7, 17.1; ‖ε_PoE‖
about 256 throughout. The PoE prediction rebuilt from the raw vectors projects with in-span share
1.0000 on every row. From `orthogonal-share-over-steps.json`, fields
`summary.early_window_orthogonal_share_mean_over_seeds`, `summary.*_mean_per_step`,
`summary.eps_poe_in_span_share_min_sanity`, `summary.verdict`, written by
`scripts/showcase/correction_span_share.py` under [the plan](../../plans/05-when-does-the-outcome-lock-in/plans/tests/07-what-the-correction-is-made-of.md).

## 2. The two experts draw different animals in the same place

Navigation: ⬅️ [Previous](#1-a-third-of-the-early-correction-is-out-of-the-experts-reach-and-the-rest-is-mostly-damping) | 📋 [TOC](#table-of-contents) | [Next](#3-the-adapter-learned-the-damping-and-half-of-the-new-direction) ➡️

![Seed 15: the decoded running estimate under the cat expert, the dog expert, PoE, the joint prompt and the empty prompt at the same cached state, seven steps, with the experts' per-position disagreement map](../../artifacts/results/what-the-correction-is-made-of/seed-15-experts-tweedie-strip.png)
*What to notice: at steps 5 and 10 the first row is a cat and the second a dog with the head in
the same place; the third row is a cat with a dog's muzzle at step 10 and a dog at step 30; the
fifth row, the empty prompt, is already a dog at step 20; the heat map is brightest on the eyes,
muzzle and ears.*
📊 Drawn in [Figure 2 of the figure explainer](../../artifacts/results/what-the-correction-is-made-of/figure-explainer.md#figure-2-seed-15-what-each-prediction-thinks-the-picture-is).

**The number.** Same-place share, the fraction of the 16,384 latent positions where both experts'
departure from the unconditional estimate is above that expert's own median, chance 0.25, mean
over the 8 seeds: 0.33 at step 0, 0.35 at step 5, 0.40 at step 10, 0.43 at step 30, 0.36 at step
49. Mean cosine between the two experts' departures on those shared positions: +0.77 at step 0,
−0.16 at step 5, −0.41 at step 10, −0.43 at step 30, +0.44 at step 49; the share of shared
positions where that cosine is negative is 0.62 at step 5 and 0.72 to 0.79 from step 10 to 40.
Seed 15 alone: same-place share 0.36 to 0.45 over steps 5 to 30, cosine −0.27 at step 5 and −0.55
at step 30. From `same-place-share.json`, fields `summary.same_place_share_mean_per_step`,
`summary.cos_on_shared_positions_mean_per_step`,
`summary.cos_on_shared_positions_share_negative_per_step` and `rows`, written by
`scripts/showcase/correction_tweedie_pictures.py`.

## 3. The adapter learned the damping and half of the new direction

Navigation: ⬅️ [Previous](#2-the-two-experts-draw-different-animals-in-the-same-place) | 📋 [TOC](#table-of-contents) | [Next](#4-runs-on-the-product-keep-moving-and-the-plain-one-keeps-changing-animal) ➡️

![The adapter's orthogonal share beside the correction's, and the cosines between their orthogonal parts, in-span parts and whole vectors, against step](../../artifacts/results/what-the-correction-is-made-of/adapter-span-share-over-steps.png)
*What to notice: the blue line runs under the pink one through the shaded window and joins it
after step 15; the purple cosine never leaves 0.9; the dark red cosine starts above 0.6 and
settles near 0.35.*
📊 Drawn in [Figure 3 of the figure explainer](../../artifacts/results/what-the-correction-is-made-of/figure-explainer.md#figure-3-the-adapters-output-projected-the-same-way).

![Seed 15: per-position norm of the adapter's output above the correction's at seven steps](../../artifacts/results/what-the-correction-is-made-of/seed-15-adapter-vs-correction-norm-maps.png)
*What to notice: both rows light the eyes, muzzle and ears at steps 20 and 30 and are near-dark
before step 10.*
📊 Drawn in [Figure 4 of the figure explainer](../../artifacts/results/what-the-correction-is-made-of/figure-explainer.md#figure-4-seed-15-where-the-adapter-and-the-correction-act).

**The number.** Orthogonal share of the adapter's output at λ 1, mean over seeds and steps 0 to
10: 0.194, against 0.374 for the correction at the same states; per step 0.16 at step 0, 0.22 at
step 5, 0.21 at step 10, 0.10 at step 20, 0.43 at step 49. Cosine between the orthogonal parts:
0.61 at step 0, 0.67 at step 2, 0.50 at step 10, 0.34 at step 20, 0.22 at step 49; early-window
mean 0.55, all-step mean 0.40. Cosine between the in-span parts 0.91 to 0.99 at every step.
Cosine between the whole vectors 0.82 in the early window, 0.87 over all steps. Norm ratio,
adapter over correction: 1.04 at step 0, 0.68 at step 10, 0.66 at step 30. Seed 15 per-column
cosine 0.96, 0.96, 0.85, 0.78, 0.96, 0.94, 0.70 at steps 0, 2, 5, 10, 20, 30, 49. Sanity: cosine
between the adapter-off PoE prediction computed live and the cached one, minimum over 400 rows,
0.9996. From `adapter-span-share.json`, fields `summary.early_window`, `summary.all_steps`,
`summary.*_mean_per_step`, `summary.sanity_cos_live_vs_cached_poe_min` and `rows`, written by
`scripts/showcase/correction_adapter_span_share.py` on mscluster85 device 0, PID 257603.

## 4. Runs on the product keep moving, and the plain one keeps changing animal

Navigation: ⬅️ [Previous](#3-the-adapter-learned-the-damping-and-half-of-the-new-direction) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

![Kinetic energy of each run's embedded running estimate, one point per run by condition, with the floor as a tick under each; all segments on the left, from step 10 on the right](../../artifacts/results/what-the-correction-is-made-of/track-kinetic-energy.png)
*What to notice: on the right panel the three left-hand conditions sit under 1 and the three
product-based conditions sit near 3, with every grey tick near 0.1.*
📊 Drawn in [Figure 5 of the figure explainer](../../artifacts/results/what-the-correction-is-made-of/figure-explainer.md#figure-5-how-far-each-runs-running-estimate-travels).

![Which-animal score of each run's running estimate against saved step, thin line per seed, thick mean per condition](../../artifacts/results/what-the-correction-is-made-of/which-animal-over-steps.png)
*What to notice: orange and green separate by step 15 and hold; the black thin lines cross zero
to the end; seed 15's dashed black line is above +0.2 at step 20 and below −0.2 from step 35.*
📊 Drawn in [Figure 6 of the figure explainer](../../artifacts/results/what-the-correction-is-made-of/figure-explainer.md#figure-6-which-animal-each-run-reads-as-step-by-step).

**The number.** Kinetic energy, the sum over the 8 saved segments from step 10 to 50 of the
squared displacement of the running estimate's L2-normalised DINOv2 embedding, median over
seeds: cat alone 0.53, dog alone 0.62, joint prompt 0.78, PoE 2.79, PoE plus λ 1.0 correction
3.13, PoE plus λ 1.2 correction 2.71; the floor (a straight track at constant speed) 0.06 to
0.15 for every condition. Over all 13 segments: 3.15, 3.22, 3.88, 5.65, 5.96, 5.37. Runs with a
sign change of the which-animal score after step 10 where both sides exceed 0.1 in magnitude
(`FLIP_MIN_MAGNITUDE = 0.1` in source): PoE 5 of 8, λ 1.0 2 of 8, λ 1.2 2 of 8, joint prompt 0,
single prompts 0; with any sign change counted, 7, 6, 7, 2, 0 and 1. Seed 15's PoE score: +0.21
at step 20, −0.06 at 25, +0.01 at 30, −0.18 at 35, −0.30 at 40, −0.22 at 50, so its crossing
passes through two near-zero frames and counts under the loose rule only. From
`track-energy-and-which-animal.json`, fields `summary.<condition>.kinetic_energy_from_step_10_median`,
`summary.<condition>.runs_with_a_real_flip_after_step_10` and `rows`, written by
`scripts/showcase/correction_track_energy.py` from
`artifacts/results/where-does-each-condition-land/frames-dino-feats.npz`.

## What this cannot tell you

Navigation: ⬅️ [Previous](#4-runs-on-the-product-keep-moving-and-the-plain-one-keeps-changing-animal) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**The span is three vectors at one state.** It bounds a per-step re-weighting of the two
experts. A method that changes the state (a corrector, a search over the noise) or the schedule
is not bounded by it.

**Every read is off-policy.** The states are the plain PoE run's, where the joint prompt never
steered. The joint prompt's row in rung 2 is what it predicts at a state it did not produce,
which is why it draws one animal there.

**The orthogonal part is not split into sampler error and model error.** That is the sampler-share
measurement in [is the gap the samplers or the models](../../plans/06-is-the-gap-the-samplers-or-the-models/MASTER_PLAN.md).

**The same-place share uses each expert's own median as its threshold**, chosen at design time
so chance is 0.25 by construction; a different threshold moves the number.

**The kinetic energy is read off 14 unevenly spaced frames** and measures motion in an
embedding, not correctness. It separates runs on the joint prompt from runs on the product; it
does not separate composing from not, since the corrected runs move as much as the plain ones.

**One pair, eight seeds, one adapter at one checkpoint and one λ.**

## Where this came from

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#depends-on) ➡️

| What | Source | Mark |
|---|---|---|
| Orthogonal shares, coefficients, norms, cosines, the verdict | `artifacts/results/what-the-correction-is-made-of/orthogonal-share-over-steps.json`, read 2026-09-05 | verified |
| Same-place shares and cosines | `.../same-place-share.json`, read 2026-09-05 | verified |
| The adapter's shares and cosines, the sanity cosine | `.../adapter-span-share.json`, read 2026-09-05 | verified |
| Kinetic energies, flips, which-animal scores | `.../track-energy-and-which-animal.json`, read 2026-09-05 | verified |
| The decoded estimates | `/datasets/mmolefe/poe_repair_min/outputs/showcase/what_the_correction_is_made_of/tweedie/seed_15/`, 35 PNGs, rendered 2026-09-05 on mscluster85 | verified |
| The cache | `/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache/heldout/a_cat__x__a_dog/seed_<9..16>/residuals/`, 50 steps each, the rule read off `scripts/build_training_cache.py` | verified |
| The adapter | rank 32, alpha 32, step 30050, `/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/checkpoints/lora_step_030050.pt` | verified |
| The frames' embeddings | `artifacts/results/where-does-each-condition-land/frames-dino-feats.npz` | verified |
| The W&B run | `yb933cr6` in `prime_lab/poe-repair-animals-compose` | verified |
| The run | tasks **Write `correction_span_common.py`**, **Write `correction_tweedie_pictures.py`**, **Write `correction_adapter_span_share.py`** and **Write `correction_track_energy.py`** in [the plan](../../plans/05-when-does-the-outcome-lock-in/plans/tests/07-what-the-correction-is-made-of.md); verdict in [its review file](../../plans/05-when-does-the-outcome-lock-in/review/07-what-the-correction-is-made-of.md) | verified |
| Regenerate with | the four scripts under `scripts/showcase/` named in [the card](../../artifacts/results/what-the-correction-is-made-of/README.md), rungs 1 and 3 on a device with CUDA confirmed; then `correction_log_wandb.py` | |

## Depends on

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- What the correction is and how it is defined: [the interaction term](../../context/world/interaction-term.md)
- What the adapter is and why it never sees the joint prompt: [the LoRA corrector](../../context/world/lora-corrector.md)
- Which trajectory the cache stores and the rule it stepped with: [the plan's rule section](../../plans/05-when-does-the-outcome-lock-in/plans/tests/07-what-the-correction-is-made-of.md#the-rule-the-cache-used)
- The sampler-against-model split this does not make: claim 2 of [the idea map](../../artifacts/ideas/which-variable-explains-what-poe-is-missing/IDEA_MAP.md)
- The frames and the both-ness plane the tracks reuse: [where each condition lands](where-does-each-condition-land.md)
- Each picture, read one at a time: [the figure explainer](../../artifacts/results/what-the-correction-is-made-of/figure-explainer.md)

## Still open

Navigation: ⬅️ [Previous](#depends-on) | 📋 [TOC](#table-of-contents)

- [ ] The bar was inconclusive at 0.37. Whether a per-step re-weighting that damps both experts
      to the joint prompt's weights (1 to 3 instead of 7.5) composes on its own is the experiment
      this number proposes; it belongs to scope 06 as a sampler-side idea and has no plan yet.
- [ ] The orthogonal part's split into sampler error and model error: the sampler-share
      measurement in scope 06.
- [ ] The same four rungs on a pair the adapter trained on, so the 0.19 against 0.37 adapter
      shortfall is known to be a held-out effect or an architecture effect.
- [ ] The joint prompt's estimate along its own run beside the off-policy row of rung 2.
