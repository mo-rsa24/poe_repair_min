# Does searching over the initial noise, with the compose scorer as verifier, produce two animals?   ❓ inconclusive on the pre-registered bar · a null in everything but its letter · verified 2026-09-05

**The claim**

Plain product-of-experts on cat × dog does not reach a two-animal image by choosing its starting
noise. Random search over the eight held-out seeds finds none. One round of zero-order search,
eight perturbed noises around each seed at two step sizes and the validated compose scorer picking
the best finished image, finds one counted success in 128 candidates, and that image is one body
with two heads.

Read at step 10 instead of at the end, the same scorer finds nothing at all, and on the composing
control pair it throws the butterfly away on three seeds of eight.

The bar written before the run calls the cat × dog result inconclusive, because one seed of eight
is a difference of 0.125, above the null margin of 0.10 and under the support margin of 0.25.
The bar stands as written. What the one seed is, is written beside it.

## Table of contents

- [What we set out to see](#what-we-set-out-to-see)
- [What would have counted](#what-would-have-counted)
- [What was tried, in order](#what-was-tried-in-order)
- [1. Random search over the held-out seeds: nothing to pick](#1-random-search-over-the-held-out-seeds-nothing-to-pick)
- [2. Zero-order search: one counted success in 128, and it has two heads](#2-zero-order-search-one-counted-success-in-128-and-it-has-two-heads)
- [3. The verifier read at step 10 finds nothing, and breaks the control pair](#3-the-verifier-read-at-step-10-finds-nothing-and-breaks-the-control-pair)
- [4. Where the kept images land: the PoE band](#4-where-the-kept-images-land-the-poe-band)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Where this came from](#where-this-came-from)
- [Depends on](#depends-on)
- [Still open](#still-open)

## What we set out to see

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-would-have-counted) ➡️

Ma et al. ([arXiv 2501.09732](https://arxiv.org/abs/2501.09732), unpacked in
[the note](../../artifacts/notes/inference-time-scaling-as-a-search-over-noise/note.md)) frame
extra inference compute as a search over the starting noise with a verifier: with a deterministic
sampler, a noise is an image, so choosing the noise is choosing the image. Random search is best
of N draws. Zero-order search keeps a pivot noise, draws N candidates near it and keeps the best.

Here the sampler is plain product-of-experts at DDIM `eta` 0, the verifier is the project's
validated instance-count compose scorer, and the question is whether a composing image sits a
small nudge away from each held-out seed's failing noise. If it does, the early-window correction
has a training-free rival that forks at step 0.

**The hypothesis.** Eight nudges at σ 0.1 or 0.3 (cosine 0.995 or 0.958 to the pivot) reach a
two-animal render on at least two of eight seeds. The step-10 read is expected to be blind, since
x0-hat is a blur where the compose decision is made.

## What would have counted

Navigation: ⬅️ [Previous](#what-we-set-out-to-see) | 📋 [TOC](#table-of-contents) | [Next](#what-was-tried-in-order) ➡️

Support if the kept image composes at least 0.25 more often than the unperturbed seed at either
σ; null if within 0.10 at both; inconclusive between. The constants are `PASS_MARGIN` and
`NULL_MARGIN` in `poe_repair/experiments/noise_search/search.py`, and the pre-registered question
is in [the review file](../../plans/06-is-the-gap-the-samplers-or-the-models/review/11-zero-order-search-over-the-initial-noise.md),
both written before the launch. The paper never writes how a zero-order candidate is drawn from
its pivot; the variance-preserving form `z' = (z + σ·u) / sqrt(1 + σ²)` used here is this
project's choice and is named as such.

**The result on the bar: inconclusive.** Pivot 0 of 8; kept 1 of 8 at σ 0.1, 0 of 8 at σ 0.3.
The difference at σ 0.1 is 0.125.

## What was tried, in order

Navigation: ⬅️ [Previous](#what-would-have-counted) | 📋 [TOC](#table-of-contents) | [Next](#1-random-search-over-the-held-out-seeds-nothing-to-pick) ➡️

| When (2026-09-05) | What | Where it landed | Outcome |
|---|---|---|---|
| 18:28 | smoke on mscluster107 device 1: seed 9, N 2, σ 0.3, 10 steps, 512² | `outputs/interaction_term/noise_search/smoke_N2_s10_20260905-182806/` | every output present, device seen, pick rule and cosine as designed |
| 18:33 to 20:27 | the full run on the same device, PID 46316: two pairs, seeds 9 to 16, σ 0.1 and 0.3, N 8, 50 steps, 1024² | `outputs/interaction_term/noise_search/zo_N8_s50_20260905-183320/`, W&B `3yxrkwgx` | 16 seeds at about 7 min each; verdict inconclusive |
| 20:27 | both-ness on the launch node | the run log | failed: an xformers attention kernel refusing CPU inputs |
| 20:28 | both-ness rerun on mscluster85, CPU, xformers disabled | `.../bothness.json` | the kept columns land in the PoE band |

Every render shares its seed's cached initial noise (or a candidate at a recorded cosine to it),
50 DDIM steps at `eta` 0, guidance 7.5, 1024²; only the starting noise differs between the pivot
column and a kept column.

## 1. Random search over the held-out seeds: nothing to pick

Navigation: ⬅️ [Previous](#what-was-tried-in-order) | 📋 [TOC](#table-of-contents) | [Next](#2-zero-order-search-one-counted-success-in-128-and-it-has-two-heads) ➡️

The eight held-out seeds are eight independent noises, so the existing renders already answer
best-of-8. Plain product-of-experts at λ 0 counts one animal on every seed; the rank-32
correction at λ 1.2 counts two on seven. The best plain-PoE seed by both-ness (seed 16, 0.31)
still sits under the worst corrected seed (seed 10, 0.34), so no verifier on these eight noises
could pick a composing one.

**The number.** Compose rate over seeds 9 to 16: plain PoE 0.0, PoE plus the λ 1.2 correction
0.875. From `/datasets/mmolefe/poe_repair_min/outputs/showcase/figure_r32_030050/results.json`,
fields `summary.full.0.0.compose_rate` and `summary.full.1.2.compose_rate`; both-ness from
`artifacts/results/where-does-each-condition-land/cat-x-dog-in-dino-space.json`, field
`cloud_axes.both_ness_by_condition`.

## 2. Zero-order search: one counted success in 128, and it has two heads

Navigation: ⬅️ [Previous](#1-random-search-over-the-held-out-seeds-nothing-to-pick) | 📋 [TOC](#table-of-contents) | [Next](#3-the-verifier-read-at-step-10-finds-nothing-and-breaks-the-control-pair) ➡️

![Cat × dog, seeds 9 to 16: Mono, plain PoE from the cached noise, the existing λ 1.2 render, then the best of 8 perturbed noises at σ 0.1 and 0.3 picked on the final image, then picked on x0-hat at step 10; the detector's count on every tile](../../artifacts/results/is-the-gap-the-samplers-or-the-models/noise-search-cat-dog-sheet.png)
*Rows are seeds, columns are conditions, the number on each tile is the detector's animal count
(green at 2 or more). What to notice: every tile from the fourth column on is one animal or one
fused animal, the same as the second column, except seed 16's σ 0.1 pick, which counts 3 and
shows one body with a dog-like face in front and a cat's head lying behind it. The third column,
the existing correction, shows two animals on seven rows.*

**The number.** Kept-image compose rate over 8 seeds, final-image verifier: 0.125 at σ 0.1 and
0.0 at σ 0.3, against 0.0 for the pivot. Over all 64 candidates per σ, 3 count two or more at
σ 0.1 (all on seed 16: counts 2, 3 and 2) and 0 at σ 0.3. Opened, the three are one two-headed
body and two single animals the detector boxed twice. Mean cosine of candidates to the pivot:
0.995 at σ 0.1, 0.958 at σ 0.3. From
`artifacts/results/is-the-gap-the-samplers-or-the-models/noise-search-summary.json`, fields
`pairs.a_cat__x__a_dog.compose_rate_by_column` and `secondary`; the verdict in
`noise-search-verdict.json`.

## 3. The verifier read at step 10 finds nothing, and breaks the control pair

Navigation: ⬅️ [Previous](#2-zero-order-search-one-counted-success-in-128-and-it-has-two-heads) | 📋 [TOC](#table-of-contents) | [Next](#4-where-the-kept-images-land-the-poe-band) ➡️

![Butterfly × meadow, seeds 9 to 16: Mono, plain PoE from the cached noise, then the best of 8 at σ 0.1 and 0.3 picked on the final image, then picked on x0-hat at step 10](../../artifacts/results/is-the-gap-the-samplers-or-the-models/noise-search-butterfly-meadow-sheet.png)
*Same layout for the composing control pair; there is no adapter column because none was rendered
for this pair. What to notice: a meadow is not an animal, so the count reads 1 on most tiles
including Mono, and a green 2 or 3 here means several butterflies (seed 14, a repeating botanical
print). Under the final-image verifier every kept tile keeps a butterfly. Under the step-10
verifier at σ 0.3, seeds 10, 13 and 15 finish with no detectable animal; seed 15's is a flat
floral pattern.*

**The number.** Step-10 verifier on cat × dog: kept compose rate 0.0 at both σ; on seed 16 it
counted 1 on all eight candidates and so could not have picked the three the final read
counted. Agreement between the step-10 and final compose verdicts per candidate: 0.922 at σ 0.1
and 1.0 at σ 0.3, almost all of it both reads saying "one animal". Control pair, finished-image
count of the step-10 pick at σ 0.3: 0 on seeds 10, 13 and 15. Same summary file, fields
`compose_rate_by_column.best_early_sigma_*` and `secondary.sigma_*_early_agrees_with_final`.

## 4. Where the kept images land: the PoE band

Navigation: ⬅️ [Previous](#3-the-verifier-read-at-step-10-finds-nothing-and-breaks-the-control-pair) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**The number.** Both-ness on the landing finding's axes, mean over 8 seeds: Mono 0.523, plain PoE
pivot 0.200, the existing λ 1.2 renders 0.420, kept by the final-image verifier 0.235 (σ 0.1) and
0.234 (σ 0.3), kept by the step-10 verifier 0.217 and 0.246. The axes reproduce the landing
finding's 0.52 / 0.21 / 0.42 to within 0.01. The one counted tile, seed 16 at σ 0.1, reads 0.442,
inside the corrected band, while it is one body with two heads by eye: the embedding axis reads a
second head as "both", so this read cannot arbitrate that tile. From
`artifacts/results/is-the-gap-the-samplers-or-the-models/noise-search-bothness.json`, fields
`by_column.*.mean_both_ness` and `by_column.best_final_sigma_0.1.per_seed.16`.

## What this cannot tell you

Navigation: ⬅️ [Previous](#4-where-the-kept-images-land-the-poe-band) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**One round, eight candidates, two step sizes.** The paper iterates and scales to hundreds of
candidates. This says the neighbourhood at cosine 0.96 to 0.995 holds no two-animal noise among
eight draws; it does not say a larger or longer search would not.

**The verifier is the judge.** The kept image is chosen by the scorer that scores it, and the
scorer was selected for: two of the three counted candidates are detector errors and the third
is a second head. A verifier that a second head cannot satisfy might search differently.

**The pivot is this sampler's plain PoE.** This plan's batched DDIM and the LoRA sampler at λ 0
start from the same noise and part late on some seeds (2.2 to 14.5 grey levels, seed 12 the same
scene with a different face), so the pivot column is not byte-identical to the λ 0 column in
other figures. Every pivot still counts one animal.

**The control pair's count is not its read.** A meadow is not an animal. The control read is by
eye, and it is recorded in the review file.

**Eight seeds, one pair.** Where things land, not the shape of a distribution.

## Where this came from

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#depends-on) ➡️

| What | Source | Mark |
|---|---|---|
| The compose rates, secondary reads and verdict | `artifacts/results/is-the-gap-the-samplers-or-the-models/noise-search-summary.json` and `noise-search-verdict.json`, copied from the run dir 2026-09-05 | verified |
| Both-ness per tile | `.../noise-search-bothness.json`, written 2026-09-05 on mscluster85 by `run.py --bothness-only` from the run's tiles and the landing finding's saved features | verified |
| The two sheets | `.../noise-search-cat-dog-sheet.png`, `.../noise-search-butterfly-meadow-sheet.png`, drawn by the runner at the end of each pair | verified |
| Every candidate, its x0-hat at step 10 and its scores | `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/noise_search/zo_N8_s50_20260905-183320/<pair>/seed_<n>/sigma_<σ>/` | verified |
| The eye reads (seed 16's three candidates, seed 12's pivot pair, the control sheet) | the review file's `Written before the run, answered after` and `Asked after the result`, recorded as the tiles were opened | stated |
| The random-search row | `figure_r32_030050/results.json` and `cat-x-dog-in-dino-space.json`, read 2026-09-05 | verified |
| The run | tasks **Run the smoke mode** and **Launch the full mode** in [the design](../../plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/11-zero-order-search-over-the-initial-noise.md); verdict in [its review file](../../plans/06-is-the-gap-the-samplers-or-the-models/review/11-zero-order-search-over-the-initial-noise.md); W&B `prime_lab/poe-repair-animals-compose/runs/3yxrkwgx` | verified |
| Regenerate with | `ssh <node> 'GPU=<idx> nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/noise_search/run_noise_search.sh full > /datasets/mmolefe/poe_repair_min/outputs/interaction_term/noise_search/logs/full.log 2>&1 &'`, then `XFORMERS_DISABLED=1 <co3 python> -m poe_repair.experiments.noise_search.run --bothness-only --run-id <run> --device cpu` on the session node | verified |

## Depends on

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- What product-of-experts composition is: [PoE composition](../../context/world/poe-composition.md)
- What the compose scorer counts and what it cannot: [compose rate](../../context/world/compose-rate.md)
- The both-ness axes and their caveat that a fused face reads as "both":
  [where each condition lands](../when-does-the-outcome-lock-in/where-does-each-condition-land.md)
- The paper: [the note](../../artifacts/notes/inference-time-scaling-as-a-search-over-noise/note.md)
  and [its register row](../../plans/standing/literature/reading-register.md)
- The two sibling selection-only baselines:
  [does selecting among PoE proposals compose](does-selecting-among-poe-proposals-compose.md) and
  plan 10's Feynman-Kac steering, in flight at the time of writing
- Which python build runs on which card: [nodes](../../environment/hpc/nodes.md)

## Still open

Navigation: ⬅️ [Previous](#depends-on) | 📋 [TOC](#table-of-contents)

- [ ] The paper's iterated climb is unrun, with the reason in the review file: nothing to climb
      from, and a climb toward what the counter rewards is a climb toward detector errors.
- [ ] A verifier a second head cannot satisfy (a per-concept read that requires both a cat box
      and a dog box, or the human-checked label pass) would make the search worth a second
      round; the compose-rate context notes why the per-concept query was set aside.
- [ ] Reading this against the in-span share from scope 05's rung 2, once it lands: a search
      over noises reaches only what plain PoE renders from some noise.
