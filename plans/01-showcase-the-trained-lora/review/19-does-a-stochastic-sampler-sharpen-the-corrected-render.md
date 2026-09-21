# 🧪 Review: does a stochastic sampler sharpen the corrected render without losing the two animals?

**Nothing is running and nothing is judged.** Slurm job 50338 on `mscluster46` was the last attempt; the queue has been empty since, and no verdict was ever written. The eta question is unanswered, not in progress. This file judges [the design](../plans/experiments/19-does-a-stochastic-sampler-sharpen-the-corrected-render.md). Its answer sets the showcase wall's sampler setting if an eta is supported, and otherwise bounds the paper's fidelity caveat to the adapter itself.

## Recommended prompt (when the run lands)

```
/analyze-run stochastic_sampler_sweep_r32_030050
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/experiments/19-does-a-stochastic-sampler-sharpen-the-corrected-render.md) | the stochastic step, the shared noise path, the cells, the code, the constants |
| **this file** | **the verdict: which eta, if any, keeps the animals and returns the sharpness and the fidelity** |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

- **eta**: the DDIM sampler's stochasticity; 0 is the deterministic sampler every other result uses, 1 the ancestral sampler on the same 50-step grid. The design file gives the update.
- **The noise path**: the fresh-noise sample at every step of one seed, shared by every condition and eta of that seed.
- **The corrected run**: plain PoE plus 1.2 times the rank-32 step-30050 adapter's change on all 50 steps.
- **Seeds sharper**: of 8, the seeds where the cell's Laplacian variance at 1024 px exceeds the corrected run's at eta 0 on the same seed. Per seed because three seeds are line drawings whose edge density swamps any mean.
- **Drift**: DINOv2 cosine distance to the Mono render minus distance to the plain-PoE render of the same seed, both at the same eta on the same noise path. Negative is nearer the joint-prompt image.
- **Compose count**: seeds of 8 where the validated detector counts two or more animal instances.
- **Butterfly present**: on the control pair, a "butterfly" box at confidence 0.30 or more.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim** (group 1: an intervention on the sampler, no training). A missed bar closes the plan; the fidelity caveat is then the adapter's own and plans 15 and 20 carry it. A supported eta changes the showcase wall's sampler setting and adds one method sentence.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| jobs 50334 and 50335 | tests the claim | 2026-09-06 03:01, `bigbatch` | 1 second each | nothing | died on `mscluster65`, whose GPU is dead; the exclusion list was extended and the node table updated |
| job 50338, `all` (identity check, 144 renders, scoring) | tests the claim | 2026-09-06 03:05, `mscluster46` (RTX 3090), `co3`, log `logs/stoch_sweep-50338.out` | about 2 GPU-hours plus 10 minutes of scoring | `/datasets/mmolefe/poe_repair_min/outputs/showcase/stochastic_sampler_sweep/` (renders, `results.json`, sheets), `artifacts/results/does-a-stochastic-sampler-sharpen-the-corrected-render/` (the figure, the sheets, `cell-table.md`, `results.json` without the feature vectors), W&B run `stochastic_sampler_sweep_r32_030050` | running |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ **At eta 0.5 or 1, does the corrected run keep its compose count within one seed of eta 0, come out sharper on most seeds, and stay as near the joint-prompt render, on cat × dog?**
      The bar, in source as the constants of `scripts/showcase/stochastic_sampler_sweep.py` and applied by `verdict_for_eta`, each eta above 0 judged against the corrected run at eta 0 on the same seeds:
      **support** if compose count ≥ eta 0's − `SUPPORT_MAX_COMPOSE_LOSS` (1) and seeds sharper ≥ `SUPPORT_MIN_SEEDS_SHARPER` (6 of 8) and mean drift ≤ eta 0's + `SUPPORT_MAX_DRIFT_RISE` (0.02);
      **null** if compose count ≤ eta 0's − `NULL_MIN_COMPOSE_LOSS` (2) (the animals go with the noise) or seeds sharper ≤ `NULL_MAX_SEEDS_SHARPER` (3 of 8) (no sharper);
      **inconclusive** otherwise (the animals stay and some seeds sharpen, but fewer than six, or the drift rises).
      The plan reads support if any eta supports, null if every eta is null, inconclusive otherwise.
      What would surprise: a supported eta with plain PoE also gaining two or more seeds, which would say the noise, not the adapter, is doing the composing.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ **Does plain PoE compose more seeds at eta above 0 than at eta 0?** Constant `POE_SAMPLER_GAIN_SEEDS = 2`. The route's prediction is no: a stochastic sampler samples the product more faithfully, and the product's answer is one blended animal. Two or more seeds gained is the sampler-side surprise scope 06 would need to know about. Reported as a number either way in `summary.a_cat__x__a_dog.poe_eta<eta>.compose_gain_over_eta0`.
- [ ] ⚠️ **Does the gain, if any, belong to the adapter or to the sampler?** A seed that sharpens under the corrected run at eta 1 and also under plain PoE at eta 1 sharpened because of the sampler. The read is the seeds-sharper count of `poe_eta1` against `poe_eta0` beside the corrected count; only the excess is the adapter's.
- [ ] ⚠️ **Does any eta break the control pair?** On butterfly × flower meadow, `control_verdict` marks a cell broken if the butterfly is present on more than `CONTROL_MAX_PRESENCE_LOSS` (1) fewer seeds than plain PoE at the same eta.
- [ ] ⚠️ **Does the Mono reference itself hold at eta above 0?** The fail criterion in the design: Mono composing fewer than 6 of 8 at an eta means the reference moved and the drift at that eta is not a fidelity read.

## Asked after the result

Navigation: ⬅️ [Written before the run, answered after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions that arise from looking at the sheets go here, marked as post-hoc, with the number and the file they came from.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- **The sampler is not the old one at eta 0.** Guarded by the identity check (plain PoE at eta 0 on seed 9 against the cached `poe.png`, within `IDENTITY_MAX_MEAN_ABS_DIFF` 6 grey levels; the render stops on failure) and by the offline check that the step equals `ddim_prev_from_x0_eps` at eta 0 to zero difference.
- **The adapter left attached during a reference render.** Mono is rendered at every eta before the adapter is attached; every frozen forward disables it; the sampler disables it on exit.
- **Sharpness measures edges, not blur.** The three sketch seeds have Laplacian variance an order of magnitude above the photographic seeds, so every sharpness read is per seed against the same seed at eta 0, and the eye read in instruction 2.1 is written before the numbers are opened.
- **A different card than the filed grids.** The filed rank-32 grids were rendered on Quadro RTX 8000s; this job runs on an RTX 3090. fp16 across cards moves pixels by about 2 of 255, so no number here is compared to a filed number, only to cells inside this run.
- **The drift instrument is not the decay finding's.** Same DINOv2 family, own embedder and own references, so the eta 0 drift here is not expected to equal −0.169 (the filed λ 1.2 value). Within-run comparisons only.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- The sampler setting the showcase wall uses, with the eta and the reason, if an eta is supported.
- One sentence in the method on the DDIM update with eta and the shared noise path.
- The plain-PoE row at eta 1, carried to scope 06's related-work paragraph on stochastic samplers.

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- [ ] The sweep covers one checkpoint (30,050) and one λ (1.2). Whether eta interacts with λ (a smaller λ at eta 1 keeping the animals with less off-target energy) is untested.
- [ ] Only DDIM's eta is varied. EDM-style churn and a Langevin corrector at each level (scope 06's corrector) are other stochastic samplers with different noise schedules; a supported result here does not transfer to them without a run.

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

When the job leaves the queue: read `logs/stoch_sweep-50338.out` for the `plan verdict:` line, open the across-eta sheet before the table, fill the questions above, then `/sync-plan-tree plans/01-showcase-the-trained-lora/`.
