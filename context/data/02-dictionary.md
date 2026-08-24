# 🌍 Dictionary

One entry per field a reader will meet in a `summary.json`, a scorer output, or a fail-rate table.
Every example is a real value read directly from a file in this repository, labelled as such.

## Table of contents

- [`pair_slug`](#pair_slug)
- [`seed`](#seed)
- [`group` / `group_label`](#group-group_label)
- [`arm`](#arm)
- [`lambda`](#lambda)
- [`step`](#step)
- [`model_id`](#model_id)
- [`d_T` / `d_t`](#d_t-d_t)
- [`n_instances`](#n_instances)
- [`label` / `truth`](#label-truth)
- [`fail_rate` / `compose_rate`](#fail_rate-compose_rate)

### `pair_slug`

Navigation: 📋 [TOC](#table-of-contents) | [Next](#seed) ➡️

**What it means in the world** ✅

Which two animal concepts a cell tests, written as the on-disk slug form (see
[world/animal-pair.md § Words this file uses](../world/animal-pair.md#words-this-file-uses)).

**Example** `a_cat__x__a_dog` (example, this project's own reference pair)

**Type and shape** text, `a_<animal1>__x__a_<animal2>` (or `an_` where grammar needs it)

**Where it comes from** the pair pool definition, `artifacts/results/does-the-fix-reach-unseen-pairs/fail_rate.md`
for the current 17-row pool, and `plans/retrofit-poe-repair-min.md` for the wider 20-pair boundary
(see [world/animal-pair.md § What people get wrong](../world/animal-pair.md#what-people-get-wrong))

**Stands for** a property of an [animal pair](../world/animal-pair.md#what-an-animal-pair-is)

**Watch out** the slug's underscore-and-`x` form is never changed to match prose; a plan or figure
that writes "a cat and a dog" and one that writes `a_cat__x__a_dog` mean the same pair ⚠️

### `seed`

Navigation: ⬅️ [`pair_slug`](#pair_slug) | 📋 [TOC](#table-of-contents) | [Next](#group-group_label) ➡️

**What it means in the world** ✅

Which re-roll of the starting noise produced this cell. The pair of concepts does not change; only
the random starting point does.

**Example** `42` (example; the project's own reference seed for `a_cat__x__a_dog`). Held-out
evaluation seeds are `9`, `10`, `11`, `12`, per `plans/retrofit-poe-repair-min.md`.

**Type and shape** integer

**Where it comes from** set at sampling time; cached per (pair, seed) cell

**Stands for** a property of one render, not of the pair itself

**Watch out** a seed and a pair are different axes of generalisation, and transferring across pairs
is the harder test (see [world/animal-pair.md](../world/animal-pair.md#words-this-file-uses)) ✍️

### `group` / `group_label`

Navigation: ⬅️ [`seed`](#seed) | 📋 [TOC](#table-of-contents) | [Next](#arm) ➡️

**What it means in the world** ✅

Which of six taxonomy categories a pair belongs to, classifying *why* composition is hard for it,
not just which species are involved. See
[world/animal-pair.md § Words this file uses](../world/animal-pair.md#words-this-file-uses) for
the six names.

**Example** `"group_label": "Group 6 - Coherent Collision"` (example, read directly from
`data/pilot/seed_42/a_cat__x__a_dog/summary.json`)

**Type and shape** text, `group<N>_<snake_case_name>` in code, `"Group <N> - <Title Case Name>"`
in the human-readable label

**Where it comes from** assigned when a pair is added to the taxonomy; see
`plans/shelved/phases/09-lora-taxonomy-single-seed.md` for the representative pair per group

**Stands for** a property of a [pair](../world/animal-pair.md#what-an-animal-pair-is), fixed at
assignment, not derived from any run's outcome

**Watch out** the taxonomy is shelved except for Group 6, which is the one group overlapping the
live animal-only pool; a pair with a group label is not automatically part of the current
experiment scope 🔍

### `arm`

Navigation: ⬅️ [`group` / `group_label`](#group-group_label) | 📋 [TOC](#table-of-contents) | [Next](#lambda) ➡️

**What it means in the world** ✍️

Which render method produced a given image: plain PoE, Mono, PoE plus the interaction term
injected at some strength, or PoE plus the trained LoRA. The word "arm" itself is struck from
prose (`plans/retrofit-poe-repair-min.md`: "An arm is the corrected run or the uncorrected run.");
it survives as a column and function-family name in code.

**Example** `poe` (example; other observed values include `mono`, and the method names in
`poe_repair/methods/_sampling.py`: `run_cfg`, `run_cfg_poe`, `run_teacher_residual`,
`run_lora_residual_inject`)

**Type and shape** text, one of a small fixed set of method names

**Where it comes from** chosen at sampling time by which function in `poe_repair/methods/` is
called

**Stands for** a property of one render, not of the pair or the seed

**Watch out** at full correction strength a sampler could in principle fall back to reproducing
the Mono prediction directly, which would make every arm look identical for the wrong reason; this
risk is checked for, per
[world/poe-composition.md § What people get wrong](../world/poe-composition.md#what-people-get-wrong)
⚠️

### `lambda`

Navigation: ⬅️ [`arm`](#arm) | 📋 [TOC](#table-of-contents) | [Next](#step) ➡️

**What it means in the world** ✅

How much of the interaction term is added back at inference. `0` reproduces plain PoE exactly;
`1` is the full correction.

**Example** `1.0` (example, full strength)

**Type and shape** float, `0.0` to `1.0` in the sweeps seen in this repo

**Where it comes from** set as a sampling parameter, swept across a grid in the dose-response
experiments

**Stands for** a property of one render, applied over whichever
[window](../world/interaction-term.md#words-this-file-uses) of steps that run specifies

**Watch out** `λ=0` must reproduce plain PoE to under 1e-5; this is the project's own canary check
against contamination (`RESEARCH_GUIDELINES.md`: "the λ=0 check: injecting nothing must
reproduce plain PoE to under 1e-5") ⚠️

### `step`

Navigation: ⬅️ [`lambda`](#lambda) | 📋 [TOC](#table-of-contents) | [Next](#model_id) ➡️

**What it means in the world** ✅

Which point in the 50-step denoising schedule a value belongs to. Step 0 is near-pure noise; step
49 is the finished image.

**Example** `10` (example; the correction window found in EXP-04 spans steps 0-10)

**Type and shape** integer, `0` to `49`

**Where it comes from** fixed by the sampler's schedule length, verified in `EXPERIMENTS.md`'s
axes table: "step | 0 to 49 | the schedule is 50 steps everywhere in the cache"

**Stands for** a property of one point in one render's trajectory

### `model_id`

Navigation: ⬅️ [`step`](#step) | 📋 [TOC](#table-of-contents) | [Next](#d_t-d_t) ➡️

**What it means in the world** ✅

Which pretrained model generated the image: the exact HuggingFace model identifier.

**Example** `stabilityai/stable-diffusion-xl-base-1.0` (example, read directly from
`data/pilot/seed_42/a_cat__x__a_dog/summary.json`)

**Type and shape** text, a HuggingFace hub repo id

**Where it comes from** fixed by the model-loading code (`poe_repair/_sdxl/`); every run in this
repo has used the one value seen so far, per this build's reading

**Stands for** a property of the whole run, not of any one pair or seed

### `d_T` / `d_t`

Navigation: ⬅️ [`model_id`](#model_id) | 📋 [TOC](#table-of-contents) | [Next](#n_instances) ➡️

**What it means in the world** ✅

The measured size of the [interaction term](../world/interaction-term.md#what-the-interaction-term-is)
between the Mono and PoE predictions: `d_t` per denoising step, `d_T` at the final step only.

**Example** `d_T_poe_vs_mono: 0.246` (example, `a_cat__x__a_dog` seed 42); `d_t_poe_vs_mono` is the
same quantity as a 50-value array, rising from `0.0` at step 0 to that final value

**Type and shape** float (`d_T`) or array of 50 floats (`d_t`), both non-negative and generally
increasing over the run for the one cell inspected

**Where it comes from** computed during sampling by comparing the Mono and PoE noise predictions
at each step; written into `summary.json`'s `metrics` block

**Stands for** a property of one (pair, seed) render pair (the PoE render and the Mono render for
the same starting noise), not of a single image alone

### `n_instances`

Navigation: ⬅️ [`d_T` / `d_t`](#d_t-d_t) | 📋 [TOC](#table-of-contents) | [Next](#label-truth) ➡️

**What it means in the world** ✅

How many distinct "animal" instances the GroundingDINO detector found in one rendered image, after
merging overlapping boxes (NMS).

**Example** `2` (example, `catdog_compose_compose_seed_09` in `scorer_validated.json`); `1` is the
chimera/dropped-concept case, `3` appears at least once (`wolfhusky_sample_seed_12`)

**Type and shape** non-negative integer

**Where it comes from** `artifacts/results/can-we-trust-the-compose-score/compose-scorer-validation/scorer_validated.json`, `validation_labels.*.n_instances`

**Stands for** a property of one rendered image

**Watch out** two instances does not mean the two *requested* animals; see
[world/compose-rate.md § What people get wrong](../world/compose-rate.md#what-people-get-wrong) ⚠️

### `label` / `truth`

Navigation: ⬅️ [`n_instances`](#n_instances) | 📋 [TOC](#table-of-contents) | [Next](#fail_rate-compose_rate) ➡️

**What it means in the world** ✅

`label` is the scorer's own compose/blend call for one image, derived mechanically from
`n_instances`. `truth` is the human-assigned ground truth, present only in the validation set used
to check the scorer, not in ordinary pipeline output.

**Example** `"truth": "compose", "label": "compose"` (example, `catdog_compose_compose_seed_09`,
where the two happen to agree, which is what validates the rule)

**Type and shape** text, one of `compose` or `blend`

**Where it comes from** `label` is computed by the rule in
[world/compose-rate.md](../world/compose-rate.md#what-a-compose-rate-is); `truth` is set once, by
a person, when the validation set was built

**Stands for** a property of one rendered image

### `fail_rate` / `compose_rate`

Navigation: ⬅️ [`label` / `truth`](#label-truth) | 📋 [TOC](#table-of-contents)

**What it means in the world** ✅

The fraction of a pair's tested seeds scored one way. `fail_rate` is used when reporting how often
plain PoE fails; `compose_rate` is the same fraction under whichever render method is being
evaluated, or when reporting a success rather than a failure.

**Example** `1.00 (8/8)` (example, `a_cat__x__a_dog`'s plain-PoE fail rate, read directly from
`artifacts/results/does-the-fix-reach-unseen-pairs/fail_rate.md`)

**Type and shape** float, `0.0` to `1.0`, usually shown with the raw count it was computed from

**Where it comes from** aggregated per pair from `label` values across seeds

**Stands for** a property of one (pair, render method) combination, over a fixed set of seeds
(8, in the pool this project currently uses)

**Watch out** a rate printed alone hides the scorer's own uncertainty; see
[world/compose-rate.md § What people get wrong](../world/compose-rate.md#what-people-get-wrong)
for the measured bound (87% to 94% true rate behind one 94% headline) ⚠️
