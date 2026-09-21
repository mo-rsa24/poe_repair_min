# 🧪 What the correction is made of, read off the cache

**What this plan asks:** at each denoising step, how much of the correction the joint prompt
supplies could product-of-experts have reached by re-weighting the three predictions it already
has, where in the picture the two experts disagree, what the trained adapter adds beyond that,
and how far each run's running estimate travels before it settles.

Step 53 in the root running order; waits on nothing. Everything it reads is already on disk:
the training cache for cat × dog seeds 9 to 16, the rank-32 adapter, and the per-step frames
rendered for [where each condition lands](../figures/06-where-each-condition-lands.md). No new
render. Its rung 2 result is read by the sessions testing sampler-side fixes in
[is the gap the samplers or the models](../../../06-is-the-gap-the-samplers-or-the-models/MASTER_PLAN.md),
because a selection or re-weighting method can only reach what the experts already span.

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/05-when-does-the-outcome-lock-in/plans/tests/07-what-the-correction-is-made-of.md — <one line on what actually happened>
```

## Recommended skill

`/pair-figure` for the share-against-step curve beside the seed-15 picture strip. Everything
else is custom scripts under `scripts/showcase/`.

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 52 (previous) | [where each condition lands](../figures/06-where-each-condition-lands.md) | five conditions as clouds and tracks in DINOv2 space; the frames this plan's rung 4 reads |
| **53 (current)** | **What the correction is made of** | **the in-span share of the correction, the experts' Tweedie pictures, the adapter's output projected the same way, the tracks' kinetic energy and which-animal score** |
| 24 (beside) | [the free bound on the model's share](../../../06-is-the-gap-the-samplers-or-the-models/plans/hypothesis/01-the-free-bound-on-the-models-share.md) | the other cache-only read of the correction, by its size as the run reaches zero noise |

## Table of contents

- [Position in the plan tree](#position-in-the-plan-tree)
- [Quick context: where you are](#quick-context-where-you-are)
- [Words this plan uses](#words-this-plan-uses)
- [The rule the cache used](#the-rule-the-cache-used)
- [Considerations](#considerations)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
- [What happens (visual)](#what-happens-visual)
- [Description: what to build](#description-what-to-build)
- [Purpose and goal](#purpose-and-goal)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [What has to pass before this runs](#what-has-to-pass-before-this-runs)
- [Figure Catalog](#figure-catalog)
- [Orchestration](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

## Quick context: where you are

Navigation: ⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#words-this-plan-uses) ➡️

**The experiment.** Four reads of the same cached run, cat × dog, held-out seeds 9 to 16, 50
steps each. The cache holds, per step, the noisy latent and the four raw noise predictions the
network made at it: under "a cat", under "a dog", under "a cat and a dog", and under the empty
prompt. From those four vectors every quantity below is a closed form, so rungs 1, 2 and 4 need
no sampling and rung 3 needs one network forward per cached step with the adapter attached.

**The hypothesis.** Most of the correction, at the steps where it matters (0 to 10), points
outside the space the three predictions span. If that holds, no per-step re-weighting of the
experts could have supplied it, and a learned term or the joint prompt is needed. If instead the
correction lies mostly inside that span, a per-step guidance re-weighting is a candidate fix and
the sampler-side sessions have room to win.

**Context details.** Adapter: rank 32, alpha 32, on the cross-attention query, key and value
projections, step 30050, at
`/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/checkpoints/lora_step_030050.pt`.
It never trained on cat × dog. Cache cells at
`/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache/heldout/a_cat__x__a_dog/seed_<n>/residuals/step_<kk>.pt`,
50 DDIM steps, guidance 7.5, stored in fp16 and upcast to fp32 before any norm. Frames at
`/datasets/mmolefe/poe_repair_min/outputs/showcase/where_each_condition_lands/frames/<cond>/seed_<n>/step_<kk>.png`,
already embedded into `artifacts/results/where-does-each-condition-land/frames-dino-feats.npz`.

**This plan's job.** Turn "the correction is small and early" into "the correction is made of
this much re-weightable and this much new, here in the picture, and this is what the adapter
adds", with the bar for the first of those written before the numbers exist.

**Associated materials.** Verdict: [the review file](../../review/07-what-the-correction-is-made-of.md).
The A-against-B split this plan does not resolve: claim 2 of
[which variable explains what PoE is missing](../../../../artifacts/ideas/which-variable-explains-what-poe-is-missing/IDEA_MAP.md).

**For the full picture.** The scope's [master plan](../../MASTER_PLAN.md).

## Words this plan uses

Navigation: ⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-rule-the-cache-used) ➡️

- **ε_a, ε_b, ε_j, ε_∅**: the network's raw noise predictions at one cached state under "a
  cat", "a dog", "a cat and a dog", and the empty prompt. Each is a 4 × 128 × 128 tensor,
  treated as one vector of 65,536 numbers.
- **the guided prediction ε̃_k**: ε_∅ + 7.5 (ε_k − ε_∅), classifier-free guidance at scale 7.5.
- **ε_PoE**: the product-of-experts prediction, ε̃_a + ε̃_b − ε_∅. It is what the cached run
  actually stepped with.
- **r_t, the correction (the interaction term)**: ε̃_j − ε_PoE, what the joint prompt would have
  added to the product-of-experts prediction at that same state. Closed form 7.5 (ε_j − ε_a −
  ε_b + ε_∅), the cache's `delta_t_from_raw`.
- **the span**: the set of vectors reachable as α ε_∅ + β (ε_a − ε_∅) + γ (ε_b − ε_∅) for any
  three numbers α, β, γ. Every per-step guidance re-weighting of the two experts lands inside
  it. It is a three-dimensional subspace of the 65,536-dimensional prediction space.
- **in-span share**: ‖projection of r_t onto the span‖² / ‖r_t‖². Its complement, one minus it,
  is the **orthogonal share**: the fraction of the correction's energy no re-weighting can reach.
- **the Tweedie estimate x̂₀**: the running guess at the finished latent from a state and a
  prediction: (x_t − √(1−ᾱ_t) ε) / √ᾱ_t. Decoding it through the VAE gives the frames plan 06
  already uses.
- **δ̂, the adapter's output**: the product-of-experts prediction with the adapter attached
  minus the same prediction with it detached, at the same cached state. At inference it is
  added at λ 1.2; here it is read at λ 1.
- **kinetic energy of a track**: the sum over consecutive saved frames of the squared distance
  the run's embedded running estimate moved, in DINOv2 space. Its floor is the squared
  straight-line distance from first to last frame divided by the number of segments, which is
  what a run moving at constant speed in a straight line would score.
- **which-animal score**: cosine of a frame's embedding to the cat-alone centroid minus its
  cosine to the dog-alone centroid, both centroids from the step-50 frames of the single-prompt
  runs. Positive is cat, negative is dog, and a sign change is the animal flipping.

## The rule the cache used

Navigation: ⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

Read off `scripts/build_training_cache.py` and `poe_repair/_sdxl/metrics.py` before any number
was computed. Written here so every projection below is against the rule that produced the states.

One network call per step with a batch of four prompts ("a cat", "a dog", "a cat and a dog",
empty) at the same latent x_t. The four raw outputs are saved as they are.

```
ε̃_a   = ε_∅ + g (ε_a − ε_∅)          g = 7.5           guided_eps
ε̃_b   = ε_∅ + g (ε_b − ε_∅)
ε_PoE  = ε̃_a + ε̃_b − ε_∅                                poe_eps
x̂₀    = (x_t − √(1−ᾱ_t) ε_PoE) / √ᾱ_t                    tweedie_mean
x_prev = √ᾱ_prev x̂₀ + √(1−ᾱ_prev) ε_PoE                  ddim_prev_from_x0_eps, η = 0
```

The latent is advanced with ε_PoE, so every cached state lies on the plain product-of-experts
path. The joint prompt's prediction ε_j is queried at those states and never steers them. The
"a cat and a dog" run that made `mono.png` is a separate trajectory whose predictions are not
cached. So r_t here is "what the joint prompt would have added at the state PoE reached", which
is the quantity the adapter was trained to imitate.

Expanding ε_PoE: ε_∅ + g (ε_a − ε_∅) + g (ε_b − ε_∅). It is itself inside the span with α = 1,
β = γ = g. And r_t = g (ε_j − ε_∅) − g (ε_a − ε_∅) − g (ε_b − ε_∅), so r_t's part outside the span
is exactly the part of the joint guidance direction g (ε_j − ε_∅) outside the span. The in-span
share of r_t therefore mixes two things: the joint direction's own overlap with the experts, and
the fact that r_t removes the two expert terms outright. Both are reported.

## Considerations

Navigation: ⬅️ [Previous](#the-rule-the-cache-used) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**Expected runtime.** Rung 2: 400 step files at 0.66 MB each, one small least-squares fit per
file, under five minutes on CPU. Rung 1: 35 VAE decodes at 1024 square, about a minute on the
session node's RTX 3090. Rung 3: 400 cached steps, each two network forwards of batch 3 (adapter
on, adapter off), about ten minutes on the same card. Rung 4: array arithmetic over the 672
stored frame embeddings, seconds.

**Cost and what it buys.** Under half an hour of one GPU and no queue. Buys the review file's
deciding question and the four figures in the catalog, and gives the sampler-side sessions the
number that says whether their methods can reach the correction at all.

**Prerequisites.** The cache cells for seeds 9 to 16 (present, 50 steps each, checked
2026-09-05); the rank-32 checkpoint; the frame features file. The session node's GPU free of
foreign processes at launch, checked with `nvidia-smi` and `pgrep -af 'sweep|train|corrector'`.

**Project tracking.** Sidecars and figures to `artifacts/results/what-the-correction-is-made-of/`.
Decoded pictures and any large intermediate to
`/datasets/mmolefe/poe_repair_min/outputs/showcase/what_the_correction_is_made_of/`. One W&B run
in `prime_lab/poe-repair-animals-compose` carrying the four figures as images and the sidecars as
an artifact; its id goes in the review file.

**The shared read-out.** The eight parallel sessions share a render read-out (a triptych sheet
per condition, the instance-count scorer, the butterfly × meadow control). This plan renders
nothing and scores nothing, so those items do not apply; only the W&B logging and the output
location rule do.

**Known issues.** See [Error Matrix](#error-matrix).

<details>
<summary>Environment Facts This Plan Depends On</summary>

- The session node mscluster85 carries an RTX 3090 and the `co3` build at
  `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`; see [nodes](../../../../environment/hpc/nodes.md).
  A run on an already-allocated interactive node needs no `sbatch`; see step 1 of
  [the execution protocol](../../../../environment/hpc/execution-protocol.md).
- The card is shared with other users' processes; a decode that needs more than the free memory
  dies with an out-of-memory error, which is what killed plan 06's first render attempt. Check
  `nvidia-smi` before launch.
- `torch.cuda.is_available()` is checked on the pinned device before real work
  (known failure poe-launch-002 in [known failures](../../../../environment/known-failures.md)).
- fp16 upcast rule: every cached tensor is float16 and is cast to float32 before any norm,
  projection or difference; see [the overview](../../../../environment/overview.md#known-gaps--non-obvious-constraints).
- Large artifacts to `/datasets` only.

</details>

## The claim

Navigation: ⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**At the steps where the correction matters, most of it points where no re-weighting of the two
experts can reach, and the adapter's output points there too.**

**Why this matters right now:** three parallel sessions are testing fixes that select among or
re-weight the product's own proposals. Whether those can work is decided by this number, before
any of them spends a GPU day.

## Why this plan exists

Navigation: ⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The gap.** The mechanism claims so far are the correction's size and its cosine with itself
across seeds. Nothing says what it is made of relative to the predictions PoE already has, and
nothing shows, in the picture, where the two experts disagree at the step the correction acts.

**The approach.** Linear algebra on the cached vectors, then the same projection applied to the
adapter's output, then the frames plan 06 already decoded, re-read as tracks with a speed and a
which-animal reading.

**Key insights:**
1. A three-vector span is the whole reach of any per-step guidance re-weighting. Projecting onto
   it is the cheapest possible upper bound on what a sampler-side method can add.
2. The adapter's fit cosine (0.8 to 0.9 in
   [the fit finding](../../../../report/does-the-fix-reach-unseen-pairs/is-the-held-out-gap-a-fit-a-drift-or-a-pair-problem.md))
   does not say which part it fits. Projecting the adapter's output the same way does.

## What happens (visual)

Navigation: ⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
 cached step k:  x_t,  ε_a  ε_b  ε_j  ε_∅            (fp16 on disk, fp32 in memory)
                        |    |    |    |
        span = { α ε_∅ + β(ε_a−ε_∅) + γ(ε_b−ε_∅) }      ε_PoE sits inside it
                        |
   r_t = ε̃_j − ε_PoE  ──┼──>  in-span part  ‖·‖²/‖r_t‖²  = in-span share      (rung 2)
                        └──>  orthogonal part             = orthogonal share
                                     ^
   δ̂ = PoE(adapter on) − PoE(off) ───┘  same projection, plus cos(δ̂_⊥, r_⊥)  (rung 3)

   x̂₀ under ε̃_a | ε̃_b | ε_PoE | ε̃_j, decoded, seed 15, seven steps          (rung 1)
   ‖x̂₀_a − x̂₀_b‖ per position, and where both experts act in the same place

   frames (plan 06) ──> DINOv2 ──> track per run: Σ‖Δz‖² against straight line  (rung 4)
                                                  cos(z, cat) − cos(z, dog) per step
```

## Description: what to build

Navigation: ⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The shared helper and the bar** `scripts/showcase/correction_span_common.py`: loads a cell's
   steps in fp32, builds the three span vectors, projects any vector onto the span by least
   squares, and holds the bar constants `ORTHO_SHARE_REWEIGHT_IMPOSSIBLE = 0.5`,
   `ORTHO_SHARE_REWEIGHT_CANDIDATE = 0.25` and `EARLY_STEPS = range(0, 11)`.
2. **Rung 2, the in-span share** `scripts/showcase/correction_span_share.py`: for every seed and
   step, the in-span and orthogonal share of r_t, the three least-squares coefficients, and the
   same shares for the joint guidance direction alone. Figure: orthogonal share of ‖r_t‖² against
   step, mean over the eight seeds with the min-to-max band, the early window shaded, the two bar
   lines drawn. Sidecar with every number and the verdict the constants produce.
3. **Rung 1, the experts' pictures** `scripts/showcase/correction_tweedie_pictures.py`: seed 15,
   steps 0, 2, 5, 10, 20, 30 and 49 (the last cached step; the frames' "step 50" is the finished
   latent, which the cache does not hold). The Tweedie estimate under each expert, under PoE,
   under the joint prompt and under the empty prompt, decoded. A strip of those pictures, a
   sixth row with the per-position map ‖x̂₀_a − x̂₀_b‖ over the four latent channels. Number, all
   eight seeds: the share of the 16,384 latent positions where both experts' departures from the
   unconditional estimate are above their own map's median, chance level 0.25, and the mean
   cosine between the two departures on those positions.
4. **Rung 3, the adapter's output** `scripts/showcase/correction_adapter_span_share.py`: the
   rank-32 adapter attached to the UNet, evaluated at every cached state for the three prompts
   "a cat", "a dog" and empty, adapter on and adapter off in the same process, so δ̂ is free of
   cross-run fp16 noise. δ̂ projected onto the same span as rung 2. Per step and seed: in-span
   and orthogonal share of δ̂, the cosine of its orthogonal part against r_t's orthogonal part,
   the overall cosine and norm ratio against r_t, and the cosine between the live and the cached
   PoE prediction as the sanity number. Figure: the two orthogonal-part cosines and shares
   against step; and for seed 15 the per-position norm map of δ̂ beside r_t's at the rung 1 steps.
5. **Rung 4, the tracks** `scripts/showcase/correction_track_energy.py`: from the stored frame
   embeddings, per run the kinetic energy, the path length, the straight-line distance and its
   floors, in the full 384-dimensional space and in the cloud-axes plane; the which-animal score
   per saved step with the number of sign changes after step 10 and the step of the last one.
   Two figures: kinetic energy against straight-line distance, one point per seed per condition;
   which-animal score against step, thin line per seed, thick mean per condition, seed 15 marked.
6. **The W&B run** `scripts/showcase/correction_log_wandb.py`: the four figures as images, the
   sidecars as one artifact, project `prime_lab/poe-repair-animals-compose`.

## Purpose and goal

Navigation: ⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

**Purpose.** Objective 2 of [the master plan](../../MASTER_PLAN.md) asks when the outcome is
decided; this plan says what the deciding signal is made of at those steps, and whether the
sampler could have produced it.

**Goals:**
1. The orthogonal-share curve with its verdict against the constants in source.
2. The seed-15 picture strip with the same-place share per step.
3. The adapter's shares and its orthogonal-part cosine against r_t, per step.
4. One kinetic-energy point and one which-animal curve per run, with the flip counted.

## Tasks

Navigation: ⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Paste: `/verify-plan @plans/05-when-does-the-outcome-lock-in/plans/tests/07-what-the-correction-is-made-of.md`
  - Done when: the report comes back clean, or its proposals have been applied.

▶ **Next: [task 1.1](#1--the-in-span-share-of-the-correction)**.

### 1. 📐 The in-span share of the correction

◀ **Needs: [task 0.1](#0--check-this-plan-before-working-from-it)**, and the bar written in
[the review file](../../review/07-what-the-correction-is-made-of.md).

- [x] **1.1** Write `correction_span_common.py` with the loader, the projection and the three bar
  constants, and `correction_span_share.py` over seeds 9 to 16, all 50 steps.
  - Done when: `artifacts/results/what-the-correction-is-made-of/orthogonal-share-over-steps.png`
    and `.json` exist, the sidecar carries 400 rows and a `verdict` field, and the constants are
    read from source, never from the plan.
- [x] **1.2** Read the verdict into the review file's deciding question.

▶ **Next: [task 2.1](#2--the-experts-pictures)**.

### 2. 🖼️ The experts' pictures

◀ **Needs: [task 1.1](#1--the-in-span-share-of-the-correction)**, so the loader exists.

- [x] **2.1** Write `correction_tweedie_pictures.py`; decode seed 15's estimates on the session
  node's GPU after checking `nvidia-smi` and `torch.cuda.is_available()`; compute the same-place
  share for all eight seeds on CPU.
  - Done when: `seed-15-experts-tweedie-strip.png` and `same-place-share.json` are filed, and
    the decoded PNGs sit under `/datasets/.../what_the_correction_is_made_of/tweedie/seed_15/`.

▶ **Next: [task 3.1](#3--the-adapters-output-projected-the-same-way)**.

### 3. 🔌 The adapter's output, projected the same way

◀ **Needs: [task 1.1](#1--the-in-span-share-of-the-correction)**, so the projection is the same.

- [x] **3.1** Write `correction_adapter_span_share.py`; launch it with `nohup` on the session
  node's device 0 and record node, device and PID in the review file's Runs table.
  - Done when: `adapter-span-share-over-steps.png`, `adapter-span-share.json` and
    `seed-15-adapter-vs-correction-norm-maps.png` are filed, 400 rows in the sidecar, and the
    live-against-cached PoE cosine is above 0.99 on every row (else the read is suspect and the
    review file says so).

▶ **Next: [task 4.1](#4--the-tracks-speed-and-animal)**.

### 4. 🛤️ The tracks' speed and animal

◀ **Needs:** nothing beyond the frame features file.

- [x] **4.1** Write `correction_track_energy.py` over `frames-dino-feats.npz`.
  - Done when: `track-kinetic-energy.png`, `which-animal-over-steps.png` and
    `track-energy-and-which-animal.json` are filed, one row per run (six conditions by eight seeds).

▶ **Next: [task 5.1](#5--log-and-file)**.

### 5. 🧾 Log and file

◀ **Needs: [tasks 1.1 to 4.1](#4--the-tracks-speed-and-animal)** done.

- [x] **5.1** Log the four figures and the sidecars to W&B with `correction_log_wandb.py`; write
  the run id into the review file.
- [x] **5.2** Write the card `artifacts/results/what-the-correction-is-made-of/README.md`, the
  figure explainer beside it, and the finding
  `report/when-does-the-outcome-lock-in/what-is-the-correction-made-of.md` with
  [the finding template](../../../../report/finding-and-explainer-template.md); add its row to
  `report/00-INDEX.md`.

▶ **Next: [instruction 6.1](#6--read-the-pictures)**.

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted.

- [ ] **Capture the failures this plan hit.**
  - Paste: `/ingest-error-pattern --from-run-log @plans/05-when-does-the-outcome-lock-in/plans/tests/07-what-the-correction-is-made-of.md`
  - Done when: each failure has a catalog entry, or there were none.
- [ ] **Bring the tree current.**
  - Paste: `/sync-plan-tree @plans/05-when-does-the-outcome-lock-in/plans/tests/07-what-the-correction-is-made-of.md — <one line>`
  - Done when: statuses match reality.

▶ **Next: what has to pass before this runs.**

## Instructions

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

**For you to follow manually.** Do these yourself.

### 6. 👁️ Read the pictures

◀ **Needs: [task 5.2](#5--log-and-file)** done.

6.1 **Open** `artifacts/results/what-the-correction-is-made-of/seed-15-experts-tweedie-strip.png`.
   - Expected result: five rows of decoded estimates (cat expert, dog expert, PoE, joint prompt,
     empty prompt) across seven steps, and a sixth row of heat maps.
   - ✅ If the cat-expert and dog-expert rows put their animal in the same place at steps 5 and
     10, and the heat map is brightest there, the same-place number has a picture behind it;
     write "agrees" in the review file's artefact check on the measuring tool.
   - ❌ If the two experts put their animals in different places and the heat map is bright
     where only one of them acts, the same-place share is reading a layout difference, not a
     disagreement; write that down and mark the number 🟡.

6.2 **Open** `which-animal-over-steps.png` and find seed 15's black line. Write into the review
   file the step at which it crosses zero, and whether that matches the frame strip in plan 06
   (a cat until step 10, a dog by step 40).

▶ **Next: what has to pass before this runs**, then the sampler-side sessions read rung 2.

## What has to pass before this runs

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **The bar is on rung 2 only.** Rungs 1, 3 and 4 are reads with numbers and no bar; they are
> reported whichever way they come out and nothing in them may be promoted to a verdict.

The statistic: the mean over seeds 9 to 16 of the per-seed mean, over steps 0 to 10 inclusive,
of the orthogonal share of ‖r_t‖². The three constants live in
`scripts/showcase/correction_span_common.py`.

- **Support** (PoE could not have supplied the correction by re-weighting): the statistic is
  above `ORTHO_SHARE_REWEIGHT_IMPOSSIBLE = 0.5`.
- **Null** (a per-step guidance re-weighting is a candidate fix): the statistic is below
  `ORTHO_SHARE_REWEIGHT_CANDIDATE = 0.25`.
- **Inconclusive**: between the two. Report the per-step curve and say so.
- **What would surprise:** the orthogonal share falling with step. The joint prompt's direction
  should become more its own as the image forms, not less.
- **Fail criteria (STOP and reassess):** the cached PoE prediction rebuilt from the raw vectors
  disagrees with the rule in [The rule the cache used](#the-rule-the-cache-used), or rung 3's
  live-against-cached cosine drops below 0.99, in which case the cache and the current build
  disagree and every number waits.

**When you get results, answer** [the review file](../../review/07-what-the-correction-is-made-of.md).

## Figure Catalog

Navigation: ⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

#### Pending: to be generated from prompts

| Item | Lane | Prompt file | What it shows | Save to |
|------|------|-------------|---------------|---------|
| (none) | | | | |

#### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|------|------|-------------|--------------|--------|---------|
| Orthogonal share over steps | — | y is the orthogonal share of ‖r_t‖² (0 to 1), x the denoising step 0 to 49; thick line the mean over seeds 9 to 16, band min to max; steps 0 to 10 shaded; the 0.5 and 0.25 bar lines; a second panel with the same for the joint guidance direction alone | `correction_span_share.py` | ✅ | `artifacts/results/what-the-correction-is-made-of/orthogonal-share-over-steps.png` + `.json` |
| Seed-15 experts' Tweedie strip | — | rows: cat expert, dog expert, PoE, joint prompt, empty prompt, then the heat map of ‖x̂₀_a − x̂₀_b‖ per latent position; columns: steps 0, 2, 5, 10, 20, 30, 49 | `correction_tweedie_pictures.py` | ✅ | `.../seed-15-experts-tweedie-strip.png` + `same-place-share.json` |
| Adapter span share over steps | — | y: the adapter output's orthogonal share, and the cosine of its orthogonal part against r_t's, against step; mean and band over seeds | `correction_adapter_span_share.py` | ✅ | `.../adapter-span-share-over-steps.png` + `.json` |
| Seed-15 norm maps, adapter beside correction | — | two rows of heat maps, ‖δ̂‖ and ‖r_t‖ per latent position, at the seven strip steps, one colour scale per row | `correction_adapter_span_share.py` | ✅ | `.../seed-15-adapter-vs-correction-norm-maps.png` |
| Track kinetic energy | — | y: sum of squared displacements of the embedded running estimate over the 13 saved segments, x: squared straight-line distance divided by 13 (the floor), one point per run, colour per condition, the diagonal drawn | `correction_track_energy.py` | ✅ | `.../track-kinetic-energy.png` + `track-energy-and-which-animal.json` |
| Which-animal over steps | — | y: cosine to the cat centroid minus cosine to the dog centroid, x: saved step; thin line per seed, thick mean per condition; seed 15 marked | `correction_track_energy.py` | ✅ | `.../which-animal-over-steps.png` |

#### Organization workflow

1. Run on the session node; 2. Read; 3. File into `artifacts/results/` with its card; 4. Link here.

## Orchestration: keeping catalogs and plan files in sync

Navigation: ⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/05-when-does-the-outcome-lock-in/plans/tests/07-what-the-correction-is-made-of.md` | **task 0.1** | conformance reported |
| Capture patterns | `/ingest-error-pattern --from-run-log @plans/05-when-does-the-outcome-lock-in/plans/tests/07-what-the-correction-is-made-of.md` | **the close out** | errors catalogued |
| Bring the tree current | `/sync-plan-tree @plans/05-when-does-the-outcome-lock-in/plans/tests/07-what-the-correction-is-made-of.md` | **the close out** | statuses match reality |
| Organize outputs | file under `artifacts/results/` + update Figure Catalog | after each rung | deliverables linked |

## Code references

Navigation: ⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

**File:** [scripts/build_training_cache.py](../../../../scripts/build_training_cache.py)
**Relevant section:** the step loop, lines 142 to 193: one four-prompt forward, the guided and
product-of-experts predictions, the DDIM step with ε_PoE, and the save of the four raw tensors.

**File:** [poe_repair/_sdxl/metrics.py](../../../../poe_repair/_sdxl/metrics.py)
**Relevant section:** `guided_eps`, `poe_eps`, `tweedie_mean`, `ddim_prev_from_x0_eps`, the four
closed forms quoted above.

**File:** [poe_repair/training_cache.py](../../../../poe_repair/training_cache.py)
**Relevant section:** `load_step_raw` (the fp32 upcast on load) and `delta_t_from_raw` (r_t).

**File:** [scripts/showcase/fit_cosine_on_cache.py](../../../../scripts/showcase/fit_cosine_on_cache.py)
**Relevant section:** how the adapter is attached and evaluated at cached states with a
three-prompt batch; rung 3 copies it and adds the adapter-off forward.

**File:** [scripts/showcase/where_each_condition_lands_frames_embed.py](../../../../scripts/showcase/where_each_condition_lands_frames_embed.py)
**Relevant section:** the cloud axes and the `frames-dino-feats.npz` layout rung 4 reads.

```python
# rung 2, per cached step, all in fp32
B = stack([eps_a - eps_u, eps_b - eps_u, eps_u]).reshape(3, -1).T     # (65536, 3)
r = g * (eps_j - eps_a - eps_b + eps_u).reshape(-1)                   # r_t
coef, *_ = lstsq(B, r); r_in = B @ coef; r_out = r - r_in
in_span = (r_in @ r_in) / (r @ r); ortho = 1 - in_span
```

## Next step

Navigation: ⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents)

The sampler-side sessions in [scope 06](../../../06-is-the-gap-the-samplers-or-the-models/MASTER_PLAN.md)
read the rung 2 verdict; [the grid and the figures](../figures/05-the-grid-and-the-figures.md)
reads the which-animal flip steps beside its own speciation numbers.

## Error Matrix

Navigation: ⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

<details>
<summary>None yet</summary>

#### From global catalog

(Patterns applicable across all projects)

#### From project catalog

(none yet)

**Auto-update note:** regenerated by `/sync-plan-tree`; do not edit manually.

</details>
