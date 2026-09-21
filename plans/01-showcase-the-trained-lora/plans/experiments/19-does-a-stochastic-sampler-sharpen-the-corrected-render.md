# 🧪 Does a stochastic sampler sharpen the corrected render without losing the two animals?

**This plan asks one question: if the 50-step DDIM sampler adds fresh noise at every step (eta above 0) instead of running deterministically (eta 0), does the rank-32 corrected render get crisper and nearer the joint-prompt image while still showing two animals?**

**Step 61 in the root running order. Waits on nothing: the checkpoint, the cache and the scorer exist. Sits beside [14-correct-early-then-clean-up](14-correct-early-then-clean-up.md), which varied when the correction acts; this plan varies how the sampler integrates it.**

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/01-showcase-the-trained-lora/plans/experiments/19-does-a-stochastic-sampler-sharpen-the-corrected-render.md — the 144 renders scored, the per-eta verdicts in the review file
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 53 (sibling) | [14-correct-early-then-clean-up](14-correct-early-then-clean-up.md) | the softness is committed by step 20; a λ schedule keeps the animals and sharpens on 6 or 7 of 8 seeds without reaching the plain-PoE mean |
| 57 (sibling) | [15-experiment-d-weight-decay](15-experiment-d-weight-decay.md) | the training-side fidelity fix: the rank-32 run again with weight decay 0.1 |
| **61 (current)** | **19: the stochastic sampler** | the sampler-side fidelity fix: the same adapter, the same seeds, eta 0, 0.5 and 1 on one shared noise path |
| 62 (next) | [20-an-ema-of-the-adapter-weights](20-an-ema-of-the-adapter-weights.md) | the other training-side fix: an averaged copy of the weights beside the raw ones |

---

## Table of contents
- [Position in the plan tree](#position-in-the-plan-tree)
- [Words this plan uses](#words-this-plan-uses)
- [Quick context: where you are](#quick-context-where-you-are)
- [Considerations](#considerations)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
- [What happens (visual)](#what-happens-visual)
- [Description: what to build](#description-what-to-build)
- [Purpose and goal](#purpose-and-goal)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [What has to pass before this runs](#what-has-to-pass-before-this-runs)
- [Figure Catalog](#figure-catalog)
- [Orchestration: keeping catalogs and plan files in sync](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Recommended skill](#recommended-skill)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

---

## Words this plan uses

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#quick-context-where-you-are) ➡️

- **The correction**: the rank-32 adapter at training step 30,050, held out on cat × dog. At each denoising step it changes the product-of-experts noise prediction; **λ** is the multiplier on that change, fixed here at 1.2, the shipped setting.
- **The corrected run**: λ 1.2 on every one of the 50 steps. "corrected" on every sheet and in every file.
- **eta**: the DDIM sampler's stochasticity (Song et al. 2020, equation 12). At eta 0 the update is deterministic and the render is a fixed function of the initial noise; this is the sampler every other result in this repo uses. At eta above 0 each step adds fresh Gaussian noise scaled by `sigma_t = eta · sqrt((1 − ā_prev)/(1 − ā_t)) · sqrt(1 − ā_t/ā_prev)`, and the deterministic part shrinks to keep the marginal at the next step correct. eta 1 is the ancestral sampler on the same 50-step grid.
- **The noise path**: the fresh-noise sample at every step of one seed, drawn from a generator seeded by `seed × 100003 + step`. Every condition and every eta of the same seed uses the same path, so two renders of one seed differ only through their noise predictions (and eta, which scales the same sample). This is what makes a same-seed comparison at eta above 0 mean what it means at eta 0.
- **Mono**: the joint prompt "a cat and a dog" under plain classifier-free guidance, the reference the drift is measured against, rendered at every eta on the same noise path.
- **Sharpness**: the Laplacian variance of the greyscale 1024 px render, the function plans 07 and 14 used. Higher is crisper. Three seeds (11, 14, 15) render as line drawings and have edge density far above the photographic seeds, so sharpness is only ever compared per seed, never as a mean against a band.
- **Seeds sharper**: of the 8 seeds, how many are sharper at this eta than the corrected run at eta 0 on the same seed.
- **Drift**: the DINOv2 cosine distance from a render to the Mono render of the same seed minus its distance to the plain-PoE render of the same seed, both references at the same eta on the same noise path. Negative means nearer the joint-prompt image. This is the pathwise (same-noise) form of the drift the decay finding reads; its scale is the same family but not the same instrument, so it is compared within this run only.
- **Both-ness**: the projection of a render's DINOv2 embedding toward the "a cat and a dog" cloud on the axes fitted for the landing finding; cat × dog only.
- **Compose count**: seeds of 8 where the validated detector counts at least two animal instances ([compose rate](../../../../context/world/compose-rate.md)).

---

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The experiment.** The corrected render is softer and greyer than the joint-prompt render on the same seed. Plan 14 found the softness is already in the running estimate by step 20 and that switching the correction off early does not fully recover it. On held-out cat × dog the correction fits the true term at cosine 0.80 overall and 0.75 late (the pressure-test route, part 5), so a third to a half of the injected energy each step points off target. At eta 0 every per-step error is carried to the final image (Nie et al. 2023, arXiv 2311.01410, prove the deterministic map carries a marginal mismatch to t = 0 while the stochastic one contracts it). This plan asks whether letting the sampler re-randomise the state each step, so the next score evaluation pulls it back toward the data manifold, removes the off-target part and returns contrast and detail.

**The hypothesis.** At eta 0.5 or 1 the corrected run keeps its compose count within one seed, is sharper than at eta 0 on most seeds, and its drift toward the joint-prompt render does not rise.

**If true.** The showcase wall renders at that eta, and the paper's fidelity caveat becomes a sampler setting rather than a property of the fix.

**If false.** Either the animals go (the noise pushes seeds back into the one-animal basin: the route's own prediction for plain PoE, since the SDE samples the product better and the product target is the chimera), or the render is no sharper. Then the softness is not accumulated integration error and the training-side plans (15 and 17) carry the question.

**Dataset.** Cat × dog, held-out seeds 9 to 16, the cached initial noise per seed; and the composing control pair butterfly × flower meadow on the same seeds, so an eta that breaks what already works is caught. The control pair's cache stops at seed 12; its seeds 13 to 16 take the cat × dog noise, as plan 14 does.

**Associated materials.**
- Review questions: [the review file](../../review/19-does-a-stochastic-sampler-sharpen-the-corrected-render.md)
- The mechanism argued from the literature: [the pressure-test route, part 3 and part 5](../../../../artifacts/ideas/improving-the-pooled-lora-run/routes/01-pressure-test-poe-failures.md)
- The softness read this builds on: [the plan 14 review](../../review/14-correct-early-then-clean-up.md)
- The sampler-side question this touches: [scope 06](../../../06-is-the-gap-the-samplers-or-the-models/MASTER_PLAN.md), which expects a stochastic sampler with no adapter to leave some one-animal basins and make more chimeras, not fewer

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Cost.** 3 etas × 3 conditions × 2 pairs × 8 seeds = 144 renders at 50 steps and 1024 px. Mono and plain PoE are one UNet pass per step, the corrected run two. On an RTX 3090 about 2 GPU-hours, plus 10 minutes of scoring. One Slurm job on `bigbatch`.

**Buys.** The review file's one question, and a sampler setting for the showcase wall if any eta is supported. It also records, for scope 06, whether plain PoE composes any more seeds with noise alone.

**Prerequisites.** The rank-32 checkpoint at step 30,050; the training cache's step-0 latents; the landing finding's cloud-axes feature file.

**W&B project:** `prime_lab/poe-repair-animals-compose`, run name `stochastic_sampler_sweep_r32_030050`. Output root: `/datasets/mmolefe/poe_repair_min/outputs/showcase/stochastic_sampler_sweep/`.

**What eta cannot separate.** A render that is sharper at eta 1 may be sharper because the noise contracted the off-target correction, or because eta 1 changes the plain sampler's texture on its own. The plain-PoE and Mono rows at each eta exist so the second effect is read on its own; only a gain the corrected run shows and the plain run does not is the adapter's.

**Known issues:** see the [Error Matrix](#error-matrix).

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- Every `biggpu` device is busy, so the render runs as a Slurm job on `bigbatch` (RTX 3090, 24 GB) with `co3` python ([nodes](../../../../environment/hpc/nodes.md)).
- Seven idle `bigbatch` nodes have a dead or unreachable GPU and are excluded on the `#SBATCH --exclude` line; the launcher's GPU guard aborts in under a second if a job lands on one anyway ([nodes](../../../../environment/hpc/nodes.md), [poe-launch-002](../../../../environment/known-failures.md)).
- Outputs on `/datasets` only, and the disk guard reads the filesystem the script writes to ([storage](../../../../environment/storage.md)).
- fp16 end to end; the identity check compares this sampler's plain PoE at eta 0 against the cached `poe.png` within a mean absolute grey-level bound, never by equality.
- DINOv2 scoring runs on the GPU; on the CPU it hits a CUDA-only xformers kernel ([poe-mem-002](../../../../environment/known-failures.md)).

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**A stochastic DDIM sampler, on the same seeds and the same noise path, gives the corrected run a crisper render nearer the joint-prompt image with the two animals kept.**

**Why this matters right now.** The showcase wall is built from the 30,050 checkpoint at eta 0. If eta alone recovers fidelity, the wall improves without a retrain, and the retrains (plans 15 and 20) are read for what they add on top.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** Every render in this repo is deterministic in its initial noise. The route's basin argument (failure b) and its exposure-bias argument (part 5) both say a deterministic sampler carries error to the final image, and nothing has tested the alternative on the adapter.

**The solution.** One sweep with eta as the only axis, the adapter fixed, the noise path shared, every condition rendered at every eta so the sampler's own effect and the adapter's are read apart.

**Key insights.**
1. Same seed is not enough at eta above 0. Two renders share a trajectory only if they share the per-step noise too, so the path is fixed per seed and reused by every condition.
2. The per-seed sharpness comparison replaces the band. The plain-PoE band spans 12 to 500 because three seeds are line drawings, which made plan 14's premise fire; a per-seed count is what the sketch seeds cannot swamp.
3. Plain PoE at eta above 0 is scope 06's question in miniature: if noise alone composes more seeds, the sampler-side story changes, and the number is recorded either way.

---

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
one seed, one shared noise path z_0 .. z_49

               eps prediction each step          DDIM update with eta
  Mono         CFG on "a cat and a dog"    ┐
  plain PoE    w(eps_a + eps_b) − eps_∅    ├──▶  x_{t-1} = √ā' x0 + √(1 − ā' − σ²) eps + σ z_k
  corrected    PoE + 1.2 · Δ̂ (adapter)    ┘           σ = eta · sqrt((1 − ā')/(1 − ā)) · sqrt(1 − ā/ā')

  eta 0      deterministic, the repo's sampler        every per-step error rides to the image
  eta 0.5    half-strength fresh noise                 errors partly contracted
  eta 1      ancestral on the 50-step grid             errors contracted most; texture re-drawn

  3 conditions × 3 etas × 8 seeds × 2 pairs = 144 renders
  read: compose count, sharpness per seed, drift to Mono at the same eta, both-ness
```

Every cell: the seed's cached initial noise, guidance 7.5, 1024 square, 50 steps, the adapter disabled on every frozen forward and on the sampler's exit.

---

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The stochastic step** (`stochastic_ddim_step` in `scripts/showcase/stochastic_sampler_sweep.py`): the DDIM update with eta in float32, reduced to the repo's `ddim_prev_from_x0_eps` at eta 0 (checked to zero difference on the SDXL DDIM schedule before launch).
2. **The shared noise path** (`noise_for`): one CPU generator per seed and step, so a render never depends on the order cells were rendered in.
3. **The identity check** (`identity_check`): plain PoE at eta 0 on cat × dog seed 9 against the cached `poe.png`, within `IDENTITY_MAX_MEAN_ABS_DIFF` grey levels, before the adapter is attached. A failure stops the render.
4. **The render** (`--render`): Mono at every eta before the adapter is attached, then plain PoE and the corrected run at every eta; resumable through `render_manifest.json`.
5. **The scoring** (`--score`): compose, sharpness, DINOv2 features, drift against the same-eta and the eta-0 references, both-ness, butterfly presence; the verdict per eta by the constants in source; `results.json`, `cell-table.md`, one sheet per pair and eta plus one across-eta sheet per pair, the three-panel figure, W&B.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves objective 3 (the softness question measured rather than argued) and goal 4's reading of experiment C. Checkable outcomes:

1. The identity check passing, so the new sampler is the old one at eta 0.
2. Every cell rendered, scored, on a sheet, in W&B, with `results.json` beside the sheets.
3. The review file's one question answered support, null or inconclusive per eta, by the constants in source.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/01-showcase-the-trained-lora/plans/experiments/19-does-a-stochastic-sampler-sharpen-the-corrected-render.md`
  - Done when: the report comes back clean, or its proposals have been applied.

▶ **Next: [task 1.1](#1--render-and-score-the-sweep)**.

### 1. 🧪 Render and score the sweep

◀ **Needs:** the checkpoint `phase1_r32_100k/checkpoints/lora_step_030050.pt`, the training cache, and the cloud-axes file `artifacts/results/where-does-each-condition-land/cat-x-dog-in-dino-space-dino-feats.npy` (all exist).

- [x] **1.1 Submit the job.** `sbatch scripts/showcase/stochastic_sampler_sweep.sbatch all` from the repo root. The job renders then scores in one process. Record the job id, node and W&B run id in the review file's Runs table.
  - Done when: `squeue -u mmolefe` shows the job running and `logs/stoch_sweep-<job>.out` prints the identity check line with `pass=True`.
- [ ] **1.2 Harvest.** When the job leaves the queue, read `logs/stoch_sweep-<job>.out`: the last lines print `plan verdict:` and the W&B url. If the render died mid-way, resubmit with `all`; the manifest skips finished cells.
  - Done when: `artifacts/results/does-a-stochastic-sampler-sharpen-the-corrected-render/results.json` exists with 144 rows and the sheets and figure sit beside it.
- [ ] **1.3 Write the card.** `README.md` in that results folder, one entry per figure and sheet, with how each was made.

▶ **Next: [instruction 2.1](#2--judge-the-cells)**.

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

**For you to follow manually.** Do these yourself.

### 2. 👁️ Judge the cells

◀ **Needs: [task 1.2](#1--render-and-score-the-sweep)** done.

2.1 **Open the across-eta sheet** `sheet-a_cat__x__a_dog-across-eta.png` under `artifacts/results/does-a-stochastic-sampler-sharpen-the-corrected-render/` (or the W&B run's Media tab). Rows are seeds 9 to 16; columns Mono at eta 0, then the corrected run at eta 0, 0.5 and 1. Write one word per seed and eta into the review file (crisper, softer, same, animals lost) before reading the numbers.

2.2 **Open the per-eta sheets** `sheet-a_cat__x__a_dog-eta0.5.png` and `-eta1.png`. Columns Mono, plain PoE, corrected at eta 0, corrected at this eta, all four at the same eta. ✅ the last column has two animals on the rows the third does and looks nearer the first; ❌ animals vanish or the last column is as soft as the third.

2.3 **Check the numbers agree with the eye**: `cell-table.md` beside the sheets has compose count, seeds sharper, drift and verdict per cell. Then write the verdict into the [review file](../../review/19-does-a-stochastic-sampler-sharpen-the-corrected-render.md).

2.4 **Read the control sheet** `sheet-a_butterfly__x__a_flower_meadow-eta1.png`: the butterfly present on the same rows as at eta 0.

▶ **Next: what has to pass before this runs.**

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

**Pass criteria:**
- The identity check passes: plain PoE at eta 0 through the new step reproduces the cached `poe.png` within `IDENTITY_MAX_MEAN_ABS_DIFF` grey levels. The render stops if it does not.
- The step reduces to `ddim_prev_from_x0_eps` at eta 0 (checked on the SDXL DDIM schedule, maximum absolute difference 0.0).

**Fail criteria:**
- A cell's Mono render at eta above 0 composes fewer than 6 of 8 seeds on cat × dog: the reference itself moved, and the drift at that eta is not a fidelity read.

**When you get results, answer the questions in the [review file](../../review/19-does-a-stochastic-sampler-sharpen-the-corrected-render.md).**

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

Every figure below lands in `artifacts/results/does-a-stochastic-sampler-sharpen-the-corrected-render/`, with its card entry in that folder's `README.md`.

| Figure | What is plotted | What it argues | What it may not claim | File |
|---|---|---|---|---|
| sharpness, drift and compose against eta | three panels, x eta in each; left y Laplacian variance at 1024 px on a log axis, thin line per seed, thick mean, corrected and plain PoE; middle y DINOv2 drift to Mono at the same eta, same lines; right y compose count of 8 for Mono, plain PoE, corrected | whether the corrected run gets sharper and nearer Mono as eta rises, and whether plain PoE does on its own | anything beyond cat × dog seeds 9 to 16 at this checkpoint and λ | `sharpness-drift-and-compose-vs-eta.png` |
| one sheet per pair and eta | rows seeds 9 to 16; columns Mono, plain PoE, corrected at eta 0, corrected at this eta, all at the same eta; count, sharpness and drift under each tile | what the numbers are counting | fidelity beyond what the eye sees at sheet size | `sheet-<pair>-eta<eta>.png` |
| the across-eta sheet, one per pair | rows seeds; columns Mono at eta 0, corrected at eta 0, 0.5, 1 | the same seed and noise path as eta rises | the plain sampler's own change with eta (that is the per-eta sheet's job) | `sheet-<pair>-across-eta.png` |
| the cell table | one row per cell: eta, compose or presence count, mean sharpness, seeds sharper than eta 0, drift, both-ness, verdict | which eta, if any, is supported | anything about pairs other than cat × dog and the control | `cell-table.md`, `results.json` |

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Extract errors | `/ingest-error-pattern --from-run-log` | task 3.1 | new patterns into the catalogs |
| Close out | `/sync-plan-tree` | task 3.2 | statuses aggregated up |

### 3. 🧹 Close out

◀ **Needs: [instruction 2.3](#2--judge-the-cells)** done.

- [ ] **3.1 Run the following prompt: `/ingest-error-pattern --from-run-log`** (after any red run).
- [ ] **3.2 Run the following prompt: `/sync-plan-tree plans/01-showcase-the-trained-lora/`**

▶ **Next: [05-assemble-the-showcase-figures](../figures/05-assemble-the-showcase-figures.md)**, which takes a supported eta as the wall's sampler setting.

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**File:** `scripts/showcase/stochastic_sampler_sweep.py`: the stochastic step, the noise path, the identity check, the render, the scoring and the constants. `scripts/showcase/stochastic_sampler_sweep.sbatch`: the `bigbatch` launcher with the disk, python, GPU and CUDA guards and the faulted-node exclusions.

**Reused:** `Sampler` (the SDXL context, prompts, pinned noise and adapter attach), `_cloud_axes`, `_write`, `mean_abs_diff` and `_laplacian_var` from `scripts/showcase/correct_early_then_clean_up.py`; `guided_eps`, `poe_eps` and `ddim_prev_from_x0_eps` from `poe_repair/_sdxl/metrics.py`; the instance-count scorer; `DinoEmbedder`.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

```
/run-experiment plans/01-showcase-the-trained-lora/plans/experiments/19-does-a-stochastic-sampler-sharpen-the-corrected-render.md — one Slurm job on bigbatch, render then score, harvest from the log
```

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

A supported eta becomes the showcase wall's sampler setting in [05-assemble-the-showcase-figures](../figures/05-assemble-the-showcase-figures.md) and a sentence in the paper's method. A null bounds the fidelity caveat to the adapter itself and hands the question to [15-experiment-d-weight-decay](15-experiment-d-weight-decay.md) and [20-an-ema-of-the-adapter-weights](20-an-ema-of-the-adapter-weights.md). Either way the plain-PoE row at eta 1 is carried to scope 06 as its free sampler-side read.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

### From project catalog

From [environment/known-failures.md](../../../../environment/known-failures.md):

- **poe-lora-003**: every condition runs through the identical sampler, differing only in the noise prediction. Mono is the reference column, never a compared cell.
- **poe-lora-004**: the sampler carries `@torch.no_grad()`.
- **poe-mem-002**: the DINOv2 embedder runs on the GPU inside the same job.
- **poe-launch-002**: the launcher asserts `torch.cuda.is_available()` and excludes the `bigbatch` nodes whose GPU is dead or unreachable; jobs 50334 and 50335 died on `mscluster65` in under a second before the exclusion list was extended.
- **The windowed sampler leaves the adapter attached** (memory note, 2026-09-05): Mono is rendered before the adapter is attached, and every frozen forward disables it first.
