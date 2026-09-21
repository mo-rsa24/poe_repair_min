# 🧪 Search the noise in the commit window: does choosing the injected noise where the corrected run commits give two animals at plain-PoE crispness?

**This plan asks one question: when the corrected sampler runs stochastically and the noise it injects at each step of the commit window is chosen by a small greedy search on a fidelity reward, is the finished image sharper than the same run with unsearched noise, with the two animals kept?**

**Step 58 in the root running order. Waits on nothing: the checkpoint, the pinned noise, the validated detector, ImageReward and the landing finding's axes are all on disk. Sits beside [14-correct-early-then-clean-up](14-correct-early-then-clean-up.md), which varies when the correction acts; this plan varies the other free variable of the run, the noise.**

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/01-showcase-the-trained-lora/plans/experiments/16-search-the-noise-in-the-commit-window.md — the run rendered and scored on both pairs; verdict in the review file; the sheet, strip and four figures filed
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 53 (sibling) | [14-correct-early-then-clean-up](14-correct-early-then-clean-up.md) | λ schedules and a plain-PoE tail against the full window; found the softness committed by step 20 |
| 55, 56 (cousins) | [feynman-kac steering on plain PoE](../../../06-is-the-gap-the-samplers-or-the-models/plans/baselines/10-feynman-kac-steering-on-a-detector-reward.md), [on the correction](../../../06-is-the-gap-the-samplers-or-the-models/plans/baselines/13-feynman-kac-steering-on-top-of-the-rank-32-correction.md) | selection among particles on the compose count; null on plain PoE, the adapter version in flight |
| **58 (current)** | **16: search the noise in the commit window** | **per-step noise search on the corrected sampler, fidelity reward, judged on sharpness with the animals kept** |
| 35 (next in scope) | [05-assemble-the-showcase-figures](../figures/05-assemble-the-showcase-figures.md) | the wall; a supported cell here becomes a row on it |

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

- **The correction**: the rank-32 adapter at training step 30050, held out on cat × dog. At each denoising step it changes the product-of-experts (PoE) noise prediction; **λ** is the multiplier on that change, 1.2 here on every step, the shipped setting.
- **eta**: the stochasticity of the DDIM sampler. At eta 0 the run is deterministic and a starting noise is an image. At eta 1 the sampler injects a fresh Gaussian draw at every step, so the run has 50 more free variables than the starting noise.
- **The injected noise**: that per-step draw. The **control** takes it from one seeded stream and never chooses. The **searched run** takes the same stream, and inside the commit window treats each draw as the **pivot** of a search.
- **The commit window**: step indices 8 to 25 at 50 steps (timestep 821 down to 481). The corrected run commits to one animal or two at a median step of 15 (range 8 to 35), and its running estimate is already smoother than every plain seed by step 20 ([plan 14's review](../../review/14-correct-early-then-clean-up.md)). Ramesh and Mardani find that local search pays at the intermediate steps where the image de-mixes, and fresh draws win at the extremes, so the search is confined to this window.
- **A candidate**: a proposed injected noise for the current step. **Local**: the pivot nudged, `(pivot + 0.3 u) / sqrt(1 + 0.09)` with `u` a fresh Gaussian, cosine about 0.96 to the pivot (the same σ the initial-noise search used). **Fresh**: an independent Gaussian draw, chosen with probability 0.25. A step runs 3 rounds of 3 candidates beside the pivot; the best of a round becomes the next pivot; the final pivot is injected. Ten corrected forward passes per searched step.
- **The reward**: judged on the running estimate one step ahead, decoded through the VAE. It is the validated instance count clipped at 2, plus half a sigmoid of ImageReward (a learned human-preference score of an image against its prompt, here "a cat and a dog"). The count dominates, so a candidate that loses an animal never beats one that keeps two; among two-animal candidates the higher preference wins. This is plan 11's reward, unchanged.
- **The running estimate**: the model's guess of the finished image at a step (the Tweedie mean), decoded. Saved at 16 steps for every condition at 256 px.
- **Sharpness**: Laplacian variance of the greyscale render, the function plan 07 and plan 14 used. It counts edges, so a line drawing outranks a photograph; the judged read is therefore **paired per seed** (searched against control on the same seed) rather than a band, because the sketch seeds (11, 14, 15) make any band useless.
- **Both-ness** and **which-animal**: a render's DINOv2 embedding projected onto the landing finding's two axes: from the cat-alone centroid to the dog-alone centroid (x), and from their midpoint toward the "a cat and a dog" centroid (y). Cat × dog only.
- **Distance to Mono**: one minus the cosine between a render's DINOv2 embedding and the seed's joint-prompt render's. Lower is nearer the image the joint prompt paints.
- **Compose count**: seeds of 8 where the validated detector counts at least two animal instances ([compose rate](../../../../context/world/compose-rate.md)). On the control pair the rule cannot apply (a meadow is not an animal); the control read is whether a butterfly box is present at confidence 0.30 or more.

---

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The experiment.** The corrected run composes 7 of 8 held-out seeds and comes out soft: at step 20 its running estimate has a Laplacian variance of 11.9 against a plain-PoE band that starts at 25.9, and the softness is in the committed structure, so a clean tail cannot reach it. Plan 14 asked whether switching the correction off earlier helps; this plan leaves the correction on and asks whether the noise injected while the structure commits can be chosen better. Ramesh and Mardani ([arXiv 2506.03164](https://arxiv.org/abs/2506.03164)) show that for a fixed model the per-step noise of a stochastic sampler is worth searching, with a greedy contextual-bandit search matching tree search, and that the intermediate steps are where local search pays.

**The hypothesis.** With the composition settled by the correction, a small search over the injected noise in steps 8 to 25 on a count-gated fidelity reward finds a sharper committed structure without losing an animal.

**If true.** The searched render is sharper than the unsearched control on at least 6 of 8 seeds and the compose count stays within one seed. The showcase wall gets a "noise searched" row, and the paper's fidelity caveat becomes a sampling-cost statement: this many extra forward passes buys this much crispness.

**If false.** Either the animals go (the reward's count gate failed to hold them, or the search moved the run out of the corrected basin), or the sharpness is a coin flip across seeds; then the softness is not a property of the noise realisation, and the correction itself is what remains to change.

**Dataset.** Cat × dog, held-out seeds 9 to 16, the cached initial noise per seed; and the composing control pair butterfly × flower meadow on the same seeds, so a search that breaks what already works is caught. The control pair's cache stops at seed 12, so its seeds 13 to 16 take the same per-seed formula the cache used (checked byte-identical on seeds 9 and 13 by plan 10).

**Associated materials.**
- Review questions: [the review file](../../review/16-search-the-noise-in-the-commit-window.md)
- The paper's note: [the tweet](https://x.com/mardanimorteza/status/1939493270612230237) and [arXiv 2506.03164](https://arxiv.org/abs/2506.03164)
- The three selection-only nulls this does not repeat: [initial-noise search](../../../../report/is-the-gap-the-samplers-or-the-models/does-searching-over-the-initial-noise-compose.md), [twisted SMC](../../../../report/is-the-gap-the-samplers-or-the-models/does-selecting-among-poe-proposals-compose.md), plan 10's Feynman-Kac steering
- Where the softness sits: [plan 14's review](../../review/14-correct-early-then-clean-up.md)

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Cost.** Per seed: two references (Mono, plain PoE), the shipped render at eta 0, the control at eta 1, and the searched run. A corrected 50-step render is about 90 s on an RTX 3090 (two UNet passes over three branches per step). The search adds 18 steps × 10 evaluations, each a corrected forward pass, a VAE decode at 1024 px, one detector call and one ImageReward call, about 3 s together, so about 9 min per seed. Per seed about 15 min; 16 seeds about 4 GPU-hours on a 3090, less on an A6000. Scoring and the figures are inside the same job.

**Buys.** The one review question, four figures, and one row on the showcase wall if supported.

**Prerequisites.** The rank-32 checkpoint at step 30050; the training cache's step-0 latents; the landing finding's 48 × 384 DINOv2 features (`artifacts/results/where-does-each-condition-land/cat-x-dog-in-dino-space-dino-feats.npy`); ImageReward's weights in `~/.cache/ImageReward`; GroundingDINO in the Hugging Face cache.

**This is not a fourth selection-only baseline.** The three selection-only runs asked whether plain PoE proposes a composing state to select; this plan puts the correction in the proposal and asks a fidelity question about a state already composing. A support here says nothing about plain PoE.

**The reward and the judge are different instruments.** The search maximises count plus ImageReward on the estimate one step ahead. The verdict reads paired sharpness at 1024 px, the compose count, both-ness and the distance to Mono on the finished image. ImageReward is reported but never judged on, because the search optimised it.

**Why eta 1 needs its own control.** Turning eta from 0 to 1 changes the run on its own (plan 10's controls composed differently from the eta-0 renders). The shipped eta-0 render is on the sheet as a reference; the judged comparison is searched against unsearched at eta 1 with the same noise stream, so exactly one thing differs: the draws chosen inside the window.

**W&B project:** `prime_lab/poe-repair-animals-compose`, group `noise-trajectory-search`. Output root: `/datasets/mmolefe/poe_repair_min/outputs/showcase/noise_trajectory_search/<run id>/`.

**Known issues:** see the [Error Matrix](#error-matrix).

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- The launch order: an allocated node, then an idle `biggpu` node, then a shared `biggpu` device over SSH with `nohup`, then `bigbatch` through `sbatch` ([execution protocol](../../../../environment/hpc/execution-protocol.md)). On 2026-09-06 every `biggpu` device carried a process, so this plan's runs went to `bigbatch` RTX 3090 nodes by `sbatch --nodelist`.
- `bigbatch` nodes that read idle with a faulted GPU: mscluster44, 45, 50, 51, 65 ([nodes](../../../../environment/hpc/nodes.md), `poe-launch-002`). Nodes 52, 53, 54 and 60 passed plan 11's smoke; name one of those.
- `co3` python on the RTX 3090, RTX 8000 and A6000 nodes; `co3_bw` on 110 to 112. The launcher picks by hostname.
- The launcher's guards run inside the job: disk at or over 90% aborts, a device with a foreign running process is refused, a device reading `[N/A]` is refused, `torch.cuda.is_available()` is asserted after pinning.
- Outputs on `/datasets` only.
- fp16 end to end; the searched and control runs are compared on the same device in the same process.
- Mono and plain PoE are rendered for every seed before the adapter is attached, and the corrected forward disables the adapter on exit, because a windowed sampler once left it enabled and contaminated the references (2026-09-05).

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**Choosing the injected noise in the commit window, on a count-gated fidelity reward, gives the corrected sampler a crisper finished image on most seeds while keeping the two animals.**

**Why this matters right now.** The showcase's honest caveat is that the fix costs fidelity. If the caveat can be bought back with inference compute on the same adapter, the paper's story is "small correction plus a little search" rather than "a correction that blurs".

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** Plan 07 varied how much correction, plan 14 varied when it acts. The run's third free variable, the noise realisation, has only been searched at step 0 on plain PoE, where nothing composes and there is nothing to pick. Inside a run that does compose, the per-step noise has never been touched.

**The solution.** Run the shipped sampler stochastically with a seeded noise stream (the control), then run it again with the same stream and a greedy search over each draw in the commit window, judged one step ahead on a reward that cannot trade an animal for polish. Score both finals the same way, paired per seed.

**Key insights.**
1. At eta 1 the run has a free variable at every step, and the paper shows those variables matter most where the image de-mixes; here that is where the softness commits.
2. The count gate keeps the search inside the corrected basin: the search can only move among two-animal futures once the detector sees two.
3. Pairing per seed cancels the sketch seeds that make any sharpness band useless, so the read is about the noise and not about seed character.

---

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
denoising step k (50 DDIM):   0 ..... 8 .......... 25 ..... 50
timestep t:                 981     821          481       1

mono_eta0      joint prompt, plain CFG, eta 0                      (reference)
poe_eta0       plain PoE, eta 0                                    (reference)
adapter_eta0   PoE + 1.2 x correction, eta 0                       (the shipped render)
control_eta1   PoE + 1.2 x correction, eta 1: z_k from one seeded stream, every step
search_eta1    same, same stream, except k in 8..25:
                   pivot = z_k
                   3 rounds x 3 candidates (fresh with p 0.25, else pivot nudged at σ 0.3)
                   each judged on x̂0 one step ahead: min(count, 2) + 0.5 sigmoid(ImageReward)
                   best of a round -> next pivot; final pivot injected
```

Every condition starts from the seed's cached initial noise, guidance 7.5, 1024 square. The search adds ten corrected forward passes per searched step and changes nothing else.

---

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The sampler and the search** (`poe_repair/experiments/noise_trajectory_search/search.py`): one trajectory of the corrected sampler at any eta; with a reward, the greedy search at every step in `SEARCH_STEPS`; the running estimates saved at `FRAME_STEPS`; every constant the review file judges against; the verdict rules in source.
2. **The runner** (`poe_repair/experiments/noise_trajectory_search/run.py`): the five conditions per pair and seed, scored (instance count and per-query confidences, ImageReward, sharpness, both-ness, distance to Mono), the running estimates embedded and projected, `cell.json` per seed so a killed job resumes, then per pair the sheet, the three-column strip, four figures, `summary_<pair>.json`, and `summary.json` and `verdict.json` at the end; W&B images, scalars, a per-step table and an artifact.
3. **The launcher** (`scripts/noise_trajectory_search/run_noise_trajectory_search.sh`) with the shared-device guards, and its Slurm wrapper (`noise_trajectory_search.sbatch`) for the `bigbatch` fallback.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves objective 3 (the softness question measured rather than argued). Checkable outcomes:

1. The smoke run finishes with every output present and the search recorded per step.
2. Both pairs rendered, scored, on a sheet, in W&B, with `summary.json` beside the sheets.
3. The four figures and the strip filed under `artifacts/results/` with their card.
4. The review file's one question answered support, null or inconclusive, by the constants in source.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/01-showcase-the-trained-lora/plans/experiments/16-search-the-noise-in-the-commit-window.md`
  - Done when: the report comes back clean, or its proposals have been applied.

▶ **Next: [task 1.1](#1--smoke-the-search-on-one-seed)**.

### 1. 🔬 Smoke the search on one seed

◀ **Needs:** a node chosen per the launch order above.

- [x] **1.1 Run the smoke mode** (cat × dog seed 9, 10 steps, 512², search at steps 2 to 5, one round of 2).
  - Command: `sbatch --nodelist=mscluster53 scripts/noise_trajectory_search/noise_trajectory_search.sbatch smoke`, or over SSH `ssh <node> 'GPU=<idx> nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/noise_trajectory_search/run_noise_trajectory_search.sh smoke > /datasets/mmolefe/poe_repair_min/outputs/showcase/noise_trajectory_search/logs/smoke.log 2>&1 &'`
  - Done when: the run directory holds five PNGs for seed 9, `search_eta1/run.json` with four searched steps, and the sheet, strip and figures draw without error.

▶ **Next: [task 2.1](#2--render-score-and-draw-the-full-run)**.

### 2. 🧪 Render, score and draw the full run

◀ **Needs: [task 1.1](#1--smoke-the-search-on-one-seed)** passed.

- [x] **2.1 Launch the full run** on both pairs, seeds 9 to 16.
  - Command: `sbatch --nodelist=<mscluster52|53|54|60> scripts/noise_trajectory_search/noise_trajectory_search.sbatch full`
  - Record node, job id, PID and the W&B run id in the review file's Runs table.
  - Done when: `verdict.json` exists under the run directory and the log ends with `[done]`.
- [x] **2.2 Harvest.** On the launch node: `grep -E "seed [0-9]+ done|verdict|Traceback" logs/nts-<job>.out`; count `cell.json` files (16 expected).

▶ **Next: [task 3.1](#3--file-the-evidence)**.

### 3. 🗂️ File the evidence

◀ **Needs: [task 2.2](#2--render-score-and-draw-the-full-run)** done.

- [x] **3.1 Copy the sheet, strip, the four figures, `summary.json` and `verdict.json`** into `artifacts/results/does-searching-the-noise-in-the-commit-window-sharpen-the-fix/` and write the card (`README.md`) with an entry per item.
- [x] **3.2 Answer the review file**, then the close-out in [section 6](#6--close-out).

▶ **Next: [instruction 5.1](#5--judge-the-sheet)**.

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

**For you to follow manually.** Do these yourself.

### 5. 👁️ Judge the sheet

◀ **Needs: [task 3.1](#3--file-the-evidence)** done.

5.1 **Open the sheet** `sheet_a_cat__x__a_dog.png` under the run directory or in the W&B run's Media tab. Rows are seeds 9 to 16; columns Mono, plain PoE, the shipped eta-0 render, the eta-1 control, the searched run. Under each tile: the detector's count (green at 2 or more), sharpness, ImageReward, both-ness and the distance to Mono. ✅ the searched column shows two animals on the rows the control does and looks crisper on most rows; ❌ animals vanish, or the searched column is no crisper than the control.

5.2 **Open `paired_per_seed_a_cat__x__a_dog.png`**: one line per seed from the control to the searched run, green where the searched run is better. Count the green lines in the sharpness panel and check the count matches `summary_a_cat__x__a_dog.json`, field `paired_search_vs_control.sharper_seeds`. Then write the verdict into the [review file](../../review/16-search-the-noise-in-the-commit-window.md).

▶ **Next: what has to pass before this runs.**

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

**Pass criteria:**
- The smoke run finishes with every output present (task 1.1).
- Inside the window the search actually evaluates: `search_eta1/run.json` has `evaluations` equal to `len(SEARCH_STEPS) × (1 + NUM_ROUNDS × NUM_CANDIDATES)`. A search that never evaluated is the control wearing another name.

**Fail criteria:**
- The plain-PoE render through the new sampler at λ 0 differs from the seed's existing plain render (`figure_r32_030050/renders/full/`) by more than the fp16 band: the sampler is wrong, and nothing else is read.

**When you get results, answer the questions in the [review file](../../review/16-search-the-noise-in-the-commit-window.md).**

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

Every figure below lands in `artifacts/results/does-searching-the-noise-in-the-commit-window-sharpen-the-fix/`, with its card entry in that folder's `README.md`.

| Figure | What is plotted | What it argues | What it may not claim | File |
|---|---|---|---|---|
| the sheet | rows seeds 9 to 16; columns Mono, plain PoE, the shipped eta-0 render, the eta-1 control, the searched run; under each tile the count, sharpness at 1024 px, ImageReward, both-ness and distance to Mono | what the numbers are counting, and whether the eye agrees | fidelity beyond sheet size | `sheet_<pair>.png` |
| the strip | rows seeds; columns Mono, plain PoE, the searched run | the paper's three-way picture: what the joint prompt paints, what PoE paints, what the corrected and searched sampler paints | anything about the control; the strip has no unsearched column | `strip_<pair>.png` |
| tracks in the DINOv2 plane | x which-animal, y both-ness; one track per seed through the 16 saved running estimates; four panels (plain PoE, shipped, control, searched); faint reference clouds from the landing finding; the searched steps drawn thick | whether the search moves the run within the corrected basin or out of it | pixel fidelity; the plane is semantic | `tracks_in_dino_plane_a_cat__x__a_dog.png` |
| over steps | left: both-ness of the running estimate against step; right: sharpness of the 256 px running estimate against step (log); thin per seed, thick mean, the window shaded | when the searched run separates from the control | the 256 px scale is not the 1024 px scale | `over_steps_<pair>.png` |
| reward over steps | left: reward of the control's draw against the chosen draw at every searched step; right: which kind of candidate won per step (pivot, local, fresh) | whether the search found anything, and where in the window | that the reward gain is a fidelity gain; that is the paired figure's job | `reward_over_steps_<pair>.png` |
| paired per seed | one line per seed from control to searched: sharpness (log), both-ness, ImageReward, distance to Mono; green where better | the verdict's count of sharper seeds, seen | the ImageReward panel is the reward, not a judge | `paired_per_seed_<pair>.png` |

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Extract errors | `/ingest-error-pattern --from-run-log` | task 6.1 | new patterns into the catalogs |
| Close out | `/sync-plan-tree` | task 6.2 | statuses aggregated up |

### 6. 🧹 Close out

◀ **Needs: [instruction 5.2](#5--judge-the-sheet)** done.

- [ ] **6.1 Run the following prompt: `/ingest-error-pattern --from-run-log`** (after any red run).
- [ ] **6.2 Run the following prompt: `/sync-plan-tree plans/01-showcase-the-trained-lora/`**

▶ **Next: [05-assemble-the-showcase-figures](../figures/05-assemble-the-showcase-figures.md)**, which takes a supported cell as a wall row.

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**Files:** `poe_repair/experiments/noise_trajectory_search/search.py` (the sampler, the search, the constants, the verdict rules), `poe_repair/experiments/noise_trajectory_search/run.py` (the runner, scoring, sheets, figures, W&B), `scripts/noise_trajectory_search/run_noise_trajectory_search.sh` (the guarded launcher), `scripts/noise_trajectory_search/noise_trajectory_search.sbatch` (the Slurm wrapper).

**Reused:** the corrected three-branch forward and adapter attach from `poe_repair/experiments/fk_steering/` (plans 10 and 11), the DDIM step coefficients and Mono sampler from `poe_repair/experiments/twisted_smc/sampler.py`, the instance-count scorer, `DinoEmbedder`, the landing finding's saved features, ImageReward as plan 11 wired it.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

```
/run-experiment plans/01-showcase-the-trained-lora/plans/experiments/16-search-the-noise-in-the-commit-window.md — smoke first, then the full run on one node, then file the evidence
```

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[05-assemble-the-showcase-figures](../figures/05-assemble-the-showcase-figures.md) takes a supported cell as a row; a null goes to the paper's fidelity caveat with the sampling cost stated.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

### From project catalog

From [environment/known-failures.md](../../../../environment/known-failures.md):

- **poe-lora-003**: every compared condition runs through the identical sampler, differing only in eta and in the draws chosen inside the window. Mono is a reference column, never a compared cell.
- **poe-lora-004**: the sampler carries `@torch.no_grad()`.
- **poe-mem-002**: DINOv2 runs on the GPU; on CPU it hits the CUDA-only xformers kernel.
- **poe-launch-002**: the launcher refuses a device reading `[N/A]` or `[GPU requires reset]`, and asserts `torch.cuda.is_available()`; on `bigbatch`, mscluster44, 45, 50, 51 and 65 read idle with a faulted GPU.
- **The windowed sampler leaves the adapter attached** (2026-09-05): Mono and plain PoE are rendered before the adapter is attached, and the corrected forward disables the adapter on exit.
- **Run-id collision** (plan 11, 2026-09-06): three Slurm jobs starting in the same second shared a run directory. One job per submission here; the run id carries the timestamp and the log carries node and PID.
