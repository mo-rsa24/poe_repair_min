# 🧪 Fitting the whole path and handing back the tail

**The adapter composes and its pictures are hazy. The adapter's job at each step of drawing is to push the image by a certain amount in a certain direction. Measured against what it should have done, the push stays close to full size all the way through and points about a quarter of the way off target in the final steps, which is where fine detail is drawn. This plan tries the free fix that follows from that, on checkpoints that already exist, and trains only if the free fix falls short.**

**Step 76 in the root running order. Waits on nothing: its first two stages read checkpoints that are already on disk. It is the only plan in this scope aimed at fidelity rather than composition.**

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/09-designing-the-correction-loss/plans/hypothesis/06-fitting-the-whole-path-and-handing-back-the-tail.md — the shrinkage check, the hand-off sweep, and whether stage B was needed
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 74 | [04: freezing the empty branch](04-freezing-the-empty-branch.md) | the baseline and `04`, launched |
| 75 | [05: training on the renders that composed](05-training-on-the-renders-that-composed.md) | variation `06`, a different target |
| **76 (current)** | **06: fitting the whole path and handing back the tail** | variation `08`, the fidelity question |

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
- [Tasks](#tasks)
- [Instructions](#instructions)
- [What has to pass before this runs](#what-has-to-pass-before-this-runs)
- [Figure Catalog](#figure-catalog)
- [Code references](#code-references)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

---

## Words this plan uses

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#quick-context-where-you-are) ➡️

- **The correction, `r`**: the difference between the joint prompt's prediction and the plain product-of-experts prediction, `eps_J - eps_PoE`. What the adapter is trying to supply.
- **What the adapter actually supplies, `r_hat`**: `eps_poe_lora - eps_poe_frozen`, already formed in the trainer at `trainer.py:588` for the energy penalty.
- **How big the push is**: `‖r_hat‖ / ‖r‖`. 1.0 means the adapter applies exactly as much as the picture needed.
- **Whether the push points the right way**: the cosine between `r_hat` and `r`. 1.0 means exactly the right direction, 0 means unrelated. This is the one that fails late, and it is the cause of the haze.
- **The hand-off step `k`**: the denoising step after which the adapter is detached and the frozen model draws the rest. `k = 50` is the adapter on the whole path.
- **x0 loss space**: scoring the error on the predicted clean image rather than on the predicted noise. The switch is `--loss-space x0`, with a cap, and no run in this lineage has used it: every `config.json` here reads `loss_space: eps`.

---

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The hypothesis, stated so it can fail**

*The haze is a full-size push pointing off target in the final steps. Detaching the adapter for the last stretch of denoising brings each render at least 0.05 nearer its own joint-prompt target in the compose scorer's DINOv2 embedding, on five or more of the eight held-out seeds, without losing the second animal.*

**Why it can fail**

The off-target push may be measured late and still not be what a person sees as haze, in which case detaching the adapter at the end changes the numbers and not the pictures. Or the picture may already be committed to a soft rendering by the pushes applied early, which the original model would then finish faithfully, soft and all. Either outcome leaves the haze unexplained and the measurement standing.

**What is already known, so this does not re-run it**

[Can a corrector or a clean tail sharpen the adapter's renders](../../../../report/is-the-gap-the-samplers-or-the-models/can-a-corrector-or-a-clean-tail-sharpen-the-adapters-renders.md), verified 2026-09-06, is null on both its bars. Corrector steps on the adapter's own score move sharpness 8.0% against a 10% bar. Handing the tail to the frozen model after step 29 moves renders 0.038 nearer the target against a 0.05 bar, which is the largest movement any lever produced, with composition held. Its conclusion is the premise here: the softness lives in the adapter's low-noise steps.

This plan takes the one lever that moved and sweeps the step it was fixed at.

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Every read is per seed, never pooled**

In `pool43-all50`, `an_elephant__x__a_penguin` seed 09 renders as an engraved plate at every checkpoint from 1,250 to 60,000 while seed 10 is photographic at every one. Same run, same weights, opposite look. A number averaged over the two describes neither.

**Compose rate cannot judge this and is not asked to**

It reads 1.000 at every eval point from step 31,250 onward in both `pool43-all50` and `pool43-early25`, including the step where `a_cat__x__a_dog` seed 09 collapses into one fused body with two heads attached. The scorer counts distinct animal instances and it is correct to; the question here is a different axis.

**Laplacian variance is reported and is never a bar**

It counts edges, so it prefers drawn fur to a clean photograph. On plain-PoE references it is heavy-tailed, mean 152 against a standard deviation of 183 over eight seeds. That was measured and written down before any grid here ran.

**Stage B moves two axes at once, deliberately**

All fifty denoising steps in training, and x0 loss space, change together. Both aim at the same thing and running them singly first would cost fourteen hours to find out whether the pair is worth trying. A win therefore belongs to the pair, which is stated in the review file rather than left for a reader to notice, and the two single-axis follow-ups are named in task 4.2.

**Two checkpoints, because one of them never trained on the late steps**

`v1_freeze_null_r16_s0_25` trained on cached steps 0 to 25, so a ratio that falls off after step 25 there is what an untrained range looks like as much as what shrinkage looks like. `pool43-all50` trained on all fifty, which is why the stop condition is read off that one. The pair separates the two causes; either alone cannot.

**The best checkpoint is not the last one, and the runs cannot prove it**

In `pool43-all50` the best frames are at step 45,000 and the collapse is at 60,000. There is no checkpoint at 45,000: that run saved weights every 10,000 steps while rendering every 1,250. Plan 01's task 1.3 fixed this for later runs.

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- [The node facts](../../../../environment/hpc/nodes.md): `biggpu` allows one job per user through Slurm, so long runs go outside Slurm with `nohup` on a pinned GPU index. `mscluster109` carries two RTX A6000 at 49 GB. `mscluster106` and `108` carry Quadro RTX 8000 at 49 GB. `mscluster110` to `112` are Blackwell and need `co3_bw`, never `co3`.
- [The storage facts](../../../../environment/storage.md): checkpoints and renders to `/datasets` only.
- [The overview](../../../../environment/overview.md): cached tensors are float16, and any analysis stacking many of them upcasts to float32 first. Task 1.1 stacks many of them.
- The tracker is W&B, project `prime_lab/poe-repair-animals-compose`.

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**Near the end of drawing a picture the adapter applies a full-size push that points about a quarter of the way off target, and that off-target push is the haze. Detaching the adapter for the last stretch brings a render at least 0.05 nearer its own joint-prompt target in the compose scorer's DINOv2 embedding, on five or more of the eight held-out seeds, with composition within one seed of the adapter on the whole path.**

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

The adapter at rank 32 on the original pool, at step 100,050, draws a cat and a dog in the target's poses with the target's framing, made of felted wool. The composition problem this project set out to solve is solved in that picture and the picture is still not usable.

Every other plan in this scope changes what the loss compares. None of them is aimed at the rendering. This one is, and it is deliberately arranged so that the cheapest possible answer, a render setting rather than a new adapter, is tested first and can end the plan.

---

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

```
  task 1.1   how big the push is, and whether it points the right way, per step   DONE
             │
             │   size holds near 1.0 the whole way. Direction drops to 0.78 and 0.63
             │   in the last ten steps, which is where fine detail is drawn.
             ▼
  stage A (2.x)  adapter off after step k, for k in 25 29 33 36 40 50
                 three checkpoints that already exist, one render pass
                                     │
             ┌───────────────────────┴───────────────────────┐
             │                                               │
      clears the bar                                  falls short
             │                                               │
   an inference recipe.                         stage B (3.x): one training run,
   Write it up. STOP.                           all 50 steps + x0 loss, 7 hours
                                                             │
                                                  stage A's sweep again, on it
```

---

## Tasks

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

For Claude to execute.

### 0. 🧭 Check this plan before working from it

- [ ] 0.1 **Run the following prompt:**
  ```
  /verify-plan @plans/09-designing-the-correction-loss/plans/hypothesis/06-fitting-the-whole-path-and-handing-back-the-tail.md
  ```
- [ ] 0.2 **Run the following prompt:**
  ```
  /frame-hypothesis fitting the whole path and handing back the tail in plans/09-designing-the-correction-loss
  ```
  - The bar goes into the review file before any render, not after

▶ **Next: [task 1.1](#1--falsify-the-premise-first-for-free)**.

### 1. 📐 Falsify the premise first, for free

- [x] 1.1 **Measure how much of the correction the adapter actually applies, and whether it points the right way, per denoising step**
  - `r_hat = eps_poe_lora - eps_poe_frozen`, the quantity already formed at `poe_repair/experiments/one_pair_one_seed/trainer.py:588`
  - `r = eps_J - eps_PoE`, both already in the training cache
  - On two checkpoints, over the eight held-out `a_cat__x__a_dog` seeds, at every one of the 50 denoising steps:
    - `v1_freeze_null_r16_s0_25` at step 30,000, which trained on cached steps 0 to 25 only
    - `pool43-all50` at step 40,000, which trained on all fifty
  - Read the null branch the way each run's own `config.json` composes it: `freeze_null` is true for `v1` and absent for `pool43-all50`, so the first takes the empty branch from the cache and the second takes the adapted one
  - Upcast the cached float16 tensors to float32 before stacking them
  - Produces: `artifacts/results/designing-the-correction-loss/why-the-renders-are-not-crisp/shrinkage-per-step.json`, the ratio per step per seed per checkpoint, and one plot of the ratio against step, one panel per checkpoint, one line per seed
  - ✓ ran 2026-09-20 on `mscluster106` device 1, `scripts/shrinkage_per_step.py`, 800 measurements over the eight held-out seeds and both checkpoints. Mean per band on `pool43-all50`: size 1.08, 0.87, 0.88, 0.95, 0.94 and direction 0.77, 0.85, 0.91, 0.88, 0.78 across steps 0-9, 10-19, 20-29, 30-39, 40-49. On `v1` the direction ends worse, at 0.63
  - **The size does not fall off, so the push is not too weak late.** What fails late is the direction: full size, about a quarter of the way off target, in exactly the steps that draw fine detail. That is what stage A now tests

▶ **Next: [task 2.1](#2--stage-a-the-hand-off-sweep-no-training)**.

### 2. 🚀 Stage A: the hand-off sweep, no training

◀ **Needs: [task 1.1](#1--falsify-the-premise-first-for-free)**, done.

- [ ] 2.1 **Render the sweep on checkpoints that already exist**
  - Checkpoints, all on disk:
    - `/datasets/mmolefe/poe_repair_min/outputs/correction_loss_variants/v1_freeze_null_r16_s0_25/checkpoints/lora_step_030000.pt`
    - `/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/pool43-all50/checkpoints/lora_step_040000.pt`
    - `/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/pool43-all50/checkpoints/lora_step_060000.pt`
  - Condition: adapter attached over denoising steps 0 to `k`, detached over `k` to 50. Sweep `k` over 25, 29, 33, 36, 40 and 50, where 50 is the adapter on the whole path and is the control
  - 50 DDIM steps, guidance 7.5, eta 0, 1024²
  - Cells: the eight held-out `a_cat__x__a_dog` seeds 9 to 16, plus `an_elephant__x__a_penguin` seeds 9 and 10 as the second pair
  - `mscluster106`, so `mscluster109` stays free for plan 04's two training runs
  - Produces: 180 renders, about an hour
- [ ] 2.2 **Score them against the two bars, per seed**
  - Fidelity: cosine distance to that seed's own Mono render in the compose scorer's DINOv2 embedding, against the same checkpoint at `k = 50`
  - Composition: instance count from `poe_repair/experiments/compose_scorer_validation/detection_scorer.py`
  - Laplacian variance reported beside both and never as a bar
  - Produces: `handoff-sweep-per-seed.json`, one row per (checkpoint, k, seed)
- [ ] 2.3 **Write the verdict into the review file**
  - ✅ Clears the fidelity bar on five or more of the eight seeds with composition held: **this is an inference recipe and stage B does not run.** Go to task 4.3
  - ⚪ Clears on fewer: continue to stage B

▶ **Next: [task 3.1](#3--stage-b-train-it-only-if-24-says-so)**, only on a ⚪ verdict.

### 3. 🔧 Stage B: train it, only if 2.3 says so

◀ **Needs: [task 2.3](#2--stage-a-the-hand-off-sweep-no-training)** falling short of the bar.

- [x] 3.1 **Write the launcher**
  - Copy `scripts/showcase/v6_render_teacher.sh` to `scripts/showcase/v_extra_wholepath_x0.sh`
  - `RUN=v_extra_wholepath_x0_r16`, `EXTRA="--freeze-null --loss-space x0"`, and **`STEPRANGE` unset** so all 50 cached steps train
  - Rank 16, `cells_v57`, `POOLGEN=v57`, weight decay 0, `SAMPLEEVERY=25`, as the copy gives
  - Produces: one launcher differing from the V6 one in the run name, the switch string, the missing `STEPRANGE` and the python environment
  - ✓ done 2026-09-19: `scripts/showcase/v_extra_wholepath_x0.sh`. It pins `co3` rather than the `co3_bw` inherited from the V6 launcher, because `mscluster109` is an A6000
- [ ] 3.2 **Launch it**
  ```
  GPU=0 nohup bash scripts/showcase/v_extra_wholepath_x0.sh > logs/v_extra.log 2>&1 &
  ```
  - `mscluster109` once plan 04's runs have finished, to stay on one device model
  - The step range changes which cached steps are sampled from and not the number of optimizer steps, so this costs the same as plan 04's runs: about 7 hours for 30,000 steps
  - Produces: 25 checkpoints, 25 render sets, one W&B run id
- [ ] 3.3 **Write the run id into the review file the hour it launches**, with the node and the GPU index
- [ ] 3.4 **Repeat tasks 2.1 and 2.2 on its checkpoints**

### 4. 📐 Read it, and say what it is not

- [ ] 4.1 **Read against `01` at matched steps, per seed**
  - `01` is `kdzx03ji`, rank 16, 30,000 steps, the same pool
- [ ] 4.2 **State in the review file that stage B moved two axes together**
  - A win belongs to the pair, not to either change
  - Name the two single-axis follow-ups, to be run only on a win: all 50 steps with the epsilon loss, and steps 0 to 25 with the x0 loss
- [ ] 4.3 **Write `report/designing-the-correction-loss/08-fitting-the-whole-path-and-handing-back-the-tail.md`**
  - The verdict against the pre-registered bar, the per-seed table, and the two sheets from the Figure Catalog

### Close out. 🔄 Record what this plan taught

- [ ] C.1 **Run the following prompt**, after any red run:
  ```
  /ingest-error-pattern --from-run-log
  ```
- [ ] C.2 **Run the following prompt:**
  ```
  /sync-plan-tree @plans/09-designing-the-correction-loss/plans/hypothesis/06-fitting-the-whole-path-and-handing-back-the-tail.md
  ```

---

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

For you to follow manually.

### 5. 👁️ Judge the sweep by eye

◀ **Needs: [task 2.1](#2--stage-a-the-hand-off-sweep-no-training)** rendered.

- [ ] 5.1 **Open the sweep sheet and pick the best `k` by eye, before reading task 2.2's numbers**
  - Six columns, one per `k`, eight rows, one per seed
  - ✅ Fur reads as fur and the eyes have detail: the hand-off is working
  - ❌ The second animal has gone: `k` is too early. This happened at `k = 25` on `a_cat__x__a_dog` seed 09 in V1 at 30,000, where the cat disappears entirely
  - Write down which `k` you picked, then compare it against the scored answer. A disagreement between your eye and the number is itself worth recording, and this project has had several
- [ ] 5.2 **Open the run in W&B if stage B ran**
  - Browser: `wandb.ai/prime_lab/poe-repair-animals-compose`, the run id from task 3.3
  - Charts tab, then `train/loss_fit`
  - The x0 loss is a different scale from every earlier run here, so compare its shape and never its height
- [ ] 5.3 **Pick the checkpoint from the render strips, not from the loss**
  - In `pool43-all50` the best frames are at step 45,000 and the loss curve does not say so

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

1. Task 1.1 shows the direction failing in the late steps, which it does on both checkpoints.
2. The three checkpoints named in task 2.1 are readable.
3. `mscluster106` is healthy, checked with `nvidia-smi` over SSH and not with `sinfo`.

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| # | What it shows | Axes and data | From |
|---|---|---|---|
| F1 | How much of the correction the adapter applies and whether it points the right way, against denoising step | y: `‖r_hat‖ / ‖r‖` on a 0-to-1-and-above scale, 1.0 meaning the adapter supplies the whole correction, and the cosine between the two on a 0-to-1 scale. x: denoising step 0 to 49. Two panels, `v1` at 30,000 and `pool43-all50` at 40,000. One line per held-out seed, eight lines per panel | `shrinkage-per-step.json`, task 1.1. Built |
| F2 | The hand-off sweep | Six columns, `k` = 25, 29, 33, 36, 40, 50. Eight rows, the held-out seeds. Each tile carries its instance count and its fidelity distance | task 2.1 |
| F3 | Where the haze is, against the target | Three columns: the joint-prompt target, plain PoE, the adapter. One row per seed. The existing comparison strips, collected | already rendered by every run |

---

## Code references

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- `poe_repair/experiments/one_pair_one_seed/trainer.py:588`, where `r_hat` is already formed
- `poe_repair/experiments/one_pair_one_seed/trainer.py:575-581`, `_loss_weights` and the `loss_space` switch
- `poe_repair/experiments/cross_pair_lora_pooling/_inline_sampling.py:55-58`, how the sampler detaches the adapter for a branch
- `poe_repair/experiments/compose_scorer_validation/detection_scorer.py`, the instance-count scorer and why the two earlier reads were rejected
- `scripts/showcase/train_pool_run.sh`, the shared runner every launcher calls

---

## Next step

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

Nothing in this scope waits on this plan. If stage A clears the bar, the result changes how every other variation is rendered, so the report page is what the rest of the scope reads.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

<details>
<summary>Two failures catalogued</summary>

**Purpose**: known issues and their fixes, regenerated by `/ingest-error-pattern` and `/sync-plan-tree`.

#### From this scope

**A checkpoint may not exist at the step whose render you liked.** `pool43-all50` renders every 1,250 steps and saves weights every 10,000, so its best frames at step 45,000 cannot be loaded. Check which steps have a `.pt` file before planning a sweep around one.

**Turning the adapter off at the halfway point loses the second animal.** In `v1_freeze_null_r16_s0_25` at step 30,000, `a_cat__x__a_dog` seed 09 renders with no cat at all in the second row of its comparison grid, which is `k = 25`. The sweep starts at 25 to keep that case visible rather than to recommend it.

</details>
