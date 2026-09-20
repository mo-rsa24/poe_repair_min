# ⚖️ The penalty read

**Four matched experiments finished on 2026-09-12 and the repository has no record of them. This plan judges them as the cheap first pass, then launches the pair that carries the claim.**

**Step 73 in the root running order. Waits on 72 for the page shape it copies. Gates nothing, but it settles whether a penalty is the right way to close the invisible direction before plan 04 spends twenty hours closing it a different way.**

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/09-designing-the-correction-loss/plans/hypothesis/03-the-v0a-read.md — the four experiments scored, both bars read, 00b's page filed with the prompt-rewrite finding beside it
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 72 (previous) | [02: the V0 foundation](../reading/02-the-v0-foundation.md) | the page shape this one copies |
| **73 (current)** | **03: the V0a read** | four experiments already on disk, judged |
| 74 (next) | [04: freezing the empty branch](04-freezing-the-empty-branch.md) | the other way to close the same direction |

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
- [Description: what to do](#description-what-to-do)
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

- **The penalty, filed as `00b`**: the objective running today plus `μ‖u − ū‖²` at `μ = 10`, a fine on how far the adapted empty branch has moved from its cached self. Selected by `--null-anchor 10`.
- **The control**: the same objective with the penalty off, trained in the same hour on the same pool with every other setting pinned by one launcher. One axis differs, so the comparison is clean.
- **The two prompt-rewrite experiments**: the same objective with the branch prompts changed, to "a cat, two animals" against a null of "two animals", and to "a cat and" against a null of "and". Selected by `--branch-prompt-style plurality` and `connective`.
- **The mechanism bar**: did the empty branch stop drifting. Free, reads off W&B, needs no sampling. It only says the penalty did what it was told.
- **The outcome bar**: did the picture get better. Compose count on the eight held-out seeds, plus a blind read by eye. This is the one that decides.
- **Blind**: the condition names hidden and the order shuffled before anyone looks, with the shuffle saved so the blinding can be checked afterwards.

---

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**What is on disk**

`/datasets/mmolefe/poe_repair_min/outputs/correction_loss_variants/` holds four rank-16 experiments at 30,000 steps, launched within seventeen seconds of each other on 2026-09-12, with three checkpoints and twelve render sets apiece. `grep -rl r16_s0_25 --include=*.md .` returns nothing.

**What is already known without touching them**

The penalty worked mechanically. Off W&B, the drift term climbs from 1.20e-4 to 4.00e-4 over training in the control and falls from 1.98e-5 to 9.84e-6 under the penalty, so 41 times smaller at 25k to 30k, at a fit loss 23% higher. The fit loss has flattened in all four, falling 3 to 8 percent over the last 5,000 steps.

**What is not known**

Whether the picture is better. One held-out seed read by eye suggests yes, which is a reason to measure and not a result.

**The honesty this plan carries**

These four ran with no written criterion. The bars below were fixed before the read and not before the run, which is a weaker guarantee than this project's conventions ask for, and the review file and the page both say so.

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Expected runtime.** No training. Twelve checkpoints across four experiments, eight held-out seeds each, at 50 DDIM steps and 1024². About four hours of rendering on one card, then minutes of scoring.

**Prerequisites.** Plan 02's page shape, so this page copies it rather than inventing a second shape.

**Four caveats that decide what these four can be compared to.** They trained on denoising steps 0 to 25 of 50, where the runs in plan 02 used all 50. All four carry `ema_decay 0.999`, which the plan-02 runs did not. Their pool is `cells_v54`, not the scope pool, and they are rank 16, not the scope rank. So they compare to each other and to nothing else: not to `phase1_r16_100k`, and not to the rank-32 pair this plan launches.

**What each half of this plan is for.** The rank-16 four already exist, cost nothing to read, and answer whether the penalty does anything at all. The rank-32 pair on the scope pool is what a claim is made from. Reading the cheap pair first is the point: if the penalty does nothing at rank 16, the expensive pair is not worth eleven hours a side.

**Two of the four held-out cells have a broken reference.** Seed 10 of cat × dog draws three dogs and seed 9 of elephant × penguin draws an elephant alone. `scripts/correction_loss_variants/arm_grid.py` already knows this and names the sound pair; no score is averaged over a cell whose reference is wrong.

**Project tracking.** W&B, project `prime_lab/poe-repair-animals-compose`, runs `tqs7qf95`, `72as6yk3`, `mw1cqrpk`, `pckb2za7`.

**Known issues.** See the [Error Matrix](#error-matrix).

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- [Storage](../../../../environment/storage.md): the four run directories and their renders are on `/datasets`; the scoring pass writes there too.
- [Nodes](../../../../environment/hpc/nodes.md): the render pass needs one healthy card. Pin a node from a live idle probe rather than trusting `sinfo`, whose idle list includes hardware-faulted GPUs.
- [Overview](../../../../environment/overview.md): cached tensors are float16 and stacking them for an analysis upcasts to float32 first.
- [Known failures](../../../../environment/known-failures.md): the scorer's DINOv2 pass on CPU needs `XFORMERS_DISABLED=1`.

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**A penalty on the empty branch cuts its drift by about forty times and costs about a quarter more fit loss. Whether that buys a better picture is answerable today, from four experiments already on disk, without training anything.**

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** The scope's next expensive step, plan 04, closes the invisible direction by freezing the empty branch. There is a cheaper approximation already trained, with its own matched control, and nobody has read it. Spending twenty hours on the exact fix before reading the approximate one that already ran is the wrong order.

**The solution.** Score the twelve checkpoints, read the strips blind, and write the verdict against bars fixed here.

**Why the prompt-rewrite experiments come along.** They are two of the same four runs, on the same pool, in the same hour. Their finding is that renaming a prompt renames a free tensor, so all three prompt styles reach the same set of composed predictions and differ only in where training starts. Their fit losses came out within 20% of each other, which is what that predicts. Reading them costs one extra paragraph and closes a question that would otherwise sit open.

---

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-do) ➡️

```
  four run dirs on /datasets            the mechanism bar (free, off W&B)
  ────────────────────────              ────────────────────────────────
  plain       null_anchor 0    ─┐        drift at 25k-30k:
  anchor      null_anchor 10   ─┤          plain  4.00e-4  climbing
  plurality   prompts changed  ─┤          anchor 9.84e-6  falling
  connective  prompts changed  ─┘          ratio  41x
       │  3 checkpoints each
       ▼
  render 8 held-out cat x dog seeds       the outcome bar (decides)
  at each checkpoint, 50 steps, 1024²     ───────────────────────────
       │                                   compose count, anchor vs plain
       ├──▶ scorer: compose count, drift    blind read by eye, 8 seeds
       │
       └──▶ strips, names hidden, order shuffled
                          │
                          ▼
                  00b's page  ──  and the prompt-rewrite finding beside it
```

---

## Description: what to do

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **A render pass** over the twelve checkpoints on the eight held-out cat × dog seeds, reusing the probe recipe rather than writing a new one.
2. **A scoring pass**: compose count and drift per seed per checkpoint, into one json.
3. **The mechanism numbers** pulled from W&B for all four runs, medians per 5,000-step window.
4. **The blind strips**, names hidden and order shuffled, with the shuffle saved.
5. **`report/designing-the-correction-loss/00b-penalising-the-empty-branch-for-moving.md`**, copying plan 02's page shape.
6. **The prompt-rewrite finding**, written in the same pass because it is the same four runs.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-do) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves objective 3 of [the scope](../../MASTER_PLAN.md#objectives): read the objectives that already ran rather than re-running them.

1. `00b` has a verdict against both bars.
2. The prompt-rewrite question is closed with a number.
3. Nothing was trained to get either.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

For Claude to execute.

### 0. 🧭 Check this plan before working from it

- [ ] 0.1 **Run the following prompt:**
  ```
  /verify-plan @plans/09-designing-the-correction-loss/plans/hypothesis/03-the-v0a-read.md
  ```
- [ ] 0.2 **Run the following prompt:**
  ```
  /xref-pillar @plans/09-designing-the-correction-loss/plans/hypothesis/03-the-v0a-read.md
  ```

▶ **Next: [tasks 1.1 to 1.3](#1--get-the-numbers)**, the render and scoring passes.

### 1. 📐 Get the numbers

- [ ] 1.1 **Render the eight held-out cat × dog seeds from each of the twelve checkpoints**
  - Reuse `runbook/running-things-on-the-cluster/probing-a-pooled-lora-checkpoint.md` section 1 rather than writing a new pass
  - Four experiments at checkpoints 10,000, 20,000 and 30,000
  - Produces: 96 renders under `/datasets/mmolefe/poe_repair_min/outputs/correction_loss_variants/read/`
  - Done when the directory holds 96 files, not when the job exits 0
- [ ] 1.2 **Score them**
  - Compose count by the validated instance-count scorer, and DINOv2 drift against the joint-prompt render
  - `XFORMERS_DISABLED=1` for the CPU DINOv2 pass
  - Produces: `artifacts/results/designing-the-correction-loss/00b-penalising-the-empty-branch-for-moving/00b-per-checkpoint.json`
- [ ] 1.3 **Run the following prompt**, once per run id, for the mechanism numbers:
  ```
  /analyze-run 72as6yk3
  ```
  - The other three are `tqs7qf95` (control), `mw1cqrpk` (plurality), `pckb2za7` (connective)
  - Produces: median fit loss and median drift per 5,000-step window, into the same json

▶ **Next: [task 2.1](#2--prepare-the-blind-read)**, which needs the renders from 1.1.

### 2. 🎲 Prepare the blind read

◀ **Needs: [task 1.1](#1--get-the-numbers)** done, so there are renders to shuffle.

- [ ] 2.1 **Build the shuffled strips**
  - One strip per held-out seed at checkpoint 30,000: the joint-prompt target, the plain product, and the four experiments' renders with their condition names replaced by ids and their order shuffled per seed
  - Save the shuffle key to a separate file, so the blinding can be checked after the read and not before it
  - Produces: eight strips plus `shuffle-key.json`
  - Exclude seed 10, whose joint-prompt reference draws three dogs. `scripts/correction_loss_variants/arm_grid.py` already names the sound cells

▶ **Next: [instruction 4.1](#4--read-the-strips-by-eye)**, the read itself.

### 3. ✍️ Write it up

◀ **Needs: [instruction 4.1](#4--read-the-strips-by-eye)** done, so there are labels to write about.

- [ ] 3.1 **Write `report/designing-the-correction-loss/00b-penalising-the-empty-branch-for-moving.md`**
  - Plan 02's page shape. Verdict first, the equation rendered, the strips as thumbnails, one table of both bars, links to the detail
  - Carries at the top, plainly, that these four ran on 2026-09-12 with no written criterion, and that the bars were fixed before the read and not before the run
  - Produces: one page
- [ ] 3.2 **Write the prompt-rewrite finding**
  - `report/designing-the-correction-loss/00-matching-the-composition-to-the-joint-prompt.md` gains a short section, or its own file if it runs long: renaming a prompt renames a free tensor, so the three prompt styles reach the same composed predictions and differ in where training starts
  - The number that carries it is the spread of the three fit losses
  - Produces: the finding, wherever it is shorter
- [ ] 3.3 **Write the artifact card**

### 3b. 🚀 Launch the pair that carries the claim

◀ **Needs: [task 3.1](#3--write-it-up)** done, so the cheap read has said whether the penalty does anything.

- [ ] 3b.1 **Launch `00a`, the objective running today, on the scope pool at rank 32**
  - `cells_v57.json`, 43 cells over 29 pairs, rank 32, alpha 32, 30,000 steps
  - One 49 GB card, and the same model as `00b` below: `mscluster109` device 0 (RTX A6000), or `mscluster106` device 1 (Quadro RTX 8000) if 109 is taken
  - Started under `nohup` outside Slurm, because `biggpu` allows one job per user
  - Needs [task 1.5](../tools/01-the-three-instrument-fixes.md) first, or 8 of the 43 cells are recorded as held-out
  - Produces: the baseline every later variation in this scope is read against
  - Expected runtime: about 11 hours at roughly 2,750 steps an hour
- [ ] 3b.2 **Launch `00b`, the same objective plus the penalty, matched to it**
  - Identical to 3b.1 with `--null-anchor 10`, on the second device of the same node, started in the same hour
  - One axis differs, so the comparison is clean
  - Produces: the rank-32 half of this plan's question
- [ ] 3b.3 **Record both in the review file before either finishes**
  - Run id, node, device, W&B id, pool file and rank, in the `## Runs` table
  - Done when a reader can tell the two apart without opening W&B

▶ **Next: [instruction 4.1](#4--read-the-strips-by-eye)**, the blind read, once both reach 30,000.

### Close out. 🔄 Record what this plan taught

- [ ] C.1 **Run the following prompt**, after any red run:
  ```
  /ingest-error-pattern --from-run-log
  ```
- [ ] C.2 **Run the following prompt:**
  ```
  /sync-plan-tree @plans/09-designing-the-correction-loss/plans/hypothesis/03-the-v0a-read.md
  ```

---

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

For you to follow manually.

### 4. 👁️ Read the strips by eye

◀ **Needs: [task 2.1](#2--prepare-the-blind-read)** done, so the strips exist and are shuffled.

- [ ] 4.1 **Open the eight strips one at a time, without opening the shuffle key**
  - Path: `artifacts/results/designing-the-correction-loss/00b-penalising-the-empty-branch-for-moving/blind/`
  - For each panel record one of three labels: **clean** (two separate animals, one of each, no blending), **unclear**, **not two**
  - Write them into a table beside the panel ids
  - ✅ All eight seeds labelled before the key is opened: the read is valid
  - ❌ You opened the key first: the read is void. Reshuffle at task 2.1 and do it again on a different day
- [ ] 4.2 **Unblind and score**
  - Join the labels to the conditions through `shuffle-key.json`
  - Count clean labels per condition
  - ✅ The penalised experiment wins on at least 5 of 8 and its compose count is no lower than the control's: the bar is met
  - ❌ Either fails: it is a null, and it is recorded as a null. Plan 04 carries the scope
- [ ] 4.3 **Look at the two prompt-rewrite conditions while you are there**
  - They should be indistinguishable from the control by eye. If one is visibly better, the theory that renaming a prompt changes nothing is wrong, and that is worth more than this plan's own question

▶ **Next: [task 3.1](#3--write-it-up)**, which needs these labels.

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> The bars are fixed here, before the read, and nothing about the result may move them afterwards.

**The mechanism bar.** The drift under the penalty is at least ten times smaller than the control's at matched steps. Already met off W&B at 41 times, recorded here so the read is not negotiable later.

**The outcome bar, which decides.** The penalised experiment composes at least as many of the eight held-out seeds as the control, and wins the blind read on at least 5 of 8.

**Below either is a null.** A null here is a finding: it says a fine is not a ban, which is exactly the tradeoff the objectives note names, and it strengthens the case for plan 04 rather than weakening the scope.

**Partial pass.** If the compose counts tie and the blind read is 4 of 8, the result is inconclusive rather than null, and it is written as inconclusive. Ties do not get rounded toward the interesting answer.

The review questions are in [the review file](../../review/03-the-v0a-read.md).

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

**Pending, from [this scope's illustrated map](../../diagram-prompts.md)**

| Figure | Lane | What it shows | Status |
|---|---|---|---|
| [corrloss-03-two-legs-and-the-fine](../../diagrams/corrloss-03-two-legs-and-the-fine.png) | subject | the two legs of the loss and the fine that this plan judges | 🖼️ rendered · opened at task 3.1, so the page's equation and the picture agree |
| [corrloss-04-the-four-places-they-differ](../../diagrams/corrloss-04-the-four-places-they-differ.png) | subject | one machine, four settings, nine objectives | 🖼️ rendered · opened at task 3.2, to place the prompt-rewrite finding on the right setting |

**Generated during plan execution**

| File | Lane | What it holds | Task | Status |
|---|---|---|---|---|
| `00b-per-checkpoint.json` | — | compose count, drift and the W&B medians for all four experiments | tasks 1.2, 1.3 | ⏳ |
| `blind/seed-NN.png` (8) | subject | one strip per seed, ids instead of names, order shuffled | task 2.1 | ⏳ |
| `shuffle-key.json` | — | the mapping from panel id to condition, opened only after the read | task 2.1 | ⏳ |
| `drift-and-fit-four-conditions.png` | subject | drift and fit loss against training step, one line per condition | task 1.3 | ⏳ |

**Organization workflow.** Filed under `artifacts/results/designing-the-correction-loss/00b-penalising-the-empty-branch-for-moving/` with its card entry.

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/09-designing-the-correction-loss/plans/hypothesis/03-the-v0a-read.md` | **task 0.1**, before any work | conformance and thin instructions reported |
| Cross-reference the plan | `/xref-pillar @plans/09-designing-the-correction-loss/plans/hypothesis/03-the-v0a-read.md` | **task 0.2**, before any work | terms already documented elsewhere linked |
| Capture patterns | `/ingest-error-pattern --from-run-log` | **the close out**, after any red run | errors added to the catalogs |
| Bring the tree current | `/sync-plan-tree @plans/09-designing-the-correction-loss/plans/hypothesis/03-the-v0a-read.md` | **the close out** | statuses, running order and Error Matrix match reality |

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**File:** `scripts/correction_loss_variants/run_arm.sh`
**Relevant section:** the launcher that produced all four experiments. Everything except one flag is pinned inside it, which is what makes the comparison clean. `MU` defaults to 0.1 and was passed as 10; `STEPRANGE="0 25"` restricted training to the first half of the schedule.

**File:** `scripts/correction_loss_variants/arm_grid.py`
**Relevant section:** `SOUND` and `BROKEN` at lines 43 to 46, the two held-out cells whose joint-prompt reference is usable and the two whose is not. Task 2.1 reads this rather than re-deciding it.

**File:** `poe_repair/experiments/one_pair_one_seed/trainer.py`
**Relevant section:** lines 496 to 503, the penalty this plan judges. `null_drift_norm` in the `info` dict at 516 is logged whether or not the penalty is on, which is why the control's drift is measurable at all.

**Run directories:** `/datasets/mmolefe/poe_repair_min/outputs/correction_loss_variants/r16_s0_25_{plain,anchor,plurality,connective}/`, each with `checkpoints/`, `samples/per_epoch/` and `config.json`.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

▶ `/analyze-run 72as6yk3` ✅ — sweeps the run for saves not yet analyzed and returns the curve reads task 1.3 needs, one call per run id.
   alt: `/report-pulse` on the finished page, the same shape check plan 02 uses.

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents)

[04: freezing the empty branch](04-freezing-the-empty-branch.md) closes the same direction exactly rather than approximately, and is read against whatever this plan concludes.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

<details>
<summary>No failures catalogued yet</summary>

**Purpose**: known issues and their fixes, regenerated by `/ingest-error-pattern` and `/sync-plan-tree`.

#### From global catalog

#### From project catalog

---

**Auto-update note:** regenerated by `/sync-plan-tree`. Do not edit by hand.

</details>
