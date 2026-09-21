# 🔬 The three trainer switches, the guidance interval, and the 14-pair pool

**Nothing in this scope can run until the trainer can be told which cells to skip, which denoising steps to sample, and how much to charge the part of the correction the two experts cannot supply, and until the sampler can hold strong guidance over a middle span only. This plan builds those four switches, the pool they read, and the exclusion list they consume, and proves each one selects what its name claims.**

**Step 66 in the root running order. Waits on nothing. Gates steps 67 to 70: no other plan in this scope can start until the dry run prints the cell count this plan promises.**

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/08-improve-the-rank-32-pooled-adapter/plans/tools/01-the-three-switches-and-the-guidance-interval.md — the four switches written, the pool and the exclusion list built, the dry run's cell count and the sampler identity check recorded in the review file
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| **66 (current)** | **01: the three switches, the guidance interval, the pool** | the code and config every other plan here reads |
| 67 (next) | [02: the blind label pass](02-the-blind-label-pass.md) | the read: strips, hidden names, three labels |
| 68 | [03: what the stacked sampler alone does](../baselines/03-what-the-stacked-sampler-alone-does.md) | the baseline under the sampler this plan's guidance interval makes possible |

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

- **A cell**: one animal pair on one seed. The cache holds, per cell, a 50-step record of what the model predicted at every step plus the picture the joint prompt drew on that seed. Eleven training pairs across eight seeds is 88 cells.
- **The joint target**: the picture the single prompt naming both animals drew on that seed, `mono.png` in the cache. It is what the adapter is trained to reach.
- **The both-names read**: an automatic check that crops each detected animal-shaped region out of a picture and asks CLIP which of the pair's two names it matches. The picture passes when both names win at least one region. It over-flags on visually similar pairs, which is a known property and is what child C1 measures.
- **The two experts' span**: at one denoising step, the plane spanned by each single-animal prediction measured against the model's prediction with no prompt at all. Anything in that plane the two experts can already produce by being re-weighted; anything out of it they cannot.
- **The in-span and orthogonal parts** of a vector: its projection into that plane and the remainder. They add back to the whole vector.
- **The windowed sampler**: the sampler in `poe_repair/methods/_poe_langevin.py` that already restricts the adapter to a span of steps (`lambda_window`) and already takes a DDIM eta. What it does not have is a guidance interval.
- **The guidance interval**: strong classifier-free guidance (7.5) held over a middle span of steps and weak guidance (1.0) outside it. The stacked sampler needs it; the shipped sampler does not.
- **The dry run**: building the training dataset and printing what it contains without training on it.

---

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The system.** Four switches and two config files, none of which exist yet. Three switches go into the pooled trainer, one into the windowed sampler, and the two config files are the pool the trainer reads and the list of cells it must skip.

**What it does.** It makes the run set in [the run design](../../../../artifacts/ideas/improving-the-pooled-lora-run/run-design-parent-and-children.md) expressible. The parent run is five settings at once; four of those five have no flag today. Without them the parent cannot be launched and the children cannot differ from it by one axis, because the axis does not exist.

**Key components.** `--exclude-cells`, `--train-step-range` and `--orth-weight` in `poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py`; a guidance interval in `poe_repair/methods/_poe_langevin.py`; a 14-pair pool and its prompts under `artifacts/_shared/cross_pair_pool_configs/`; `exclude_cells.json` under the scope's output root.

**How this is judged.** As an instrument, by whether it can fail rather than by what it found. A flag whose target group is empty is a silent no-op: the run completes, reports a plausible metric, and describes a different configuration than its name claims. So every switch prints the count it actually selected, and the sampler change is held against a render made before it.

**Associated materials.**
- Review questions: [the review file](../../review/01-the-three-switches-and-the-guidance-interval.md)
- The design this serves: [the run design](../../../../artifacts/ideas/improving-the-pooled-lora-run/run-design-parent-and-children.md), the "Code before launch" and "Before the launch line, in order" sections
- The scope: [MASTER_PLAN](../../MASTER_PLAN.md)

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Cost.** The code is a few hours of writing. The dry run needs no GPU. The sampler identity check is 8 renders on cat × dog, about 20 minutes on any free card. Call it half a GPU-hour in all.

**Buys.** Every other plan in this scope. Nothing here is launched, so nothing here can be wasted GPU time; the risk this plan carries is a switch that appears to work and selects nothing.

**Prerequisites.** The training cache at `/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache/`, which is on the data filesystem rather than in the checkout; `target_quality.json` at `/datasets/mmolefe/poe_repair_min/outputs/showcase/target_quality/target_quality.json`; the span projection already written in `scripts/showcase/correction_span_common.py`; the rank-32 checkpoint at step 30050 for the identity check.

**Output root.** `/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/`.

**W&B project.** `prime_lab/poe-repair-animals-compose`. This plan logs nothing to it; the identity check's sheet goes to the output root.

**The precision trap.** A sampler identity check has to be judged against a reference rendered in the same loop and the same precision. An fp32-against-fp16 comparison of the same combine drifts by about three grey levels over 50 steps, which is larger than several real differences. Keep the old render as the identity reference and produce the new one through the same code path with the interval set to cover everything.

**Known issues:** see the [Error Matrix](#error-matrix).

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- **Two filesystems, and the guard has to check the one being written to.** `exclude_cells.json`, the pool yaml's resolved copy and the identity-check renders go under `/datasets`, never `/home-mscluster`, which has hit 100% and silently killed checkpointing before ([storage](../../../../environment/storage.md)).
- **The pool config path is hard-coded.** `artifacts/_shared/cross_pair_pool_configs/` is read as `GROUP_POOL_CONFIGS` at `poe_repair/paths.py:150`, so the new yaml goes beside the existing three and nowhere else.
- **The identity check needs a card, and `biggpu` allows one job per user** ([execution protocol](../../../../environment/hpc/execution-protocol.md)). It is small enough to run on the session node's card or on any free device over SSH.
- **`co3_bw` on the Blackwell, never `co3`.** A `co3` CUDA operation on `mscluster112` produces no output rather than an error ([nodes](../../../../environment/hpc/nodes.md)).

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**Four switches exist, each prints the count it selected, and the dry run on the parent config builds the cell count the design predicts.**

**Why this matters right now:** the chain of five runs is roughly twelve hours on one device. A switch that silently selects nothing costs all twelve, and the loss is invisible because the run completes and reports a plausible number.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The gap.** The trainer takes a pool, a rank, a length, a weight decay and an EMA decay. It cannot be told to skip a cell, to sample only part of the run, or to charge one part of its error more than another. The sampler can hold the adapter to a window and can add fresh randomness, but it applies one guidance strength to the whole run. Four of the parent's five settings therefore have nowhere to live.

**The approach.** Write the four switches, build the two config files, and prove each selection is non-empty before anything trains.

**Key insights.**

1. **A flag with an empty target group is worse than a missing flag.** The run finishes and its name says one thing while its configuration says another. Printing the selected count is what converts that silent failure into a visible one.
2. **The span projection is already written.** `correction_span_common.py` computes exactly the decomposition `--orth-weight` needs, so this is a wiring job rather than new mathematics.
3. **The exclusion list is expected to over-flag.** The both-names read finds two-of-the-same on visually separable pairs and over-flags on similar ones. That is not a bug to fix here; it is the axis child C1 exists to measure, so the list is built as the read gives it and its count is recorded.

---

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
 target_quality.json ──▶ rows with both_present == false, train pairs only
                                  │
                                  ▼
                        exclude_cells.json  ──┐
 14-pair pool yaml ────────────────────────┐  │
                                           ▼  ▼
   train_pooled.py   --exclude-cells   ▶ dataset built
                     --train-step-range 0 24   ▶ steps sampled from [0,24)
                     --orth-weight 3           ▶ loss = in-span + 3 x orthogonal
                     --dry-run                 ▶ prints: cells kept, cells dropped,
                                                  step range, orth weight, and exits

 _poe_langevin.py    guidance_interval (5,35) ▶ 7.5 inside, 1.0 outside
                     guidance_interval None   ▶ 7.5 everywhere  ← must equal today's render
```

The last line is the identity check. Setting the new switch to cover the whole run has to reproduce what the sampler already draws, in the same process and the same precision, or the switch has changed the shipped path rather than added to it.

---

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **`--exclude-cells <path>`** in `train_pooled.py`: reads a JSON list of `{pair, seed}` objects and skips those cells when the dataset is built. Prints the number of cells requested for exclusion, the number actually found and dropped, and the number remaining. A requested cell that is not in the pool is a warning naming it, not a silent skip.
2. **`--train-step-range LO HI`** in `train_pooled.py`: training samples denoising steps from `[LO, HI)` only. Defaults to the full range so existing runs are unaffected. Prints the range and the number of distinct steps it selects.
3. **`--orth-weight W`** in `train_pooled.py`: per sample, project both the target correction and the adapter's prediction onto the span of the two experts' own directions and onto its complement, then form the loss as the in-span error plus `W` times the orthogonal error. Defaults to 1.0, which is the current plain loss. Prints the weight and, on the first batch, the mean fraction of the target that falls out of span, so a projection that has collapsed is visible immediately.
4. **A guidance interval** in the windowed sampler: an optional `(lo, hi)` beside `lambda_window` and eta, applying the strong guidance scale on steps `[lo, hi)` and 1.0 outside. Absent, behaviour is unchanged.
5. **The 14-pair pool** and its prompts under `artifacts/_shared/cross_pair_pool_configs/`: the 11 training pairs plus lion × horse, wolf × horse and bear × salmon, with a check that every cell the pool names exists in the cache and a printed count per pair.
6. **`exclude_cells.json`** under the output root, built from the hand verdicts in
   [`eye-verdicts.csv`](../../../../artifacts/results/which-joint-prompt-targets-can-the-adapter-learn-from/eye-verdicts.csv), with its count printed beside the total and broken down per pair.
   It is not built from `target_quality.json`'s `both_present` field. That field is the automatic check, and
   [the target sheets](../../../../artifacts/results/which-joint-prompt-targets-can-the-adapter-learn-from/README.md) show it over-flagging: on `a_lion__x__a_tiger` it passes two cells of eight
   while several it fails plainly show both animals, and on `a_mailbox__x__a_snowfield` it passes one of eight
   while every tile is correct. The automatic column stays in the file beside the hand verdicts, because the
   disagreement between them is what says whether the check can be cited.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

**Purpose.** Serves the scope's objective 1: the parent cannot put five changes on at once, and the children cannot each remove one, until each change is a switch.

**Goals.**

1. Four switches exist and parse, and each prints the count it selected.
2. `exclude_cells.json` exists with its count printed beside the pool's total.
3. The 14-pair pool yaml names only cells present in the cache.
4. The dry run on the parent config prints 88 plus 20 minus the excluded count.
5. The sampler with the interval covering the whole run reproduces today's render within the same-loop tolerance.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/08-improve-the-rank-32-pooled-adapter/plans/tools/01-the-three-switches-and-the-guidance-interval.md`
  - Done when: the report comes back clean, or its proposals have been applied.
- [ ] **0.2** Cross-reference this plan's terms against `context/`, `environment/`, `runbook/` and `report/`, in case a term here is already defined elsewhere in the repo.
  - Paste: `/xref-pillar @plans/08-improve-the-rank-32-pooled-adapter/plans/tools/01-the-three-switches-and-the-guidance-interval.md`
  - Done when: the scan returns no candidates, or its proposed links have been applied.

▶ **Next: [task 1.1](#1--write-the-three-trainer-switches)**, the first real work.

### 1. 🔧 Write the three trainer switches

◀ **Needs: [task 0.1](#0--check-this-plan-before-working-from-it)**, so the plan is known good before code is written against it.

- [ ] **1.1 Add `--exclude-cells`, `--train-step-range` and `--orth-weight`** to `poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py`, each with the default that reproduces today's behaviour.
  - Each switch prints, at dataset-build time, the count it selected: cells dropped and remaining, distinct steps in range, and the orthogonal weight with the first batch's mean out-of-span fraction.
  - **Done when:** `--help` lists all three, and a run with none of them set prints the same cell count the trainer prints today.
- [ ] **1.2 Wire `--orth-weight` through the span projection already in `scripts/showcase/correction_span_common.py`** rather than writing a second decomposition.
  - **Done when:** on the first batch, the printed in-span and orthogonal norms recombine to the target's norm within floating-point tolerance, which is the check that the projection is a decomposition and not a truncation.

▶ **Next: [task 2.1](#2--write-the-guidance-interval-and-hold-it-against-todays-render)**.

### 2. 🔧 Write the guidance interval, and hold it against today's render

◀ **Needs: [task 0.1](#0--check-this-plan-before-working-from-it)**. The sampler change does not depend on the trainer switches; the two are grouped so they land in one reviewable state, which is ordering rather than dependency.

- [ ] **2.1 Add an optional guidance interval** to the windowed sampler in `poe_repair/methods/_poe_langevin.py`, beside `lambda_window` and the eta setting: the strong scale on `[lo, hi)`, 1.0 outside, unchanged behaviour when absent.
  - **Done when:** the sampler's own signature carries it and a call without it takes the existing code path.
- [ ] **2.2 Run the identity check.** Render cat × dog seeds 9 to 16 with the interval set to cover all 50 steps at guidance 7.5, in the same process and precision as the reference, and compare against the existing shipped render of the same seeds.
  - The reference is an existing render, not a fresh one: keep the old file and compare against it, so the check cannot pass by both sides moving together.
  - **Done when:** the mean absolute difference per seed is at or under 6 grey levels, the same-loop fp16 tolerance already used in this repo. Above it, the interval has changed the shipped path and task 2.1 is wrong.

▶ **Next: [task 3.1](#3--build-the-pool-and-the-exclusion-list)**.

### 3. 🔧 Build the pool and the exclusion list

◀ **Needs: [task 0.1](#0--check-this-plan-before-working-from-it)**. This group depends on nothing group 1 or 2 produces; it is placed here because the dry run in group 4 needs both the switches and these files.

- [ ] **3.1 Write the 14-pair pool configuration** under `artifacts/_shared/cross_pair_pool_configs/`, the 11 training pairs plus lion × horse, wolf × horse and bear × salmon.
  - The trainer reads three separate files and all three are required arguments, so this writes `pair_pool_14.yaml` (passed to `--pair-pool`) and `pair_prompts_14.yaml` (passed to `--pair-prompts`), and either reuses the existing `seed_pool.yaml` for `--seed-pool-path` or writes a variant beside it. Say in the commit which of the three was reused.
  - The two horse pairs sit under `heldout/` in the cache and the loader searches both splits, so no file is moved.
  - **Done when:** a resolution pass prints one line per pair with the number of cached cells found, every line is non-zero, and the three added pairs total 20 (lion × horse and wolf × horse contribute 8 each on seeds 1 to 8; bear × salmon contributes 4, since its only other cached seed is 42, outside the training seed list).
- [ ] **3.2 Build `exclude_cells.json`** from the filled `eye_verdict` column of
  [`eye-verdicts.csv`](../../../../artifacts/results/which-joint-prompt-targets-can-the-adapter-learn-from/eye-verdicts.csv), dropping cells marked `bad` and keeping `good`; a cell left
  `unsure` is kept and counted separately so its effect can be checked.
  - ◀ Needs [instruction 5.4](#5--read-the-dry-run-and-the-identity-sheet), the hand pass over the target sheets.
  - Also emit the agreement between the hand verdicts and the automatic `both_present` column: how often
    they agree, and the two disagreement counts separately. That table is the evidence for whether the
    automatic check may be cited anywhere in the paper.
  - Output: `/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/exclude_cells.json`
  - **Done when:** the file exists and the run prints the excluded count beside the pool total, in the form `excluding N of M cells`.

▶ **Next: [task 4.1](#4--dry-run-the-parent-configuration)**.

### 4. 🚀 Dry-run the parent configuration

◀ **Needs: [tasks 3.1 and 3.2](#3--build-the-pool-and-the-exclusion-list)** done, so both config files exist.

- [ ] **4.1 Run the parent config with `--dry-run`**, every switch set as the parent uses it, and capture the output.

    ```bash
    /home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python -m poe_repair.experiments.cross_pair_lora_pooling.train_pooled \
      --pair-pool      /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/artifacts/_shared/cross_pair_pool_configs/pair_pool_14.yaml \
      --pair-prompts   /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/artifacts/_shared/cross_pair_pool_configs/pair_prompts_14.yaml \
      --seed-pool-path /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/artifacts/_shared/cross_pair_pool_configs/seed_pool.yaml \
      --exclude-cells  /datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/exclude_cells.json \
      --train-step-range 0 24 --orth-weight 3.0 \
      --lora-rank 32 --lora-alpha 32 --weight-decay 1e-2 --ema-decay 0.999 --lr 1e-4 \
      --total-epochs 600 --epoch-size 50 \
      --dry-run
    ```

  - The trainer takes three separate config files and counts its length in epochs, not steps: `--total-epochs 600` at `--epoch-size 50` is the 30,000 steps the design asks for. `--lora-rank` is the rank flag; there is no `--rank` and no `--max-steps`.

  - **Done when:** the output prints a kept-cell count equal to 88 plus 20 minus the excluded count, a step range of `[0, 24)` selecting 24 distinct steps, and an orthogonal weight of 3.0, and then exits without training.

▶ **Next: [instruction 5.1](#5--read-the-dry-run-and-the-identity-sheet)**.

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

**For you to follow manually.** Do these yourself.

### 5. 👁️ Read the dry run and the identity sheet

◀ **Needs: [task 4.1](#4--dry-run-the-parent-configuration)** and [task 2.2](#2--write-the-guidance-interval-and-hold-it-against-todays-render) done.

5.1 **Read the dry run's counts against the design.**
   - Open the captured output.
   - Check the kept-cell count arithmetic yourself: 88 plus 20 is 108, minus the excluded count.
   - ✅ The three numbers agree and the excluded count is non-zero: the exclusion switch has a target group, so it is not a no-op.
   - ❌ The excluded count is zero: either `target_quality.json` has no failing training rows, or the pair names in it do not match the pool's. Check one pair name from each file by eye before changing any code.

5.2 **Look at the identity sheet.**
   - Open the eight-seed comparison under `/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/identity_check/`.
   - Each row is one seed: the existing render on the left, the render through the new interval on the right.
   - ✅ The pairs are indistinguishable by eye and the reported difference is at or under 6 grey levels.
   - ❌ Any pair differs visibly: the interval is being applied when it should not be. Record which seeds and stop; the stacked sampler cannot be trusted until this passes.

5.3 **Record both readings** in the [review file](../../review/01-the-three-switches-and-the-guidance-interval.md): the three counts, the excluded count, and the per-seed identity differences.

5.4 **Judge the target sheets by eye**, which is what task 3.2's exclusion list is built from.
   - Open the twelve sheets under [the target-quality grouping](../../../../artifacts/results/which-joint-prompt-targets-can-the-adapter-learn-from/README.md). Rows are pairs, columns are seeds 1 to 8, the row caption is that pair's own prompt, and each tile's border is the automatic check's verdict.
   - For every tile, fill the `eye_verdict` column of [`eye-verdicts.csv`](../../../../artifacts/results/which-joint-prompt-targets-can-the-adapter-learn-from/eye-verdicts.csv) with `good`, `bad` or `unsure`. Good means the picture shows both things the prompt names, each clearly itself.
   - ✅ All 480 rows carry a verdict, or you stop early having covered the pairs you intend to train on and say which sheets you skipped.
   - ❌ You find yourself agreeing with the border everywhere: say so, because that would mean the automatic check is sound and the whole premise of this instrument was wrong, which is a useful result and changes task 3.2 back to the automatic column.

▶ **Next: [the close out](#close-out--record-what-this-plan-taught)**.

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted, including any that went red.

- [ ] **Capture the failures this plan hit**, while they are still fresh.
  - Paste: `/ingest-error-pattern --from-run-log @plans/08-improve-the-rank-32-pooled-adapter/plans/tools/01-the-three-switches-and-the-guidance-interval.md`
  - Run after any red run. Done when: each failure has a catalog entry, or there were none.
- [ ] **Bring the tree current.**
  - Paste: `/sync-plan-tree @plans/08-improve-the-rank-32-pooled-adapter/plans/tools/01-the-three-switches-and-the-guidance-interval.md — the four switches, the pool, the exclusion list, the dry-run counts and the identity check`
  - Done when: statuses, the running order and the Error Matrix match reality.

▶ **Next: [what has to pass before this runs](#what-has-to-pass-before-this-runs)**.

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **Why this checkpoint matters:** every plan in this scope reads these switches, and the chain behind them is roughly twelve hours of one device. A switch that selects nothing wastes all of it and reports a plausible number while doing so.

**Pass criteria:**
- Every switch prints a non-zero selected count where the design says it should select something.
- The exclusion's per-pair breakdown is printed and recorded, along with the number of pairs left holding zero cells. The count read ahead of this plan says the rule drops 80 of 108 cells and empties four pairs; a run that matches that is expected, and a run that differs from it materially means the switch is not selecting what the hand count selected.
- The dry run's kept-cell arithmetic matches 108 minus the excluded count.
- The in-span and orthogonal norms recombine to the target's norm on the first batch.
- The identity check is at or under 6 grey levels per seed against the kept reference render.

**Fail criteria (STOP):**
- The exclusion count is zero, or the pool resolves any pair to zero cached cells: the config names something the cache does not have, and no training may start.
- The identity check exceeds 6 grey levels: the guidance interval has altered the shipped sampler rather than extending it, so both the baseline and every stacked render would be measuring a changed path.

**When you get results, answer the questions in the [review file](../../review/01-the-three-switches-and-the-guidance-interval.md).**

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

### Pending: to be generated from prompts

| Item | Lane | Prompt file | What it shows | Save to |
|------|------|-------------|---------------|---------|
| Which cells the adapter is allowed to learn from | subject | [diagram-prompts.md](../../diagram-prompts.md#prompt-1-subject-which-cells-the-adapter-is-allowed-to-learn-from) | the cache, the grid of cells, the filter and what survives | `diagrams/improve-r32-01-which-cells-the-adapter-learns-from.png` |
| What the loss weights up | subject | [diagram-prompts.md](../../diagram-prompts.md#prompt-2a-subject-what-the-loss-weights-up) | the target split into in-span and orthogonal parts, the orthogonal one charged three times | `diagrams/improve-r32-02a-what-the-loss-weights-up.png` |
| The two ways every checkpoint is rendered | subject | [diagram-prompts.md](../../diagram-prompts.md#prompt-3-subject-the-two-ways-every-checkpoint-is-rendered) | the shipped and stacked samplers step for step | `diagrams/improve-r32-03-two-ways-every-checkpoint-is-rendered.png` |

### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|------|------|-------------|--------------|--------|---------|
| the identity sheet | — | eight rows, existing render beside the render through the new interval, with the per-seed difference under each pair | task 2.2 | ⏳ generated during run | `identity_check/` under the output root |

### Organization workflow

1. The identity sheet stays under the output root; it is a check, not evidence, so it is not filed under `artifacts/results/`.
2. The three pending diagrams are rendered from the scope's map by `/render-diagrams`, not by this plan.

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @...tools/01-the-three-switches-and-the-guidance-interval.md` | **task 0.1**, before any work | conformance and thin instructions reported |
| Cross-reference the plan | `/xref-pillar @...tools/01-the-three-switches-and-the-guidance-interval.md` | **task 0.2**, before any work | terms already documented elsewhere linked |
| Capture patterns | `/ingest-error-pattern --from-run-log` | **the close out**, after any red run | errors added to the catalogs |
| Bring the tree current | `/sync-plan-tree @...tools/01-the-three-switches-and-the-guidance-interval.md` | **the close out** | statuses, running order and Error Matrix match reality |

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**File:** `poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py`
**Relevant section:** the argument parser (which already carries `--ema-decay` at line 128, `--weight-decay` at 131 and `--gradient-checkpointing` at 187) and the dataset build.

```python
ap.add_argument("--exclude-cells", type=str, default=None,
                help="JSON list of {pair, seed} to skip when building the dataset")
ap.add_argument("--train-step-range", type=int, nargs=2, default=None,
                metavar=("LO", "HI"), help="sample training steps from [LO, HI) only")
ap.add_argument("--orth-weight", type=float, default=1.0,
                help="multiplier on the part of the error outside the two experts' span")
```

**File:** `poe_repair/methods/_poe_langevin.py`
**Relevant section:** the sampler carrying `lambda_window` (line 282) and its eta setting. The interval joins them.

**File:** `scripts/showcase/correction_span_common.py`
**Relevant section:** the projection onto the span of the two experts' directions and its complement, already written and used by the correction-span finding. `--orth-weight` calls it rather than reimplementing it.

**File:** `poe_repair/paths.py:150`
**Relevant section:** `GROUP_POOL_CONFIGS`, the hard-coded location of the pool yamls.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

— custom; no skill fits. This is repo-specific code against a trainer and a sampler that only exist here.

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents)

[02: the blind label pass](02-the-blind-label-pass.md) builds the read that judges every render this scope produces, so the baseline can be read the moment the sampler works.

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
