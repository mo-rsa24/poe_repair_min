# 🔬 Calibrate the instrument

Step 47 in the root running order; waits on step 46 (the smoke must pass); next is
[the grid and the figures](05-the-grid-and-the-figures.md).

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/05-when-does-the-outcome-lock-in/plans/04-calibrate-the-instrument.md — <one line on what actually happened>
```

## Recommended skill

— custom; no skill fits (an agreement sweep with bars already designed).
also: `/analyze-run` on the calibration W&B run once it lands.

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 46 (previous) | [wire the oracle](03-wire-the-oracle.md) | the adapter and its smoke |
| **47 (current)** | **Calibrate the instrument** | **buys or refuses trust: 240 states, oracle against teacher, two bars in code, per family** |
| 48 (next) | [the grid and the figures](05-the-grid-and-the-figures.md) | the sweeps and the paper figures |

## Table of contents

- [Position in the plan tree](#position-in-the-plan-tree)
- [Quick context: where you are](#quick-context-where-you-are)
- [Considerations](#considerations)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
- [What happens (visual)](#what-happens-visual)
- [Description: what to build](#description-what-to-build)
- [Purpose and goal](#purpose-and-goal)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [The engagement gate](#the-engagement-gate)
- [Figure Catalog](#figure-catalog)
- [Orchestration](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

## Quick context: where you are

Navigation: ⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The experiment.** On a subsample of roughly 240 cached states (three families, 8 pairs, 2
seeds, 5 spread steps), compare the oracle's settled frame with the teacher's finish-the-run
ending, scoring agreement as scorer-verdict match plus DINOv2 distance, at guidance 7.5.

**The hypothesis.** The distilled student tracks its teacher well enough to instrument the
grid, and comparably across families. If true, plan 05 runs on the oracle at about one
GPU-hour. If false, plan 05 shrinks scope to the families and steps that pass, or runs
finish-the-run everywhere. Rationale: PoE states sit furthest off the tube LCM trained on, and
a bias that differs between compared groups corrupts every cross-family figure, so
comparability is a bar, never an assumption.

**Context details.** Teacher endings by plan 01's script; oracle by plan 03's module; scorer is
the validated instance-count scorer. Distilled students are documented to memorize, which this
pass would catch.

**This plan's job.** Turn "instrument-grade" from a disclaimer into a recorded verdict: adopt,
shrink, or fall back.

**Associated materials.** Verdict: [the review file](../review/04-calibrate-the-instrument.md).
Design reasoning: [the decision ledger](../decisions-taken-here.md).

**For the full picture.** The scope's [master plan](../MASTER_PLAN.md).

## Considerations

Navigation: ⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**Expected runtime.** Teacher side dominates: roughly 100 U-Net calls per state, 240 states, 1
to 3 GPU-hours as the rough figure; measure ten states first and write the number in the Runs
table before launching the rest.

**Prerequisites.** Plan 03's smoke passed; the cache present; the scorer importable.

**Project tracking.** W&B project `prime_lab/poe-repair-animals-compose`, one run; large
outputs to `/datasets/mmolefe/poe_repair_min/outputs/commitment/calibration/`.

**Known issues.** See [Error Matrix](#error-matrix).

<details>
<summary>Environment Facts This Plan Depends On</summary>

- The `co3` absolute python path; see [the environment index](../../../environment/00-INDEX.md).
- biggpu allows one Slurm job per user; a run this size fits in-session on a GPU node, so no
  queue is needed; if queued anyway, harvest reads all three execution modes.
- Large artifacts to `/datasets` only; disk guard on the target filesystem.

</details>

## The claim

Navigation: ⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**A per-family agreement table and one recorded verdict: adopt, shrink, or fall back, judged
against two bars that sit in source before any number exists.**

**Why this matters right now:** every plan-05 figure inherits this verdict; without it the
oracle's frames carry false authority.

## Why this plan exists

Navigation: ⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** Trust in a distilled model off its training distribution cannot be argued from
theory here (training set size and intrinsic dimension are both unknown), and sharp frames are
never evidence of trust.

**The solution.** The teacher is in hand, so the student's error on exactly our states is
computable. Purchase trust empirically, per family.

**Key insights:**
1. Two bars, not one: an absolute floor (`AGREEMENT_FLOOR`), and a comparability requirement
   (`FAMILY_GAP_MAX`), because a per-family bias is the confound that corrupts cross-family
   comparison even when average agreement looks fine.
2. The counts per family are printed from the data, never assumed from the config: a flag that
   selected an empty group is a silent no-op.

## What happens (visual)

Navigation: ⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
 240 states: 3 families x 8 pairs x 2 seeds x 5 steps
      |                          |
   oracle (1-4 calls)        teacher finish (~100 calls)
      |                          |
      +---- agreement per state -+
              |
   per family: scorer-match rate, DINOv2 distance
              |
   bar 1: floor per family      bar 2: max gap between families
              |
   verdict: adopt | shrink | fall back
```

## Description: what to build

Navigation: ⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The calibration script** `scripts/commitment/calibrate_oracle.py`: builds the subsample
   (printing the selected count per family), runs both sides, scores agreement, writes
   `calibration/agreement.json` (one row per state) and `calibration/verdict.json` (per-family
   rates, the two bars, the verdict). Bars `AGREEMENT_FLOOR` and `FAMILY_GAP_MAX` in source.
2. **The W&B run** logging the per-family agreement table and the running cost.
3. **Output root**: `/datasets/mmolefe/poe_repair_min/outputs/commitment/calibration/`.

## Purpose and goal

Navigation: ⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

**Purpose.** Objective 3 of [the master plan](../MASTER_PLAN.md): adopt or reject LCM-SDXL by
pre-set bars.

**Goals:**
1. `agreement.json` holds one row per state with the per-family counts printed.
2. `verdict.json` records adopt, shrink, or fall back, and the review file's bar is answered.

## Tasks

Navigation: ⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Preflight: check this plan before working from it

- [ ] **0.1** Paste: `/verify-plan @plans/05-when-does-the-outcome-lock-in/plans/04-calibrate-the-instrument.md`
  - Done when: the report comes back clean, or its proposals have been applied.

▶ **Next: [task 1.1](#1--build-the-calibration-script)**.

### 1. 🔧 Build the calibration script

◀ **Needs: [task 0.1](#0--preflight-check-this-plan-before-working-from-it)**, and
[03-wire-the-oracle.md's gate](03-wire-the-oracle.md#the-engagement-gate) passed.

- [ ] **1.1** Write `scripts/commitment/calibrate_oracle.py` per the Description, both bars in
  source, per-family selection counts printed before anything runs.
  - **Done when:** a `--dry-run` prints the subsample (240 rows expected; the exact count per
    family) and exits without GPU work. A family with zero selected states stops the plan here.
- [ ] **1.2** Time ten states end to end and extrapolate.
  - **Done when:** the measured per-state cost and the projected total are in the review file's
    Runs table, and the projection is under the walltime available in-session.

▶ **Next: [task 2.1](#2--run-the-calibration)**.

### 2. 🚀 Run the calibration

◀ **Needs: [tasks 1.1 to 1.2](#1--build-the-calibration-script)**.

- [ ] **2.1** Run the full pass with W&B logging on.

    ```bash
    co3 python scripts/commitment/calibrate_oracle.py \
      --out /datasets/mmolefe/poe_repair_min/outputs/commitment/calibration/ \
      --wandb prime_lab/poe-repair-animals-compose
    ```

  - **Done when:** `agreement.json` row count equals the printed subsample count,
    `verdict.json` exists, and the W&B run id is recorded in the review file's Runs table.

▶ **Next: [instruction 3.1](#3--read-the-wb-run-and-record-the-verdict)**.

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted.

- [ ] **Capture the failures this plan hit.**
  - Paste: `/ingest-error-pattern --from-run-log @plans/05-when-does-the-outcome-lock-in/plans/04-calibrate-the-instrument.md`
  - Done when: each failure has a catalog entry, or there were none.
- [ ] **Bring the tree current.**
  - Paste: `/sync-plan-tree @plans/05-when-does-the-outcome-lock-in/plans/04-calibrate-the-instrument.md — <one line>`
  - Done when: statuses match reality.

▶ **Next: the engagement gate.**

## Instructions

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-engagement-gate) ➡️

**For you to follow manually.** Do these yourself.

### 3. 🌐 Read the W&B run, and record the verdict

◀ **Needs: [task 2.1](#2--run-the-calibration)** done, so the run exists.

3.1 **Open W&B** at wandb.ai, project `prime_lab/poe-repair-animals-compose`, find the
   calibration run by its recorded id, open the per-family agreement table.
   - Expected result: three rows (joint, PoE, LoRA-corrected), each with scorer-match rate and
     mean DINOv2 distance.
   - ✅ If both bars pass, record "adopt" in the review file and mark the gate passed.
   - ❌ If the floor fails for a family, record which, and whether "shrink" (drop that family
     or its failing steps) preserves the figures plan 05 owes; record the choice.
   - ❌ If the family gap exceeds its bar, record "fall back": plan 05 runs finish-the-run.

3.2 **Spot-check five disagreeing states by eye.**
   - [ ] Open the five worst side-by-sides under `calibration/`
   - [ ] Record one line each: does the disagreement look like blur, a different subject, or a
     different composition. This shapes how plan 05's captions describe oracle error.

▶ **Next: the engagement gate**, then [plan 05](05-the-grid-and-the-figures.md).

## The engagement gate

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **This is the scope's trust gate: plan 05's instrument choice is decided here, and figures
> built on an uncalibrated oracle would carry false authority into the paper.**

- **Pass criteria:**
  - Every family's agreement is at or above `AGREEMENT_FLOOR`, and no family sits more than
    `FAMILY_GAP_MAX` below another.
- **Fail criteria (not a stop; a fork):**
  - A failed floor or gap sends plan 05 down its named fallback (shrink, or finish-the-run
    everywhere); the verdict is recorded either way and the scope continues.

**When you get results, answer** [the review file](../review/04-calibrate-the-instrument.md).

## Figure Catalog

Navigation: ⬅️ [Previous](#the-engagement-gate) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

#### Pending: to be generated from prompts

| Item | Lane | Prompt file | What it shows | Save to |
|------|------|-------------|---------------|---------|
| The calibration gauge | subject | [diagram-prompts.md](../diagram-prompts.md#prompt-4-subject-the-calibration-gauge) | student against teacher, two bars | `../diagrams/when-does-the-outcome-lock-in-04-the-calibration-gauge.png` |
| Process lane v01 | process | [diagram-prompts.md](../diagram-prompts.md#process-lane) | the five plans as a journey | `../diagrams/when-does-the-outcome-lock-in-process-01.png` |

#### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|------|------|-------------|--------------|--------|---------|
| Per-family agreement bars | — | agreement rate per family with both bars drawn | `calibrate_oracle.py` | ⏳ | filed to `artifacts/results/when-does-the-outcome-lock-in/oracle-agreement-per-family__240-state-calibration.png` |

#### Organization workflow

1. Run; 2. Read in W&B; 3. File the bar chart with its card; 4. Link here.

## Orchestration: keeping catalogs and plan files in sync

Navigation: ⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/05-when-does-the-outcome-lock-in/plans/04-calibrate-the-instrument.md` | **task 0.1** | conformance reported |
| Capture patterns | `/ingest-error-pattern --from-run-log @plans/05-when-does-the-outcome-lock-in/plans/04-calibrate-the-instrument.md` | **the close out** | errors catalogued |
| Bring the tree current | `/sync-plan-tree @plans/05-when-does-the-outcome-lock-in/plans/04-calibrate-the-instrument.md` | **the close out** | statuses match reality |
| Organize outputs | manual move + update Figure Catalog | after completion | deliverables linked |

## Code references

Navigation: ⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

**File:** `scripts/commitment/calibrate_oracle.py` (new)

```python
AGREEMENT_FLOOR = ...   # set before any run; the absolute per-family floor
FAMILY_GAP_MAX  = ...   # set before any run; max allowed gap between families
# per state: oracle(x_t, t, prompt) vs teacher_finish(x_t, t); scorer + DINOv2
```

**File:** `poe_repair/lcm_oracle.py` (from [plan 03](03-wire-the-oracle.md))

## Next step

Navigation: ⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents)

[The grid and the figures](05-the-grid-and-the-figures.md), on whichever instrument this
plan's verdict chose.

## Error Matrix

Navigation: ⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

<details>
<summary>No failures catalogued yet</summary>

#### From global catalog

(Patterns applicable across all projects)

#### From project catalog

(Patterns specific to this project)

**Auto-update note:** regenerated by `/sync-plan-tree`; do not edit manually.

</details>
