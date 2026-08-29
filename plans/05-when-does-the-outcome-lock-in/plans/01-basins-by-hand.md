# 🔬 Basins by hand

Step 44 in the root running order; waits on nothing; next is
[the free probe](02-the-free-probe.md).

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/05-when-does-the-outcome-lock-in/plans/01-basins-by-hand.md — <one line on what actually happened>
```

## Recommended skill

— custom; no skill fits (a twenty-line probe script against repo code).

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 43 (previous) | [revalidate the scorer off animals](../../01-showcase-the-trained-lora/plans/13-revalidate-the-scorer-off-animals.md) | the label-pass validation in the sibling scope |
| **44 (current)** | **Basins by hand** | **proves basins and a ridge exist for the composed flow, before any new model is downloaded** |
| 45 (next) | [the free probe](02-the-free-probe.md) | posterior-mean drift as the first speciation number |

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

**The experiment.** Take one cached mid-run state, nudge it twice by 1% of its norm, finish the
run three times with the repo's own DDIM code, and see whether the three endings agree. Repeat
at an early, a middle, and a late step.

**The hypothesis.** Basins are real for the composed flow: at a late step the three endings are
the same image, at an early step they are free to differ. If true, the commitment probe this
scope builds has a well-posed thing to measure. If false, the valley picture is wrong, the
basin-oracle premise dies here, and the scope re-marks before any model download. Rationale:
a deterministic flow assigns every state one endpoint, and endpoints should cluster into modes.

**Context details.** One pair-and-seed cell from the cached trajectories (each cell holds a
50-step `latent_trajectory.pt`); steps 5, 25 and 40; three endings per step, roughly 300 U-Net
calls per step-triplet.

**This plan's job.** Prove the premise cheaply, in an afternoon, before plans 03 to 05 spend
anything on the oracle.

**Associated materials.** The verdict lands in [the review file](../review/01-basins-by-hand.md).
The reasoning behind the design is in [the decision ledger](../decisions-taken-here.md); the
full walk record is [the walk map](../../.walk/consistency-model-basin-oracle.md).

**For the full picture.** The scope's [master plan](../MASTER_PLAN.md).

## Considerations

Navigation: ⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**Expected runtime.** A few minutes of GPU per step-triplet, three triplets total; measure the
first and write the number into the review file's Runs table.

**Prerequisites.** A GPU node and the cached cells present on `/datasets`.

**Project tracking.** Output is small and local: `/datasets/mmolefe/poe_repair_min/outputs/commitment/basins_by_hand/`.
No W&B run needed at this size.

**Known issues.** See [Error Matrix](#error-matrix) for a full catalog.

<details>
<summary>Environment Facts This Plan Depends On</summary>

- The `co3` absolute python path runs everything; see [the environment index](../../../environment/00-INDEX.md).
- Large artifacts go to `/datasets` only; the script's disk guard checks the filesystem it
  actually writes to.
- fp16 upcast rule applies when decoding latents.

</details>

## The claim

Navigation: ⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**One cell, three probed steps, nine finished endings: enough to prove or kill the premise that
the composed flow has basins with an unstable ridge between them.**

**Why this matters right now:** every later plan in this scope measures "which basin, decided
when"; if there are no basins, there is nothing to measure and the scope stops at a cost of one
afternoon.

## Why this plan exists

Navigation: ⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The question.** The scope's premise is a picture: states slide into valleys, a ridge
separates them, a small nudge near the ridge decides the ending. Nothing in the repo has tested
that picture on the composed flow.

**The approach.** The cheapest possible test, using only repo code: perturb and finish.

**Key insights:**
1. Agreement at a late step and freedom to differ at an early step is exactly what basins
   plus a ridge predict; disagreement at a late step under a 1% nudge kills the picture.
2. The same script with steps subsampled seeds the calibration teacher in
   [plan 04](04-calibrate-the-instrument.md), so nothing here is throwaway.

## What happens (visual)

Navigation: ⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
 one cached state x_t   (t = 5, 25, 40 in turn)

    x - d           x            x + d      d = random direction,
      |             |              |            1% of ||x||
      |   finish the remaining     |
      |   DDIM steps, composed eps |
      v             v              v
   ending L      ending M       ending R

 L = M = R at t=40, free to differ at t=5  -> basins real
 L != R at t=40 under a 1% nudge           -> picture wrong, STOP
```

## Description: what to build

Navigation: ⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The probe script** `scripts/commitment/perturb_finish.py`. Loads one cell's
   `latent_trajectory.pt`, builds the two nudged copies, finishes all three from a given step
   with the same composed epsilon the cache was made with (reusing `load_ddim_scheduler` and
   `ddim_prev_from_x0_eps`), decodes the endings, saves PNGs and a JSON of relative latent
   distances. The agreement bar lives in this file as a constant
   (`REL_ENDING_DIST_MAX`), set before any run.
2. **The output**: `/datasets/mmolefe/poe_repair_min/outputs/commitment/basins_by_hand/`
   holding nine PNGs, one JSON per probed step, and one 3x3 contact-sheet image.

## Purpose and goal

Navigation: ⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

**Purpose.** Objective 1 of [the master plan](../MASTER_PLAN.md): establish whether basin
structure is measurable on the cached trajectories at all.

**Goals:**
1. Nine endings exist on disk with their distance JSONs.
2. The review file's bar is answered ✅ or ❌, either way with the numbers.

## Tasks

Navigation: ⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Preflight: check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/05-when-does-the-outcome-lock-in/plans/01-basins-by-hand.md`
  - Done when: the report comes back clean, or its proposals have been applied.

▶ **Next: [task 1.1](#1--build-the-probe-script)**, the first real work.

### 1. 🔧 Build the probe script

◀ **Needs: [task 0.1](#0--preflight-check-this-plan-before-working-from-it)**, so the plan is known good.

- [ ] **1.1** Locate one cached cell and print what was found.
  - List the pair-and-seed cells under `/datasets/mmolefe/poe_repair_min/outputs/` that hold a
    `latent_trajectory.pt` (the same cells the trajectory-divergence analyses read), print the
    count and the chosen cell's path.
  - **Done when:** the chosen cell's path and the total cell count are printed and recorded in
    the review file's orientation paragraph. A count of zero stops the plan here.
- [ ] **1.2** Write `scripts/commitment/perturb_finish.py` per the Description, bar constant
  `REL_ENDING_DIST_MAX` in the source.
  - **Done when:** the script exists and `co3 python scripts/commitment/perturb_finish.py --help`
    prints its arguments.

▶ **Next: [task 2.1](#2--run-the-three-triplets)**.

### 2. 🚀 Run the three triplets

◀ **Needs: [tasks 1.1 to 1.2](#1--build-the-probe-script)** done, so the script and the cell exist.

- [ ] **2.1** Run the probe at steps 5, 25 and 40 on the chosen cell.

    ```bash
    co3 python scripts/commitment/perturb_finish.py --cell <chosen-cell-path> \
      --steps 5 25 40 --nudge 0.01 \
      --out /datasets/mmolefe/poe_repair_min/outputs/commitment/basins_by_hand/
    ```

  - **Done when:** nine PNGs, three JSONs and the contact sheet exist under the output path,
    and the wall time is recorded in the review file's Runs table.

▶ **Next: [instruction 3.1](#3--judge-the-endings-by-eye)** (the eyeball read of the contact sheet).

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted, including the ones that went red.

- [ ] **Capture the failures this plan hit**, while they are still fresh.
  - Paste: `/ingest-error-pattern --from-run-log @plans/05-when-does-the-outcome-lock-in/plans/01-basins-by-hand.md`
  - Done when: each failure has a catalog entry, or there were none to record.
- [ ] **Bring the tree current** with what actually happened.
  - Paste: `/sync-plan-tree @plans/05-when-does-the-outcome-lock-in/plans/01-basins-by-hand.md — <one line on what you did>`
  - Done when: statuses, the running order and the Error Matrix match reality.

▶ **Next: the engagement gate.**

## Instructions

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-engagement-gate) ➡️

**For you to follow manually.** Do these yourself.

### 3. 👁️ Judge the endings by eye

◀ **Needs: [task 2.1](#2--run-the-three-triplets)** done, so the contact sheet exists.

3.1 **Open the contact sheet** at
   `/datasets/mmolefe/poe_repair_min/outputs/commitment/basins_by_hand/contact_sheet.png`
   (over the SSH port-forward image viewer or by copying it local).
   - Expected result: a 3x3 grid, rows = probed steps 5, 25, 40; columns = nudged-down,
     untouched, nudged-up endings.
   - ✅ If the step-40 row shows three versions of the same image and the step-5 row shows any
     visible divergence, record "basins real" in the review file.
   - ❌ If the step-40 row shows different images, record "picture wrong" in the review file and
     stop the scope at the gate below.

3.2 **Record the numbers beside the eyeball read.**
   - [ ] Open the three JSONs, copy each step's two relative distances into
     [the review file](../review/01-basins-by-hand.md) under the bar question.

▶ **Next: the engagement gate**, then [plan 02](02-the-free-probe.md).

## The engagement gate

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **This gate protects every later plan in the scope**: if basins are not real for the composed
> flow, plans 02 to 05 measure nothing and must not run.

- **Pass criteria:**
  - Step-40 endings agree under `REL_ENDING_DIST_MAX` (both relative distances below the bar)
    and the eyeball read concurs.
  - Step-5 endings were free to differ (no requirement that they do).
- **Fail criteria (STOP):**
  - Step-40 endings disagree under a 1% nudge, by number or by eye. The scope's premise is
    wrong; stop and re-mark.
- **Partial pass guidance:**
  - Agreement at 40 but oddities at 25 (inside the divergence window) is expected territory,
    not a failure; note it under "Asked after the result".

**When you get results, answer** [the review file](../review/01-basins-by-hand.md) **or move to** [Next step](#next-step).

## Figure Catalog

Navigation: ⬅️ [Previous](#the-engagement-gate) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

#### Pending: to be generated from prompts

| Item | Lane | Prompt file | What it shows | Save to |
|------|------|-------------|---------------|---------|
| The free probes | subject | [diagram-prompts.md](../diagram-prompts.md#prompt-3-subject-the-free-probes) | the perturbation triplet and the drift curve | `../diagrams/when-does-the-outcome-lock-in-03-the-free-probes.png` |
| Process lane v01 | process | [diagram-prompts.md](../diagram-prompts.md#process-lane) | the five plans as a journey with their gates | `../diagrams/when-does-the-outcome-lock-in-process-01.png` |

#### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|------|------|-------------|--------------|--------|---------|
| Contact sheet | — | 3x3 grid of endings, rows = steps, columns = nudges | `perturb_finish.py` | ⏳ Generated during run | copied to `artifacts/results/when-does-the-outcome-lock-in/endings-under-a-1pc-nudge__steps-5-25-40.png` with its README card |

#### Organization workflow

1. Run the probe; 2. Judge by eye; 3. Copy the contact sheet into `artifacts/results/` with its
card entry; 4. Link it here.

## Orchestration: keeping catalogs and plan files in sync

Navigation: ⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/05-when-does-the-outcome-lock-in/plans/01-basins-by-hand.md` | **task 0.1**, before any work | conformance reported |
| Capture patterns | `/ingest-error-pattern --from-run-log @plans/05-when-does-the-outcome-lock-in/plans/01-basins-by-hand.md` | **the close out**, after any red run | errors added to catalogs |
| Bring the tree current | `/sync-plan-tree @plans/05-when-does-the-outcome-lock-in/plans/01-basins-by-hand.md` | **the close out** | statuses match reality |
| Organize outputs | manual move + update Figure Catalog | after completion | deliverables linked |

This table is the reference, not the trigger: the trigger is the task line each row names.

## Code references

Navigation: ⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

**File:** [poe_repair/runtime.py](../../../poe_repair/runtime.py)
**Functions:** `load_ddim_scheduler`, `ddim_prev_from_x0_eps`
**Relevant section:** the DDIM stepping the probe reuses; the probe never reimplements the update.

```python
# perturb_finish, the core loop
u = torch.randn_like(x); u /= u.norm()
for delta in (-1, 0, +1):
    z = x + delta * 0.01 * x.norm() * u
    ending[delta] = finish_ddim(z, t_start, composed_eps)  # repo stepping, unchanged
```

## Next step

Navigation: ⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents)

[The free probe](02-the-free-probe.md): posterior-mean drift as the first speciation number,
needing no new model.

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
