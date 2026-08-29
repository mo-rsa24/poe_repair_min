# 🔬 Wire the oracle

Step 46 in the root running order; waits on step 44 (the premise must stand); next is
[calibration](04-calibrate-the-instrument.md).

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/closing-the-compositional-gap/plans/when-does-the-outcome-lock-in/plans/03-wire-the-oracle.md — <one line on what actually happened>
```

## Recommended skill

— custom; no skill fits (a model download plus one adapter module).

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 45 (previous) | [the free probe](02-the-free-probe.md) | probe one, from the cache alone |
| **46 (current)** | **Wire the oracle** | **LCM-SDXL downloaded, adapted to cached states, and smoke-tested against one teacher ending** |
| 47 (next) | [calibrate the instrument](04-calibrate-the-instrument.md) | the 240-state agreement pass |

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

**The system.** An adapter that hands any cached DDIM state to LCM-SDXL and gets back a settled
endpoint frame: the counterfactual finish, the image the plain guided model would land on if
composing stopped at that step.

**The design.** LCM-SDXL (distilled from this exact base model, same VAE latent space) behind
one module that maps our timesteps and states into its conventions, passes guidance 7.5 through
its scale embedding (`time_cond_proj_dim: 256` in its unet config), and asserts no
negative-prompt double pass runs on top, because real classifier-free guidance stacked on the
embedded scale is the one configuration that genuinely corrupts the read.

**Context details.** Checkpoint `latent-consistency/lcm-sdxl` (openrail++), 2 to 8 inference
steps, LCMScheduler. Its card's "use guidance 1.0 to 2.0" line is stale advice belonging to the
LoRA variant; the config is the authority. The trained guidance range is undocumented; plan 04
covers it empirically at exactly 7.5.

**This plan's job.** Make the instrument exist and pass one smoke, so calibration has something
to judge.

**Associated materials.** Verdict: [the review file](../review/03-wire-the-oracle.md). The
mechanism reasoning: [the decision ledger](../decisions-taken-here.md).

**For the full picture.** The scope's [master plan](../MASTER_PLAN.md).

## Considerations

Navigation: ⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**Expected runtime.** Download a few GB once; the smoke itself is under a minute of GPU.

**Prerequisites.** Plan 01 passed its gate; a GPU node; disk room on `/datasets` for the
checkpoint (the download task prints where it landed and how big it is).

**Project tracking.** Smoke outputs to
`/datasets/mmolefe/poe_repair_min/outputs/commitment/lcm_smoke/`.

**Known issues.** See [Error Matrix](#error-matrix).

<details>
<summary>Environment Facts This Plan Depends On</summary>

- The `co3` absolute python path; see [the environment index](../../../../../environment/00-INDEX.md).
- Model weights and all large artifacts to `/datasets` only: set the HF cache env var to a
  `/datasets` path before downloading; the task asserts the landing filesystem.
- fp16 variant of the checkpoint; fp16 upcast rule for any arithmetic on latents.

</details>

## The claim

Navigation: ⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**One adapter module and one passed smoke: a cached state in, a settled frame out, agreeing
with the teacher's finish on the same state.**

**Why this matters right now:** plans 04 and 05 both call this module; a parameterisation
mistake here produces plausible wrong frames everywhere downstream.

## Why this plan exists

Navigation: ⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** Our code treats network output as epsilon under DDIM; LCM is an x0-style
consistency function with its own boundary conditions and scheduler. Carrying a state across
that boundary casually is the classic off-by-a-factor failure, and here it fails silently as
plausible-looking frames.

**The solution.** One adapter, three asserts (embedding path taken, no double guidance,
timestep mapping exact), one smoke against ground truth.

**Key insights:**
1. The smoke's teacher ending comes from plan 01's finish-the-run script, so the ground truth
   costs nothing new.
2. The prompt is an argument, never hardcoded: both sweeps (joint prompt, expert pair) call
   this same module in plan 05.

## What happens (visual)

Navigation: ⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
 cached x_t, our DDIM grid          LCM-SDXL conventions
 ------------------------          --------------------
 x_t, step index t   --map-->      x_t, LCM timestep
 prompt, w = 7.5     --embed->     w-embedding (no negative pass; asserted)
                       |
                       v
              settled frame x0_hat
                       |
        compare: teacher finish from same x_t
```

## Description: what to build

Navigation: ⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The download**, pinned to `/datasets` via the HF cache env var, size and path printed.
2. **The adapter** `poe_repair/lcm_oracle.py`: `oracle(x_t, t, prompt, w=7.5, steps=4)` returning
   the settled latent and its decoded frame. Asserts: `unet.config.time_cond_proj_dim` present;
   the pipeline call path takes the embedding branch, never a second unconditional pass; the
   timestep mapping from our 50-step grid to LCM's is exact and printed once.
3. **The smoke** `scripts/commitment/lcm_smoke.py`: one cell's step-25 state through the oracle
   (joint prompt) beside the teacher's finish from plan 01's script; scorer verdicts and DINOv2
   distance printed; smoke bar `SMOKE_SCORER_MATCH = True` and `SMOKE_DINO_DIST_MAX` in source.

## Purpose and goal

Navigation: ⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

**Purpose.** Objective 3 of [the master plan](../MASTER_PLAN.md): adopt or reject LCM-SDXL by
pre-set bars; this plan builds the thing the bars judge.

**Goals:**
1. The adapter exists with its three asserts, and the smoke passes its bar.
2. The checkpoint's landing path and size are recorded in the review file.

## Tasks

Navigation: ⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Preflight: check this plan before working from it

- [ ] **0.1** Paste: `/verify-plan @plans/closing-the-compositional-gap/plans/when-does-the-outcome-lock-in/plans/03-wire-the-oracle.md`
  - Done when: the report comes back clean, or its proposals have been applied.

▶ **Next: [task 1.1](#1--download-and-adapt)**.

### 1. 🔧 Download and adapt

◀ **Needs: [task 0.1](#0--preflight-check-this-plan-before-working-from-it)**, and
[01-basins-by-hand.md's gate](01-basins-by-hand.md#the-engagement-gate) passed, so the premise stands.

- [ ] **1.1** Download `latent-consistency/lcm-sdxl` (fp16 variant) with the HF cache pointed at
  `/datasets`, then print the landing path, its filesystem, and the size on disk.
  - **Done when:** the printed path is under `/datasets` and the size is recorded in the review
    file's Runs table.
  - 💡 `/video-scout "running LCM-SDXL few-step inference in diffusers, I need to wire cached SDXL latents into it myself"`:
    twenty minutes of someone competent doing this wiring, worth it before writing the adapter.
- [ ] **1.2** Write `poe_repair/lcm_oracle.py` per the Description, three asserts included.
  - **Done when:** importing the module and calling it on a random latent returns a frame and
    prints the timestep mapping once.

▶ **Next: [task 2.1](#2--run-the-smoke)**.

### 2. 🚀 Run the smoke

◀ **Needs: [tasks 1.1 to 1.2](#1--download-and-adapt)**.

- [ ] **2.1** Run the one-state smoke.

    ```bash
    co3 python scripts/commitment/lcm_smoke.py --cell <plan-01-chosen-cell> --step 25 \
      --out /datasets/mmolefe/poe_repair_min/outputs/commitment/lcm_smoke/
    ```

  - **Done when:** the smoke prints its scorer verdicts and DINOv2 distance, the side-by-side
    PNG exists, and the bar's pass or fail is recorded in the review file.

▶ **Next: [instruction 3.1](#3--look-at-the-two-frames)**.

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted.

- [ ] **Capture the failures this plan hit.**
  - Paste: `/ingest-error-pattern --from-run-log @plans/closing-the-compositional-gap/plans/when-does-the-outcome-lock-in/plans/03-wire-the-oracle.md`
  - Done when: each failure has a catalog entry, or there were none.
- [ ] **Bring the tree current.**
  - Paste: `/sync-plan-tree @plans/closing-the-compositional-gap/plans/when-does-the-outcome-lock-in/plans/03-wire-the-oracle.md — <one line>`
  - Done when: statuses match reality.

▶ **Next: the engagement gate.**

## Instructions

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-engagement-gate) ➡️

**For you to follow manually.** Do these yourself.

### 3. 👁️ Look at the two frames

◀ **Needs: [task 2.1](#2--run-the-smoke)** done, so the side-by-side exists.

3.1 **Open the side-by-side PNG** under
   `/datasets/mmolefe/poe_repair_min/outputs/commitment/lcm_smoke/`.
   - Expected result: the oracle's frame and the teacher's ending, same subject, comparable
     composition; sharpness may differ (4 steps against 25).
   - ✅ If they read as the same image to the eye, note it beside the numbers in the review file.
   - ❌ If they read as different images while the numbers claim agreement, the instrument
     check in the review file fails regardless of the numbers; record what you saw.

▶ **Next: the engagement gate**, then [plan 04](04-calibrate-the-instrument.md).

## The engagement gate

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **A failed smoke here is cheap; the same failure discovered inside plan 05's grid is not.**

- **Pass criteria:**
  - All three asserts hold on a real call.
  - The smoke's scorer verdicts match and the DINOv2 distance is under its bar, and the eyeball
    read concurs.
- **Fail criteria (STOP):**
  - Any assert fires, or the smoke frames disagree. Fix the adapter or record the checkpoint as
    unusable; calibration does not run on a failed smoke.

**When you get results, answer** [the review file](../review/03-wire-the-oracle.md).

## Figure Catalog

Navigation: ⬅️ [Previous](#the-engagement-gate) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

#### Pending: to be generated from prompts

| Item | Lane | Prompt file | What it shows | Save to |
|------|------|-------------|---------------|---------|
| The endpoint spyglass | subject | [diagram-prompts.md](../diagram-prompts.md#prompt-2-subject-the-endpoint-spyglass) | the oracle reading a state ahead | `../diagrams/when-does-the-outcome-lock-in-02-the-endpoint-spyglass.png` |
| Process lane v01 | process | [diagram-prompts.md](../diagram-prompts.md#process-lane) | the five plans as a journey | `../diagrams/when-does-the-outcome-lock-in-process-01.png` |

#### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|------|------|-------------|--------------|--------|---------|
| Smoke side-by-side | — | oracle frame beside teacher ending, one state | `lcm_smoke.py` | ⏳ | filed to `artifacts/results/when-does-the-outcome-lock-in/oracle-vs-teacher-ending__one-state-smoke.png` |

#### Organization workflow

1. Run; 2. Look; 3. File with card; 4. Link here.

## Orchestration: keeping catalogs and plan files in sync

Navigation: ⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/closing-the-compositional-gap/plans/when-does-the-outcome-lock-in/plans/03-wire-the-oracle.md` | **task 0.1** | conformance reported |
| Capture patterns | `/ingest-error-pattern --from-run-log @plans/closing-the-compositional-gap/plans/when-does-the-outcome-lock-in/plans/03-wire-the-oracle.md` | **the close out** | errors catalogued |
| Bring the tree current | `/sync-plan-tree @plans/closing-the-compositional-gap/plans/when-does-the-outcome-lock-in/plans/03-wire-the-oracle.md` | **the close out** | statuses match reality |
| Organize outputs | manual move + update Figure Catalog | after completion | deliverables linked |

## Code references

Navigation: ⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

**File:** `poe_repair/lcm_oracle.py` (new)
**Function:** `oracle(x_t, t, prompt, w=7.5, steps=4)`

```python
assert unet.config.time_cond_proj_dim is not None      # w goes in as an embedding
assert not pipeline_does_cfg_double_pass               # double-guiding corrupts the read
# map our DDIM step index to LCM's timestep grid, exactly, printed once
```

**File:** [poe_repair/runtime.py](../../../../../poe_repair/runtime.py)
**Relevant section:** the schedule the timestep mapping is checked against.

## Next step

Navigation: ⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents)

[Calibrate the instrument](04-calibrate-the-instrument.md): 240 states, two bars, the verdict
that decides whether plan 05 runs on the oracle or on the teacher.

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
