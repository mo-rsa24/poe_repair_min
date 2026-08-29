# 📊 The grid and the figures

Step 48 in the root running order; waits on steps 45 and 47; last plan in the scope.

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/closing-the-compositional-gap/plans/when-does-the-outcome-lock-in/plans/05-the-grid-and-the-figures.md — <one line on what actually happened>
```

## Recommended skill

— custom; no skill fits (the sweep runner and the figure scripts are bespoke).
also: `/design-figure` before building the three-timestamp figure, if its layout needs deciding
beyond what the ledger already fixed.

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 47 (previous) | [calibrate the instrument](04-calibrate-the-instrument.md) | the trust verdict |
| **48 (current)** | **The grid and the figures** | **both sweeps with stability copies, the speciation table, and the scope's paper figures** |
| after | [the master plan's recall gallery](../MASTER_PLAN.md#definition-of-done) | closes the scope once every plan is ✅ |

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

**The experiment.** Run the commitment probe over the full grid: three families, all cached
pairs and seeds, all 50 steps, two sweeps (joint prompt; expert pair), three stability copies
per read. From it: the per-cell speciation table, the per-step counterfactual compose-rate
curves per family, destination filmstrips for chosen cells, and the three-timestamp figure.

**The hypothesis.** Speciation clusters at or before the correction window's end (step 10)
across families, which explains the 8-to-26-step gap as decide-then-descend. If it clusters
inside the divergence band (18 to 36) instead, that story is dead and the scope's outcome is
the honest kill. Rationale: pre-registered in the decision ledger before any run.

**Context details.** Instrument per plan 04's verdict (oracle, shrunk oracle, or
finish-the-run). Every oracle read carries its stability check; reads that flip under a 1%
nudge are marked on-the-fence, never averaged in silently. Captions name the sweep that
produced each figure.

**This plan's job.** Deliver the scope's evidence and figures, with the instrument's verdict
and the probes' cross-check visible in every caption.

**Associated materials.** Verdict: [the review file](../review/05-the-grid-and-the-figures.md).
Figure rules: [the decision ledger](../decisions-taken-here.md) (PCA with printed variance,
UMAP excluded, state-space pictures paired with outcome curves).

**For the full picture.** The scope's [master plan](../MASTER_PLAN.md).

## Considerations

Navigation: ⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**Expected runtime.** On the oracle: roughly 18k states times up to 3 copies at 1 to 4 calls,
about one GPU-hour as the rough figure. On finish-the-run fallback: tens of GPU-hours, batched
by family. Measure one family slice first; write both numbers in the Runs table.

**Prerequisites.** Plan 04's verdict recorded; plan 02's speciation table standing.

**Project tracking.** W&B project `prime_lab/poe-repair-animals-compose`, one run per sweep;
outputs to `/datasets/mmolefe/poe_repair_min/outputs/commitment/grid/`.

**Known issues.** See [Error Matrix](#error-matrix).

<details>
<summary>Environment Facts This Plan Depends On</summary>

- The `co3` absolute python path; see [the environment index](../../../../../environment/00-INDEX.md).
- Long sweeps outside Slurm run with `nohup` on the session node (biggpu allows one job per
  user); harvest reads the queue, the node's processes, and the output count.
- Large artifacts to `/datasets` only; disk guard on the target filesystem.

</details>

## The claim

Navigation: ⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**The speciation table, the compose-rate curves, the filmstrips, and the three-timestamp
figure, each caption naming its sweep and its instrument, judged against the pre-registered
cluster question.**

**Why this matters right now:** this is the deliverable the scope exists for; the paper's
mechanism section either gains its explanation of the window-versus-divergence gap here or
records the kill.

## Why this plan exists

Navigation: ⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The gap.** Plans 01 to 04 build and validate instruments; nothing yet measures the grid or
draws a figure the paper can cite.

**The approach.** One sweep runner, then figure scripts reading its outputs; state-space views
(PCA overlay) paired with outcome curves per the ledger's rules.

**Key insights:**
1. The cluster question is judged per family and overall, with the bar in source
   (`SPECIATION_EARLY_MAX = 10`, the step at or below which clustering supports
   decide-then-descend; the divergence band 18 to 36 is read from the existing analyses).
2. Disagreement between the oracle's speciation and plan 02's drift settling is reported as a
   finding, never reconciled silently.

## What happens (visual)

Navigation: ⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
 grid: 3 families x pairs x seeds x 50 steps x {x, x+d, x-d}
   |
   sweep A: joint prompt      sweep B: expert pair
   |                          |
   compose rate per step      which-mode label per step
   per family                 + filmstrips (chosen cells)
   |
   speciation per cell (bar in code)  + stability flags
   |
   three-timestamp figure: |window 0-10| speciation | divergence 18-36|
   + PCA overlay (variance printed)  + scatter vs plan 02's settling
```

## Description: what to build

Navigation: ⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The sweep runner** `scripts/commitment/grid_sweep.py`: both sweeps, stability copies,
   instrument chosen by plan 04's `verdict.json`, per-family counts printed, resumable,
   rows to `grid/speciation.json` and per-step scores to `grid/scores.json`.
2. **The figure scripts** `scripts/commitment/figures.py`: the three-timestamp figure, the
   per-family compose-rate curves, filmstrips for the cells named in the review file before
   the run, the PCA overlay with its explained-variance number printed on the axes, and the
   oracle-vs-drift speciation scatter.
3. **Filing**: figures and sidecars to
   `artifacts/results/when-does-the-outcome-lock-in/`, register slots updated in the paper's
   figure register.

## Purpose and goal

Navigation: ⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

**Purpose.** Objective 4 of [the master plan](../MASTER_PLAN.md), and the scope's Expected
Outcome.

**Goals:**
1. `speciation.json` has one row per cell per family per sweep, counts printed and matching.
2. The five figure kinds exist with sidecars, captions naming sweep and instrument.
3. The cluster question is answered in the review file, either way.

## Tasks

Navigation: ⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Preflight: check this plan before working from it

- [ ] **0.1** Paste: `/verify-plan @plans/closing-the-compositional-gap/plans/when-does-the-outcome-lock-in/plans/05-the-grid-and-the-figures.md`
  - Done when: the report comes back clean, or its proposals have been applied.

▶ **Next: [task 1.1](#1--build-the-sweep-runner)**.

### 1. 🔧 Build the sweep runner

◀ **Needs: [task 0.1](#0--preflight-check-this-plan-before-working-from-it)**, plan 04's
verdict recorded, and the filmstrip cells named in
[the review file](../review/05-the-grid-and-the-figures.md) before anything runs.

- [ ] **1.1** Write `scripts/commitment/grid_sweep.py` per the Description; `--dry-run` prints
  the grid size per family and the instrument the verdict chose.
  - **Done when:** the dry run's counts match the cache and the chosen instrument is printed.
- [ ] **1.2** Time one family slice and extrapolate; record both instrument costs in the Runs
  table.
  - **Done when:** the projection fits the session (oracle path) or is batched with `nohup`
    (fallback path), and the choice is recorded.

▶ **Next: [task 2.1](#2--run-the-sweeps)**.

### 2. 🚀 Run the sweeps

◀ **Needs: [tasks 1.1 to 1.2](#1--build-the-sweep-runner)**.

- [ ] **2.1** Run sweep A (joint prompt) and sweep B (expert pair), stability copies on, W&B
  logging on.

    ```bash
    co3 python scripts/commitment/grid_sweep.py --sweep joint --stability 3 \
      --out /datasets/mmolefe/poe_repair_min/outputs/commitment/grid/ \
      --wandb prime_lab/poe-repair-animals-compose
    co3 python scripts/commitment/grid_sweep.py --sweep experts --stability 3 \
      --out /datasets/mmolefe/poe_repair_min/outputs/commitment/grid/ \
      --wandb prime_lab/poe-repair-animals-compose
    ```

  - **Done when:** both JSON row counts equal the printed grid size, and the W&B run ids are
    in the Runs table.
- [ ] **2.2** Build all five figure kinds and file them with sidecars into
  `artifacts/results/when-does-the-outcome-lock-in/`, card entries written.
  - **Done when:** the files exist, each sidecar names sweep, instrument, and the bar it was
    judged against, and the Figure Catalog below links them.

▶ **Next: [instruction 3.1](#3--judge-the-filmstrips-and-the-captions)**.

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted.

- [ ] **Capture the failures this plan hit.**
  - Paste: `/ingest-error-pattern --from-run-log @plans/closing-the-compositional-gap/plans/when-does-the-outcome-lock-in/plans/05-the-grid-and-the-figures.md`
  - Done when: each failure has a catalog entry, or there were none.
- [ ] **Bring the tree current.**
  - Paste: `/sync-plan-tree @plans/closing-the-compositional-gap/plans/when-does-the-outcome-lock-in/plans/05-the-grid-and-the-figures.md — <one line>`
  - Done when: statuses match reality, and the master plan's recall-gallery criterion is the
    only thing left open in the scope.

▶ **Next: the engagement gate.**

## Instructions

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-engagement-gate) ➡️

**For you to follow manually.** Do these yourself.

### 3. 👁️ Judge the filmstrips and the captions

◀ **Needs: [task 2.2](#2--run-the-sweeps)** done, so the figures exist.

3.1 **Open each filmstrip** under `artifacts/results/when-does-the-outcome-lock-in/`.
   - Expected result: sharp destination frames from early steps onward, with on-the-fence
     steps visibly marked where the stability check flipped.
   - ✅ If the marked fence region sits where the endpoint visibly changes, record "filmstrips
     judged consistent" in the review file.
   - ❌ If frames look committed where the stability flag says fence (or the reverse), record
     the cells; the stability bar or the filmstrip rendering is wrong.

3.2 **Read every caption against its sidecar.**
   - [ ] Each caption names its sweep (joint or experts) and its instrument (oracle or teacher)
   - [ ] The PCA overlay prints its explained variance on the axes
   - [ ] Record "captions carry their sweep" in the review file, or list the offenders

3.3 **Read the three-timestamp figure and answer the cluster question** in
   [the review file](../review/05-the-grid-and-the-figures.md), per family and overall.

▶ **Next: the engagement gate**, then the scope's recall gallery per
[the master plan](../MASTER_PLAN.md#definition-of-done).

## The engagement gate

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **This gate decides what the paper's mechanism section may claim about the gap.**

- **Pass criteria:**
  - The speciation table is complete, the figures filed, and the cluster question answered
    with its bar (either answer passes; the gate is about the evidence existing, not about
    which way it went).
- **Fail criteria (STOP):**
  - Missing cells silently dropped, or captions missing their sweep or instrument, or a
    stability-flagged read averaged in as committed.
- **Partial pass guidance:**
  - A shrunk instrument (plan 04's verdict) shrinks the claims, never the honesty: captions
    say which families and steps the oracle covered.

**When you get results, answer** [the review file](../review/05-the-grid-and-the-figures.md).

## Figure Catalog

Navigation: ⬅️ [Previous](#the-engagement-gate) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

#### Pending: to be generated from prompts

| Item | Lane | Prompt file | What it shows | Save to |
|------|------|-------------|---------------|---------|
| The three-timestamp strip | subject | [diagram-prompts.md](../diagram-prompts.md#prompt-5-subject-the-three-timestamp-strip) | window, speciation, divergence on one strip | `../diagrams/when-does-the-outcome-lock-in-05-the-three-timestamp-strip.png` |
| The whole instrument | subject | [diagram-prompts.md](../diagram-prompts.md#subject-capstone-the-whole-instrument-on-one-page) | the capstone | `../diagrams/when-does-the-outcome-lock-in-00-capstone.png` |
| Process lane v01 | process | [diagram-prompts.md](../diagram-prompts.md#process-lane) | the five plans as a journey | `../diagrams/when-does-the-outcome-lock-in-process-01.png` |

#### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|------|------|-------------|--------------|--------|---------|
| Three-timestamp figure | — | correction window, speciation marks, divergence band, one strip per family | `figures.py` | ⏳ | `artifacts/results/when-does-the-outcome-lock-in/window-speciation-divergence__per-family.png` |
| Compose-rate curves | — | counterfactual compose rate per step, one curve per family, per sweep | `figures.py` | ⏳ | `...counterfactual-compose-rate-vs-step__per-family-per-sweep.png` |
| Filmstrips | — | destination frames per step for the named cells, fence steps marked | `figures.py` | ⏳ | `...filmstrip__<cell>.png` |
| PCA overlay | — | three families' trajectories in the top PCs, variance printed | `figures.py` | ⏳ | `...pca-overlay__variance-printed.png` |
| Oracle vs drift scatter | — | oracle speciation against plan 02's settling step | `figures.py` | ⏳ | `...oracle-speciation-vs-drift-settling__per-cell.png` |

#### Organization workflow

1. Run sweeps; 2. Build figures; 3. File with cards; 4. Update the paper's figure register.

## Orchestration: keeping catalogs and plan files in sync

Navigation: ⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/closing-the-compositional-gap/plans/when-does-the-outcome-lock-in/plans/05-the-grid-and-the-figures.md` | **task 0.1** | conformance reported |
| Capture patterns | `/ingest-error-pattern --from-run-log @plans/closing-the-compositional-gap/plans/when-does-the-outcome-lock-in/plans/05-the-grid-and-the-figures.md` | **the close out** | errors catalogued |
| Bring the tree current | `/sync-plan-tree @plans/closing-the-compositional-gap/plans/when-does-the-outcome-lock-in/plans/05-the-grid-and-the-figures.md` | **the close out** | statuses match reality |
| Organize outputs | manual move + update Figure Catalog | after completion | deliverables linked |

## Code references

Navigation: ⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

**File:** `scripts/commitment/grid_sweep.py` (new)

```python
SPECIATION_EARLY_MAX = 10   # clustering at or below supports decide-then-descend
# instrument chosen from plan 04's verdict.json; never hardcoded here
# per read: stability copies x_t, x_t +/- d; a flipped endpoint marks the step on-the-fence
```

**File:** `poe_repair/lcm_oracle.py` (from [plan 03](03-wire-the-oracle.md))
**File:** [the scorer contract](../../../../../report/) referenced by `scorer_validated.json`

## Next step

Navigation: ⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents)

The scope's close: every plan ✅, then the recall gallery per
[the master plan's Definition of Done](../MASTER_PLAN.md#definition-of-done).

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
