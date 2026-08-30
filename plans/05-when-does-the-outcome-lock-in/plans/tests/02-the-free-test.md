# 🧪 The free test: posterior-mean drift

**What this plan asks:** in each cached run, does the model's running guess at the final image
stop moving before the paths visibly separate?

Step 45 in the root running order; waits on nothing (runs beside 44); next is
[wiring the endpoint predictor](../tools/03-wire-the-endpoint-predictor.md).

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/05-when-does-the-outcome-lock-in/plans/02-the-free-probe.md — <one line on what actually happened>
```

## Recommended skill

— custom; no skill fits (per-step tensor arithmetic over the cache).

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 44 (previous) | [basins by hand](../reading/01-basins-by-hand.md) | proves basins and a ridge exist |
| **45 (current)** | **The free test** | **posterior-mean drift per pair-and-seed run, the first speciation numbers, judged against the pre-registered ordering** |
| 46 (next) | [wire the endpoint predictor](../tools/03-wire-the-endpoint-predictor.md) | the LCM-SDXL adapter and its one-state check |

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
- [What has to pass before this runs](#what-has-to-pass-before-this-runs)
- [Figure Catalog](#figure-catalog)
- [Orchestration](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

## Quick context: where you are

Navigation: ⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The experiment.** For every cached pair-and-seed run, compute how far Tweedie's running
estimate of the final image moves between consecutive steps (posterior-mean drift), find the
step where it settles, and set that settling step beside the same run's divergence step from the
existing trajectory-divergence analyses.

> **Tweedie** is the formula that turns a noisy state and the model's noise prediction into a
> running guess at the final clean image.

> **Speciation** is the field's word for the step at which the outcome stops being undecided.
> A **basin** is the set of states that all flow to the same ending, so the posterior mean
> settles once the posterior mass has concentrated on one of them.

**The hypothesis.** The settling step lands at or before the divergence step in every
pair-and-seed run where both are defined, because settling reads the decision and divergence
reads its visible display. If true, the story that the run decides early and then only descends
survives its first contact, and the lag (divergence minus settling) becomes a measured quantity
per pair-and-seed run. If false in even one clean run, the ordering claim is wrong as stated and
the review file records what broke it. Rationale: shared noise masks a committed difference
until it is removed.

**Context details.** All cached pair-and-seed runs; per state the estimate is
`x0_hat = (x_t - sigma_t * eps) / alpha_t`. Whether per-step epsilons are cached is unknown;
task 1.1 settles it and prices the plan (zero extra compute if cached, 2 to 4 U-Net calls per
state if not).

**This plan's job.** Produce speciation numbers with no new model, so the endpoint predictor's
numbers in plan 05 arrive with an independent cross-check already standing.

**Associated materials.** Verdict: [the review file](../../review/02-the-free-test.md). Design
reasoning: [the decision ledger](../../decisions-taken-here.md). Divergence numbers:
`/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses/trajectory_divergence/`.

**For the full picture.** The scope's [master plan](../../MASTER_PLAN.md).

## Considerations

Navigation: ⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**Expected runtime.** If epsilons are cached: minutes, CPU-friendly. If not: one GPU pass over
the grid, a few hours; measure a single pair-and-seed run first and write the number in the Runs
table.

**Prerequisites.** The cache present; the divergence analyses readable.

**Project tracking.** Outputs to
`/datasets/mmolefe/poe_repair_min/outputs/commitment/posterior_drift/`; no W&B run needed
unless the recompute path is taken, in which case log it to `prime_lab/poe-repair-animals-compose`.

**Known issues.** See [Error Matrix](#error-matrix).

<details>
<summary>Environment Facts This Plan Depends On</summary>

- The `co3` absolute python path; see [the environment index](../../../../environment/00-INDEX.md).
- Large artifacts to `/datasets` only; disk guard on the target filesystem.
- fp16 upcast rule when computing `x0_hat` (drift at late steps is small; do the arithmetic in fp32).

</details>

## The claim

Navigation: ⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**A speciation table per pair-and-seed run from the cache alone, judged against the
pre-registered ordering: settling at or before divergence, everywhere both are defined.**

**Why this matters right now:** it is the cheapest test of the scope's central story, and it
gives plan 05's endpoint-predictor curves an independent read to agree or disagree with.

## Why this plan exists

Navigation: ⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The gap.** The repo measures divergence (paths separating) but has no number for the
decision itself, per pair-and-seed run.

**The approach.** The posterior mean settles when the posterior mass has concentrated on one
basin, so its settling step is a free commitment read.

**Key insights:**
1. Two independent tests agreeing on speciation is the corroboration figure; a systematic
   disagreement is itself a finding and is reported, never hidden.
2. The ordering threshold lives in the script as `ORDERING_HOLDS_FRAC_MIN = 1.0` (every clean
   pair-and-seed run), set now, before any number exists.

## What happens (visual)

Navigation: ⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
 per pair-and-seed run:
 drift(t) = || x0_hat(t) - x0_hat(t+1) ||   (fp32)

 drift |\
       | \___
       |     \____
       |          \_____________   <- settles at t*
       +---|----|----|----|----|--
           0   10   20   30   40
                t* (speciation)  vs  divergence step (18 to 36)
```

## Description: what to build

Navigation: ⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The drift script** `scripts/commitment/posterior_drift.py`: per pair-and-seed run, the
   drift curve, the settling step under a threshold in source (`DRIFT_SETTLED_MAX`, the relative
   drift below which all remaining steps must stay), and a JSON row per run carrying the
   settling step, that run's divergence step, and the lag.
2. **The comparison table** `posterior_drift/speciation_vs_divergence.json` plus one figure:
   settling step against divergence step, one point per pair-and-seed run, the identity line
   drawn.
3. **Output root**: `/datasets/mmolefe/poe_repair_min/outputs/commitment/posterior_drift/`.

## Purpose and goal

Navigation: ⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

**Purpose.** Objective 2 of [the master plan](../../MASTER_PLAN.md): measure the speciation step
per pair-and-seed run with two independent tests; this plan is the first of the two.

**Goals:**
1. A settling step exists per pair-and-seed run, with the threshold that defined it in source.
2. The ordering question in the review file is answered ✅ or ❌ with the full table.

## Tasks

Navigation: ⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Paste: `/verify-plan @plans/05-when-does-the-outcome-lock-in/plans/02-the-free-probe.md`
  - Done when: the report comes back clean, or its proposals have been applied.

▶ **Next: [task 1.1](#1--price-the-plan-then-build-the-script)**.

### 1. 🔧 Price the plan, then build the script

◀ **Needs: [task 0.1](#0--check-this-plan-before-working-from-it)**.

- [ ] **1.1** Establish whether per-step epsilons are cached, and print the answer.
  - Inspect one pair-and-seed run's files; print which tensors exist per step and their shapes.
  - **Done when:** the review file's orientation paragraph records "epsilons cached: yes/no"
    and, if no, the measured recompute time for a single pair-and-seed run, which prices the grid.
- [ ] **1.2** Write `scripts/commitment/posterior_drift.py` per the Description, thresholds
  `DRIFT_SETTLED_MAX` and `ORDERING_HOLDS_FRAC_MIN = 1.0` in the source.
  - **Done when:** the script runs on one pair-and-seed run and emits its drift curve and
    settling step.

▶ **Next: [task 2.1](#2--run-over-every-pair-and-seed-run)**.

### 2. 🚀 Run over every pair-and-seed run

◀ **Needs: [tasks 1.1 to 1.2](#1--price-the-plan-then-build-the-script)**.

- [ ] **2.1** Run the drift script over everything in the cache.

    ```bash
    co3 python scripts/commitment/posterior_drift.py --all-cells \
      --out /datasets/mmolefe/poe_repair_min/outputs/commitment/posterior_drift/
    ```

  - **Done when:** `speciation_vs_divergence.json` holds one row per pair-and-seed run (row
    count printed and equal to the count from plan 01's task 1.1), and the scatter figure exists.

▶ **Next: [instruction 3.1](#3--read-the-scatter-and-record-the-verdict)**.

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted.

- [ ] **Capture the failures this plan hit.**
  - Paste: `/ingest-error-pattern --from-run-log @plans/05-when-does-the-outcome-lock-in/plans/02-the-free-probe.md`
  - Done when: each failure has a catalog entry, or there were none.
- [ ] **Bring the tree current.**
  - Paste: `/sync-plan-tree @plans/05-when-does-the-outcome-lock-in/plans/02-the-free-probe.md — <one line>`
  - Done when: statuses match reality.

▶ **Next: what has to pass before this runs.**

## Instructions

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

**For you to follow manually.** Do these yourself.

### 3. 👁️ Read the scatter, and record the verdict

◀ **Needs: [task 2.1](#2--run-over-every-pair-and-seed-run)** done, so the scatter exists.

3.1 **Open the scatter figure** under
   `/datasets/mmolefe/poe_repair_min/outputs/commitment/posterior_drift/`.
   - Expected result: one point per pair-and-seed run, settling step on y, divergence step on x,
     the identity line drawn.
   - ✅ If every point sits on or below the identity line, answer the review file's question ✅.
   - ❌ If any clean point sits above the line, answer ❌ and copy that run's id into
     "Asked after the result" for a follow-up look.

3.2 **Record where the settling steps cluster** (near 10, or inside 18 to 36); one line in the
   review file. This is the first read on the scope's headline question, informative here,
   judged formally in [plan 05](../figures/05-the-grid-and-the-figures.md).

▶ **Next: what has to pass before this runs**, then [plan 03](../tools/03-wire-the-endpoint-predictor.md).

## What has to pass before this runs

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **The scope's story rests on this**: if the ordering fails broadly, the run does not decide
> early and then only descend, and plan 05's interpretation must change before it runs.

- **Pass criteria:**
  - Every clean pair-and-seed run's settling step is at or before its divergence step.
- **Fail criteria (STOP and reassess, not abandon):**
  - Several clean runs settle after diverging; either the measurement or the story is wrong, and
    the review file's artefact checks come first.
- **Partial pass guidance:**
  - A handful of ambiguous runs (settling undefined because drift never settles under the
    threshold) is recorded as 🟡 one by one, not forced into either verdict.

**When you get results, answer** [the review file](../../review/02-the-free-test.md).

## Figure Catalog

Navigation: ⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

#### Pending: to be generated from prompts

| Item | Lane | Prompt file | What it shows | Save to |
|------|------|-------------|---------------|---------|
| The free tests | subject | [diagram-prompts.md](../../diagram-prompts.md#prompt-3-subject-the-free-tests) | the drift curve settling | `../diagrams/when-does-the-outcome-lock-in-03-the-free-probes.png` |
| Process lane v01 | process | [diagram-prompts.md](../../diagram-prompts.md#process-lane) | the five plans as a journey | `../diagrams/when-does-the-outcome-lock-in-process-01.png` |

#### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|------|------|-------------|--------------|--------|---------|
| Settling vs divergence scatter | — | settling step (y) against divergence step (x), one point per pair-and-seed run, identity line | `posterior_drift.py` | ⏳ | filed to `artifacts/results/when-does-the-outcome-lock-in/settling-step-vs-divergence-step__all-cells.png` |

#### Organization workflow

1. Run; 2. Read; 3. File the scatter with its card; 4. Link here.

## Orchestration: keeping catalogs and plan files in sync

Navigation: ⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/05-when-does-the-outcome-lock-in/plans/02-the-free-probe.md` | **task 0.1** | conformance reported |
| Capture patterns | `/ingest-error-pattern --from-run-log @plans/05-when-does-the-outcome-lock-in/plans/02-the-free-probe.md` | **the close out** | errors catalogued |
| Bring the tree current | `/sync-plan-tree @plans/05-when-does-the-outcome-lock-in/plans/02-the-free-probe.md` | **the close out** | statuses match reality |
| Organize outputs | manual move + update Figure Catalog | after completion | deliverables linked |

## Code references

Navigation: ⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

**File:** [poe_repair/runtime.py](../../../poe_repair/runtime.py)
**Relevant section:** the alpha/sigma schedule the drift computation reads its coefficients from.

```python
# the whole computation, per pair-and-seed run
x0_hat = (traj[t] - sigma[t] * eps[t]) / alpha[t]        # fp32
drift[t] = (x0_hat[t] - x0_hat[t + 1]).norm() / x0_hat[t + 1].norm()
t_star = first t after which drift stays below DRIFT_SETTLED_MAX
```

## Next step

Navigation: ⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents)

[Wire the endpoint predictor](../tools/03-wire-the-endpoint-predictor.md): the LCM-SDXL adapter, so the second test
exists.

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
