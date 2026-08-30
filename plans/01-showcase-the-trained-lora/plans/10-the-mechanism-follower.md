# 🔬 The checkpoint watcher that reads the mechanism

**This plan asks one question: what did the LoRA actually add to cross-attention, and does injecting that thing on its own make pictures compose?**

**Step 40 in the root running order. Waits on: steps 38 and 39 (it watches their checkpoint folders). Next: [11-the-counted-joint-prompt-figure](11-the-counted-joint-prompt-figure.md).**

## Recommended prompt (after run completes)

```
/ingest-error-pattern --from-run-log
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 39 (previous) | [09-experiment-b-rank-16-32](09-experiment-b-rank-16-32.md) | The capacity runs it watches |
| **40 (current)** | **10: the-mechanism-follower** | h-space and Jacobian reads per broad checkpoint, each ending in an intervention |
| 41 (next) | [11-the-counted-joint-prompt-figure](11-the-counted-joint-prompt-figure.md) | The measured baseline figure |

---

## Table of contents
- [Position in the plan tree](#position-in-the-plan-tree)
- [Quick context: where you are](#quick-context-where-you-are)
- [Considerations](#considerations)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
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

## Quick context: where you are

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The object under study (ledger):** the [interaction term](../../../context/world/interaction-term.md) is a rule, not a vector (same-pair cross-seed cosine 0.002), so the mechanism question is what state-dependent computation the LoRA added to cross-attention, the layer carrying object attribution.

**The two reads (ledger):** h-space, the UNet bottleneck activations with the adapter on minus off (the correction in the model's own semantic space); and Jacobian singular directions (does the adapter open a new high-gain direction for the missing animal). Both run as one watcher process: it watches a run's checkpoints folder, and when a broad-interval checkpoint lands (default first, middle, final) it computes both reads on a device that is not the training device and logs to the same W&B run. The training loop is never modified.

**The causal rule (ledger, non-negotiable):** each read ends in its intervention: inject the found direction before about step 10, score composition. A direction figure without an intervention shows a correlation and nothing more, and figures whose interventions fail to move [compose rate](../../../context/world/compose-rate.md) stay out of the main text.

**Associated materials:**
- **Review questions:** [../review/10-the-mechanism-follower.md](../review/10-the-mechanism-follower.md)
- **Ledger entry:** [the mechanism reads](../decisions-taken-here.md#the-mechanism-reads-run-during-training-beside-it-never-inside-it)

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Expected runtime:** the watcher is idle-then-bursty; each burst (two reads on a few showcase renders at one checkpoint) is minutes-to-an-hour on a non-training device. The interventions at the end are a small render batch each.

**Storage (ledger):** forced to pooled activations or a subset of the renders; never full per-step activation dumps.

**Prerequisites:** A or B live (their checkpoint folders are what it watches); the phase1 checkpoints work for a dry run.

**W&B project:** `prime_lab/poe-repair-animals-compose` (logs into the watched run).

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- A non-training device must exist while A and B run; the protocol's device census (`nvidia-smi` per node) decides where the watcher lives ([execution-protocol](../../../environment/hpc/execution-protocol.md)).
- Outputs to `/datasets`; `co3` python.

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**Mechanism figures that appear live beside the training curves, each carrying its own causal test.** Most of the paper's figures measure the cached true correction rather than the adapter; this is the deliberate scope addition that lets the mechanism section, if the section order keeps one, be LoRA-measured.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

**The problem.** What the LoRA does is measured everywhere; how it does it is measured nowhere, and in-loop reads would slow every run (the rejected alternative).

**The solution.** A watcher that costs the training loop nothing and turns checkpoints into mechanism reads as they land.

**Key insight.** Reads keyed to first/middle/final checkpoints show mechanism *formation*, which post-hoc reads on the final checkpoint cannot.

---

## Description: what to build

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The watcher** (`scripts/showcase/mechanism_follower.py`): watches a checkpoints dir; on a broad-interval checkpoint, runs the h-space read (adapter on minus off, pooled bottleneck activations, a few showcase renders) and the Jacobian read (top singular directions at steps inside the 0-10 window), logs both to the watched W&B run, saves arrays to `/datasets/mmolefe/poe_repair_min/outputs/showcase/mechanism/`.
2. **The interventions**: for each discovered direction, inject before step 10 at a small λ grid, score compose rate; `intervention_scores.json`.
3. **The dry run**: the watcher pointed at the existing phase1 checkpoints folder proves the whole path before A and B need it.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves goal 8 (mechanism interventions' verdict on the causal caption). Checkable outcomes:

1. The dry run produces both reads on phase1's first/middle/final checkpoints.
2. Every reported direction has an intervention score.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 0. ✅ Verify this plan first

- [ ] **0.1 Run the following prompt: `/verify-plan plans/01-showcase-the-trained-lora/plans/10-the-mechanism-follower.md`**

▶ **Next: [task 1.1](#1--build-the-watcher)**.

### 1. 🔬 Build the watcher

◀ **Needs: nothing to build; [08 task 1.2](08-experiment-a-resume-to-200k.md#1--launch-and-harvest) or [09 task 1.2](09-experiment-b-rank-16-32.md#1--launch-the-pair)** live before the watch phase.

- [ ] **1.1 Write the follower** with the watch loop, the two reads, the storage cap, and the W&B logging into the watched run.
- [ ] **1.2 Dry-run it against phase1's checkpoints folder** (first/middle/final = 10k/50k/100k). Completion is observable: six read outputs on `/datasets` (two reads × three checkpoints) and their panels in W&B.
- [ ] **1.3 Attach it to A's and B's folders** on a non-training device; record node/device/PID in the review Runs table.
- [ ] **1.4 Run the interventions** for every direction the reads surface; write `intervention_scores.json`.

▶ **Next: [instruction 2.1](#2--read-the-mechanism-panels)**.

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

### 2. 👁️ Read the mechanism panels

◀ **Needs: [tasks 1.2 to 1.4](#1--build-the-watcher)** producing panels.

2.1 **Open the watched run in W&B**; the mechanism panels sit beside the training curves. ✅ reads appear within an hour of each broad checkpoint; ❌ the watcher's log shows it skipped one: restart it, note in review.

2.2 **Judge each direction by its intervention**: compose rate moved = the figure may enter the main text; unmoved = it stays out (ledger). **Write verdicts** into the [review file](../review/10-the-mechanism-follower.md).

▶ **Next: what has to pass before this runs.**

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> The mistake this section exists to prevent is a beautiful direction figure with no causal test behind it. No intervention score, no main-text figure. No exceptions.

**Pass criteria:**
- Dry run green; every reported direction carries an intervention score.

**Fail criteria:**
- The watcher modified the training loop or ran on a training device (both rejected alternatives), or a direction is reported without its intervention.

**When you get results, answer the open questions in the [review file](../review/10-the-mechanism-follower.md).**

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

### Pending

| Figure | Lane | What it shows | Save to |
|--------|------|---------------|---------|
| mechanism formation panels | subject | the two reads at first/middle/final checkpoints, per run | W&B + `/datasets/.../mechanism/`; candidates for plan 05 only with passing interventions |

### Generated during plan execution

| Figure | Lane | Description | Generated by | Status |
|--------|------|-------------|--------------|--------|
| intervention_scores.json | — | per direction: λ grid, compose rate (sidecar) | task 1.4 | ⏳ |

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Extract errors | `/ingest-error-pattern --from-run-log` | task 3.1 | new patterns into the catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | auto | this file's matrix regenerated |
| Close out | `/sync-plan-tree` | task 3.2 | statuses aggregated up |

### 3. 🧹 Close out

◀ **Needs: [instruction 2.2](#2--read-the-mechanism-panels)** done.

- [ ] **3.1 Run the following prompt: `/ingest-error-pattern --from-run-log`** (after any red run).
- [ ] **3.2 Run the following prompt: `/sync-plan-tree plans/01-showcase-the-trained-lora/`**

▶ **Next: [11-the-counted-joint-prompt-figure](11-the-counted-joint-prompt-figure.md).**

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**File:** `poe_repair/_sdxl/runtime.py::load_sdxl_models` — the model loading the reads reuse.
**Pattern:** the h-space read hooks the UNet mid-block; the Jacobian read uses JVPs at cached states, never full materialised Jacobians.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

▶ Paste to run this plan (runs beside experiments A and B):

```
Execute plans/01-showcase-the-trained-lora/plans/10-the-mechanism-follower.md: the checkpoint watcher on a non-training device, h-space and Jacobian reads at first/middle/final checkpoints, each ending in its injection-and-score intervention; it must never crash a training run.
```

alt, headless overnight: in a fresh session run `/unattended run-experiment plans/01-showcase-the-trained-lora/plans/10-the-mechanism-follower.md` and paste the tmux block it emits (a long-lived watcher; the interventions score numerically). What has to pass before this runs and the review file's threshold are the stop conditions.

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[11-the-counted-joint-prompt-figure](11-the-counted-joint-prompt-figure.md): the measured joint-prompt baseline, minutes of scoring, independent of everything above.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

### From global catalog

(empty until `/ingest-error-pattern` populates)

### From project catalog

(empty until `/ingest-error-pattern` populates)
