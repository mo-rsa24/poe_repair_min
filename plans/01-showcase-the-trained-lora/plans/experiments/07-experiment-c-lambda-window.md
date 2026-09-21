# 🧪 Experiment C: the lambda-window injection series

**This plan asks one question: at a single fixed checkpoint, does the blur in the output grow as the injected correction is turned up?**

**Step 37 in the root running order. Waits on: step 36 (the frozen tracking set's softness reads). Next: [08-experiment-a-resume-to-200k](08-experiment-a-resume-to-200k.md).**

## Recommended prompt (after run completes)

```
/ingest-error-pattern --from-run-log
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 36 (previous) | [06-extend-the-tracking-set](../tools/06-extend-the-tracking-set.md) | The frozen tracking set |
| **37 (current)** | **07: experiment-c-lambda-window** | The no-training injection run across values, on existing checkpoints |
| 38 (next) | [08-experiment-a-resume-to-200k](08-experiment-a-resume-to-200k.md) | The length axis |

---

## Table of contents
- [Position in the plan tree](#position-in-the-plan-tree)
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

## Quick context: where you are

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The hypothesis:** *the softness on held-out pairs comes from the injection rather than from undertraining: at a fixed checkpoint, softness tracks λ.*

> Held-out means the pairs were never shown during training, so the number says how well the
> adapter does on animals it has not seen.

**The design (ledger):** no training. The λ grid {0, 0.25, 0.5, 0.75, 1.0} crossed with the step window at inference on existing checkpoints, sharing plan 03's runner.

**If true:** softness rises with λ at the fixed 100k checkpoint, and experiments A and B are read knowing the blur has an injection component no training length removes.

**If false:** softness is flat in λ, and undertraining or capacity stay live accounts for the blur.

**Associated materials:**
- **Review questions:** [../review/07-experiment-c-lambda-window.md](../../review/07-experiment-c-lambda-window.md)
- **Ledger entry:** [experiment C](../../decisions-taken-here.md#the-longer-training-question-runs-as-three-experiments-re-scoped-by-evidence)

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Expected runtime:** in-session on a free device (ledger's launch shape); the grid over the F9 pairs is dozens of renders, an afternoon. Record wall time in the review file.

**Prerequisites:** plan 03's runner with λ scaling; the 100k checkpoint; a softness read (the embedding-drift read from plan 06, reused offline, plus the qualitative strip).

**W&B project:** `prime_lab/poe-repair-animals-compose`.

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- In-session runs still write to `/datasets` only; `co3` python; guidance 7.5; fp16 upcast rule for decodes ([overview](../../../../environment/overview.md)).

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**The cheapest of the three experiments, run first: if softness tracks λ, the blur is the injection, and A's and B's null results are read in that light.** It is also the intervention that lets any discovered direction use causal language (ledger).

> A null result from A or B means the numbers after more training, or at a bigger rank, look the
> same as the numbers before, so that knob moved nothing the measure can see.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

**The problem.** Three accounts of the held-out blur (length, capacity, injection) cost wildly different GPU-hours to test, and the cheapest one is the only one needing no training.

**The solution.** Turn the injection strength down at a fixed checkpoint and watch the softness.

**Key insight.** λ=0 is plain PoE and λ=1 is the shipped configuration; the grid interpolates between the two things the paper already shows.

---

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

One fixed checkpoint (100k steps), one fixed window (steps 0-10), λ swept across
{0, 0.25, 0.5, 0.75, 1.0} on the same four F9 pairs plus the repaired pair:

    λ=0 (plain PoE) ── 0.25 ── 0.5 ── 0.75 ── 1.0 (shipped config)
         │                                          │
    already scored in the paper              already scored in the paper
         └──────────── this plan renders the three points between ────────────┘

Each point gets the same read: embedding-drift softness score, sharpness proxy, and one
frame of the descriptive strip. A monotone rise in blur across the row is the signal
this plan is testing for. See the [lambda-softness strip prompt](../../diagram-prompts.md#prompt-7-subject-the-lambda-softness-strip-planned) for the illustrated version.

---

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The grid run**: {0, 0.25, 0.5, 0.75, 1.0} × window {0-10 fixed} on the four F9 pairs plus the repaired pair, existing seeds, 100k checkpoint.
2. **The softness table**: per λ, the embedding-drift read and a sharpness proxy, plus [compose rate](../../../../context/world/compose-rate.md); `lambda_softness.json`.
3. **The strip**: one row per λ for one named pair, labelled descriptive, for plan 05's wall.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves goal 4 (A, B, C run; verdicts against pre-registered thresholds). Checkable outcomes:

1. Every point in the grid rendered and measured; `lambda_softness.json` complete.
2. The review file's threshold question answered.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 0. ✅ Verify this plan first

- [ ] **0.1 Run the following prompt: `/verify-plan plans/01-showcase-the-trained-lora/plans/experiments/07-experiment-c-lambda-window.md`**

▶ **Next: [task 1.1](#1--run-the-grid)**.

### 1. 🧪 Run the grid

◀ **Needs: the λ-scaled injection runner.** Plan 03's runner (`scripts/showcase/lora_dose_sweep.py`) was NOT reused: its off-window steps run a single unguided forward (by design, for a different compute-budget experiment), which cannot reproduce this project's cached `poe.png` (guided PoE at all 50 steps) at λ=0 — the identity check this plan's review file pre-registers. Built fresh here as `scripts/showcase/lambda_window_grid.py`, reusing plan 03's LoRA attach/load logic and disk-guard/GPU-check boilerplate but with its own sampler (`run_lora_residual_inject_windowed_poe`): off-window steps run the same full guided-PoE forward as on-window steps, adapter disabled.

- [x] **1.1 Run the grid** as a Slurm job chain (sweep → measure → strip, `afterok`-dependent) on `mscluster76`/bigbatch, not literally in-session; outputs to `/datasets/mmolefe/poe_repair_min/outputs/showcase/experiment_c/`. 25/25 renders on disk (5 cells × 5 λ), jobs 48612/48613/48614, commit `32d1973`→`c6a470b` (dirty).
- [x] **1.2 Measure softness per λ**; wrote `lambda_softness.json` (per run: λ, compose, drift read, sharpness proxy).
- [x] **1.3 Build the descriptive strip**: `lambda_softness_strip.png`, the repaired cat×dog@seed1 pair.

▶ **Next: [instruction 2.1](#2--judge-the-trend)**.

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

### 2. 👁️ Judge the trend

◀ **Needs: [tasks 1.1 to 1.3](#1--run-the-grid)** done.

2.1 **Eyeball the strip.** ✅ blur visibly grows with λ while composition appears: the injection account holds; ❌ crisp at every λ: the injection account dies here.

2.2 **Check the numbers agree with the eyeball** (`lambda_softness.json`), then **write the verdict** into the [review file](../../review/07-experiment-c-lambda-window.md).

▶ **Next: what has to pass before this runs.**

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> A and B together cost roughly 20 GPU-hours; this afternoon-scale run fixes how their results will be read. Run it first.

**Pass criteria:**
- Grid complete; the review file's threshold question answered before A or B is interpreted.

**Fail criteria:**
- The runner's λ=0 render fails the identity check against cached PoE (then nothing here means anything; fix the runner).

**When you get results, answer the open questions in the [review file](../../review/07-experiment-c-lambda-window.md).**

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

### Pending

| Figure | Lane | What it shows | Save to |
|--------|------|---------------|---------|
| the λ-softness strip | subject | one pair at five λ values, blur and composition together, labelled descriptive | `diagrams/the-lambda-softness-strip.png` (feeds plan 05) |

### Generated during plan execution

| Figure | Lane | Description | Generated by | Status |
|--------|------|-------------|--------------|--------|
| lambda_softness.json | — | per-λ measurements (sidecar) | task 1.2 | ⏳ |

### Organization workflow

The strip is generated by task 1.3 and saved for plan 05's wall; `lambda_softness.json` is
generated by task 1.2 and read back by instruction 2.2. Neither needs a separate filing step:
task 1.3 writes the strip directly to the location plan 05 reads from.

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Extract errors | `/ingest-error-pattern --from-run-log` | task 3.1 | new patterns into the catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | auto | this file's matrix regenerated |
| Close out | `/sync-plan-tree` | task 3.2 | statuses aggregated up |

### 3. 🧹 Close out

◀ **Needs: [instruction 2.2](#2--judge-the-trend)** done.

- [ ] **3.1 Run the following prompt: `/ingest-error-pattern --from-run-log`** (after any red run).
- [ ] **3.2 Run the following prompt: `/sync-plan-tree plans/01-showcase-the-trained-lora/`**

▶ **Next: [08-experiment-a-resume-to-200k](08-experiment-a-resume-to-200k.md).**

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**File:** `scripts/showcase/lora_dose_sweep.py` (plan 03) — the runner; this plan is a fixed-window slice of its grid with the softness read added.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

▶ Paste to run this plan (in-session; a fixed-window slice of plan 03's grid):

```
/run-experiment plans/01-showcase-the-trained-lora/plans/experiments/07-experiment-c-lambda-window.md — reuse plan 03's lambda-scaled runner (scripts/showcase/lora_dose_sweep.py) if it exists, else build it here and 03 reuses it; the lambda-0 render must match cached poe.png in mode before the grid runs.
```

alt, headless overnight: in a fresh session run `/unattended run-experiment plans/01-showcase-the-trained-lora/plans/experiments/07-experiment-c-lambda-window.md` and paste the tmux block it emits (mechanical grid plus scoring, verified by its own sidecars). What has to pass before this runs and the review file's threshold are the stop conditions.

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[08-experiment-a-resume-to-200k](08-experiment-a-resume-to-200k.md): the length axis, launched only once plan 06's tracking set is frozen.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

### From global catalog

(none apply to this plan)

### From project catalog

From [environment/known-failures.md](../../../../environment/known-failures.md):

- **poe-lora-003** — a masked/windowed sampler's off-window steps can silently diverge from a
  full-guidance baseline it's compared against. Hit here at the λ=0 identity check (DINOv2
  distance 0.43 to cached `poe.png` before the fix); fixed by writing
  `run_lora_residual_inject_windowed_poe` in `scripts/showcase/lambda_window_grid.py`.
- **poe-lora-004** — a hand-written sampler function missing `@torch.no_grad()` OOMs on its
  first forward pass. Hit while writing the same new sampler (job 48610); one-line fix.
- **poe-mem-002** — `_Embedders()` instantiated with no `device` defaults to CPU, and DINOv2's
  xformers attention kernel doesn't support CPU. Hit while writing `measure()` in
  `scripts/showcase/lambda_window_grid.py` (job 48608); fixed by passing `device=torch.device("cuda")`.
