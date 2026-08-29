# 🧪 Experiment A: resume to 200k

**Step 38 in the root running order. Waits on: step 36 (the frozen instrument; step 31's verdict arrives through it). Next: [09-experiment-b-rank-16-32](09-experiment-b-rank-16-32.md).**

## Recommended prompt (after run completes)

```
/ingest-error-pattern --from-run-log
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 37 (previous) | [07-experiment-c-lambda-window](07-experiment-c-lambda-window.md) | The injection account, read first |
| **38 (current)** | **08: experiment-a-resume-to-200k** | The length axis, nothing else changed |
| 39 (next) | [09-experiment-b-rank-16-32](09-experiment-b-rank-16-32.md) | The capacity axis |

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
- [The engagement gate](#the-engagement-gate)
- [Figure Catalog](#figure-catalog)
- [Orchestration: keeping catalogs and plan files in sync](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Recommended skill](#recommended-skill)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

---

## Quick context: where you are

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The hypothesis:** *training length is not the knob: doubling the steps moves neither the held-out compose rate outside its seed-noise band nor the frozen tracking set's crispness reads.*

**The design (ledger):** resume `phase1_r8_100k` from `lora_step_100000.pt` to 200k steps, nothing else changed. The evidence already re-scoped this experiment: compose rate saturated by 50k and F8b puts the LoRA at or above the oracle ceiling on the scored metric, so A cannot claim compose-rate gains; it chases the scorer-invisible softness.

**Null bar, fixed before launch:** held-out compose rate at 200k within the seed-noise band (spread over the 8 held-out seeds) of 100k, with no crispness change in the frozen tracking set, means length is not the knob.

**Associated materials:**
- **Review questions:** [../review/08-experiment-a-resume-to-200k.md](../review/08-experiment-a-resume-to-200k.md)
- **Ledger entry:** [experiment A](../decisions-taken-here.md#the-longer-training-question-runs-as-three-experiments-re-scoped-by-evidence)

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Expected runtime:** roughly 6.6 GPU-hours at the measured 4.2 optimizer steps/s; measure it. biggpu, 3-day walltime is ample.

**Launch shape (ledger):** the one Slurm job per user goes here: sbatch on an idle biggpu node per [execution-protocol](../../../environment/hpc/execution-protocol.md).

**Prerequisites:** plan 06's frozen tracking set (hash in config); the plateau verdict from plan 01 recorded.

**W&B project:** `prime_lab/poe-repair-animals-compose`.

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- biggpu allows one Slurm job per user; this run takes that slot ([overview](../../../environment/overview.md)).
- Checkpointing to `/datasets` only (the home filesystem once hit 100% and silently killed checkpointing); the launcher's disk guard checks `/datasets`.
- The `co3` python; fp16 with upcast rule.

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**The length axis settled by a run whose null bar was fixed before launch.** A null here is a finding: it closes "just train longer" with a number instead of an opinion.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

**The problem.** The held-out renders are soft, and "train longer" is the reflex answer; two walks disagreed on it and the reconciliation was to test it rather than argue it.

**The solution.** One resume, one axis, the frozen instrument watching.

**Key insight.** Plan 07's answer arrives first and cheaply; if softness tracks λ, a null here is expected and doubly informative.

---

## Description: what to build

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The resume launcher**: sbatch script wrapping the pooled trainer with `--resume lora_step_100000.pt --max-steps 200000`, tracking set active, disk guard on `/datasets`.
2. **The run**: 100k additional steps, checkpoints every 10k, tracking set rendered at each.
3. **The comparison table**: 200k vs 100k on held-out compose rate (with seed-noise band) and on each tracking-set read; `experiment_a_verdict_inputs.json`.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves goal 4. Checkable outcomes:

1. The run reaches 200k with checkpoints and tracking renders on disk.
2. The comparison table written; the review bar answered.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 0. ✅ Verify this plan first

- [ ] **0.1 Run the following prompt: `/verify-plan plans/01-showcase-the-trained-lora/plans/08-experiment-a-resume-to-200k.md`**

▶ **Next: [task 1.1](#1--launch-and-harvest)**.

### 1. 🧪 Launch and harvest

◀ **Needs: [06 instruction 2.3](06-extend-the-tracking-set.md#2--prove-the-smoke-in-wb)** (frozen instrument proven) and **[01 instruction 2.4](01-read-the-plateau-curves.md#2--read-the-curves-in-wb)** (plateau verdict recorded).

- [ ] **1.1 Write the sbatch launcher** (`scripts/showcase/experiment_a_resume.sbatch`): co3 python, cache env var, disk guard on `/datasets`, tracking set on, W&B flags.
- [ ] **1.2 Submit and record.** `sbatch` on an idle biggpu node; job id and run id into the review Runs table. Completion is observable: `squeue -u mmolefe` shows the job; the W&B run appears with the frozen manifest hash in its config.
- [ ] **1.3 Harvest at completion**: confirm `lora_step_200000.pt` exists on `/datasets`, tracking renders complete, then build `experiment_a_verdict_inputs.json` (200k vs 100k, per metric, with the seed-noise band).

▶ **Next: [instruction 2.1](#2--watch-and-judge)**.

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-engagement-gate) ➡️

### 2. 📈 Watch and judge

◀ **Needs: [task 1.2](#1--launch-and-harvest)** done, so the run is live.

2.1 **Mid-run health check** (once, a few hours in): W&B charts, the seven tracking families updating. ✅ curves advancing; ❌ any family frozen or the job gone from `squeue`: harvest the log, `/ingest-error-pattern --from-run-log`.

2.2 **Capture the tracking panel** at completion into `runbook/reading-a-training-run.md`'s screenshot slot (wandb or Playwright MCP), captioned.

2.3 **Judge against the null bar** using `experiment_a_verdict_inputs.json`; **write the verdict** into the [review file](../review/08-experiment-a-resume-to-200k.md).

▶ **Next: the engagement gate.**

---

## The engagement gate

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> The null bar was fixed before launch. Whatever lands, the review records it against that bar; the bar does not move to meet the result.

**Pass criteria:**
- Run reached 200k; the comparison table complete; the review bar answered.

**Fail criteria:**
- The run died before 200k (harvest, ingest, relaunch decision in the review), or the tracking manifest hash changed mid-run (the instrument unfroze; the comparison is void).

**When you get results, answer the open questions in the [review file](../review/08-experiment-a-resume-to-200k.md).**

---

## Figure Catalog

⬅️ [Previous](#the-engagement-gate) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

### Pending

| Figure | Lane | What it shows | Save to |
|--------|------|---------------|---------|
| (none owed; A's panels live in W&B and feed plan 12's oracle-ceiling reading) | — | — | — |

### Generated during plan execution

| Figure | Lane | Description | Generated by | Status |
|--------|------|-------------|--------------|--------|
| experiment_a_verdict_inputs.json | — | 200k vs 100k per metric with the seed-noise band | task 1.3 | ⏳ |

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Extract errors | `/ingest-error-pattern --from-run-log` | task 3.1 | new patterns into the catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | auto | this file's matrix regenerated |
| Close out | `/sync-plan-tree` | task 3.2 | statuses aggregated up |

### 3. 🧹 Close out

◀ **Needs: [instruction 2.3](#2--watch-and-judge)** done.

- [ ] **3.1 Run the following prompt: `/ingest-error-pattern --from-run-log`** (after any red run).
- [ ] **3.2 Run the following prompt: `/sync-plan-tree plans/01-showcase-the-trained-lora/`**

▶ **Next: [09-experiment-b-rank-16-32](09-experiment-b-rank-16-32.md).**

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**File:** `poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py` — the trainer; the launcher adds resume and max-steps flags.
**File:** `artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k/checkpoints/lora_step_100000.pt` — the resume point.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

▶ Paste to run this plan (the one Slurm job):

```
/run-experiment plans/01-showcase-the-trained-lora/plans/08-experiment-a-resume-to-200k.md — dry resume of ~100 steps in-session first; only then sbatch on an idle biggpu node with the preflight block; the null bar is pre-registered in the review file.
```

alt, headless overnight: in a fresh session run `/unattended run-experiment plans/01-showcase-the-trained-lora/plans/08-experiment-a-resume-to-200k.md` and paste the tmux block it emits (multi-hour training, SSH-drop-prone, bars pre-registered). The engagement gate and the review file's bar are the stop conditions.

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[09-experiment-b-rank-16-32](09-experiment-b-rank-16-32.md): the capacity axis, over the SSH-plus-nohup path while A holds the Slurm slot.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

### From global catalog

(empty until `/ingest-error-pattern` populates)

### From project catalog

(empty until `/ingest-error-pattern` populates)
