# 🧪 Experiment B: rank 16 and 32

**This plan asks one question: does giving the adapter more capacity, rank 16 or rank 32 instead of rank 8, change anything at the same number of steps?**

**Step 39 in the root running order. Waits on: step 36 (the frozen tracking set). Next: [10-the-checkpoint-watcher](../tests/10-the-checkpoint-watcher.md).**

## Recommended prompt (after run completes)

```
/ingest-error-pattern --from-run-log
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 38 (previous) | [08-experiment-a-resume-to-200k](08-experiment-a-resume-to-200k.md) | The length axis |
| **39 (current)** | **09: experiment-b-rank-16-32** | The capacity axis at matched step counts |
| 40 (next) | [10-the-checkpoint-watcher](../tests/10-the-checkpoint-watcher.md) | The h-space and Jacobian reads beside these runs |

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

**The hypothesis:** *capacity is not the knob either: rank 16 and rank 32 land inside rank 8's seed-noise band on held-out [compose rate](../../../../context/world/compose-rate.md) at matched step counts.*

> Held-out means the pairs were never shown during training, so the number says how well the
> adapter does on animals it has not seen.

**The design (ledger):** fresh runs at rank 16 and rank 32 (alpha equal to rank), 100k steps each, everything else identical; `sweep_s1_rank.sh` is the runner for the capacity axis. B buys the rank-ablation figure regardless of verdict: same pairs, same λ, same inference across r8/r16/r32 at matched step counts.

**The interpretation key (ledger):** the panel from plan 12 showing what the cached true correction reaches at best. If that correction gives crisp images while the adapter gives soft ones, capacity is the right lever to pull; if the cached correction is soft too, a null here is the expected and final answer.

> A null here means rank 16 and rank 32 read the same as rank 8, so extra capacity bought
> nothing the measure can see.

**Associated materials:**
- **Review questions:** [../review/09-experiment-b-rank-16-32.md](../../review/09-experiment-b-rank-16-32.md)
- **Ledger entry:** [experiment B](../../decisions-taken-here.md#the-longer-training-question-runs-as-three-experiments-re-scoped-by-evidence)

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Expected runtime:** two runs of the same length as the original 100k training; measure from the first checkpoints. Roughly the bulk of the ledger's ~20 GPU-hours across three nodes.

**Launch shape (ledger):** experiment A holds the one Slurm job, so B's two runs go over the SSH-plus-nohup idle-node path per [execution-protocol step 3](../../../../environment/hpc/execution-protocol.md), including the admin node-cap fallback: prefer device 1, verify it is free. `squeue` is blind to these; monitoring is the nohup log plus `pgrep` on the node.

**Prerequisites:** plan 06's frozen tracking set.

**W&B project:** `prime_lab/poe-repair-animals-compose`.

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- Long runs outside Slurm are the norm here (one job per user); harvest reads all three execution modes: `squeue`, `pgrep -af 'sweep|train'` on the node, and the log tail ([overview](../../../../environment/overview.md)).
- Checkpoints to `/datasets` only; `co3` python; fp16 upcast rule.

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**The capacity axis settled at matched step counts, plus the rank-ablation figure the paper can show either way.** Rank-8 in weight space and k=8 in r_t space are different objects (ledger); this figure is what lets the paper say so with data.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

**The problem.** If the softness is a capacity ceiling, no length of rank-8 training fixes it; if it is not, running the same thing at two more ranks closes that door with a number.

**The solution.** Two fresh trainings at the next two capacity steps, judged through the same frozen tracking set.

**Key insight.** Matched step counts are what make the three runs one comparison; a rank-32 run read at a different step count is a mixed comparison and the figure may not imply otherwise.

---

## Description: what to build

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **Two launches** via `scripts/cross_seed_lora_pooling/sweep_s1_rank.sh` adapted to the pooled trainer: rank 16 (alpha 16) and rank 32 (alpha 32), 100k steps, tracking set on.
2. **The rank-ablation table**: r8/r16/r32 at matched checkpoints on held-out compose rate and the crispness reads; `rank_ablation.json`.
3. **The figure draft**: one panel, three ranks, matched steps, seed-noise band drawn; shipped by plan 05.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves goal 4 and the rank-ablation figure. Checkable outcomes:

1. Both runs reach 100k with tracking renders.
2. `rank_ablation.json` complete at matched checkpoints; the review file's threshold question answered.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 0. ✅ Verify this plan first

- [ ] **0.1 Run the following prompt: `/verify-plan plans/01-showcase-the-trained-lora/plans/09-experiment-b-rank-16-32.md`**

▶ **Next: [task 1.1](#1--launch-the-pair)**.

### 1. 🧪 Launch the pair

◀ **Needs: [06 instruction 2.3](../tools/06-extend-the-tracking-set.md#2--prove-the-first-short-run-in-wb)** (frozen tracking set proven).

- [ ] **1.1 Adapt the rank harness** to the pooled trainer with the tracking set on; ranks 16 and 32, alpha equal to rank, everything else the 100k config.
- [ ] **1.2 Find two free devices and launch over SSH with nohup** (prefer device 1 per the protocol's node-cap fallback; verify free with `nvidia-smi` first). Record node, device, PID, and run id per run in the review Runs table. Completion is observable: `pgrep -af train` on each node shows the process; two new W&B runs appear.
- [ ] **1.3 Harvest at completion**: checkpoints on `/datasets`, then build `rank_ablation.json` at matched checkpoints (10k grid).

▶ **Next: [instruction 2.1](#2--watch-and-judge)**.

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

### 2. 📈 Watch and judge

◀ **Needs: [task 1.2](#1--launch-the-pair)** done, so both runs are live.

2.1 **Mid-run health check per run**: SSH to each node, `tail` the nohup log, `pgrep` the PID; in W&B, the tracking families updating on both runs. ✅ both advancing; ❌ one gone: harvest its log, `/ingest-error-pattern --from-run-log`, decide relaunch in the review.

2.2 **Capture the three-rank comparison panel** at completion into the place the runbook keeps for screenshots, captioned.

2.3 **Judge against the null threshold**; **write the verdict** into the [review file](../../review/09-experiment-b-rank-16-32.md), reading it beside plan 12's panel of what the cached correction reaches at best.

▶ **Next: what has to pass before this runs.**

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> Two runs on nohup paths that Slurm cannot see. The review's Runs table is the only ledger of where they live; an unrecorded PID is an orphaned GPU.

**Pass criteria:**
- Both runs at 100k; the ablation table complete at matched steps; the threshold question answered.

**Fail criteria:**
- Step counts unmatched at judgment time (mixed comparison; the figure may not ship), or a run's manifest hash differs (the tracking set unfroze).

**When you get results, answer the open questions in the [review file](../../review/09-experiment-b-rank-16-32.md).**

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

### Pending

| Figure | Lane | What it shows | Save to |
|--------|------|---------------|---------|
| the rank ablation | subject | held-out compose rate (y) per rank r8/r16/r32 (x) at matched steps, seed-noise band drawn | drafted here, shipped by plan 05 |

### Generated during plan execution

| Figure | Lane | Description | Generated by | Status |
|--------|------|-------------|--------------|--------|
| rank_ablation.json | — | the matched-checkpoint table (sidecar) | task 1.3 | ⏳ |

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

▶ **Next: [10-the-checkpoint-watcher](../tests/10-the-checkpoint-watcher.md).**

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**File:** `scripts/cross_seed_lora_pooling/sweep_s1_rank.sh` — the rank runner to adapt.
**File:** `artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k/config.json` — the config every non-rank flag must match.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

▶ Paste to run this plan (two idle nodes in parallel):

```
/run-experiment plans/01-showcase-the-trained-lora/plans/09-experiment-b-rank-16-32.md — SSH-plus-nohup per the execution protocol, absolute paths on the launch line, in-script device guard; if sbatch is denied by the admin node cap, prefer device 1 and verify it is free.
```

alt, headless overnight: in a fresh session run `/unattended run-experiment plans/01-showcase-the-trained-lora/plans/09-experiment-b-rank-16-32.md` and paste the tmux block it emits (multi-hour training on remote devices, thresholds pre-registered). What has to pass before this runs and the review file's threshold are the stop conditions.

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[10-the-checkpoint-watcher](../tests/10-the-checkpoint-watcher.md): the h-space and Jacobian reads that run beside A's and B's checkpoints, off the training devices.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

### From global catalog

(empty until `/ingest-error-pattern` populates)

### From project catalog

(empty until `/ingest-error-pattern` populates)
