# 📊 Read where the training curve stops rising

**This plan asks one question: had the trained LoRA stopped improving by 100k steps, or was it
still climbing?**

**Step 31 in the root running order. Waits on: nothing. Next: [02-the-dog-x-dog-null-probe](02-the-dog-x-dog-null-probe.md).**

## Recommended prompt (after run completes)

After you finish this plan and want to ingest error patterns into the catalogs, use this prompt:

```
/ingest-error-pattern --from-run-log
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| — (previous) | scope opens here | — |
| **31 (current)** | **01: read-the-plateau-curves** | The free curve-read that frames experiments A and B |
| 32 (next) | [02-the-dog-x-dog-null-probe](02-the-dog-x-dog-null-probe.md) | The null-input control |

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

**The question:**
Is the trained LoRA (`phase1_r8_100k`) at a ceiling, or was it still improving when training stopped? The scope's [decision ledger](../decisions-taken-here.md) settled that experiments A (train to 200k) and B (rank 16/32) run either way, so this read no longer decides *whether* they run; it is their **interpretation key**.

**The signal already logged:**
`eval/frac_distance_reached` (how much of the gap toward the joint prediction the LoRA's correction closes, 0 to 1) flattens near 0.4 in the live curves from `instrument-02`. Held-out [compose rate](../../../context/world/compose-rate.md) saturates by step 50k (0.812 at 10k, 0.961 at 50k and 60k, from the run's own `compose_rate.json`). The train loss still creeps (median 0.00062 early to 0.00029 late).

> Held-out means the pairs were never shown during training, so the number says how well the
> adapter does on animals it has not seen.

**If flat across epochs and checkpoints:** ceiling. A's null result is expected and B's result is read as a capacity finding or a dead lever depending on the panel showing what the cached true correction can reach at best.

**If still rising:** waypoint. A is expected to move the tracking set, and a flat A becomes a finding.

**Associated materials:**
- **Review questions:** [../review/01-read-the-plateau-curves.md](../review/01-read-the-plateau-curves.md)
- **The run's local files:** `artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k/` (`history.json`, 91,800 rows; `compose_rate.json`; `samples/per_epoch/`)

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Expected runtime:** under an hour, no GPU, no queue. Everything is a read of files already on disk and curves already in W&B.

**Prerequisites:** none. This is the scope's opener.

**W&B project:** `prime_lab/poe-repair-animals-compose` (the run's local `wandb/` folder names the run id).

**Known issues:** see the [Error Matrix](#error-matrix).

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- The `co3` python at `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python` for the local history read ([environment/overview.md](../../../environment/overview.md)).
- W&B owns the numbers; the plan tree owns the verdict. No curve is copied into markdown; the verdict, the run id, and the threshold it was judged against go in the review file.

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**A recorded ceiling-or-waypoint verdict on `eval/frac_distance_reached`, read across epochs and checkpoints, before any GPU is spent on experiments A and B.** It matters now because A and B launch next, and without this read their results cannot be interpreted.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

**The problem.** Two design walks read the same held-out softness oppositely: one as undertraining (train longer), one as the L2-on-a-heavy-tailed-target ceiling (training longer buys nothing). The reconciliation runs the experiments anyway, but a result nobody can interpret is a wasted run.

**The solution.** The interpretation key is free: the curves are already logged. Read them, fix the verdict in the review file, then launch.

**Key insight.** The LoRA's L2 loss on a state-specific target commits it to the conditional mean of what its inputs pin down; averaged near-orthogonal corrections are small and smooth. That account predicts a curve that flattens and stays flat. A rising curve is the one observation that would contradict it.

---

## Description: what to build

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The curve read**: `eval/frac_distance_reached` over the full 100k steps, from W&B (browser) and from the local `history.json` (script), the two agreeing.
2. **The checkpoint cross-section**: the same quantity at each saved checkpoint (10k steps apart), so "flat across epochs" is a table, not an impression.
3. **The verdict**: one line in the review file, ceiling or waypoint, with the numbers that decided it.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves master-plan objective 1 (decide the train-longer framing from already-logged curves before any GPU is spent). Checkable outcomes:

1. The per-checkpoint table exists as a sidecar json.
2. The review file's threshold question is answered with the deciding numbers.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

For Claude to execute. Ask Claude to do these.

### 0. ✅ Verify this plan first

- [ ] **0.1 Run the following prompt: `/verify-plan plans/01-showcase-the-trained-lora/plans/01-read-the-plateau-curves.md`**
  - Produces: a structure and instructions check before any work runs off this plan.

▶ **Next: [task 1.1](#1--extract-the-curve-locally)**.

### 1. 📊 Extract the curve locally

◀ **Needs: nothing** — the history file is on disk.

- [ ] **1.1 Tabulate `eval/frac_distance_reached` per checkpoint from `history.json`**
  - Command:
    ```
    /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -c "see code references"
    ```
  - Output goes to: `artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k/frac_distance_by_checkpoint.json`
  - Completion is observable: the json holds one row per checkpoint (10k to 100k), each with the metric's mean over the eval runs at that step.
- [ ] **1.2 Compute the flatness numbers**
  - The last-half slope (linear fit over steps 50k to 100k) and the max-minus-min over the same span, written into the same sidecar.

▶ **Next: [instruction 2.1](#2--read-the-curves-in-wb)** (read the same curves in the browser).

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-engagement-gate) ➡️

For you to follow manually. Do these yourself.

### 2. 📈 Read the curves in W&B

◀ **Needs: [tasks 1.1 to 1.2](#1--extract-the-curve-locally)** done, so the local numbers exist to compare against.

2.1 **Open W&B and find the run.** Browser: wandb.ai, project `prime_lab/poe-repair-animals-compose`. The run id is in `artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k/wandb/` (folder name `run-...-<id>`). Click into it. The reverse SSH port-forward, if working off the cluster browser, is in [the runbook](../../../runbook/00-INDEX.md).

2.2 **Charts tab, search `eval/frac_distance_reached`.** You should see one curve over 100k steps. ✅ If its last half is flat by eye and the local slope from task 1.2 is consistent with zero at the curve's own noise, the verdict is ceiling. ❌ If it is visibly rising at 100k and the local slope agrees, the verdict is waypoint.

2.3 **Screenshot the panel** into the place kept for it in [runbook/reading-a-training-run.md](../../../runbook/reading-a-training-run.md).

2.4 **Write the verdict** into the [review file](../review/01-read-the-plateau-curves.md): ceiling or waypoint, the slope, the span, the run id.

▶ **Next: the engagement gate.**

---

## The engagement gate

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> Experiments A and B launch after this plan. Launching them without the verdict recorded turns both into runs nobody can interpret.

**Pass criteria:**
- The per-checkpoint sidecar exists and the review file's threshold question is answered.

**Fail criteria:**
- The metric is missing from W&B or the local history, in which case the read falls back to the per-epoch samples and says so in the review file.

**When you have the verdict, answer the open questions in the [review file](../review/01-read-the-plateau-curves.md).**

---

## Figure Catalog

⬅️ [Previous](#the-engagement-gate) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

### Pending

| Figure | Lane | What it shows | Save to |
|--------|------|---------------|---------|
| (none owed by this plan; the curve panel lives in W&B and its screenshot in the runbook) | — | — | — |

### Generated during plan execution

| Figure | Lane | Description | Generated by | Status |
|--------|------|-------------|--------------|--------|
| frac_distance_by_checkpoint.json | — | the per-checkpoint table (sidecar, not a figure) | task 1.1 | ⏳ |

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

**After this plan completes:**

1. **Ingest errors:** `/ingest-error-pattern --from-run-log` (task 3.1) extracts any new failure patterns; it triggers `/sync-plan-tree --update-error-matrices`.
2. **Error Matrix refresh:** automatic, by the trigger above.
3. **Close-out:** `/sync-plan-tree` (task 3.2) reconciles the plan's status up the tree.

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Extract errors | `/ingest-error-pattern --from-run-log` | task 3.1 | new patterns into the catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | auto | this file's matrix regenerated |
| Close out | `/sync-plan-tree` | task 3.2 | statuses aggregated up |

### 3. 🧹 Close out

◀ **Needs: [instruction 2.4](#2--read-the-curves-in-wb)** done, so the verdict exists.

- [ ] **3.1 Run the following prompt: `/ingest-error-pattern --from-run-log`** (only if anything failed red).
- [ ] **3.2 Run the following prompt: `/sync-plan-tree plans/01-showcase-the-trained-lora/`**

▶ **Next: [02-the-dog-x-dog-null-probe](02-the-dog-x-dog-null-probe.md).**

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**File:** `artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k/history.json` — 91,800 logged rows; filter keys beginning `eval/frac_distance_reached`, group by `train/optimizer_step` bucket of 10k.

```python
import json, statistics
h = json.load(open(".../history.json"))
rows = [r for r in h if any(k.startswith("eval/frac_distance_reached") for k in r)]
# bucket by optimizer step, mean per checkpoint, dump sidecar
```

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

▶ Paste to run this plan (no GPU; the verdict lands in the review file):

```
Execute plans/01-showcase-the-trained-lora/plans/01-read-the-plateau-curves.md: run task 0.1's verify-plan, then the Tasks lane; stop at the handoff to the manual W&B read.
```

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[02-the-dog-x-dog-null-probe](02-the-dog-x-dog-null-probe.md): the null-input control that tests whether the LoRA learned a rule or a plurality prior.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

### From global catalog

(empty until `/ingest-error-pattern` populates)

### From project catalog

(empty until `/ingest-error-pattern` populates)
