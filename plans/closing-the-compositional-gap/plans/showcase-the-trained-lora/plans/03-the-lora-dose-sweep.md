# 🧪 The LoRA dose sweep

**Step 33 in the root running order. Waits on: step 32 (the probe is this sweep's zero cell). Next: [04-the-transfer-matrix-figure](04-the-transfer-matrix-figure.md).**

## Recommended prompt (after run completes)

```
/ingest-error-pattern --from-run-log
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 32 (previous) | [02-the-dog-x-dog-null-probe](02-the-dog-x-dog-null-probe.md) | The zero-interaction control cell |
| **33 (current)** | **03: the-lora-dose-sweep** | The causal dose curve for the shipped LoRA |
| 34 (next) | [04-the-transfer-matrix-figure](04-the-transfer-matrix-figure.md) | The reviewer-credible transfer demo |

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

**The hypothesis:** *the LoRA's own output r̂ causes composition the way the cached oracle correction does: compose rate rises with λ on r̂ and the matched controls stay at the floor.*

**Why this run and not the oracle's:** roughly twenty figures measure the cached r_t and one measured the LoRA, while the LoRA is the artifact the paper ships (ledger). The oracle sweep's numbers exist for comparison (AUC 0.387 real against 0.023 random, from `dose_curves.json`); this sweep swaps the injection source and keeps everything else.

**If true:** compose rate rises with λ; wrong-seed and shuffled controls stay at the floor; AUC lands beside the oracle's for comparison.

**If false:** the curve stays flat, or a control rises with it, and the causal story cannot be carried by the LoRA's own output.

**This plan also absorbs experiment C from the ledger:** the λ grid {0, 0.25, 0.5, 0.75, 1.0} crossed with the step window at inference on existing checkpoints. If softness tracks λ at a fixed checkpoint, the blur is the injection, not undertraining; that reading feeds plan 06's interpretation.

**Associated materials:**
- **Review questions:** [../review/03-the-lora-dose-sweep.md](../review/03-the-lora-dose-sweep.md)
- **Ledger entries:** [the adapter-dose sweep](../decisions-taken-here.md#the-adapter-dose-sweep-is-owed-and-approved) and [experiment C](../decisions-taken-here.md#the-longer-training-question-runs-as-three-experiments-re-scoped-by-evidence)
- **Oracle machinery to copy:** the dose sweep under `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose/` and its runner

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Expected runtime:** the grid is 5 λ values x the window x the probe pairs and seeds; at 50 DDIM steps a render is under a minute on a biggpu device, so the sweep is hours, not days. Name the exact cell count in task 1.1's manifest before launching and record measured wall time in the review file.

**Prerequisites:** plan 02's harness (shared); the checkpoint; the validated scorer.

**GPU:** one device, launched per [execution-protocol](../../../../../environment/hpc/execution-protocol.md); nohup path when Slurm has no idle node, and then `squeue` is blind to it (harvest by `pgrep` + the log).

**W&B project:** `prime_lab/poe-repair-animals-compose`.

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- Large outputs to `/datasets` only; the runner's disk guard checks that filesystem.
- The `co3` python; fp16 with the upcast rule; guidance fixed at 7.5.
- biggpu allows one Slurm job per user; long sweeps go nohup outside Slurm ([overview](../../../../../environment/overview.md)).

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**The causal dose figure for the shipped artifact: compose rate against λ on r̂, four control rows, AUC beside the oracle's.** It matters now because the paper's causal story currently rests on a correction no deployed system has access to.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

**The problem.** The register's imbalance: the causal dose evidence belongs to the oracle, the paper ships the LoRA.

**The solution.** The same machinery with the injection source swapped: λ on r̂ instead of λ on cached r_t, wrong-seed and shuffled controls kept.

**Key insight.** The dog × dog probe (plan 02) is this sweep's zero-interaction control cell; the two share one harness, so 02 lands first and 03 reuses it.

---

## Description: what to build

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The manifest**: pairs, seeds, λ grid {0, 0.25, 0.5, 0.75, 1.0}, window 0-10, controls (wrong-seed r̂, shuffled r̂), one row per cell, printed counts per arm before launch (a flag with an empty target group is a silent no-op).
2. **The runner**: plan 02's harness plus λ scaling and the two control sources. Location: `scripts/showcase/lora_dose_sweep.py`.
3. **The scored table**: `dose_curves_lora.json` mirroring the oracle's `dose_curves.json` schema (compose rate per λ per arm, AUC per arm with its meaning in words).
4. **The softness read for plan 06**: per-λ CLIP/DINOv2 sharpness proxy or the qualitative strip, labelled descriptive, answering "does softness track λ at fixed checkpoint".

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves master-plan objective 3 and goal 3. Checkable outcomes:

1. The manifest's per-arm counts printed and recorded before launch.
2. `dose_curves_lora.json` written with all arms scored.
3. The review file's bar answered; the dose figure drafted for plan 05.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 0. ✅ Verify this plan first

- [ ] **0.1 Run the following prompt: `/verify-plan plans/closing-the-compositional-gap/plans/showcase-the-trained-lora/plans/03-the-lora-dose-sweep.md`**

▶ **Next: [task 1.1](#1--build-the-sweep)**.

### 1. 🧪 Build the sweep

◀ **Needs: [02 tasks 1.1 to 1.4](02-the-dog-x-dog-null-probe.md#1--build-and-run-the-probe)** done, so the shared harness exists.

- [ ] **1.1 Write the manifest and print per-arm counts.** Completion is observable: a printed table, one row per arm, cell counts non-zero for every arm.
- [ ] **1.2 Extend the harness with λ scaling and the wrong-seed / shuffled control sources.**
- [ ] **1.3 Launch the sweep** per the execution protocol; output to `/datasets/mmolefe/poe_repair_min/outputs/showcase/lora_dose/`. Log the launch mode (Slurm or nohup) in the review file's Runs table.
- [ ] **1.4 Score all arms; write `dose_curves_lora.json`** (schema mirrors the oracle file). Completion is observable: the json's per-arm row counts equal the manifest's.
- [ ] **1.5 Compute the softness-vs-λ read** and save its strip beside the json, labelled descriptive.

▶ **Next: [instruction 2.1](#2--read-the-sweep)**.

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-engagement-gate) ➡️

### 2. 📈 Read the sweep

◀ **Needs: [tasks 1.1 to 1.5](#1--build-the-sweep)** done.

2.1 **Monitor while it runs.** `squeue -u mmolefe` for a Slurm launch; for nohup, SSH to the node, `pgrep -af 'lora_dose'` and `tail -f` the log. ✅ cells accumulating under `/datasets/.../lora_dose/`; ❌ the log stalls or the disk guard trips: stop, `/ingest-error-pattern --from-run-log`.

2.2 **Read the curve.** Open the drafted dose figure (task 1.4's quick plot): ✅ real-r̂ curve rises with λ and both controls hug the floor; ❌ any control rises: record it, the comparison is contaminated and the review file says so.

2.3 **Write the verdict** into the [review file](../review/03-the-lora-dose-sweep.md): AUC per arm, the bar, the launch mode, wall time.

▶ **Next: the engagement gate.**

---

## The engagement gate

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> This figure is the causal spine of the showcase. A flat curve or a rising control changes what the paper may claim, and finding that out before assembly is the point of running it now.

**Pass criteria:**
- Every manifest arm scored; the review bar answered with AUC values.

**Fail criteria:**
- A control arm rises off the floor (contaminated comparison), or per-arm counts were zero anywhere (silent no-op).

**When you get results, answer the open questions in the [review file](../review/03-the-lora-dose-sweep.md).**

---

## Figure Catalog

⬅️ [Previous](#the-engagement-gate) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

### Pending

| Figure | Lane | What it shows | Save to |
|--------|------|---------------|---------|
| the LoRA dose figure | subject | compose rate (y, 0-1) vs λ (x), one curve per injection source (r̂, wrong-seed, shuffled), AUC in words | drafted here, shipped by plan 05 |

### Generated during plan execution

| Figure | Lane | Description | Generated by | Status |
|--------|------|-------------|--------------|--------|
| dose_curves_lora.json | — | the scored table (sidecar) | task 1.4 | ⏳ |
| softness-vs-λ strip | — | descriptive; feeds plan 06's interpretation | task 1.5 | ⏳ |

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Extract errors | `/ingest-error-pattern --from-run-log` | task 3.1 | new patterns into the catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | auto | this file's matrix regenerated |
| Close out | `/sync-plan-tree` | task 3.2 | statuses aggregated up |

### 3. 🧹 Close out

◀ **Needs: [instruction 2.3](#2--read-the-sweep)** done.

- [ ] **3.1 Run the following prompt: `/ingest-error-pattern --from-run-log`** (after any red run).
- [ ] **3.2 Run the following prompt: `/sync-plan-tree plans/closing-the-compositional-gap/plans/showcase-the-trained-lora/`**

▶ **Next: [04-the-transfer-matrix-figure](04-the-transfer-matrix-figure.md).**

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**File:** the oracle dose runner feeding `/datasets/.../outputs/interaction_term/dose/dose_curves.json` — the schema and control-row layout to mirror.
**File:** `scripts/showcase/dog_x_dog_probe.py` (plan 02) — the shared harness this extends.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

▶ Paste to run this plan (reuses the injection harness plan 07 builds):

```
/run-experiment plans/closing-the-compositional-gap/plans/showcase-the-trained-lora/plans/03-the-lora-dose-sweep.md — four control rows; stop if any control leaves the floor; AUC always carries its meaning in words.
```

alt, headless overnight: in a fresh session run `/unattended run-experiment plans/closing-the-compositional-gap/plans/showcase-the-trained-lora/plans/03-the-lora-dose-sweep.md` and paste the tmux block it emits (hours of renders, numeric abort conditions, no human mid-loop). The engagement gate and the review file's bar are the stop conditions.

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[04-the-transfer-matrix-figure](04-the-transfer-matrix-figure.md): the generalization demo at the reviewer-credible tier.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

### From global catalog

(empty until `/ingest-error-pattern` populates)

### From project catalog

(empty until `/ingest-error-pattern` populates)
