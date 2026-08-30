# 🧪 The LoRA's run across correction amounts

**This plan asks one question: does turning up the LoRA's own correction, a little at a time, make more pictures compose, while matched fake corrections do nothing?**

**Step 33 in the root running order. Waits on: step 32 (the dog × dog test is this series' zero-interaction run). Next: [04-the-transfer-matrix-figure](04-the-transfer-matrix-figure.md).**

## Recommended prompt (after run completes)

```
/ingest-error-pattern --from-run-log
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 32 (previous) | [02-the-dog-x-dog-null-probe](02-the-dog-x-dog-null-probe.md) | The zero-interaction control run |
| **33 (current)** | **03: the-lora-dose-sweep** | Compose rate against the amount of correction, for the shipped LoRA |
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

**The hypothesis:** *the LoRA's own output r̂ causes composition the way the cached true correction does: [compose rate](../../../context/world/compose-rate.md) rises with λ on r̂ and the matched controls stay at what you would get by luck.*

> The cached true correction is the correction computed from the joined prompt, saved once and
> read back. What you would get by luck is what compose rate reads when the injected vector
> carries no real correction, and it is not zero. AUC is
> the area under the compose-rate-against-λ curve, on a 0-to-1 scale: 1.0 would mean every
> generation composed at every λ, 0.0 that none did.

**Why this run and not the cached one:** roughly twenty figures measure the cached r_t and one measured the LoRA, while the LoRA is the artifact the paper ships (ledger). The cached correction's numbers exist for comparison (AUC 0.387 real against 0.023 random, from `dose_curves.json`); this series swaps the injection source and keeps everything else.

**If true:** compose rate rises with λ; wrong-seed and shuffled controls stay at what you would get by luck; AUC lands beside the cached correction's for comparison.

**If false:** the curve stays flat, or a control rises with it, and the causal story cannot be carried by the LoRA's own output.

**This plan also absorbs experiment C from the ledger:** the λ grid {0, 0.25, 0.5, 0.75, 1.0} crossed with the step window at inference on existing checkpoints. If softness tracks λ at a fixed checkpoint, the blur is the injection, not undertraining; that reading feeds plan 06's interpretation.

**Associated materials:**
- **Review questions:** [../review/03-the-lora-dose-sweep.md](../review/03-the-lora-dose-sweep.md)
- **Ledger entries:** [the run across correction amounts for the adapter](../decisions-taken-here.md#the-run-across-correction-amounts-for-the-adapter-is-owed-and-approved) and [experiment C](../decisions-taken-here.md#the-longer-training-question-runs-as-three-experiments-re-scoped-by-evidence)
- **The machinery to copy:** the cached correction's own run across amounts under `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose/` and its runner

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Expected runtime:** the grid is 5 λ values x the window x the dog × dog pairs and seeds; at 50 DDIM steps a render is under a minute on a biggpu device, so the whole series is hours, not days. Name the exact number of runs in task 1.1's manifest before launching and record measured wall time in the review file.

**Prerequisites:** plan 02's runner (shared); the checkpoint; the validated scorer.

**GPU:** one device, launched per [execution-protocol](../../../environment/hpc/execution-protocol.md); nohup path when Slurm has no idle node, and then `squeue` is blind to it (harvest by `pgrep` + the log).

**W&B project:** `prime_lab/poe-repair-animals-compose`.

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- Large outputs to `/datasets` only; the runner's disk guard checks that filesystem.
- The `co3` python; fp16 with the upcast rule; guidance fixed at 7.5.
- biggpu allows one Slurm job per user; long series go nohup outside Slurm ([overview](../../../environment/overview.md)).

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**The causal figure for the shipped artifact: compose rate against λ on r̂, four control rows, AUC beside the cached correction's.** It matters now because the paper's causal story currently rests on a correction no deployed system has access to.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

**The problem.** The register is lopsided. All the evidence that more correction gives more composition belongs to the cached correction, and the paper ships the LoRA.

**The solution.** The same machinery with the injection source swapped: λ on r̂ instead of λ on cached r_t, wrong-seed and shuffled controls kept.

**Key insight.** The dog × dog test (plan 02) is this series' zero-interaction control; the two share one runner, so 02 lands first and 03 reuses it.

---

## Description: what to build

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The manifest**: pairs, seeds, λ grid {0, 0.25, 0.5, 0.75, 1.0}, window 0-10, controls (wrong-seed r̂, shuffled r̂), one row per run, printed counts per condition before launch (a flag with an empty target group is a silent no-op).
2. **The runner**: plan 02's runner plus λ scaling and the two control sources. Location: `scripts/showcase/lora_dose_sweep.py`.
3. **The scored table**: `dose_curves_lora.json` mirroring the `dose_curves.json` schema of the cached-correction run (compose rate per λ per condition, AUC per condition with its meaning in words).
4. **The softness read for plan 06**: per-λ CLIP/DINOv2 sharpness proxy or the qualitative strip, labelled descriptive, answering "does softness track λ at fixed checkpoint".

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves master-plan objective 3 and goal 3. Checkable outcomes:

1. The manifest's per-condition counts printed and recorded before launch.
2. `dose_curves_lora.json` written with every condition scored.
3. The review file's threshold question answered; the compose-rate-against-λ figure drafted for plan 05.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 0. ✅ Verify this plan first

- [ ] **0.1 Run the following prompt: `/verify-plan plans/01-showcase-the-trained-lora/plans/03-the-lora-dose-sweep.md`**

▶ **Next: [task 1.1](#1--build-the-series)**.

### 1. 🧪 Build the series

◀ **Needs: [02 tasks 1.1 to 1.4](02-the-dog-x-dog-null-probe.md#1--build-and-run-the-test)** done, so the shared runner exists.

- [ ] **1.1 Write the manifest and print per-arm counts.** Completion is observable: a printed table, one row per condition, with a non-zero run count for every condition.
- [ ] **1.2 Extend the harness with λ scaling and the wrong-seed / shuffled control sources.**
- [ ] **1.3 Launch the sweep** per the execution protocol; output to `/datasets/mmolefe/poe_repair_min/outputs/showcase/lora_dose/`. Log the launch mode (Slurm or nohup) in the review file's Runs table.
- [ ] **1.4 Score all arms; write `dose_curves_lora.json`** (schema mirrors the cached-correction file). Completion is observable: the json's per-condition row counts equal the manifest's.
- [ ] **1.5 Compute the softness-vs-λ read** and save its strip beside the json, labelled descriptive.

▶ **Next: [instruction 2.1](#2--read-the-series)**.

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

### 2. 📈 Read the series

◀ **Needs: [tasks 1.1 to 1.5](#1--build-the-series)** done.

2.1 **Monitor while it runs.** `squeue -u mmolefe` for a Slurm launch; for nohup, SSH to the node, `pgrep -af 'lora_dose'` and `tail -f` the log. ✅ renders accumulating under `/datasets/.../lora_dose/`; ❌ the log stalls or the disk guard trips: stop, `/ingest-error-pattern --from-run-log`.

2.2 **Read the curve.** Open the drafted compose-rate-against-λ figure (task 1.4's quick plot): ✅ real-r̂ curve rises with λ and both controls stay at what you would get by luck; ❌ any control rises: record it, the comparison is contaminated and the review file says so.

2.3 **Write the verdict** into the [review file](../review/03-the-lora-dose-sweep.md): AUC per condition, the threshold, the launch mode, wall time.

▶ **Next: what has to pass before this runs.**

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> This figure is what the showcase's causal claim rests on. A flat curve or a rising control changes what the paper may claim, and finding that out before assembly is the point of running it now.

**Pass criteria:**
- Every condition in the manifest scored; the review file's threshold question answered with AUC values.

**Fail criteria:**
- A control condition rises above what you would get by luck (contaminated comparison), or a condition's run count was zero anywhere (silent no-op).

**When you get results, answer the open questions in the [review file](../review/03-the-lora-dose-sweep.md).**

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

### Pending

| Figure | Lane | What it shows | Save to |
|--------|------|---------------|---------|
| the LoRA's compose-rate-against-λ figure | subject | compose rate (y, 0-1) vs λ (x), one curve per injection source (r̂, wrong-seed, shuffled), AUC in words | drafted here, shipped by plan 05 |

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

◀ **Needs: [instruction 2.3](#2--read-the-series)** done.

- [ ] **3.1 Run the following prompt: `/ingest-error-pattern --from-run-log`** (after any red run).
- [ ] **3.2 Run the following prompt: `/sync-plan-tree plans/01-showcase-the-trained-lora/`**

▶ **Next: [04-the-transfer-matrix-figure](04-the-transfer-matrix-figure.md).**

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**File:** the cached correction's runner across amounts, feeding `/datasets/.../outputs/interaction_term/dose/dose_curves.json` — the schema and control-row layout to mirror.
**File:** `scripts/showcase/dog_x_dog_probe.py` (plan 02) — the shared runner this extends.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

▶ Paste to run this plan (reuses the injection runner plan 07 builds):

```
/run-experiment plans/01-showcase-the-trained-lora/plans/03-the-lora-dose-sweep.md — four control rows; stop if any control rises above what you would get by luck; AUC always carries its meaning in words.
```

alt, headless overnight: in a fresh session run `/unattended run-experiment plans/01-showcase-the-trained-lora/plans/03-the-lora-dose-sweep.md` and paste the tmux block it emits (hours of renders, numeric abort conditions, no human mid-loop). What has to pass before this runs and the review file's threshold are the stop conditions.

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
