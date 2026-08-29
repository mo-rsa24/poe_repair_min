# 🔬 Extend the tracking set

**Step 36 in the root running order. Waits on: step 31 (the plateau read frames what the new curves are for). Next: [07-experiment-c-lambda-window](07-experiment-c-lambda-window.md).**

## Recommended prompt (after run completes)

```
/ingest-error-pattern --from-run-log
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 35 (previous) | [05-assemble-the-showcase-figures](05-assemble-the-showcase-figures.md) | The wall (runs last; earlier in numbering only) |
| **36 (current)** | **06: extend-the-tracking-set** | Instrument-02 plus four curves, frozen and smoke-proven before any launch |
| 37 (next) | [07-experiment-c-lambda-window](07-experiment-c-lambda-window.md) | The injection sweep on existing checkpoints |

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

**The instrument:** one tracking set, the fixed list of images and curves saved at every checkpoint, frozen before experiments A and B launch, rendered every 10k steps, logged to W&B. It extends instrument-02's wiring and nothing else (ledger: additions must be computable from what the tracking set already renders or records, so training cost does not grow).

**Contents (ledger, fixed):** the four F9 cells (held-out pairs, seed 9), the repair cell cat × dog seed 1, compose rate over the held-out pool (8 pairs × 8 seeds), direction-cosine and fraction-of-distance-reached (already wired), plus four additions: learned-vs-actual cosine at every recorded sampling step, teacher-forced (the pooled trainer records the learned delta at all sampling steps, `train_pooled.py`'s `record_delta_at_steps=list(range(...))`; report per window bucket, early/commit/late, with steps 7, 15, 22 as named representatives); DINOv2/CLIP embedding drift (distance-to-mono minus distance-to-poe per cell); spectral share of learned deltas at a fixed bucket, labelled diagnostic; and the divergence-step profile of ‖r̂‖ across the 50 steps (closed-loop, labelled so).

**Associated materials:**
- **Review questions:** [../review/06-extend-the-tracking-set.md](../review/06-extend-the-tracking-set.md)
- **Ledger entry:** [the shared instrument](../decisions-taken-here.md#the-shared-instrument-extends-instrument-02-and-nothing-else)
- **The wiring it extends:** [instrument-02](../../does-the-fix-reach-unseen-pairs/plans/instrument-02-three-live-curves-while-training.md)

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Expected runtime:** code plus one 1-epoch smoke on a free device, in the 1-to-3-hour band instrument-02's smoke measured. Record the measured time in the review file.

**Prerequisites:** instrument-02's eval hook (already wired and smoke-proven); the phase1 checkpoint for the teacher-forced read.

**W&B project:** `prime_lab/poe-repair-animals-compose`.

**The admission rule is a gate, not advice:** an addition needing new forward passes goes to plan 10, never in here.

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- The `co3` python; outputs to `/datasets`; W&B owns the numbers, the tree owns the verdict ([overview](../../../../../environment/overview.md)).
- fp16 upcast rule for any decode in the embedding-drift read.

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**One frozen, smoke-proven tracking set that makes experiments A and B readable while they run.** It matters now because both experiments launch next, and a metric added after launch cannot be compared across the run.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

**The problem.** A and B chase a scorer-invisible softness; without new reads, their tracking would report the same saturated compose rate and decide nothing.

**The solution.** Four reads that see what the scorer cannot, admitted under the no-new-forward-passes rule, frozen before launch so every checkpoint is comparable.

**Key insight.** Freezing before launch is what makes "no crispness change in the frozen tracking set" (experiment A's null bar) a meaningful sentence.

---

## Description: what to build

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The four reads**, in the eval hook beside the existing three curves: learned-vs-actual cosine (all recorded steps, bucketed early/commit/late, teacher-forced), embedding drift (DINOv2 and CLIP), spectral share (top-k of stacked learned deltas, one fixed bucket), divergence-step profile (closed-loop, labelled).
2. **The frozen manifest**: `tracking_set.json` naming every cell and metric, written once, hash-stamped into the W&B config.
3. **The smoke**: one epoch on a free device proving all curves log non-null.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves goal 3 (tracking set extended and smoke-proven before any launch). Checkable outcomes:

1. `tracking_set.json` frozen and hashed into config.
2. Smoke run with all seven metric families logging non-null.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 0. ✅ Verify this plan first

- [ ] **0.1 Run the following prompt: `/verify-plan plans/closing-the-compositional-gap/plans/showcase-the-trained-lora/plans/06-extend-the-tracking-set.md`**

▶ **Next: [task 1.1](#1--wire-and-freeze)**.

### 1. 🔬 Wire and freeze

◀ **Needs: nothing from this scope** — instrument-02's hook exists.

- [ ] **1.1 Add the four reads to the eval hook** (`poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py`, `_run_inline_sample` region), each keyed `eval/tracking/<name>`, each computable from tensors the hook already holds.
- [ ] **1.2 Write and freeze `tracking_set.json`**; log its hash to W&B config. Completion is observable: the hash printed at startup and visible in the run's config tab.
- [ ] **1.3 Run the 1-epoch smoke** on a free device per the [execution protocol](../../../../../environment/hpc/execution-protocol.md); log dir to `/datasets/mmolefe/poe_repair_min/outputs/showcase/tracking_smoke/`.
  - Command shape: the phase-1 launcher with 1 epoch and W&B on. `bash scripts/animals_compose_transfer/train_phase1.sh dry` cannot serve here: dry mode sets `WANDB="disabled"` (line 16 of the script), so instruction 2.1's W&B check would fail with the wiring correct. Call the trainer with `--total-epochs 1 --wandb-mode online` (or `offline` plus a sync) instead.

▶ **Next: [instruction 2.1](#2--prove-the-smoke-in-wb)**.

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-engagement-gate) ➡️

### 2. 📈 Prove the smoke in W&B

◀ **Needs: [tasks 1.1 to 1.3](#1--wire-and-freeze)** done.

2.1 **Open the smoke run** (project above, newest run). Charts tab, search `eval/tracking/`. ✅ all four new families present with non-null points, and the three instrument-02 curves still log; ❌ any family missing or all-null: stop, fix, re-smoke.

2.2 **Capture the panel** into `runbook/reading-a-training-run.md`'s screenshot slot (wandb MCP or Playwright MCP per that runbook page), with a one-line healthy-shape caption.

2.3 **Record the verdict** in the [review file](../review/06-extend-the-tracking-set.md).

▶ **Next: the engagement gate.**

---

## The engagement gate

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> A and B are 20-plus GPU-hours reading their progress through this instrument. A silent metric here is silent garbage there.

**Pass criteria:**
- All seven families log non-null on the smoke; the manifest hash is in the config.

**Fail criteria:**
- Any read needed a new forward pass (route it to plan 10 instead), or any family is missing/null.

**When you get results, answer the open questions in the [review file](../review/06-extend-the-tracking-set.md).**

---

## Figure Catalog

⬅️ [Previous](#the-engagement-gate) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

### Pending

| Figure | Lane | What it shows | Save to |
|--------|------|---------------|---------|
| (none; this plan's output is wiring plus W&B panels) | — | — | — |

### Generated during plan execution

| Figure | Lane | Description | Generated by | Status |
|--------|------|-------------|--------------|--------|
| tracking_set.json | — | the frozen manifest (sidecar) | task 1.2 | ⏳ |
| smoke panel screenshot | process | the seven families logging, healthy shape | instruction 2.2 | ⏳ |

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Extract errors | `/ingest-error-pattern --from-run-log` | task 3.1 | new patterns into the catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | auto | this file's matrix regenerated |
| Close out | `/sync-plan-tree` | task 3.2 | statuses aggregated up |

### 3. 🧹 Close out

◀ **Needs: [instruction 2.3](#2--prove-the-smoke-in-wb)** done.

- [ ] **3.1 Run the following prompt: `/ingest-error-pattern --from-run-log`** (after any red run).
- [ ] **3.2 Run the following prompt: `/sync-plan-tree plans/closing-the-compositional-gap/plans/showcase-the-trained-lora/`**

▶ **Next: [07-experiment-c-lambda-window](07-experiment-c-lambda-window.md).**

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**File:** `poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py::_run_inline_sample` — the eval hook the reads join.
**File:** `artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k/dataset_meta.json` — the cells the manifest freezes.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

▶ Paste to run this plan (gates every training launch):

```
Execute plans/closing-the-compositional-gap/plans/showcase-the-trained-lora/plans/06-extend-the-tracking-set.md: wire the four curves, freeze tracking_set.json, then the 1-epoch smoke with --wandb-mode online (never dry mode, it disables W&B); report the four eval/tracking/ families.
```

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[07-experiment-c-lambda-window](07-experiment-c-lambda-window.md): the no-training injection sweep whose softness read this instrument will echo live during A and B.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

### From global catalog

(empty until `/ingest-error-pattern` populates)

### From project catalog

(empty until `/ingest-error-pattern` populates)
