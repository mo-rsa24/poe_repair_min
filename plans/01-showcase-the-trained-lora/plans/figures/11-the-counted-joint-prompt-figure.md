# 📊 The counted joint-prompt figure

**This plan asks one question: how often does the joint prompt itself fail to show both animals, counted rather than assumed?**

**Step 41 in the root running order. Waits on: nothing (minutes of scoring on renders that exist). Next: [12-close-f8a-and-the-oracle-panel](12-close-f8a-and-the-best-case-panel.md).**

## Recommended prompt (after run completes)

```
/ingest-error-pattern --from-run-log
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 40 (previous) | [10-the-mechanism-follower](../tests/10-the-checkpoint-watcher.md) | The mechanism reads |
| **41 (current)** | **11: the-counted-joint-prompt-figure** | Score the mono renders; the three-bar figure |
| 42 (next) | [12-close-f8a-and-the-oracle-panel](12-close-f8a-and-the-best-case-panel.md) | The unscored tail and the best-case panel |

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

**The measurement:** the joint prompt's own [compose rate](../../../../context/world/compose-rate.md), counted. On some held-out runs the joint prompt fails to show both animals; the LoRA, which never consumes the joint prompt at inference, restores them. That baseline is currently an anecdote; this plan makes it a number.

> Held-out means the pairs were never shown during training, so the number says how well the
> adapter does on animals it has not seen.

**The figure (ledger):** compose rate per pair, three bars (joint prompt, plain PoE, LoRA-corrected), the same pairs and seeds throughout, with a strip of repaired renders beside it as the anecdote.

**The wording rule (ledger):** "restores composition the joint prompt loses on this pool", never "outperforms SDXL".

**Associated materials:**
- **Review questions:** [../review/11-the-counted-joint-prompt-figure.md](../../review/11-the-counted-joint-prompt-figure.md)
- **Ledger entry:** [the joint prompt's own failure](../../decisions-taken-here.md#the-joint-prompts-own-failure-framing-and-count)
- **The renders:** one `mono.png` per pair-and-seed under the training cache (`/datasets/mmolefe/poe_repair_min/outputs/training_cache/heldout/<pair>/seed_<n>/`)

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Expected runtime:** minutes, in-session (ledger): the renders already exist for every pair and seed; only the scorer runs.

**Prerequisites:** the validated scorer; the held-out pool definition (8 pairs × 8 seeds).

**A known open thread this read may close:** the clean-pair-pool review's open question on whether `an_elephant__x__a_penguin` composes by default.

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- The `co3` python; the scorer's weights load once per session; outputs to `/datasets` ([overview](../../../../environment/overview.md)).

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**Mono as a measured ceiling: the three-bar figure that frames every showcase comparison honestly.** F8b already shows the cached true correction at 0.75 on some pairs; this plan gives the joint prompt itself the same treatment.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

**The problem.** Captions comparing against Mono implicitly treat it as 1.0; the F9 rows already show that is false, and a reviewer will notice before we do.

**The solution.** Score what is already on disk and put the measured baseline in every comparison.

**Key insight.** The LoRA restoring composition the *target itself* loses is the strongest honest sentence the showcase owns; it only exists once this number does.

---

## Description: what to build

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The mono scoring pass**: every `mono.png` in the held-out pool through the instance-count scorer; `mono_baseline.json` (per run: pair, seed, n_instances, compose).
2. **The three-bar figure draft**: per pair, joint prompt vs plain PoE vs LoRA at λ=1 (the PoE and LoRA numbers exist in `compose_rate.json` and the transfer outputs), a strip of repaired renders beside it.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves goal 6. Checkable outcomes:

1. `mono_baseline.json` complete over the pool.
2. The three-bar draft exists with its sidecar; the review questions answered.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 0. ✅ Verify this plan first

- [ ] **0.1 Run the following prompt: `/verify-plan plans/01-showcase-the-trained-lora/plans/11-the-counted-joint-prompt-figure.md`**

▶ **Next: [task 1.1](#1--score-and-draft)**.

### 1. 📊 Score and draft

◀ **Needs: nothing** — renders and scorer exist.

- [ ] **1.1 Score the mono renders** over the held-out pool; write `mono_baseline.json` to `/datasets/mmolefe/poe_repair_min/outputs/showcase/mono_baseline/`. Completion is observable: one row per pair-and-seed in the pool, none missing.
- [ ] **1.2 Draft the three-bar figure** with its sidecar; caption uses the wording rule and names the tier, space, guidance 7.5, and checkpoint.

▶ **Next: [instruction 2.1](#2--spot-check-the-counts)**.

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

### 2. 👁️ Spot-check the counts

◀ **Needs: [tasks 1.1 to 1.2](#1--score-and-draft)** done.

2.1 **Open five mono renders the scorer called failures** and five it called passes. ✅ your eye agrees with the counts; ❌ disagreement on more than one: the scorer's reach into mono renders is suspect, note it in the review and hold the figure.

2.2 **Read the elephant × penguin row** and carry the answer to the clean-pair-pool review's open question (link there from this plan's review).

2.3 **Write the verdict** into the [review file](../../review/11-the-counted-joint-prompt-figure.md).

▶ **Next: what has to pass before this runs.**

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> Every showcase caption that compares against Mono waits on this number. Minutes of scoring decide whether the whole wall is honest.

**Pass criteria:**
- The pool fully scored; the spot-check agrees; the three-bar draft exists.

**Fail criteria:**
- The scorer and the eye disagree on mono renders (hold the figure, take the finding to plan 13's re-validation).

**When you get results, answer the open questions in the [review file](../../review/11-the-counted-joint-prompt-figure.md).**

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

### Pending

| Figure | Lane | What it shows | Save to |
|--------|------|---------------|---------|
| the three-bar counted figure | subject | compose rate per pair: joint prompt, plain PoE, LoRA; strip of repaired renders beside | drafted here, shipped by plan 05 |

### Generated during plan execution

| Figure | Lane | Description | Generated by | Status |
|--------|------|-------------|--------------|--------|
| mono_baseline.json | — | the measured joint-prompt baseline (sidecar) | task 1.1 | ⏳ |

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Extract errors | `/ingest-error-pattern --from-run-log` | task 3.1 | new patterns into the catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | auto | this file's matrix regenerated |
| Close out | `/sync-plan-tree` | task 3.2 | statuses aggregated up |

### 3. 🧹 Close out

◀ **Needs: [instruction 2.3](#2--spot-check-the-counts)** done.

- [ ] **3.1 Run the following prompt: `/ingest-error-pattern --from-run-log`** (after any red run).
- [ ] **3.2 Run the following prompt: `/sync-plan-tree plans/01-showcase-the-trained-lora/`**

▶ **Next: [12-close-f8a-and-the-oracle-panel](12-close-f8a-and-the-best-case-panel.md).**

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**Pattern:** the scoring loop in `scripts/cross_seed_lora_pooling/render_heldout_summary.py` — the scorer invocation to copy, pointed at `mono.png` instead of corrected renders.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

▶ Paste to run this plan (minutes of scoring, no queue):

```
Execute plans/01-showcase-the-trained-lora/plans/11-the-counted-joint-prompt-figure.md: score the cached mono renders, build the three-bars-per-pair figure with the strip of repaired renders beside it; the wording rule applies.
```

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[12-close-f8a-and-the-oracle-panel](12-close-f8a-and-the-best-case-panel.md): the unscored 70k-100k tail and the qualitative panel of what the cached correction reaches at best, reconciled with the transfer-figures plan next door.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

### From global catalog

(empty until `/ingest-error-pattern` populates)

### From project catalog

(empty until `/ingest-error-pattern` populates)
