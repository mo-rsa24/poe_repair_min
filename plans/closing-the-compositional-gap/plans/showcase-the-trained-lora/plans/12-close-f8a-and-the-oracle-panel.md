# 📊 Close F8a and build the oracle panel

**Step 42 in the root running order. Waits on: nothing. Next: [13-revalidate-the-scorer-off-animals](13-revalidate-the-scorer-off-animals.md).**

## Recommended prompt (after run completes)

```
/ingest-error-pattern --from-run-log
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 41 (previous) | [11-the-counted-joint-prompt-figure](11-the-counted-joint-prompt-figure.md) | The measured baseline |
| **42 (current)** | **12: close-f8a-and-the-oracle-panel** | Score the 70k-100k tail; the qualitative ceiling panel |
| 43 (next) | [13-revalidate-the-scorer-off-animals](13-revalidate-the-scorer-off-animals.md) | Opens tier-three captions |

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

**Two owed pieces of evidence, both cheap.** First: the phase1 run's checkpoints 70k to 100k were never scored (the compose-rate table stops at 60k) while their per-epoch samples sit on disk; scoring them extends F8a (compose-rate-as-the-lora-trains) to the full run and closes the "did anything move late" gap. Second: the qualitative ceiling panel, LoRA-corrected beside true-r_t-injected renders on the same held-out cells, the interpretation key for experiment B (oracle crisp with adapter soft is the one outcome that makes capacity the lever).

**Reconciliation duty:** these two tasks were also routed to
[figure-01-the-transfer-figures](../../does-the-fix-reach-unseen-pairs/plans/figure-01-the-transfer-figures.md) in the neighbouring scope. Task 1.1 checks whether that plan already picked them up; the work runs once, in whichever plan claims it first, and the other carries a pointer.

**Associated materials:**
- **Review questions:** [../review/12-close-f8a-and-the-oracle-panel.md](../review/12-close-f8a-and-the-oracle-panel.md)
- **Ledger entry:** [two tasks owed alongside](../decisions-taken-here.md#the-longer-training-question-runs-as-three-experiments-re-scoped-by-evidence)
- **The samples:** `artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k/samples/per_epoch/epoch_{1400..2000}_step_*`

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Expected runtime:** scoring only for the tail (in-session, under an hour); the ceiling panel reuses dose-sweep renders where the cell matches and renders only missing cells (a small batch).

**Prerequisites:** the validated scorer; `dose_curves.json`'s render store for oracle cells.

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- The `co3` python; outputs to `/datasets`; fp16 renders are mode-reproducible, and the panel's sidecar says so ([overview](../../../../../environment/overview.md)).

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**F8a extended to the full run, and the ceiling panel that decides how experiment B's result is read.** Expected: the curve stays flat near 0.96; if it moves late, the train-longer question reopens and this plan's review says so.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

**The problem.** A figure whose x-axis stops at 60% of the run invites the question the paper cannot answer; and B's verdict has no interpretation without the ceiling panel.

**The solution.** Score what exists; render the few missing ceiling cells; label the panel qualitative because no crispness instrument exists.

---

## Description: what to build

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The reconciliation check** against figure-01 next door (has it scored the tail or built the panel already?).
2. **The tail scoring**: per-epoch samples for steps 70k-100k through the scorer; extend `compose_rate.json`'s table shape into `compose_rate_full.json`.
3. **The F8a extension**: re-run `scripts/adapter_transfers.py` against the full table.
4. **The ceiling panel**: LoRA-corrected vs oracle-corrected (true r_t, λ=1) on the F9 cells, reusing dose renders where the cell matches; caption states the comparison is qualitative.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves goal 9. Checkable outcomes:

1. `compose_rate_full.json` covers 10k-100k.
2. F8a rebuilt over the full x-axis; the ceiling panel exists with its sidecar.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 0. ✅ Verify this plan first

- [ ] **0.1 Run the following prompt: `/verify-plan plans/closing-the-compositional-gap/plans/showcase-the-trained-lora/plans/12-close-f8a-and-the-oracle-panel.md`**

▶ **Next: [task 1.1](#1--score-extend-build)**.

### 1. 📊 Score, extend, build

◀ **Needs: nothing** — samples and renders exist.

- [ ] **1.1 Reconcile with figure-01 next door**: read its Owed section; if either piece is done there, link its output here and skip that piece. Completion is observable: one line in this plan's review naming what was found.
- [ ] **1.2 Score the 70k-100k per-epoch samples**; write `compose_rate_full.json`. Completion is observable: rows for steps 70000 to 100000 with in_in and out_out rates and n.
- [ ] **1.3 Rebuild F8a** over the full run (`python scripts/adapter_transfers.py`, full-table input); output beside the existing figure with its sidecar.
- [ ] **1.4 Build the ceiling panel**: per F9 cell, LoRA render beside oracle render; render only missing oracle cells; sidecar names space (decoded pixels), mode (closed-loop), and the qualitative label.

▶ **Next: [instruction 2.1](#2--read-the-tail-and-the-ceiling)**.

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-engagement-gate) ➡️

### 2. 👁️ Read the tail and the ceiling

◀ **Needs: [tasks 1.1 to 1.4](#1--score-extend-build)** done.

2.1 **Read the extended F8a.** ✅ flat within noise past 60k: saturation confirmed, the caption may say the run trained past its metric; ❌ a late rise: the train-longer question reopens; say so in the review and flag plan 08's framing.

2.2 **Read the ceiling panel.** ✅ oracle visibly crisper than the LoRA: capacity stays a live lever for B's reading; ❌ oracle equally soft: no training buys crispness, and B's null becomes the expected end of the story.

2.3 **Write both verdicts** into the [review file](../review/12-close-f8a-and-the-oracle-panel.md).

▶ **Next: the engagement gate.**

---

## The engagement gate

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> Two cheap reads that reframe two expensive experiments. Run them before believing anything A or B reports.

**Pass criteria:**
- The full-run table complete; both verdicts recorded.

**Fail criteria:**
- The tail scored off samples whose λ or window differ from the 10k-60k evals (a mixed comparison; check the sample manifest before scoring).

**When you get results, answer the open questions in the [review file](../review/12-close-f8a-and-the-oracle-panel.md).**

---

## Figure Catalog

⬅️ [Previous](#the-engagement-gate) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

### Pending

| Figure | Lane | What it shows | Save to |
|--------|------|---------------|---------|
| F8a, full run | subject | compose rate (y) over training steps 10k-100k (x), trained-on vs held-out, floor drawn | `paper/iclr/figures/compose-rate-as-the-lora-trains.{png,pdf,json}` (extended) |
| the ceiling panel | subject | LoRA beside oracle, same cells, qualitative label | drafted here, shipped by plan 05 |

### Generated during plan execution

| Figure | Lane | Description | Generated by | Status |
|--------|------|-------------|--------------|--------|
| compose_rate_full.json | — | the full-run scored table (sidecar) | task 1.2 | ⏳ |

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Extract errors | `/ingest-error-pattern --from-run-log` | task 3.1 | new patterns into the catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | auto | this file's matrix regenerated |
| Close out | `/sync-plan-tree` | task 3.2 | statuses aggregated up |

### 3. 🧹 Close out

◀ **Needs: [instruction 2.3](#2--read-the-tail-and-the-ceiling)** done.

- [ ] **3.1 Run the following prompt: `/ingest-error-pattern --from-run-log`** (after any red run).
- [ ] **3.2 Run the following prompt: `/sync-plan-tree plans/closing-the-compositional-gap/plans/showcase-the-trained-lora/`**

▶ **Next: [13-revalidate-the-scorer-off-animals](13-revalidate-the-scorer-off-animals.md).**

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**File:** `scripts/adapter_transfers.py` — F8a's builder, re-run on the full table.
**Dir:** `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose/` — the oracle render store the panel reuses.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

▶ Paste to run this plan (reads existing renders):

```
Execute plans/closing-the-compositional-gap/plans/showcase-the-trained-lora/plans/12-close-f8a-and-the-oracle-panel.md: the reconciliation task against figure-01-the-transfer-figures.md first, never duplicate its tasks; then score 70k-100k and build the oracle panel that keys experiment B.
```

alt, headless overnight: in a fresh session run `/unattended run-experiment plans/closing-the-compositional-gap/plans/showcase-the-trained-lora/plans/12-close-f8a-and-the-oracle-panel.md` and paste the tmux block it emits (the scoring sweep half only; the panel eyeball stays attended). The engagement gate and the review file's bar are the stop conditions.

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[13-revalidate-the-scorer-off-animals](13-revalidate-the-scorer-off-animals.md): the validation round that opens tier-three captions.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

### From global catalog

(empty until `/ingest-error-pattern` populates)

### From project catalog

(empty until `/ingest-error-pattern` populates)
