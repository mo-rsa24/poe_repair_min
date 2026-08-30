# 🔁 The transfer matrix at the credible tier

**Step 34 in the root running order. Waits on: nothing (independent of 02 and 03). Next: [05-assemble-the-showcase-figures](05-assemble-the-showcase-figures.md).**

## What this asks, in one line

does a fix pooled from one set of concepts repair pairs built from entirely different concepts?

## Recommended prompt (after run completes)

After you finish running this plan and want to ingest error patterns into the catalogs, use this prompt:

```
/ingest-error-pattern --from-run-log
```

This extracts error patterns from the run transcript, deduplicates against global and project catalogs, and adds new entries.

---

## Position in the plan tree

📋 [TOC](#table-of-contents) | [Next](#table-of-contents) ➡️

| Step | Plan | What it does |
|------|------|-------------|
| 33 (previous) | [03-the-lora-dose-sweep](../tests/03-the-correction-amount-series.md) | compose rate against the amount of correction |
| **34 (current)** | **04: the-transfer-matrix-figure** | **Group-pooled LoRAs evaluated on concept-disjoint pairs** |
| 35 (next) | [05-assemble-the-showcase-figures](05-assemble-the-showcase-figures.md) | the assembly under the standard |

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
- [Tasks](#tasks) — things for Claude to execute
- [Instructions](#instructions) — things for you to do manually
- [What has to pass before this runs](#what-has-to-pass-before-this-runs)
- [Figure Catalog](#figure-catalog)
- [Orchestration: keeping catalogs and plan files in sync](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Recommended skill](#recommended-skill)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

---

## Quick context: where you are

⬅️ [Previous](#table-of-contents) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The experiment:** score group-pooled LoRA checkpoints on pairs sharing no concept with their training pool, as a matrix: training group by evaluation pair.

**The hypothesis:** *the pooled correction transfers beyond its concepts: [compose rate](../../../../context/world/compose-rate.md) on disjoint pairs sits well above plain PoE with no correction at all.*

**If true:** the generalization figure ships at the tier the repo named reviewer-credible.

**If false:** transfer is concept-bound; the paper claims within-distribution repair and says so.

**What this plan does:** resolves the pooled checkpoints, runs the disjoint evaluation combinations not already cached, and renders the matrix; the existing held-out grid is the qualitative view beside it.

> Held-out means the pairs were never shown during training, so the number says how well the
> adapter does on animals it has not seen.

**Associated materials:**
- **Review questions:** [../review/04-the-transfer-matrix-figure.md](../../review/04-the-transfer-matrix-figure.md)
- **The ledger:** [decisions-taken-here.md](../../decisions-taken-here.md), "The generalization demo ships at the reviewer-credible tier"
- **Assets/outputs:** `/datasets/mmolefe/poe_repair_min/outputs/showcase_the_trained_lora/transfer_matrix/`

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Expected runtime:** depends on how many squares of the matrix are uncached; each square is seeds x 50 DDIM steps. Enumerate the squares first (task 1.1) and record the measured count and wall time in the review file.

**Prerequisites:** pooled checkpoints via `checkpoints/latest.json` per group ([heldout_pair.sh](../../../scripts/cross_seed_lora_pooling/heldout_pair.sh) is the resolution pattern); `scorer_validated.json`; the disjointness audit in task 1.2.

**Known issues:** see the [Error Matrix](#error-matrix).

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- Python is the `co3` env at `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`; never a bare `python`. GPU launches may use the `co3_bw` env where a launcher hard-codes it.
- Large outputs go to `/datasets/mmolefe/poe_repair_min/` only; any disk guard must check `/datasets`, the filesystem written to.
- biggpu allows one Slurm job per user and registers no GPUs; long runs launch with `nohup` over SSH per [environment/hpc/execution-protocol.md](../../../../environment/hpc/execution-protocol.md), with every path on the launch line absolute (`poe-launch-001`).
- fp16 upcast rule per [environment/overview.md](../../../../environment/overview.md): norms and cosines computed on cached fp16 tensors are read after upcast to fp32.
- W&B: training checkpoints came from `prime_lab/poe-repair-cross-seed` (run `pueuo7bl`); eval runs log to `prime_lab/poe-repair-animals-compose`. Credentials from `~/.netrc` / `WANDB_API_KEY`.

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**Deliver the transfer matrix: training group by concept-disjoint evaluation pair, compose rate per square, with the disjointness of every square audited before any scoring.** Single-pair transfer stays a quick wiring check; this is the tier a reviewer believes.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

**The problem:** the held-out grid shows transfer qualitatively, but "a memorised correction that happens to fit" survives it; the repo downgraded that tier itself.

**The solution:** pooled training, disjoint evaluation, a matrix a reader can scan for structure.

**Key insights:**
1. Disjointness has to be verified rather than assumed. A token shared between a pool and an evaluation pair silently demotes that square to within-distribution.
2. Mixed comparisons stay marked. Squares differing in more than the training-group axis are read alone, never as cause and effect.

---

## Description: what to build

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The census:** enumerate groups x candidate disjoint pairs; audit token-level disjointness for each square; write `matrix_manifest.json` (square, disjoint yes/no, cached yes/no).
2. **Fill:** run the uncached disjoint squares (window 0 to 10, λ=1, guidance 7.5, the eval seeds).
3. **Score and render:** instance-count per square; the matrix figure with per-square compose rate and the plain-PoE baseline row; sidecar `transfer_matrix.json`.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves [Objective 4 and DoD 3 of the scope master plan](../../MASTER_PLAN.md).

1. The manifest exists with every square's disjointness audited.
2. All disjoint squares scored; the matrix + sidecar rendered.
3. The review file's threshold question answered: transfer above plain PoE or not, per group.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 0. ✅ Check this plan before working from it

◀ **Needs: nothing** from plan 03; parallel-safe.

- [ ] **0.1** Run `/verify-plan plans/01-showcase-the-trained-lora/plans/04-the-transfer-matrix-figure.md`

▶ **Next: [task group 1](#1--census-fill-score)**.

### 1. 🔁 Census, fill, score

◀ **Needs: [group 0](#0--check-this-plan-before-working-from-it)**.

- [ ] **1.1** Enumerate the squares of the matrix; print the count (the census is what stops a silent no-op)
- [ ] **1.2** Audit disjointness token-by-token per square; a shared token marks that square `within` and excludes it from the transfer read
- [ ] **1.3** Run the uncached disjoint squares with nohup, absolute paths; observable effect: one PNG folder per square + manifest updates
- [ ] **1.4** Score everything; write `transfer_matrix.json`; render the matrix with the plain-PoE baseline row and true counts beside any rescaled marks

▶ **Next: [instruction 2.1](#2--read-the-matrix)**.

### 3. 🧹 Close out

◀ **Needs: [instruction group 2](#2--read-the-matrix)** done and the review answered.

- [ ] **3.1** `/ingest-error-pattern --from-run-log` if anything errored
- [ ] **3.2** `/sync-plan-tree plans/01-showcase-the-trained-lora`

▶ **Next: what has to pass before this runs.**

---

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

### 2. 👁️ Read the matrix

◀ **Needs: [tasks 1.1 to 1.4](#1--census-fill-score)**.

2.1 **Scan the matrix figure**
   - Open `.../transfer_matrix/matrix.png`
   - Expected result: disjoint squares above the plain-PoE baseline row, structure by group if any
   - ✅ record per-group rates in the review file
   - ❌ any `within`-marked square rendered as transfer: fix the figure before any reading

2.2 **Spot-check four squares by eye**
   - Two best, two worst: open their PNGs, compare eyeball to score
   - ✅ agree: note it
   - ❌ disagree: scorer contract first, science second

▶ **Next: [close out](#3--close-out)**.

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **The generalization claim of the paper rides on this tier being real.**

**Pass criteria:**
- Every rendered transfer square passed the disjointness audit.
- Matrix + sidecar exist; the threshold question answered per group.

**Fail criteria (STOP):**
- Disjointness assumed anywhere; a shared token found after rendering.

**When you get results, answer the open questions in the [review file](../../review/04-the-transfer-matrix-figure.md).**

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

### Pending: from the scope map

| Figure | Prompt | What it shows | Lane |
|---|---|---|---|
| the transfer matrix | [diagram-prompts.md, subject 4](../../diagram-prompts.md) | the grid under the scorer lens | subject |

### Generated during plan execution

| Figure | Description | Generated by | Status | Axes |
|---|---|---|---|---|
| step-34_transfer-matrix.png | compose rate per (train group, disjoint pair) | task 1.4 | ⏳ | rows: training group; columns: evaluation pair; each square: compose rate 0 to 1, plain-PoE baseline row beneath |

---

### Organization workflow

1. Render pending map pieces via [the illustrated map](../../diagram-prompts.md) when the scope's diagrams get drained by `/render-diagrams`.
2. Execute the plan; run-produced figures land under the outputs folder named in Quick context.
3. Plan 05 imports what graduates to the paper into `paper/iclr/figures/` with sidecars and runs the standard check.


## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

When this plan runs and produces output, three things stay in sync: the error catalogs, this file's Error Matrix, and the figure catalog.

1. **Ingest errors:** the close-out task runs `/ingest-error-pattern --from-run-log`; it deduplicates against `~/.claude/GLOBAL_ERROR_CATALOG.md` and [environment/known-failures.md](../../../../environment/known-failures.md) and triggers `/sync-plan-tree --update-error-matrices`.
2. **Error Matrix regeneration:** `/sync-plan-tree` rewrites the Error Matrix below from the catalogs; never edit it by hand.
3. **Figure filing:** outputs land under `/datasets/mmolefe/poe_repair_min/outputs/showcase_the_trained_lora/transfer_matrix/`, and each figure that graduates to the paper is copied into `paper/iclr/figures/` with its sidecar by plan 05, which owns the standard check.

| Command | Triggered by | Outcome |
|---|---|---|
| `/ingest-error-pattern --from-run-log` | close-out task | catalogs updated |
| `/sync-plan-tree` | close-out task | statuses + Error Matrix current |

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

- `scripts/showcase/transfer_matrix.py` (new)::main — census, audit, fill, score.
- Checkpoint resolution: [heldout_pair.sh](../../../scripts/cross_seed_lora_pooling/heldout_pair.sh)::read_latest; scorer contract: `scorer_validated.json` (instance count).

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

▶ Paste to run this plan (the reviewer-credible tier):

```
/run-experiment plans/01-showcase-the-trained-lora/plans/04-the-transfer-matrix-figure.md — group-pooled training evaluated on concept-disjoint pairs; void any square sharing a token.
```

▶ `/run-experiment plans/01-showcase-the-trained-lora/plans/04-the-transfer-matrix-figure.md` ✅

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[05-assemble-the-showcase-figures](05-assemble-the-showcase-figures.md): everything above, assembled under the ledger's standard.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

### From global catalog

Empty until `/ingest-error-pattern` populates it; regenerated by `/sync-plan-tree`. Global catalog: `~/.claude/GLOBAL_ERROR_CATALOG.md`.

### From project catalog

Empty until `/ingest-error-pattern` populates it. Project catalog: [environment/known-failures.md](../../../../environment/known-failures.md); read `poe-launch-001` (absolute paths on SSH launches) before any nohup launch in this scope.

**Auto-update note:** regenerated by `/sync-plan-tree`; do not edit manually.

---
