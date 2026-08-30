# 🖼️ Assemble the showcase figures

**Step 35 in the root running order. Waits on: steps 32, 33, 34, and the figure plans 41 and 42. Next: nothing; assembly closes the scope.**

## What this asks, in one line

are all the showcase figures built, filed, and clean against the one standard?

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
| 34 (previous) | [04-the-transfer-matrix-figure](04-the-transfer-matrix-figure.md) | the transfer matrix |
| **35 (current)** | **05: assemble-the-showcase-figures** | **The figure set into paper/iclr/figures/, checked against the standard** |
| (scope end) | — | the scope's recall gallery closes it |

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

**The experiment:** none; this plan draws settled results. It assembles the showcase set: the structure figure from the spectra, the figure where two independent measurements agree on the same window (`step-35_two-instruments-one-window.png`), the dog × dog grid, the compose-rate-against-λ figure, the transfer matrix, each with a sidecar, each checked against [the ledger's standard](../decisions-taken-here.md).

**What this plan does:** builds the two figures that need no new runs (the structure figure, and the one where two independent measurements agree on the same window), imports the run-produced figures from plans 02 to 04 as they land, and runs the standard check over the whole set.

**Associated materials:**
- **Review questions:** none; figure runs draw settled results and take no review file per the conventions.
- **Assets/outputs:** `paper/iclr/figures/` (small, versioned) with sidecars; working copies under `/datasets/.../assembly/`

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Expected runtime:** an afternoon per figure batch; cache-only, no GPU.

**Prerequisites:** spectrum.json + spectrum_windowed.json (exist); F4a's window numbers (verdicted); plans 02 to 04 for their figures as they complete: this plan can start with the two cache-only figures immediately.

**Known issues:** see the [Error Matrix](#error-matrix).

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- Python is the `co3` env at `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`; never a bare `python`. GPU launches may use the `co3_bw` env where a launcher hard-codes it.
- Large outputs go to `/datasets/mmolefe/poe_repair_min/` only; any disk guard must check `/datasets`, the filesystem written to.
- biggpu allows one Slurm job per user and registers no GPUs; long runs launch with `nohup` over SSH per [environment/hpc/execution-protocol.md](../../../environment/hpc/execution-protocol.md), with every path on the launch line absolute (`poe-launch-001`).
- fp16 upcast rule per [environment/overview.md](../../../environment/overview.md): norms and cosines computed on cached fp16 tensors are read after upcast to fp32.
- W&B: training checkpoints came from `prime_lab/poe-repair-cross-seed` (run `pueuo7bl`); eval runs log to `prime_lab/poe-repair-animals-compose`. Credentials from `~/.netrc` / `WANDB_API_KEY`.

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**Deliver the showcase figure set under one standard a reader can trace: what is shown, axes named, every number carrying its meaning, no panel violating the ledger.**

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

**The problem:** figures accumulated across scopes with uneven captions; the paper needs one voice and one traceable standard.

**The solution:** one assembly pass owning the standard check, with the ledger as the checklist.

**Key insights:**
1. The two-measurements-one-window figure is free and strong. The injection-window curve (F4a) and the windowed held-out projection agree on steps 0 to 10, and they were measured in completely different ways.
2. Numbers live in figures and sidecars, not prose.

> Held-out means the pairs were never shown during training, so the number says how well the
> adapter does on animals it has not seen.

---

## Description: what to build

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **Structure figure:** energy-at-k with train, held-out projection, and the applicable anchors, from spectrum.json + spectrum_windowed.json; script `scripts/showcase/structure_figure.py` (new).
2. **Two measurements, one window:** F4a's curve of [compose rate](../../../context/world/compose-rate.md) against window position, and the early/late held-out projection, on aligned step axes; `scripts/showcase/two_instruments_window.py` (new).
3. **Imports:** the dog × dog grid, the compose-rate-against-λ figure, the transfer matrix as their plans finish; copy + sidecar into `paper/iclr/figures/`.
4. **The standard check:** a checklist pass over every figure against the ledger; violations fixed or the figure held back.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves [Objective 5 and DoD 4 of the scope master plan](../MASTER_PLAN.md).

1. The two cache-only figures exist with sidecars.
2. Every run-produced figure is imported with its sidecar.
3. The standard check has a written pass over the whole set.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 0. ✅ Check this plan before working from it

◀ **Needs: nothing**; the cache-only figures start immediately.

- [ ] **0.1** Run `/verify-plan plans/01-showcase-the-trained-lora/plans/05-assemble-the-showcase-figures.md`

▶ **Next: [task group 1](#1--build-the-cache-only-figures)**.

### 1. 🖼️ Build the cache-only figures

◀ **Needs: [group 0](#0--check-this-plan-before-working-from-it)**.

- [ ] **1.1** `structure_figure.py`: three-curve energy-at-k per the ledger (train + both self-fit reference levels; held-out + k/d in a second panel); sidecar JSON with every plotted number
  - 💡 `/design-figure` on the two-panel split if the single-figure version fights itself ✅
- [ ] **1.2** `two_instruments_window.py`: F4a's window curve and the windowed projection, aligned axes; sidecar
- [ ] **1.3** Copy both into `paper/iclr/figures/` with sidecars
- [x] **1.4** The [shared-plane](/home-mscluster/mmolefe/goal-setting/learning/trajectory-manifold-by-hand/plans/09-the-fraction-on-the-face.md) trajectory figure (F11 in the register): built as `paper/iclr/figures/held-out-trajectories-in-one-shared-plane.pdf` by `scripts/plot_shared_plane_trajectories.py` from the sidecar at `artifacts/drips/showcase-the-trained-adapter/manifold/manifold_data.json`; every drawn number recomputed and asserted in the script
- [ ] **1.5** Export `artifacts/drips/showcase-the-trained-adapter/manifold/trajectory-manifold-prototype.html` as a self-contained file into the supplement when it is assembled; the hosted prototype link is not citable, and if per-step decoded frames get built the hover card changes, so export last

▶ **Next: [task group 2](#2--import-as-plans-land)**.

### 2. 📥 Import as plans land

◀ **Needs: plans 02, 03, 04** delivering their figures (import per arrival; do not wait for all).

- [ ] **2.1** Import the dog × dog grid + norm curve with sidecars
- [ ] **2.2** Import the compose-rate-against-λ figure with `dose_curves.json`
- [ ] **2.3** Import the transfer matrix with `transfer_matrix.json`

▶ **Next: [instruction 3.1](#3--run-the-standard-check)**.

### 4. 🧹 Close out

◀ **Needs: [instruction group 3](#3--run-the-standard-check)** passed.

- [ ] **4.1** `/sync-plan-tree plans/01-showcase-the-trained-lora`
- [ ] **4.2** When every plan in the scope is ✅: run `/recap-plan-tree @plans/01-showcase-the-trained-lora/MASTER_PLAN.md` and record the Artifact URL in the master plan

▶ **Next: what has to pass before this runs.**

---

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

### 3. ✅ Run the standard check

◀ **Needs: [tasks 1.1 to 2.3](#1--build-the-cache-only-figures)** for whichever figures exist.

3.1 **Walk the checklist per figure**
   - Open each figure beside [the ledger](../decisions-taken-here.md)
   - Check: axes named with units; every number's meaning stated
   - Check: correct anchor for the quantity (self-fit reference levels for train curves, k/d for projections)
   - Check: no distance-reading MDS panel; fidelity and composition never conflated; the two 8s never implied related
   - ✅ note "passes standard" in the figure's sidecar
   - ❌ fix or hold back; a held-back figure is named in the master plan, never silently dropped

▶ **Next: [close out](#4--close-out)**.

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **This is where the paper's figures either meet the standard or do not ship.**

**Pass criteria:**
- Every shipped figure has a sidecar and a written standard pass.

**Fail criteria:**
- Any figure in `paper/iclr/figures/` without its sidecar or with a standard violation.

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

### Pending: from the scope map

| Figure | Prompt | What it shows | Lane |
|---|---|---|---|
| the adapter on trial | [diagram-prompts.md, subject capstone](../diagram-prompts.md) | the four stations around the adapter chip | subject |
| the scope's process lane | [diagram-prompts.md, process lane](../diagram-prompts.md) | the five plans as one route | process |

### Generated during plan execution

| Figure | Description | Generated by | Status | Axes |
|---|---|---|---|---|
| step-35_structure-figure.png | energy-at-k with anchors | task 1.1 | ⏳ | X: k (log); Y: fraction of energy 0 to 1 |
| step-35_two-instruments-one-window.png | window curve + windowed projection | task 1.2 | ⏳ | X: denoising step; Y left: compose rate 0 to 1; Y right: held-out projection at k=8 |

---

### Organization workflow

1. Render pending map pieces via [the illustrated map](../diagram-prompts.md) when the scope's diagrams get drained by `/render-diagrams`.
2. Execute the plan; run-produced figures land under the outputs folder named in Quick context.
3. Plan 05 imports what graduates to the paper into `paper/iclr/figures/` with sidecars and runs the standard check.


## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

When this plan runs and produces output, three things stay in sync: the error catalogs, this file's Error Matrix, and the figure catalog.

1. **Ingest errors:** the close-out task runs `/ingest-error-pattern --from-run-log`; it deduplicates against `~/.claude/GLOBAL_ERROR_CATALOG.md` and [environment/known-failures.md](../../../environment/known-failures.md) and triggers `/sync-plan-tree --update-error-matrices`.
2. **Error Matrix regeneration:** `/sync-plan-tree` rewrites the Error Matrix below from the catalogs; never edit it by hand.
3. **Figure filing:** outputs land under `/datasets/mmolefe/poe_repair_min/outputs/showcase_the_trained_lora/assembly/`, and each figure that graduates to the paper is copied into `paper/iclr/figures/` with its sidecar by plan 05, which owns the standard check.

| Command | Triggered by | Outcome |
|---|---|---|
| `/ingest-error-pattern --from-run-log` | close-out task | catalogs updated |
| `/sync-plan-tree` | close-out task | statuses + Error Matrix current |

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

- `scripts/showcase/structure_figure.py` (new) — reads the two spectrum JSONs only; no cache access needed.
- `scripts/showcase/two_instruments_window.py` (new) — reads F4a's verdicted numbers from its sidecar and spectrum_windowed.json.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

▶ Paste to run this plan (closes the scope):

```
Execute plans/01-showcase-the-trained-lora/plans/05-assemble-the-showcase-figures.md: assemble under the ledger's standard; no sidecar, no ship; wording rules apply to every caption.
```

▶ `/plan-figures plans/01-showcase-the-trained-lora` ✅ — the set-level pass, if the batch wants re-balancing before import.
   alt: `/pair-figure` per figure where the statistical entity is in doubt.

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

The scope closes with the recall gallery (task 4.2). The walk that designed this scope is archived at [plans/.walk/showcasing-the-trained-lora.md](../../../artifacts/drips/showcase-the-trained-adapter/the-parallel-walk.md).

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

### From global catalog

Empty until `/ingest-error-pattern` populates it; regenerated by `/sync-plan-tree`. Global catalog: `~/.claude/GLOBAL_ERROR_CATALOG.md`.

### From project catalog

Empty until `/ingest-error-pattern` populates it. Project catalog: [environment/known-failures.md](../../../environment/known-failures.md); read `poe-launch-001` (absolute paths on SSH launches) before any nohup launch in this scope.

**Auto-update note:** regenerated by `/sync-plan-tree`; do not edit manually.

---
