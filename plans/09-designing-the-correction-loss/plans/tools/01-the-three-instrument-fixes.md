# 🔬 The four instrument fixes

**Nothing in this scope can be measured until the trainer reports whether the direction the loss cannot see was ever used, saves weights beside every picture it draws, and writes down which objective it ran. Three one-line changes, each proved to do something.**

**Step 71 in the root running order. Waits on nothing. Gates steps 72 to 74: every later plan reads a number or a checkpoint this one makes exist.**

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/09-designing-the-correction-loss/plans/tools/01-the-three-instrument-fixes.md — the four changes written, the smoke run's new panel, its paired checkpoint and its config keys recorded in the review file
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| **71 (current)** | **01: the four instrument fixes** | the number, the cadence and the config every later plan reads |
| 72 (next) | [02: the V0 foundation](../reading/02-the-v0-foundation.md) | the page that says what the objective running today produces |
| 73 | [03: the V0a read](../hypothesis/03-the-v0a-read.md) | four experiments already on disk, judged |

---

## Table of contents

- [Position in the plan tree](#position-in-the-plan-tree)
- [Words this plan uses](#words-this-plan-uses)
- [Quick context: where you are](#quick-context-where-you-are)
- [Considerations](#considerations)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
- [What happens (visual)](#what-happens-visual)
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

## Words this plan uses

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#quick-context-where-you-are) ➡️

- **The three branches**: what the adapted model predicts given "a cat", given "a dog", and given an empty prompt. Written `a`, `b` and `u`. One forward pass produces all three.
- **The guidance weight**: 7.5, the number classifier-free guidance scales each concept's contribution by. It sits inside this loss on both sides, not outside it at sampling time only.
- **The dialled error**: what the loss actually minimises, the guided composition against the guided cached target. This is `train/loss_fit` today.
- **The undialled error**: the same comparison with the guidance weight set to 1, so `a + b − u` against the raw cached joint-prompt prediction. It is not computed today.
- **The drift**: how far the adapted empty branch has moved from its cached self, `‖u − ū‖`. Already logged every step as `train/null_drift_norm`, whether or not the penalty is on.
- **The invisible direction**: add the same tensor to `a` and to `u` and the loss does not move, because it only ever sees the sum. Sampling reads the empty branch again at `−(w−1) = −6.5`, so the picture moves by 6.5 times that tensor.
- **A render set**: the four labelled comparison strips a training run writes at a checkpoint epoch, two on pairs it trained on and two on pairs it never saw.

---

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**What this plan asks**

Is the invisible direction actually carrying size, or is the whole scope arguing about a term that is negligible in practice?

**Why it comes first**

Every other plan here changes an objective on the strength of a derivation. This one measures whether the derivation matters, for free, on runs that already exist. If the undialled error is a negligible fraction of the fit loss, the case for freezing the empty branch weakens before a single training step is spent.

**What else it fixes while it is in there**

Two provenance gaps found while planning. Checkpoints are saved four times less often than renders, so nine of every twelve pictures have no weights behind them. And `config.json` omits every flag that changes the objective, so a saved run cannot say what it ran.

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Expected runtime.** The code changes are minutes. The smoke run is two epochs, about two minutes on any healthy card.

**Prerequisites.** None. Every file this plan touches already exists.

**Cost of the cadence change.** Twenty checkpoints instead of three per 50,000-step run, 2.3 GB an experiment at rank 16. No extra compute: the render pass already halts training at those epochs.

**Project tracking.** W&B, project `prime_lab/poe-repair-animals-compose`.

**Known issues.** See the [Error Matrix](#error-matrix).

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- [Storage](../../../../environment/storage.md): checkpoints go to `/datasets` only, which has 233 TB free, so the cadence change costs nothing that matters. `/home-mscluster` hit 100% once and silently killed checkpointing.
- [Overview](../../../../environment/overview.md): the forward pass is cast to fp32 before the branches are combined, because fp16 cancellation destroys the gradient. The undialled error is computed on the same fp32 tensors and must not reintroduce a fp16 path.
- [Nodes](../../../../environment/hpc/nodes.md): the smoke can run on any healthy `bigbatch` 3090; the fault set grows between probes, so pin a node rather than trusting `sinfo`.

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**One number, one flag and one config line turn this scope from a set of derivations into a set of measurements. The number says whether the invisible direction was used; the flag pairs weights with pictures so any point on a curve can be re-read; the config line means a run can say what objective it was.**

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** The scope's whole premise is that training is blind to a direction the sampler amplifies by 6.5. That premise has never been measured. It follows from the algebra, and the algebra is not in dispute, but whether the blind direction carries real size in a trained run is an empirical question nobody has asked.

**The solution.** `train/loss_fit` is the guided comparison. Computing the same comparison with the guidance weight at 1 costs one extra composition of tensors that are already in memory. The gap between the two is exactly the term the invisible direction lives in.

**Two things found while planning, fixed in the same pass.**

1. `--ckpt-every-epochs` defaults to 200 at `train_pooled.py:222`, which is every 10,000 steps, and the launcher never passes it. Renders run on `--sample-every-epochs`, set to 50, every 2,500 steps. Nine of twelve pictures have no weights.
2. The config dump at `train_pooled.py:365-373` lists its keys explicitly and omits `train_step_range`, `orth_weight` and `exclude_cells`. The four experiments from 2026-09-12 trained on denoising steps 0 to 25 only, and nothing in their saved config says so.

---

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
  today                                    after this plan
  ─────                                    ───────────────
  loss_fit   = ‖ guided − guided ‖²        loss_fit     (unchanged)
                                           loss_undialled = ‖ (a+b−u) − eps_J ‖²
                                              └── their gap is the blind direction

  render   ──┬── 2,500 steps               render     ──┬── 2,500 steps
  checkpoint ┴── 10,000 steps              checkpoint ──┘   same moment

  config.json: compose, kappa, ema_decay,  config.json: + train_step_range
    loss_space, energy_penalty,                         + orth_weight
    branch_prompt_style, null_anchor                    + exclude_cells
```

---

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The undialled error**, in `poe_repair/experiments/one_pair_one_seed/trainer.py`. Compose the three adapted branches at weight 1, subtract the raw cached joint-prompt prediction, square and mean. Add it to the `info` dict beside `loss_fit` at line 505. It is a diagnostic and never enters `loss`.
2. **The pairing of checkpoints with renders**, in `scripts/showcase/train_pool_run.sh`. One added line passing `--ckpt-every-epochs "$SAMPLEEVERY"`.
3. **The objective flags in the config dump**, in `poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py` at the `write_json` call.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves objective 1 of [the scope](../../MASTER_PLAN.md#objectives): measure whether the invisible direction is actually used, before changing any objective on the strength of a derivation.

1. `train/loss_undialled` appears in W&B on a live run and is not identically zero.
2. Its ratio to `train/loss_fit` is recorded for at least one existing run.
3. A checkpoint sits beside every render directory a smoke run writes.
4. A smoke run's `config.json` names its step range.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

For Claude to execute.

### 0. 🧭 Check this plan before working from it

- [x] 0.1 **Run the following prompt:**
  ```
  /verify-plan @plans/09-designing-the-correction-loss/plans/tools/01-the-three-instrument-fixes.md
  ```
  - Produces: a conformance report and any thin instructions
  - ✓ verified 2026-09-17: conforms. 19 sections in skeleton order, 31 anchors and 8 relative links resolve, review file paired with its bar fixed in text. One real fault, a backward pointer at the end of group 2 aimed at its own section, plus three groups missing a required pointer; all four applied.
- [x] 0.2 **Run the following prompt:**
  ```
  /xref-pillar @plans/09-designing-the-correction-loss/plans/tools/01-the-three-instrument-fixes.md
  ```
  - Produces: links for terms already defined elsewhere in the pillars
  - ✓ verified 2026-09-17: 9 of 9 relative targets resolve, no broken links. Six terms have an owning pillar file and sit unlinked, all second occurrences of terms linked once already. Four terms have no owner, and two of them are this plan's own outputs: `train/loss_undialled` and `train/null_drift_norm` appear nowhere in `context/`. Routed to plan 02, which writes the pillar files.

▶ **Next: [tasks 1.1 to 1.5](#1--write-the-four-changes)**, the four code changes.

### 1. 🔧 Write the four changes

◀ **Needs: [task 0.1](#0--check-this-plan-before-working-from-it)** done, so the plan is known good before any code changes.

- [x] 1.1 **Add the undialled error to the loss info dict**
  - In `poe_repair/experiments/one_pair_one_seed/trainer.py`, after `err` is computed at line 453
  - Compose `eps_a_raw_l + eps_b_raw_l - eps_uncond_l` and subtract the raw cached joint prediction, the un-guided `ej`, not the guided `eps_j_target`
  - Add `"loss_undialled"` to the `info` dict at line 505, beside `loss_fit`
  - It never enters `loss`. A diagnostic that changes the gradient is not a diagnostic
  - Produces: one new key in `info`, no forward pass, no extra memory
  - ✓ verified by inspection 2026-09-17, not yet by a run: `trainer.py` 378, 394, 415 carry the raw `ej` out of the no_grad loop, which the task did not anticipate (only the guided `ej_g` was carried, so line 453 had no raw joint prediction in scope); 461-462 form it inside `no_grad` on `.float()` tensors; 517 adds the key beside `loss_fit`. It never enters `loss`. The run proof is task 2.1.
- [x] 1.2 **Log it to W&B**
  - Wherever `train/loss_fit` is logged, add `train/loss_undialled`
  - Produces: a panel that appears on the next run
  - ✓ verified by inspection 2026-09-17: `multi_pair_trainer.py:145`, beside `train/loss_fit`. Reads `info.get("loss_undialled", nan)`, so a trainer that does not compute it logs NaN rather than a wrong number.
- [x] 1.3 **Pair checkpoints with renders**
  - In `scripts/showcase/train_pool_run.sh`, beside `--sample-every-epochs "$SAMPLEEVERY"` at line 100:
  ```bash
  --ckpt-every-epochs   "$SAMPLEEVERY" \
  ```
  - Produces: a checkpoint written at every epoch a render set is written
  - ✓ verified by inspection 2026-09-17: `train_pool_run.sh:103`. The task said line 100; it is 102, and the new line lands at 103. `bash -n` passes. Nothing was running, checked first, because bash reads a live script by byte offset.
- [x] 1.4 **Record the objective flags in the config dump**
  - In `poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py`, in the `write_json(run_dir / "config.json", {...})` call at line 365, add `train_step_range`, `orth_weight` and `cells_file`
  - `exclude_cells` is not a flag on this script and appears nowhere in the codebase. Exclusion is expressed by which cells the `--cells` file names, so the path to that file is what identifies the training set
  - Produces: a `config.json` that can identify its own objective
  - ✓ verified by a run 2026-09-17, Slurm job 56021: `config.json` reads `train_step_range [0, 25]`, `orth_weight 1.0`, `cells_file .../cells_v57.json`. The job later went out of memory, but the config is written before training starts, so this instrument is proved.
- [x] 1.5 **Take training cells from the train split by name, and record which split each came from**
  - In `poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py`, at the `resolve_cells(pair, seeds, cache_root=cache_root)` call on line 434, pass `split="train"`
  - `CellPath.from_root` at `poe_repair/training_cache.py:63` searches `["heldout", "train"]` when no split is given, so a cell sitting in both directories is taken from `heldout/`. Naming the split makes a cell missing from `train/` raise instead of falling back
  - In the same function, add the resolved `cell.split` to each entry of `cells_meta`, which writes `dataset_meta.json`. It records `None` for every cell today, so a finished run cannot be audited without re-deriving the lookup
  - This is live, not hypothetical: `a_typewriter__x__a_cactus` seeds 1 to 8 sit in both directories, which is 8 of `cells_v57`'s 43 cells. The two copies are byte-identical, so no run is contaminated in substance, but without this change the run's own record says 8 training cells came from the held-out split
  - Produces: a run whose `dataset_meta.json` names a split per cell, and a hard failure rather than a silent fallback when a cell is missing from `train/`
  - ✓ verified by a run 2026-09-17, Slurm job 56021: `dataset_meta.json` carries 43 cells, every one reading `split: train`, the eight `a_typewriter__x__a_cactus` cells included. Those are the cells that exist in both directories and were previously taken from `heldout/`. A cell missing from `train/` now raises `FileNotFoundError` naming the path it searched.
  - Full entry: [the held-out-first cache lookup](../../../../environment/known-failures.md)

▶ **Next: [task 2.1](#2--prove-each-change-does-something)**, the smoke that proves each one.

### 2. 🚀 Prove each change does something

- [x] 2.1 **Smoke the four changes together**
  - Two epochs, rank 16, `--train-step-range 0 25`, on a pinned healthy `bigbatch` node
  - Rank 16 deliberately, not the scope's rank 32: this smoke proves four instruments work and carries no claim, so it belongs on a 24 GB card that is actually free rather than queueing for a 49 GB one
  - Produces: a W&B run carrying `train/loss_undialled`; a checkpoint directory beside the render directory at the same step; a `config.json` naming `train_step_range`; a `dataset_meta.json` reading `split: train` for every cell
  - Done when all four are visible, not when the job exits 0
  - ✓ verified by a full run, not by a smoke, 2026-09-17: `v1_freeze_null_r16_s0_25` (W&B `kdzx03ji`, rank 16, 30,000 steps, `mscluster109`) carries all four. `history.json` holds `loss_undialled` across 30,830 rows; `config.json` reads `train_step_range [0, 25]`; `dataset_meta.json` lists 43 cells every one reading `split: train`, the eight `a_typewriter__x__a_cactus` cells included; 25 checkpoints sit beside 25 render sets. The two-epoch smoke was overtaken by a run that proves more.
- [ ] 2.2 **Record the ratio on an existing run**
  - The change cannot be applied retroactively, so recompute the undialled error offline from the cache for a handful of steps of `phase1_r32_100k`, using its checkpoint at 30,050
  - Produces: `artifacts/results/designing-the-correction-loss/00-matching-the-composition-to-the-joint-prompt/undialled-vs-fit.json`, the ratio per denoising step
  - This is the number the review file's bar is read against

▶ **Next: [instruction 3.1](#3--read-the-new-panel-in-wb)**, the panel the smoke just produced.

### Close out. 🔄 Record what this plan taught

◀ **Needs: [instruction 3.3](#3--read-the-new-panel-in-wb)** done, so there is a sentence to record.

- [ ] C.1 **Run the following prompt**, after any red run:
  ```
  /ingest-error-pattern --from-run-log
  ```
- [ ] C.2 **Run the following prompt:**
  ```
  /sync-plan-tree @plans/09-designing-the-correction-loss/plans/tools/01-the-three-instrument-fixes.md
  ```

---

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

For you to follow manually.

### 3. 🌐 Read the new panel in W&B

◀ **Needs: [task 2.1](#2--prove-each-change-does-something)** done, so a run exists with the panel on it.

- [ ] 3.1 **Open the smoke run**
  - Browser: `wandb.ai/prime_lab/poe-repair-animals-compose`, most recent run by timestamp
  - Click into it, then the Charts tab
- [ ] 3.2 **Find `train/loss_undialled`**
  - Search the panel list for `undialled`
  - ✅ Present, non-null, and varying between steps: the logging works
  - ❌ Present but identically zero: the logging is wrong, not the physics. Most likely the guided target was subtracted instead of the raw one. Back to task 1.1
  - ❌ Absent: the key never reached the logger. Check task 1.2 rather than 1.1
- [ ] 3.3 **Lay it beside `train/loss_fit`**
  - Add both to one panel
  - Write down which is larger and by roughly how much. That single sentence is what the review file's bar is read against, and it is the cheapest result this scope will ever produce

▶ **Next: [what has to pass before this runs](#what-has-to-pass-before-this-runs)**, the four gate criteria.

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> Every other plan in this scope reads a number, a checkpoint or a config key this plan creates. Nothing downstream starts until the smoke shows all four.

- The smoke run carries `train/loss_undialled`, non-null and varying
- A checkpoint directory sits beside the render directory at the same step
- `config.json` names `train_step_range`
- The undialled-to-fit ratio is recorded for one existing run

**Partial pass.** If the cadence and config changes land but the undialled error does not, plans 02 and 03 may still start: they read existing runs. Plan 04 waits, because it is the plan the undialled number is supposed to justify.

The review questions are in [the review file](../../review/01-the-three-instrument-fixes.md).

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

**Pending, from [this scope's illustrated map](../../diagram-prompts.md)**

| Figure | Lane | What it shows | Status |
|---|---|---|---|
| corrloss-03-two-legs-and-the-fine | subject | the two legs of the loss and every place the empty branch is read | opened beside the code at task 1.1, to check the composition being added matches the picture |
| corrloss-05-checkpoint-and-picture-together | subject | a checkpoint and a render written at the same step | opened at instruction 3.1, to check the smoke run's output matches it |
| corrloss-process-01 | process | the four stages of this scope, this plan being the first | opened at task 0.1 |

**Generated during plan execution**

| File | Lane | What it holds | Task | Status |
|---|---|---|---|---|
| `undialled-vs-fit.json` | — | the undialled error against the fit loss, per denoising step, on `phase1_r32_100k` at 30,050 | task 2.2 | ⏳ |

**Organization workflow.** The json is filed under `artifacts/results/designing-the-correction-loss/00-matching-the-composition-to-the-joint-prompt/` with its card entry, per `~/.claude/ARTIFACT_TREE_FORMAT.md`.

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/09-designing-the-correction-loss/plans/tools/01-the-three-instrument-fixes.md` | **task 0.1**, before any work | conformance and thin instructions reported |
| Cross-reference the plan | `/xref-pillar @plans/09-designing-the-correction-loss/plans/tools/01-the-three-instrument-fixes.md` | **task 0.2**, before any work | terms already documented elsewhere linked |
| Capture patterns | `/ingest-error-pattern --from-run-log` | **the close out**, after any red run | errors added to the catalogs |
| Bring the tree current | `/sync-plan-tree @plans/09-designing-the-correction-loss/plans/tools/01-the-three-instrument-fixes.md` | **the close out** | statuses, running order and Error Matrix match reality |

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**File:** `poe_repair/experiments/one_pair_one_seed/trainer.py`
**Relevant section:** the three branches are split at lines 446 to 449 and composed at 450; the error is formed at 453; the `info` dict starts at 505.

```python
# after err at line 453, before the orth split
with torch.no_grad():
    undialled = eps_a_raw_l + eps_b_raw_l - eps_uncond_l          # (K, 4, H, W)
    err_undialled = undialled - ej_raw.float()                     # the raw cached target
    loss_undialled = (err_undialled ** 2).mean(dim=(1, 2, 3)).mean()
# ... then in info, beside "loss_fit":
"loss_undialled": float(loss_undialled.item()),
```

The raw cached joint prediction is `ej` in the target block at lines 376 to 414, before `guided_eps` is applied to it. It must be carried out of that block for this to work; the guided `eps_j_target` is the wrong tensor.

**File:** `scripts/showcase/train_pool_run.sh`
**Relevant section:** the launch line at 93 to 110, where `--sample-every-epochs "$SAMPLEEVERY"` already sits.

**File:** `poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py`
**Relevant section:** `--ckpt-every-epochs` at line 222 with its default of 200; the config dump at 365 to 373, which lists its keys explicitly.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

— custom; no skill fits. Three edits to a trainer and a launcher that exist only here.

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents)

[02: the V0 foundation](../reading/02-the-v0-foundation.md) reads the five runs already on disk and writes the page every later variation copies.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

<details>
<summary>1 failure catalogued</summary>

**Purpose**: known issues and their fixes, regenerated by `/ingest-error-pattern` and `/sync-plan-tree`.

#### From global catalog

#### From project catalog

**poe-data-001 · a cell named in `--cells` is loaded from the held-out split without a warning** 🔴

`CellPath.from_root` searches `heldout/` before `train/` when no split is given, and
`train_pooled.py` gives none. A cell present in both directories is taken from the held-out
split, trained on, and later scored against, with no warning and no record: `dataset_meta.json`
writes `split: None` for every cell. Twelve cells across four pairs sit in both directories
today. Latent, not live: every cell of `cells_v54`, `cells_v55` and `cells_v57_animals` resolves
to `train/`. Full entry, with the pre-launch check to run:
[the held-out-first cache lookup](../../../../environment/known-failures.md).

**Bears on task 1.4.** The task adds `train_step_range`, `orth_weight` and `exclude_cells` to
`config.json` so a saved config can identify its own objective. The resolved split per cell is
the same kind of gap and is not on the list.

---

**Auto-update note:** regenerated by `/sync-plan-tree`. Do not edit by hand.

</details>
