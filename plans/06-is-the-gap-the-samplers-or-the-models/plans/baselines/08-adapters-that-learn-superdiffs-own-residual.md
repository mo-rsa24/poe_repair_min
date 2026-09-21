# 🎓 Adapters that learn SuperDiff's own residual

If the PoE-trained correction does not carry over to SuperDiff, train adapters against
SuperDiff's own missing term, `r_t^SD = eps_J − eps_M` at `kappa` 0.5, along SuperDiff's own
trajectories. Gated on [step 49](07-does-the-poe-trained-correction-reach-superdiff.md).

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/08-adapters-that-learn-superdiffs-own-residual.md — <ran: verdict per rank / closed unrun: step 49 transferred>
```

## Recommended skill

▶ `/run-experiment` ✅ for the cache build and the three launches; `/analyze-run` ✅ for the
   W&B curves and the sheets.

## Position in the plan tree

**Step 50 of 50.** Waits on step 49. The one order is the `## Running order` table in the
[repo root MASTER_PLAN.md](../../../../MASTER_PLAN.md).

| Step | Plan | What it does |
|------|------|-------------|
| 49 | [baseline-03: does-the-poe-trained-correction-reach-superdiff](07-does-the-poe-trained-correction-reach-superdiff.md) ⚠️ | the gate: this plan runs only on indistinguishable or hurts |
| **50 (current)** | **baseline-04: adapters-that-learn-superdiffs-own-residual** ⚠️ | **a SuperDiff trajectory cache at κ=0.5, three trainings, the same sheets with the new adapters** |

Design only. Verdicts and run state live in
[the paired review file](../../review/08-adapters-that-learn-superdiffs-own-residual.md).

## Table of contents

- [Position in the plan tree](#position-in-the-plan-tree)
- [What this asks, in one line](#what-this-asks-in-one-line)
- [Quick context: where you are](#quick-context-where-you-are)
- [Considerations](#considerations)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
- [What happens (visual)](#what-happens-visual)
- [Description: what to build](#description-what-to-build)
- [Purpose and goal](#purpose-and-goal)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [Tasks](#tasks) — things for Claude to execute
- [Instructions](#instructions) — things for you to do manually
- [The check before moving on](#the-check-before-moving-on)
- [Figure Catalog](#figure-catalog)
- [Orchestration: keeping catalogs and plan files in sync](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

## What this asks, in one line

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#quick-context-where-you-are) ➡️

Build a cache of SuperDiff trajectories at `kappa` 0.5 that stores every raw prediction the
residual needs, train rank 8, 16 and 32 adapters to predict `eps_J − eps_M` from it the way the
current adapters predict `eps_J − eps_PoE`, and draw the same sheets as step 49 with the new
adapters in place of the old.

## Quick context: where you are

⬅️ [Previous](#what-this-asks-in-one-line) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The system.** A cache builder extending `poe_repair/composers/superdiff.py` (which already
computes every raw prediction per step when `form_r_t=True`), a target variant beside
`poe_repair/training_cache.py`'s `delta_t_from_raw`, and the pooled trainer behind
`poe_repair/students/train_direct_eps.py`.

**What it does.** The current cache stores `{x_t, eps_a_raw, eps_b_raw, eps_j_raw, eps_uncond}`
per step along PoE trajectories and forms Δ_t = gs·(eps_j − eps_a − eps_b + eps_uncond). The
SuperDiff cache stores `{x_t, eps_1_raw, eps_2_raw, eps_j_raw, eps_uncond}` per step along
SuperDiff trajectories at `kappa` 0.5 and forms
Δ_t^SD = ε̃_J − [eps_uncond + gs·((eps_2 − eps_uncond) + 0.5·(eps_1 − eps_2))].

**Key components.** The cache, the target, the three trainings, the sheets.

**Testing approach.** Same sheets as step 49, same baseline (step 28's `kappa_050` sheets), so
the two adapter families read on one axis.

**Associated materials.** [The review questions](../../review/08-adapters-that-learn-superdiffs-own-residual.md),
[the PoE cache format](../../../../poe_repair/training_cache.py), and the clean pair pool in
[scope 04](../../../04-does-the-fix-reach-unseen-pairs/MASTER_PLAN.md).

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**This plan is gated and its cost is real.** Three trainings of the current length are about 30
hours each on an RTX 8000, one job per user on biggpu, queued behind whatever is running. It runs
only if step 49 says the PoE-trained correction does not transfer.

**`kappa` is 0.5 in the cache, the target, and inference.** A fixed `kappa` makes the residual a
single well-defined function of the state. The pipeline's own per-step `kappa` would make the
target depend on a value the adapter never sees.

**The cache is SuperDiff's trajectory at 200 stochastic steps, not PoE's at 50.** Four times the
steps per cell, and every step carries four raw predictions. Task 1 states the size before
building.

**The pool is the PoE cache's pool.** Same pairs, same seeds, so the only difference between the
two adapter families is the rule they learned against.

**The cache cannot be shared with PoE.** Different trajectories by construction.

**Known issues.** See [Error Matrix](#error-matrix).

## The claim

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**An adapter trained on SuperDiff's own residual at `kappa` 0.5 separates the two concepts at a
lower λ than SuperDiff alone, on the same sheets where the PoE-trained adapter did not.**

**Independent variables.** Rank (8, 16, 32) and λ, crossed on 2 pairs and 4 seeds, exactly as
step 49.

**Dependent variable.** Two separate concepts per tile, by eye on both pairs and by the validated
detector on cat×dog.

**Falsify condition.** Read each sheet against step 28's `kappa_050` sheet and step 49's sheet
for the same pair and rank.

- **Pass.** Separation at a lower λ than the baseline on more seeds than not, for at least one
  rank. The residual is learnable; the rule-specific version works where the transferred one did
  not.
- **Fail.** Indistinguishable from the baseline at every rank. Then `r_t^SD` is not learnable by
  this adapter family at this data size, which is a finding about the residual's structure.
- **Inconclusive.** Training did not converge (loss curves in W&B never flatten). Recorded, not
  read as a fail.

**Why this matters right now.** Together with step 49 it says whether the paper's correction is
one object per composition rule or one object per model.

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** If the PoE-trained adapter does not reach SuperDiff, the paper cannot say
whether that is because the residual is rule-specific or because no adapter can learn it.

**The approach.** Train the rule-specific adapter and put it on the same sheet.

**Key insights.**

1. Fixing `kappa` turns SuperDiff's residual into a fixed target, the same kind of object the PoE
   residual is.
2. The composer already computes every raw prediction the cache needs; the cache builder is a
   writer, not a new sampler.

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
  SuperDiff trajectory at kappa 0.5, 200 steps, per (pair, seed) in the pool
    step t:  x_t, eps_1_raw, eps_2_raw, eps_j_raw, eps_uncond   ──►  cache
                                                                       │
    target  Δ_t^SD = ε̃_J − eps_M(kappa 0.5)                            ▼
                                                        train rank 8 / 16 / 32
                                                                       │
                                                                       ▼
  sheets: rows seeds 9–12, columns λ, one per (pair, rank), read beside step 28 and step 49
```

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The cache builder.** `scripts/build_superdiff_cache.py`, one (pair, seed) per call like
   `scripts/build_training_cache.py`, writing `residuals/step_{000..199}.pt` with the five
   tensors, under `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff/cache/`.
2. **The target.** `delta_t_superdiff_from_raw(eps_1, eps_2, eps_j, eps_uncond, guidance_scale, kappa=0.5)`
   beside `delta_t_from_raw` in `poe_repair/training_cache.py`, with a loader that reads the
   SuperDiff key names.
3. **The trainings.** The pooled trainer pointed at the SuperDiff cache and target, rank 8, 16,
   32, alpha equal to rank, modules `attn2.to_q/k/v`, the current runs' length and schedule,
   W&B project `prime_lab/poe-repair-animals-compose`.
4. **The sheets.** Six, named `cat_dog_grid_200_steps_kappa_050_sdlora_r{8,16,32}.png` and the
   butterfly three, same assembler as step 28.

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Purpose**

Serves objective 7 of [the scope's direction](../../MASTER_PLAN.md): if the trained correction
does not transfer, learn SuperDiff's own.

**Goals**

1. The cache exists over the pool, sized and counted before training starts.
2. Three trainings finish with flattened loss curves in W&B, run ids in the review file.
3. Six sheets exist beside step 49's, with sidecars and README entries.
4. The verdict per rank is in the review file, read beside step 28 and step 49.

## Environment Facts This Plan Depends On

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- `co3` python at its absolute path, `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`.
  Never a bare `python`.
- Cache, checkpoints and renders under
  `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff/`, disk guard on
  `/datasets`, per [environment/storage.md](../../../../environment/storage.md). Task 1.1 states
  the cache size before building it.
- biggpu allows one job per user, so the three trainings queue behind whatever is running; the
  cache build and the sheets use the shared-device or `nohup` path per
  [environment/hpc/execution-protocol.md](../../../../environment/hpc/execution-protocol.md).
  About 30 hours per training on an RTX 8000, per
  [environment/hpc/throughput.md](../../../../environment/hpc/throughput.md).
- The tracker is W&B, project `prime_lab/poe-repair-animals-compose`; the review file carries run
  ids and verdicts, never curves.
- **The PoE cache cannot be used.** Different trajectories.
- fp16 models, fp32 for every norm.
- SDXL base via `SuperDiffSDXLPipeline`'s components, 200 steps, guidance 7.5, `kappa` 0.5,
  1024².

## Tasks

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/08-adapters-that-learn-superdiffs-own-residual.md`
  - Done when: the report comes back clean, or its proposals have been applied.
- [ ] **0.2** Cross-reference this plan's terms against context/, environment/, runbook/, report/,
      and any learning journey that names this project.
  - Paste: `/xref-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/08-adapters-that-learn-superdiffs-own-residual.md`
  - Done when: the scan comes back with no candidates, or its proposed links have been applied.
- [x] **0.3** Quote step 49's verdict in this plan's review file. Indistinguishable or hurts: this
      plan runs. Transfers: this plan closes unrun with that sentence as the reason.
  - **Done when:** the verdict is quoted and the run-or-close decision is written beside it.
  - Quoted in [the review file](../../review/08-adapters-that-learn-superdiffs-own-residual.md):
    hurts, so this plan runs.

▶ **Next: [task 1.1](#1--build-the-superdiff-cache)**, or the close out if step 49 transferred.

### 1. 💾 Build the SuperDiff cache

◀ **Needs: [task 0.3](#0--check-this-plan-before-working-from-it)**, the gate open.

- [x] **1.1** State the cache size before building: pool cells (from the PoE cache's pool, read from
      `scripts/build_training_cache.py`'s callers and scope 04's clean-pair-pool plan) × 200 steps
      × five tensors of 4×128×128 in fp16, in GB, and confirm `/datasets` has it.
  - **Done when:** the number and the `df` line are in the review file.
  - 88 cells, ≈ 11.6 GB, in
    [the review file](../../review/08-adapters-that-learn-superdiffs-own-residual.md).
- [x] **1.2** Write `scripts/build_superdiff_cache.py` on top of `superdiff.py` with
      `kappa_override=0.5`, `form_r_t=True`, saving the five raw tensors per step.
  - **Done when:** one cell builds, its 200 step files load, and the target formed from them
    matches the sidecar's `r_t_sd_norms` for that render to fp16 tolerance.
  - Written with the PoE cache's key names so the existing loader reads it unchanged (`x_t` is
    the UNet input, `eps_a_raw`/`eps_b_raw` are prompts 1/2). Check passed at 3.6×10⁻⁵, in
    [the review file](../../review/08-adapters-that-learn-superdiffs-own-residual.md).
- [x] **1.3** Put the SuperDiff blend where the trainer composes its three branches, and build
      the cache over the pool under `nohup`.
  - The cache keeps the PoE key names, so `poe_repair/training_cache.py` and
    `load_cached_steps` read it unchanged. The one place the rule lives is
    `one_pair_one_seed/trainer.py:_compose`, selected by `cfg.compose` ("poe", the original, or
    "superdiff" at `cfg.kappa`), threaded from
    `cross_pair_lora_pooling/train_pooled.py --compose superdiff --kappa 0.5`. Verified on CPU:
    the "poe" branch is bit-identical to the old formula, the "superdiff" branch equals the
    composer's blend.
  - Build: two shards on one node's two devices,
    `scripts/build_superdiff_cache.py --shard {0,1} --nshards 2`, ~75 s per cell.
  - **Done when:** every pool cell has 200 step files, counted (88 × 200), and a one-epoch
    `--dry-run` of the trainer with `--compose superdiff` on one cached cell runs to a
    checkpoint.

▶ **Next: [task 2.1](#2--train-the-three-adapters)**.

### 2. 🏋️ Train the three adapters

◀ **Needs: [task 1.3](#1--build-the-superdiff-cache)**, the whole cache on disk.

- [ ] **2.1** Point the pooled trainer at the SuperDiff cache and target; launch rank 8, 16, 32,
  - ⏹ Stopped by the user 2026-09-04 at steps 61k / 62k / 36k of 100k, on the checkpoint strips (verdict and every strip read in the review file). Checkpoints kept under `corrector/superdiff/sdlora_r{8,16,32}_100k/checkpoints/`.
      alpha equal to rank, the current runs' length and schedule, one at a time as biggpu allows.
  - 💡 `/run-experiment` ✅ for each launch.
  - **Done when:** three W&B run ids in the review file and three final checkpoints on disk.

▶ **Next: [instruction 4.1](#4--watch-the-curves-and-read-the-sheets)**.

### 3. 🚀 Render the sheets

◀ **Needs: [instruction 4.1](#4--watch-the-curves-and-read-the-sheets)**, the curves judged
flattened.

- [ ] **3.1** Render the 96 transfer cells with the new adapters, exactly step 49's task 2.1 with
  - ⚠️ Not run: the trainings were stopped before their finals. The six timelines under `corrector/superdiff/timelines/` stand in.
      the SuperDiff-trained checkpoints, under `.../superdiff/transfer_sdlora/`.
  - **Done when:** 96 renders with sidecars, counted.
- [ ] **3.2** Draw the six sheets and README entries into `across-composition-rules/`.
  - ⚠️ Not run, same reason.
  - **Done when:** six sheets with no red boxes, six sidecars, six README entries.

▶ **Next: [instruction 4.2](#4--watch-the-curves-and-read-the-sheets)**.

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted, or the gate closed at task 0.3.

- [ ] **Capture the failures this plan hit**, while they are still fresh.
  - Paste: `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/08-adapters-that-learn-superdiffs-own-residual.md`
  - Done when: each failure has a catalog entry, or there were none to record.
- [ ] **Bring the tree current** with what actually happened.
  - Paste: `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/08-adapters-that-learn-superdiffs-own-residual.md — <one line>`
  - Done when: statuses, the running order and the Error Matrix match reality.

▶ **Next: [the check before moving on](#the-check-before-moving-on).**

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-check-before-moving-on) ➡️

**For you to follow manually.** Do these yourself, interleaved with the Tasks rather than after
them.

### 4. 👁️ Watch the curves and read the sheets

◀ **Needs: [task 2.1](#2--train-the-three-adapters)**, the three runs launched.

- [ ] **4.1** In W&B, project `prime_lab/poe-repair-animals-compose`, open each of the three runs,
      Charts tab, and read the training loss and the eval compose-rate curve.
  - Expected result: loss flattens before the final step on all three; compose rate rises then
    plateaus.
  - ✅ All three flattened: hand to task 3.1.
  - ❌ One never flattens: record it as inconclusive for that rank, run the sheets for the others.
- [ ] **4.2** Read each of the six sheets beside step 28's `kappa_050` sheet and step 49's sheet
      for the same pair and rank, column by column, and record the first λ at which each seed
      separates, for all three.
  - ✅ Pass / ❌ fail / 〰️ inconclusive per rank, per the claim's definitions.

▶ **Next: [the close out](#close-out--record-what-this-plan-taught)**.

## The check before moving on

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **Why this checkpoint matters:** this is the last plan in the scope, and with step 49 it decides
> whether the paper's correction is one object per rule or one object per model.

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
SD=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff

find "$SD/cache" -name "step_199.pt" | wc -l        # expect one per pool cell
ls "$SD"/sdlora_r{8,16,32}/checkpoints/ | tail -3   # three final checkpoints
find "$SD/transfer_sdlora/pairs" -name "*.png" | wc -l   # expect 96
```

**Pass criteria**

- Cache complete and sized in the review file before training.
- Three run ids and three final checkpoints.
- Six sheets, six README entries.
- Instruction 4.2's verdict per rank in the review file.

**Fail criteria (STOP)**

- The target formed from the cache does not match the composer's own `r_t_sd_norms` for the same
  render. The cache is not what the composer computed.

**Partial pass guidance**

- One rank not converging is inconclusive for that rank and does not block the others.

**When you get results, answer**
[the review file](../../review/08-adapters-that-learn-superdiffs-own-residual.md).

## Figure Catalog

⬅️ [Previous](#the-check-before-moving-on) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

The standard every figure in this scope is held to is
[in the scope's MASTER_PLAN](../../MASTER_PLAN.md#the-figure-bar-every-plan-here-is-held-to).

### Pending: to be generated from prompts

None. This scope carries no `diagram-prompts.md`.

### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|---|---|---|---|---|---|
| `across-composition-rules/cat_dog_grid_200_steps_kappa_050_sdlora_r{8,16,32}.png` | — | rows seeds 9 to 12, columns λ; the SuperDiff-trained adapter injected into SuperDiff on every step at `kappa` 0.5 | task 3.2 | ⏳ | **Supplementary.** Read beside step 28's no-adapter sheet and step 49's PoE-adapter sheet |
| `across-composition-rules/butterfly_meadow_grid_200_steps_kappa_050_sdlora_r{8,16,32}.png` | — | same, the easy pair | task 3.2 | ⏳ | **Supplementary.** |

### Organization workflow

1. Cache and checkpoints under `/datasets`.
2. Six sheets into `across-composition-rules/`, sidecar and README entry each.

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| What changes | Where it has to be reflected |
|---|---|
| the gate decision at task 0.3 | the scope MASTER_PLAN's DoD item 14 |
| a sheet lands in `across-composition-rules/` | that folder's `README.md` gains an entry |
| a training finishes | the review file's Runs table with its W&B id |
| the plan's status | the scope [MASTER_PLAN.md](../../MASTER_PLAN.md) and the root running order, by `sync-plan-tree` |

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/08-adapters-that-learn-superdiffs-own-residual.md` | **task 0.1** | Conformance reported |
| Capture patterns | `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/08-adapters-that-learn-superdiffs-own-residual.md` | **the close out** | Errors added to catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | Auto | Error Matrix regenerated |
| Bring the tree current | `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/08-adapters-that-learn-superdiffs-own-residual.md` | **the close out** | Statuses and running order match reality |

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Path | Why it is read |
|---|---|
| [poe_repair/training_cache.py](../../../../poe_repair/training_cache.py) | the PoE cache format and `delta_t_from_raw`, which the SuperDiff variants sit beside |
| [scripts/build_training_cache.py](../../../../scripts/build_training_cache.py) | the one-cell-per-call builder shape, and the pool its callers use |
| [poe_repair/students/train_direct_eps.py](../../../../poe_repair/students/train_direct_eps.py) | the trainer the three runs reuse |
| [poe_repair/composers/superdiff.py](../../../../poe_repair/composers/superdiff.py) | computes every raw prediction the cache stores |

## Next step

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

None in this scope. The verdict folds into the paper's section on what the correction is, per
the scope's expected outcome.

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

<details>
<summary>2 catalogued failures and their fixes</summary>

Auto-updated after runs via `/ingest-error-pattern` and `/sync-plan-tree`.

### From global catalog

(Patterns applicable across all projects.) None yet.

### From project catalog

#### 🔴 the cache target is not what the composer computed

**When it happens:** the target formula in `training_cache.py` drifts from the blend in
`superdiff.py` (a different `kappa`, a dropped `eps_uncond` term).
**What you see:** an adapter trained on the wrong residual; sheets that look like the baseline.
**Why:** two copies of one formula.
**How to fix:** task 1.2 checks the cache-formed target against the composer's own
`r_t_sd_norms` for the same render before the pool is built.

#### 🟡 the cache does not fit

**When it happens:** 200 steps × five fp16 tensors per cell over the whole pool.
**What you see:** the disk guard aborting mid-pool.
**Why:** four times the PoE cache's step count.
**How to fix:** task 1.1 sizes it first; if it does not fit, store every other step and say so
in the review file.

---

**Auto-update note:** regenerated by `/sync-plan-tree` after new errors are added to the catalogs.
Do not edit manually.

</details>
