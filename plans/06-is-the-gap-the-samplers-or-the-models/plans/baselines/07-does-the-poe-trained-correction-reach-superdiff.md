# 🔁 Does the PoE-trained correction reach SuperDiff

The three adapters learned `eps_J − eps_PoE` along PoE trajectories. Injected into SuperDiff at
inference, on every step, does that correction still separate the two concepts?

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/07-does-the-poe-trained-correction-reach-superdiff.md — <transfers / indistinguishable / hurts, per rank>
```

## Recommended skill

▶ `/run-experiment` ✅ for the 96-render launch under `nohup`, and `/analyze-run` ✅ to harvest
   the sheets against the review questions.

## Position in the plan tree

**Step 49 of 50.** Waits on steps 28, 38 and 39. The one order is the `## Running order` table in
the [repo root MASTER_PLAN.md](../../../../MASTER_PLAN.md).

| Step | Plan | What it does |
|------|------|-------------|
| 28 | [baseline-01: what-changes-when-superdiff-leaves-its-own-defaults](05-what-changes-when-superdiff-leaves-its-own-defaults.md) ⚠️ | the composer, the `kappa` override, the λ injection, and the κ=0.5 sheets this plan reads against |
| 38, 39 | [experiment A](../../../01-showcase-the-trained-lora/plans/experiments/08-experiment-a-resume-to-200k.md), [experiment B](../../../01-showcase-the-trained-lora/plans/experiments/09-experiment-b-rank-16-32.md) ⚠️ | the rank 8, 16 and 32 checkpoints |
| **49 (current)** | **baseline-03: does-the-poe-trained-correction-reach-superdiff** ⚠️ | **injects each adapter into SuperDiff over the whole run and reads the sheets against the no-adapter ones** |
| 50 | [baseline-04: adapters-that-learn-superdiffs-own-residual](08-adapters-that-learn-superdiffs-own-residual.md) ⚠️ | runs only if this plan says the correction does not carry over |

Design only. Verdicts and run state live in
[the paired review file](../../review/07-does-the-poe-trained-correction-reach-superdiff.md).

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

Attach each trained adapter (rank 8, 16, 32) to SuperDiff's UNet, run SuperDiff at 200 steps with
`kappa` fixed at 0.5, and on every step take Δ̂ = `eps_M`(adapter on) − `eps_M`(adapter off) and
step with `eps_M`(off) + λ·Δ̂. Then read the seeds-by-λ sheets against the no-adapter κ=0.5 sheets
from [step 28](05-what-changes-when-superdiff-leaves-its-own-defaults.md).

## Quick context: where you are

⬅️ [Previous](#what-this-asks-in-one-line) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The system.** `poe_repair/composers/superdiff.py`, gaining an adapter toggle inside
`_run_sampling_loop`, and the three LoRA checkpoints under
`/datasets/mmolefe/poe_repair_min/outputs/showcase/`.

**What it does.** The adapters were trained to supply the residual PoE drops. Step 28 showed
SuperDiff at its own settings is missing a residual of the same shape: one blended animal at
λ=0, two animals once half of `r_t^SD` is added back. Whether the PoE-trained residual is close
enough to `r_t^SD` to do that job is the question.

**Key components.** The adapter attach (same modules and adapter name as the dose sweep), the
whole-run two-pass injection, and the sheets.

**Testing approach.** Same sheet layout as step 28: rows seeds 9 to 12, columns λ, one sheet per
(pair, rank). The λ=0 column is the no-adapter render, reused from step 28.

**Associated materials.** [The review questions](../../review/07-does-the-poe-trained-correction-reach-superdiff.md),
[the dose sweep that set the attach and injection grammar](../../../../scripts/showcase/lora_dose_sweep.py),
and [step 28's review](../../review/05-what-changes-when-superdiff-leaves-its-own-defaults.md).

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**The injection runs on every step, not a window.** The PoE dose sweep injects on the first 10 of
50 steps. Here the adapter is on for all 200 steps, so every step pays two UNet passes and every
render costs about twice a plain one.

**`kappa` is fixed at 0.5 throughout.** So the no-adapter baseline is exactly step 28's
`kappa_050` sheets, and nothing but the adapter differs between a baseline tile and a transfer
tile at the same seed and λ.

**The adapter was trained on PoE trajectories at 50 DDIM steps.** SuperDiff's trajectory at 200
stochastic steps visits different states at different noise levels. The adapter sees inputs it
never trained on. That is the point of the test, and it is also why a null here is a finding
about the correction, not a bug.

**A silent no-op is the failure to catch.** Attaching to a UNet whose module names do not match
succeeds with zero matched modules and renders the baseline again. Task 1.1 prints the count.

**Rank order is reported, not claimed.** Three ranks give a trend, not a result.

**The cache cannot be used.** SuperDiff follows its own trajectory.

**Known issues.** See [Error Matrix](#error-matrix).

## The claim

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**A correction learned against PoE, injected into SuperDiff over the whole run at κ=0.5, either
separates the two concepts where SuperDiff alone does not, or it does not.**

**Independent variables.** Rank (8, 16, 32) and λ (0.25, 0.5, 0.75, 1), crossed on 2 pairs and
4 seeds. λ=0 is the no-adapter baseline from step 28.

**Dependent variable.** Whether each tile shows two separate concepts, by eye on both pairs and by
the validated detector on cat×dog only.

**Falsify condition.** Read per (pair, rank) sheet against step 28's `kappa_050` sheet for that
pair, column by column.

- **Transfers.** At a matched λ the adapter sheet separates the two concepts on more seeds than the
  no-adapter sheet does at that λ. The strongest form: separation at λ=0.25 where the baseline
  needs 0.5 (cat×dog) or already separates (butterfly×meadow, where the test is sharpness of the
  butterfly instead).
- **Indistinguishable.** The adapter sheet and the no-adapter sheet switch at the same λ on the
  same seeds. The correction did not carry over; step 50 runs.
- **Hurts.** Separation is lost at a λ where the baseline had it. Recorded, and step 50 runs.

**Why this matters right now.** If it transfers, the paper's adapter is a correction to the model,
not to one sampler's arithmetic, and step 50 is not needed. If it does not, the residual is
rule-specific and step 50 is the honest next experiment.

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** The adapter is trained and evaluated against one composition rule. A reviewer
asks whether it learned that rule's arithmetic or the model's missing term.

**The approach.** Put it inside a different rule that step 28 showed is missing a residual of the
same shape, and see whether it fixes that too.

**Key insights.**

1. Step 28 made SuperDiff a fair second rule: it composes once its residual is added back, so a
   transferred correction has something to do.
2. Fixing `kappa` at 0.5 removes the pipeline's own per-step variation, so the only difference
   between a baseline tile and a transfer tile is the adapter.

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
  one sheet per (pair, rank), 200 steps, kappa 0.5, adapter on for all 200 steps

            λ=0        λ=0.25     λ=0.5      λ=0.75     λ=1
  seed 9    [step 28]  [ ]        [ ]        [ ]        [ ]
  seed 10   [step 28]  [ ]        [ ]        [ ]        [ ]
  seed 11   [step 28]  [ ]        [ ]        [ ]        [ ]
  seed 12   [step 28]  [ ]        [ ]        [ ]        [ ]

  each tile: eps_M(off) + λ·(eps_M(on) − eps_M(off)) at every step
  read against step 28's kappa_050 sheet: same seeds, same λ, no adapter
```

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The adapter toggle in `superdiff.py`.** `run(..., lora_adapter_name=None, lambda_value=0.0)`:
   when an adapter name is given, every step runs the triple-batched UNet call twice, adapter off
   then on (PEFT `disable_adapter` / `set_adapter`, as `run_lora_residual_inject_masked` does), forms
   `eps_M` from each at the fixed `kappa`, and steps with `eps_M`(off) + λ·Δ̂.
2. **The attach.** `lora_trainer.attach_lora` with rank, alpha=rank, modules `attn2.to_q/k/v`,
   adapter name `lora`, then `load_lora_state` from the checkpoint, on SuperDiff's own UNet.
3. **The renders.** 3 ranks × 2 pairs × seeds 9 to 12 × λ ∈ {0.25, 0.5, 0.75, 1}: 96, under
   `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff/transfer/`.
4. **The sheets.** Six, named `cat_dog_grid_200_steps_kappa_050_lora_r{8,16,32}.png` and
   `butterfly_meadow_grid_200_steps_kappa_050_lora_r{8,16,32}.png`, same assembler as step 28.

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Purpose**

Serves objective 7 of [the scope's direction](../../MASTER_PLAN.md): test whether the trained
correction transfers across composition rules.

**Goals**

1. Each of the three checkpoints attaches to SuperDiff's UNet with a non-zero matched-module count,
   and λ=0 with the adapter attached reproduces step 28's `kappa_050` render exactly.
2. 96 renders exist with sidecars.
3. Six sheets exist under `across-composition-rules/` with sidecars and README entries.
4. The transfer verdict per (pair, rank) is in the review file, eye read beside the detector read
   where the detector applies.

## Environment Facts This Plan Depends On

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- `co3` python at its absolute path, `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`.
  Never a bare `python`.
- Renders write under
  `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff/transfer/`, with
  a disk guard on `/datasets`, per [environment/storage.md](../../../../environment/storage.md).
- Checkpoints, read only: `/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r8_450k/checkpoints/lora_step_450000.pt`,
  `.../phase1_r16_100k/checkpoints/lora_step_100000.pt`,
  `.../phase1_r32_100k/checkpoints/lora_step_100000.pt`. Each run's `config.json` records
  alpha equal to rank.
- About 7 hours of GPU for the 96 renders (every step is two UNet passes), so the run goes under
  `nohup` per [environment/hpc/execution-protocol.md](../../../../environment/hpc/execution-protocol.md),
  on mscluster85 after step 28's sweep releases it or on a biggpu node once a training finishes.
  mscluster85's GPU is shared with another user's long job; one render process at a time.
- **The cached trajectories cannot be used.** SuperDiff follows its own path.
- fp16 models, fp32 for every norm.
- SDXL base via `SuperDiffSDXLPipeline`'s components, 200 steps, guidance 7.5, `kappa` 0.5,
  1024².

## Tasks

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/07-does-the-poe-trained-correction-reach-superdiff.md`
  - Done when: the report comes back clean, or its proposals have been applied.
- [ ] **0.2** Cross-reference this plan's terms against context/, environment/, runbook/, report/,
      and any learning journey that names this project.
  - Paste: `/xref-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/07-does-the-poe-trained-correction-reach-superdiff.md`
  - Done when: the scan comes back with no candidates, or its proposed links have been applied.

▶ **Next: [instruction 4.1](#4--wait-for-the-three-final-checkpoints)**, then
[task 1.1](#1--put-the-adapter-inside-the-loop).

### 1. 🔌 Put the adapter inside the loop

◀ **Needs: [instruction 4.1](#4--wait-for-the-three-final-checkpoints)**, all three final
checkpoints on disk.

- [x] **1.1** Attach each checkpoint to SuperDiff's UNet and print the matched-module count.
  - Result in [the review file](../../review/07-does-the-poe-trained-correction-reach-superdiff.md);
    run on the latest checkpoints rather than the finals, per instruction 4.1's partial-pass path.
  - `lora_trainer.attach_lora` with `rank`, `alpha=rank`, modules `attn2.to_q`, `attn2.to_k`,
    `attn2.to_v`, adapter name `lora`; then `load_lora_state`. Same call sequence as
    `scripts/showcase/lora_dose_sweep.py:_attach_and_load_lora`.
  - **Done when:** all three print a non-zero count and the same count as each other (the
    architecture is the same, so the module set is). Zero is a failure, not a pass.
- [x] **1.2** Add the whole-run two-pass injection to `_run_sampling_loop`: adapter off then on
      per step, Δ̂ = `eps_M`(on) − `eps_M`(off), step with `eps_M`(off) + λ·Δ̂. Record ‖Δ̂‖ per step
      in the sidecar.
  - Landed in `superdiff.py` as `lora_adapter_name`, `lambda_value`, `lora_tag`; both checks in
    [the review file](../../review/07-does-the-poe-trained-correction-reach-superdiff.md).
  - **Done when:** with the adapter attached and λ=0, the render is byte-identical to step 28's
    `superdiff_200steps_kappa0.50` render for the same pair and seed (the off pass alone must
    reproduce the baseline), and at λ=1 ‖Δ̂‖ is non-zero on every step.

▶ **Next: [task 2.1](#2--render-the-transfer-set)**.

### 2. 🚀 Render the transfer set

◀ **Needs: [task 1.2](#1--put-the-adapter-inside-the-loop)**, the injection verified inert at
λ=0.

- [ ] **2.1** Render 96 cells: rank ∈ {8, 16, 32} × pair ∈ {cat×dog, butterfly×meadow} × seed
      ∈ {9, 10, 11, 12} × λ ∈ {0.25, 0.5, 0.75, 1}, 200 steps, `kappa` 0.5. Under `nohup`, log to
      `.../superdiff/transfer.log`, time the first cell and record it.
  - 💡 `/run-experiment` ✅ for the launch; `/analyze-run` ✅ for the harvest.
  - **Done when:** 96 renders with sidecars, counted.
  - Stopped on purpose at 49 of 96 (rank 8 both pairs, rank 16 cat×dog complete) once the
    verdict could no longer change; record in
    [the review file](../../review/07-does-the-poe-trained-correction-reach-superdiff.md).
- [ ] **2.2** Draw the six sheets and their sidecars into
      `paper/iclr/figures/how-much-is-added/across-composition-rules/`, λ=0 column taken from
      step 28's `kappa_050` cells, and add a README entry per sheet.
  - **Done when:** six sheets with no red boxes, six sidecars, six README entries.

▶ **Next: [instruction 4.2](#4--wait-for-the-three-final-checkpoints)**, the read.

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted, including the ones that went red.

- [ ] **Capture the failures this plan hit**, while they are still fresh.
  - Paste: `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/07-does-the-poe-trained-correction-reach-superdiff.md`
  - Done when: each failure has a catalog entry, or there were none to record.
- [ ] **Bring the tree current** with what actually happened.
  - Paste: `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/07-does-the-poe-trained-correction-reach-superdiff.md — <one line>`
  - Done when: statuses, the running order and the Error Matrix match reality, and
    [step 50](08-adapters-that-learn-superdiffs-own-residual.md) is marked to run or to close unrun.

▶ **Next: [the check before moving on](#the-check-before-moving-on).**

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-check-before-moving-on) ➡️

**For you to follow manually.** Do these yourself, interleaved with the Tasks rather than after
them.

### 4. 👁️ Wait for the three final checkpoints

◀ **Needs: [task 0.2](#0--check-this-plan-before-working-from-it)**.

- [ ] **4.1** Confirm the three trainings have finished and the final checkpoints exist:
      `ls /datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r8_450k/checkpoints/lora_step_450000.pt`
      and the two `lora_step_100000.pt` files for `phase1_r16_100k` and `phase1_r32_100k`.
  - Expected result: three files, each with a `config.json` beside its run saying alpha equals
    rank.
  - ✅ All three present: hand to task 1.1.
  - ❌ One missing: wait, or run this plan on the two that exist and record the third as pending.
- [ ] **4.2** Read each of the six sheets beside step 28's `kappa_050` sheet for the same pair,
      column by column, and record per sheet: the first λ at which each seed shows two separate
      concepts, and the same for the baseline.
  - ✅ Transfers, 〰️ indistinguishable, ❌ hurts, per the claim's definitions, per (pair, rank).
  - Where the detector applies (cat×dog) and disagrees with the eye, the eye is cited, per this
    project's practice.

▶ **Next: [the close out](#close-out--record-what-this-plan-taught)**.

## The check before moving on

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **Why this checkpoint matters:** step 50 costs three ~30-hour trainings. It runs only if this
> plan says the PoE-trained correction does not carry over, so a wrong verdict here either wastes
> that or skips it.

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
TR=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff/transfer

find "$TR/pairs" -name "*.png" | wc -l          # expect 96
ls paper/iclr/figures/how-much-is-added/across-composition-rules/*lora_r*.png | wc -l   # expect 6
```

**Pass criteria**

- Task 1.1's counts are non-zero and equal across ranks.
- λ=0 with the adapter attached is byte-identical to the step 28 baseline.
- 96 renders, six sheets, six README entries.
- Instruction 4.2's per-sheet verdicts are in the review file.

**Fail criteria (STOP)**

- A zero matched-module count, or λ=0 not reproducing the baseline. The injection is not inert
  and every sheet would be measuring the bug.

**Partial pass guidance**

- A missing checkpoint: run the ranks that exist, record the third as pending, and do not close
  the plan until it lands.

**When you get results, answer**
[the review file](../../review/07-does-the-poe-trained-correction-reach-superdiff.md).

## Figure Catalog

⬅️ [Previous](#the-check-before-moving-on) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

The standard every figure in this scope is held to is
[in the scope's MASTER_PLAN](../../MASTER_PLAN.md#the-figure-bar-every-plan-here-is-held-to).

### Pending: to be generated from prompts

None. This scope carries no `diagram-prompts.md`.

### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|---|---|---|---|---|---|
| `across-composition-rules/cat_dog_grid_200_steps_kappa_050_lora_r{8,16,32}.png` | — | rows seeds 9 to 12, columns λ; the PoE-trained adapter injected into SuperDiff on every step at `kappa` 0.5 | task 2.2 | ⏳ | **Supplementary.** Read beside `cat_dog_grid_200_steps_kappa_050.png` from step 28, which is the same grid with no adapter |
| `across-composition-rules/butterfly_meadow_grid_200_steps_kappa_050_lora_r{8,16,32}.png` | — | same, the easy pair | task 2.2 | ⏳ | **Supplementary.** Read beside the step 28 butterfly sheet |

### Organization workflow

1. Render the 96 cells under `/datasets`.
2. Draw the six sheets into `across-composition-rules/`, sidecar and README entry each.

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| What changes | Where it has to be reflected |
|---|---|
| the transfer verdict | [step 50](08-adapters-that-learn-superdiffs-own-residual.md)'s gate, and the scope MASTER_PLAN's DoD item 14 |
| a sheet lands in `across-composition-rules/` | that folder's `README.md` gains an entry |
| the plan's status | the scope [MASTER_PLAN.md](../../MASTER_PLAN.md) and the root running order, by `sync-plan-tree` |

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/07-does-the-poe-trained-correction-reach-superdiff.md` | **task 0.1** | Conformance reported |
| Capture patterns | `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/07-does-the-poe-trained-correction-reach-superdiff.md` | **the close out** | Errors added to catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | Auto | Error Matrix regenerated |
| Bring the tree current | `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/07-does-the-poe-trained-correction-reach-superdiff.md` | **the close out** | Statuses and running order match reality |

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Path | Why it is read |
|---|---|
| [poe_repair/composers/superdiff.py](../../../../poe_repair/composers/superdiff.py) | the composer this plan extends with the adapter toggle |
| [poe_repair/methods/_sampling.py](../../../../poe_repair/methods/_sampling.py) `run_lora_residual_inject_masked` | the off/on two-pass grammar being reproduced with `eps_M` in place of `eps_PoE` |
| [scripts/showcase/lora_dose_sweep.py](../../../../scripts/showcase/lora_dose_sweep.py) `_attach_and_load_lora` | the attach and load sequence, copied exactly |
| `/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r*/config.json` | rank and alpha per checkpoint |

## Next step

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[Step 50, adapters that learn SuperDiff's own residual](08-adapters-that-learn-superdiffs-own-residual.md),
which runs only if this plan's verdict is indistinguishable or hurts.

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

<details>
<summary>2 catalogued failures and their fixes</summary>

Auto-updated after runs via `/ingest-error-pattern` and `/sync-plan-tree`.

### From global catalog

(Patterns applicable across all projects.) None yet.

### From project catalog

#### 🔴 the adapter attaches to nothing

**When it happens:** attaching to a UNet whose module names differ from the training UNet's.
**What you see:** every transfer tile identical to the baseline; the plan reads as "does not
transfer" when nothing was injected.
**Why:** PEFT matches target modules by name and silently matches zero.
**How to fix:** task 1.1 prints the matched count; task 1.2 checks λ=0 reproduces the baseline
and λ=1 has non-zero ‖Δ̂‖ on every step.

#### 🟡 two processes on the shared GPU

**When it happens:** launching the transfer set while step 28's sweep or another render still
holds mscluster85.
**What you see:** CUDA out of memory at model load.
**Why:** the GPU also carries another user's long-running job.
**How to fix:** one render process at a time; check `nvidia-smi` before launch.

---

**Auto-update note:** regenerated by `/sync-plan-tree` after new errors are added to the catalogs.
Do not edit manually.

</details>
