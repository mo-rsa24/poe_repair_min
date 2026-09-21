# 🎲 Feynman-Kac steering on top of the rank-32 correction

Run the same particle sampler as plan 10, with the rank-32 correction added to the proposal at a
partial dose, and read whether selecting among those particles on the compose scorer adds anything
over the corrected sampler alone. Plan 10 showed that plain product-of-experts offers no composing
state to select; this plan puts composing states into the proposal on purpose and asks whether
selection can then finish the job.

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/13-feynman-kac-steering-on-top-of-the-rank-32-correction.md — <support / null / inconclusive, the compose rates per lambda>
```

## Recommended skill

▶ `/run-experiment` ✅ for the launch; `/analyze-run` ✅ for the W&B sheets and the sidecar.

## Position in the plan tree

**Step 56 of 65.** Waits on nothing that is not already on disk: the rank-32 checkpoint at step
30050, the pinned noise, the validated detector, and plan 10's runner. The one order is the
`## Running order` table in the [repo root MASTER_PLAN.md](../../../../MASTER_PLAN.md).

| Step | Plan | What it does |
|------|------|-------------|
| 55 | [baseline-06: feynman-kac-steering-on-a-detector-reward](10-feynman-kac-steering-on-a-detector-reward.md) ⚪ | the same steering over plain PoE: null, 0 of 128 unweighted particles composed on cat × dog |
| **56 (current)** | **baseline-09: feynman-kac-steering-on-top-of-the-rank-32-correction** ✅ | **the same steering with the adapter's correction in the proposal at λ 0.5 and 1.2, judged against the adapter alone** |

Design only. Verdicts and run state live in
[the paired review file](../../review/13-feynman-kac-steering-on-top-of-the-rank-32-correction.md).

## Table of contents

- [Position in the plan tree](#position-in-the-plan-tree)
- [What this asks, in one line](#what-this-asks-in-one-line)
- [Words this plan uses](#words-this-plan-uses)
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

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#words-this-plan-uses) ➡️

When the proposal is the corrected sampler at half dose, so that some of its particles already
show two animals, does resampling toward the ones the compose scorer counts two in raise the
compose rate over the corrected sampler left alone?

## Words this plan uses

⬅️ [Previous](#what-this-asks-in-one-line) | 📋 [TOC](#table-of-contents) | [Next](#quick-context-where-you-are) ➡️

- **The correction at λ**: the showcase renders' composition, `ε_frozen + λ · (ε_lora − ε_frozen)`,
  where `ε_frozen` is plain product-of-experts with the adapter off and `ε_lora` the same three
  branches with the rank-32 adapter (checkpoint step 30050) on. λ 1 is the adapter as trained;
  λ 0.5 adds half its correction.
- **Adapter alone**: that sampler with one particle, deterministic DDIM (`eta` 0), from the seed's
  pinned noise. The repo's standard render at that λ.
- **The control**: K 16 particles of the corrected sampler at `eta` 1.0, never weighted or
  resampled; particle 0 starts from the pinned noise.
- **FK on the adapter**: the same K 16 particles resampled at step indices 0, 10, 20, 30, 40 on
  the compose scorer's count of the decoded x0-hat, max potential, λ 10 in the potential (plan 10's
  settings, unchanged).
- **Best of 16**: the unweighted particle with the highest final reward, the paper's own baseline.
- **The fidelity tie-breaker**: ImageReward, the human-preference score the FK paper steers SDXL
  on, against the joint prompt. Reward `min(count, 2) + 0.5 · sigmoid(ImageReward)`: the count
  sets the level, fidelity orders within it. Among two-animal particles the cleanest survives and
  is the one shown. The bars judge the count alone.
- **Headroom**: one minus the adapter-alone compose rate at that λ. Selection cannot add more
  than this.

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The system.** Plan 10's package, `poe_repair/experiments/fk_steering/`, with an adapter path:
`steer.run_fk_steering(adapter_lambda=λ)` computes the two three-branch forwards per step and
mixes them; `run.py --adapter-checkpoint ... --adapter-lambdas 0.5 1.2` attaches the LoRA once
(disabled by default, enabled only inside the corrected forward), renders an adapter-alone
reference per λ, and draws one sheet per pair and λ. Launched by
`scripts/fk_steering/run_fk_steering.sh adapter`.

**What it does.** Plan 10 found the plain product proposes no composing state at K 16 on cat × dog
(0 of 128 unweighted particles, 0 of 128 steered). Selection needs something to select. The
rank-32 correction at λ 1 composes 7 of 8 held-out seeds, so at λ 1 the proposal spans composing
states with almost no headroom; at half dose the dose curve puts the adapter part way, and both
outcomes are in the proposal. This plan runs both λ and judges at 0.5.

**Key components.** The adapter attach, the mixed forward, the per-λ sheet, the per-λ verdict.

**Testing approach.** Per pair and λ, one sheet: rows seeds 9 to 16; columns Mono (`eta` 0),
plain PoE (`eta` 0), adapter alone at λ (`eta` 0), control particle 0 (K 16, `eta` 1, adapter at λ),
best of 16 unweighted, FK on the adapter at K 16. Only the weighting differs between the control
and the FK column.

**Associated materials.** [The review questions](../../review/13-feynman-kac-steering-on-top-of-the-rank-32-correction.md),
[plan 10](10-feynman-kac-steering-on-a-detector-reward.md) for the method and the null it landed,
[which checkpoint composes best](../../../../report/does-training-longer-help-the-pooled-lora/which-checkpoint-composes-best-and-does-more-correction-help.md)
for the adapter-alone rates this plan's headroom comes from.

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**This is a combination question, not a sampler-versus-model one.** Plan 10 asked whether the
sampler's share alone reaches composition; the answer was no. This plan asks whether selection adds
to the model-side fix. A support here does not change plan 10's null and does not size the
sampler's share. It goes to the paper as a sentence about combining the two, or as a row in
[step 29's comparison](06-three-rules-on-one-amount-axis.md).

**λ 1.2 has almost no headroom.** The adapter alone composes about 7 of 8 there, so the most FK
can add is one seed, under the 0.25 bar by construction. λ 1.2 is on the sheet because the
session contract asks for the adapter at 1.2; the judged λ is 0.5.

**Butterfly × meadow is not readable by this scorer.** Plan 10 found Mono itself scores 0.125 on
that pair: the detector counts animals, and a butterfly over a meadow is one animal. Its FK column
at K 16 read 0.75 because selection found frames with two butterflies. The pair stays on the run
at λ 1.2 only, as the contract asks, and its numbers are reported as what the detector counted,
never as composition.

**The reward and the judge are the same scorer.** As in plan 10: the best-of-16 column separates
what picking the best final particle adds from what resampling adds.

**Why a fidelity term, and why it cannot change the verdict.** The count is blind to haze and to a
patchy face; two of the adapter's own composed renders show exactly that. ImageReward orders the
two-animal particles by how clean they are, with a weight under 1 so a one-animal particle can
never outrank a two-animal one. The paper's SDXL runs steered on ImageReward alone; here it is the
second term, so the compose-rate bar is untouched. The sheet prints the score on every tile.

**K 16 only.** Plan 10's K 4 added nothing the K 16 read did not carry, and each corrected
particle costs two UNet forwards per step. Dropping K 4 keeps the run inside one night.

**The adapter must be off for the reference columns.** The attach helper leaves the adapter
disabled, the mixed forward enables it only for its own second pass and disables it again, and
the Mono and plain PoE references run through the frozen path. The λ 0 check is that the
control's particle 0 with the adapter attached but λ 0 would equal plain PoE; the adapter-alone
reference at λ is the check the eye can do against the showcase grids.

**Cost.** Measured on plan 10: about 17 s per particle trajectory at 50 steps, 1024², on the
A6000. The corrected forward doubles the UNet work, so about 34 s. Per cell and λ: 16 control
plus 16 steered particles plus one reference, about 19 min. Cat × dog at two λ plus butterfly ×
meadow at one: 24 cell-λ, about 7.5 h.

**Known issues.** See [Error Matrix](#error-matrix).

## The claim

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**With the rank-32 correction at half dose in the proposal, resampling on the compose scorer's
read of x0-hat raises the compose rate over the corrected sampler left unweighted.**

**Independent variables.** Steering on or off; λ in {0.5, 1.2} for the correction in the
proposal. Fixed: K 16, `eta` 1.0, 50 DDIM steps, guidance 7.5, 1024², λ 10 in the potential, max
potential, resampling at step indices 0, 10, 20, 30, 40; cat × dog at both λ, butterfly × meadow at
λ 1.2; seeds 9 to 16; the rank-32 adapter at step 30050.

**Dependent variable.** Compose rate over the 8 seeds by the validated detector. For FK, the
picked particle per seed; for the control, particle 0 per seed. Recorded, not judged: adapter
alone at `eta` 0, best of 16, the fraction over all 16 particles, agreement per read step.

**Falsify condition.** The bars sit in `poe_repair/experiments/fk_steering/steer.py`:
`PASS_MARGIN`, `NULL_MARGIN`, `MAX_REWARD_DISAGREEMENT`, `REWARD_BLIND_STEP`, and for this plan
`ADAPTER_LAMBDAS`, `ADAPTER_JUDGED_LAMBDA`, `ADAPTER_K`, `ADAPTER_CHECKPOINT`, `ADAPTER_RANK`.
Judged on cat × dog at λ 0.5.

- **Support.** FK on the adapter at λ 0.5 composes at least 0.25 more often than the control's
  particle 0 at λ 0.5. Selection finishes what a half dose starts.
- **Null.** FK is within 0.10 of the control at both λ 0.5 and λ 1.2. Selection adds nothing to
  the correction at either dose.
- **Inconclusive.** The step-10 reward disagrees with the final verdict on more than half of the
  final particles at λ 0.5.
- **The headroom sentence.** Whatever the verdict, the caption states the adapter-alone rate at
  each λ beside the FK rate, so a small gap at λ 1.2 is read as no room rather than no effect.

**Why this matters right now.** Plan 10 closed the sampler-only route. This is the cheapest test
of whether selection is worth anything once a model-side fix has put composing states in reach,
and it uses code that ran clean last night.

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** A reviewer who accepts that the adapter composes 7 of 8 will ask whether an
inference-time method on top of it reaches 8 of 8, or reaches 7 of 8 at a smaller dose.

**The approach.** The published steering method at its published settings, over the corrected
sampler at two doses, judged where the headroom is.

**Key insights.**

1. Selection can only reach what the proposal spans; plan 10 measured an empty span for plain
   PoE, and the dose curve says the span fills as λ grows.
2. Judging at half dose is what makes a support or a null informative; at full dose either
   outcome is within one seed of the ceiling.

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
  per step, per particle:  adapter OFF ──► ε_frozen (plain PoE)
                           adapter ON  ──► ε_lora
                           ε = ε_frozen + λ · (ε_lora − ε_frozen)          λ ∈ {0.5, 1.2}
  K 16 particles, DDIM eta 1.0, 50 steps
  steps 0, 10, 20, 30, 40: decode x0hat ──► detector count ──► clip 2 ──► exp(10 · lineage max) ──► resample

  sheet per (pair, λ), rows seeds 9..16:
  [ Mono eta0 | PoE eta0 | adapter alone λ, eta0 | control p0 (K16, eta1, λ) | best of 16 | FK K16 on adapter λ ]
```

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The mixed forward.** `steer.run_fk_steering(adapter_lambda=λ)`: per chunk, the frozen
   three-branch forward, then the adapter-on forward, mixed at λ; the adapter is disabled again
   before returning.
2. **The runner's adapter mode.** `run.py --adapter-checkpoint <pt> --adapter-rank 32
   --adapter-lambdas 0.5 1.2 --control-pair-lambdas 1.2 --particles 16`: attaches the LoRA once,
   renders the adapter-alone `eta` 0 reference per λ, the control and the steered run per λ, one
   sheet and sidecar per (pair, λ), `summary.json` and `verdict.json` judged at
   `ADAPTER_JUDGED_LAMBDA`, W&B group `fk-steering`.
3. **The launcher mode.** `scripts/fk_steering/run_fk_steering.sh adapter` with the same guards as
   plan 10's modes.

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Purpose**

Serves the paper's corrector section with the one combination a reviewer will ask for, and
serves scope 01's showcase by saying whether an inference-time add-on changes what the adapter
delivers.

**Goals**

1. A smoke run proves the adapter path: the attach report, an adapter-alone reference, a control
   and a steered run at one λ, a sheet, `verdict.json`.
2. Three sheets (cat × dog at λ 0.5 and 1.2, butterfly × meadow at 1.2) with sidecars, on W&B.
3. `verdict.json` judged at λ 0.5, quoted in the review file with the run id, node, device, PID.

## Environment Facts This Plan Depends On

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- Everything plan 10 depends on, unchanged: python by node, the `torch.cuda` check against
  `poe-launch-002`, outputs under `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/fk_steering/`
  with the disk guard, the training cache's pinned noise, one Slurm job per user so `nohup` on a
  free device, W&B as the tracker.
- The rank-32 checkpoint at
  `/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/checkpoints/lora_step_030050.pt`,
  rank 32, alpha 32, cross-attention q, k, v (its own `config` block, read 2026-09-06).
- A free `biggpu` device with at least 16 GB: the corrected forward keeps the same activations as
  plan 10 (two passes in sequence, not one wider one).

## Tasks

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/13-feynman-kac-steering-on-top-of-the-rank-32-correction.md`
  - Done when: the report comes back clean, or its proposals have been applied.
- [ ] **0.2** Cross-reference this plan's terms against context/, environment/, runbook/, report/,
      and any learning journey that names this project.
  - Paste: `/xref-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/13-feynman-kac-steering-on-top-of-the-rank-32-correction.md`
  - Done when: the scan comes back with no candidates, or its proposed links have been applied.

▶ **Next: [task 1.1](#1--pin-the-bars-and-build-the-adapter-path)**.

### 1. 🔧 Pin the bars and build the adapter path

- [x] **1.1** Write the falsification criterion into the review file and the adapter constants
      into `steer.py` before anything runs.
  - Done when: `ADAPTER_CHECKPOINT`, `ADAPTER_RANK`, `ADAPTER_LAMBDAS`, `ADAPTER_JUDGED_LAMBDA`
    and `ADAPTER_K` are in source beside plan 10's bars, and the review's pre-registered question
    quotes them.
- [x] **1.2** Add the mixed forward to `steer.py`, the adapter mode and the ImageReward
      tie-breaker (`--fidelity imagereward`) to `run.py`, and the `adapter` mode to the launcher.
  - **Done when:** `run.py --help` lists `--adapter-checkpoint`, `--adapter-lambdas`,
    `--control-pair-lambdas` and `--fidelity`, the module imports, and ImageReward loads and
    scores one image in `co3`.
- [x] **1.3** Run the smoke: cat × dog seed 9, K 2, 10 steps, 512², adapter at λ 0.5 only.
  - Command: `GPU=<idx> bash scripts/fk_steering/run_fk_steering.sh adapter --smoke --adapter-lambdas 0.5`,
    or as a Slurm job on an idle `bigbatch` node:
    `sbatch --nodelist=<node> scripts/fk_steering/fk_steering.sbatch adapter --smoke --adapter-lambdas 0.5`
  - **Done when:** the run dir holds `adapter.json` with 420 loaded tensors and checkpoint step
    30050, `adapter_eta0_lam0.5.png`, `ctrl_K2_lam0.5/` and `fk_K2_lam0.5/`, a one-row sheet
    with an `IR` score on every tile, `verdict.json`.
  - Ran 2026-09-06: Slurm job 50312 on mscluster52 (every listed output present); three idle
    `bigbatch` nodes turned out to have faulted GPUs, recorded in the review's Runs table.

▶ **Next: [task 2.1](#2--the-full-run)**.

### 2. 🏋️ The full run

◀ **Needs: [task 1.3](#1--pin-the-bars-and-build-the-adapter-path)**, the smoke clean.

- [x] **2.1** Launch the full mode with `nohup` on a free device: cat × dog at λ 0.5 and 1.2,
      butterfly × meadow at 1.2, seeds 9 to 16, K 16, 50 steps, 1024², W&B online.
  - Command: `GPU=<idx> nohup bash scripts/fk_steering/run_fk_steering.sh adapter > /datasets/mmolefe/poe_repair_min/outputs/interaction_term/fk_steering/logs/adapter_full.log 2>&1 &`,
    or `sbatch --nodelist=<idle bigbatch node> scripts/fk_steering/fk_steering.sbatch adapter` when
    no `biggpu` device is free (the route taken on 2026-09-06: job 50332 on mscluster52, after
    probe jobs found mscluster44, 45 and 65 faulted).
  - Launched 2026-09-06 03:24 as Slurm job 50332 on mscluster52, W&B `dfrceppl`, run dir
    `fka_K16_s50_20260906-025826`; in flight.
  - 💡 `/run-experiment` ✅ for the launch and the harvest.
  - **Done when:** the W&B run id, the run dir, the node, the device and the PID are in the review
    file's Runs table.
- [x] **2.2** Harvest: copy the three sheets and `summary.json` into
      `artifacts/results/is-the-gap-the-samplers-or-the-models/` with README entries, quote
      `verdict.json` in the review file, answer every pre-registered question there, and add the
      result to plan 10's finding in `report/` as its own section.
  - **Done when:** the review file's verdict is written and the finding carries the per-λ table.

▶ **Next: [instruction 3.1](#3--read-the-sheets)**.

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted.

- [ ] **Capture the failures this plan hit**, while they are still fresh.
  - Paste: `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/13-feynman-kac-steering-on-top-of-the-rank-32-correction.md`
  - Done when: each failure has a catalog entry, or there were none to record.
- [ ] **Bring the tree current** with what actually happened.
  - Paste: `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/13-feynman-kac-steering-on-top-of-the-rank-32-correction.md — <one line>`
  - Done when: statuses, the running order and the Error Matrix match reality.

▶ **Next: [the check before moving on](#the-check-before-moving-on).**

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-check-before-moving-on) ➡️

**For you to follow manually.** Do these yourself, interleaved with the Tasks rather than after
them.

### 3. 👁️ Read the sheets

◀ **Needs: [task 2.2](#2--the-full-run)**, the sheets filed.

- [ ] **3.1** Open `artifacts/results/is-the-gap-the-samplers-or-the-models/fk-steering-on-adapter-cat-dog-lambda-0.5-sheet.png`.
      Row by row: does the adapter-alone tile show one animal or two, does the control tile
      agree with it, and does the FK tile show two bodies where both show one? Count the rows
      where FK adds a second body; that count over 8 is the number the verdict was judged on.
- [ ] **3.2** Open the λ 1.2 sheet the same way. Expected: the adapter-alone column already shows
      two animals on most rows, and the FK column matches it. A row where FK loses the second
      animal is the method taking something away; note it in the review's caveats.
- [ ] **3.3** Open the butterfly × meadow sheet. Expected: a clear butterfly over a meadow in every
      column. The green counts there mean several butterflies, not composition; check that no FK
      tile lost the butterfly or the meadow.

▶ **Next: [the close out](#close-out--record-what-this-plan-taught)**.

## The check before moving on

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **Why this checkpoint matters:** it is the one place the paper can say what an inference-time
> method adds on top of the adapter, with the headroom stated beside it.

```bash
OUT=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/fk_steering
ls -d "$OUT"/fka_*                                 # the adapter runs
cat "$OUT"/<run>/adapter.json                      # 420 tensors, checkpoint step 30050
ls "$OUT"/<run>/*.png                              # three sheets
cat "$OUT"/<run>/verdict.json
```

**Pass criteria**

- `adapter.json` reports 420 loaded tensors and checkpoint step 30050.
- Three sheets with no red "missing" boxes, three sidecars.
- `verdict.json` quoted in the review file, with the adapter-alone rate beside the FK rate at each λ.

**Fail criteria (STOP)**

- The adapter-alone column at λ 1.2 composes fewer than 5 of 8 cat × dog seeds: the adapter did
  not attach the way the showcase grids had it (rank, alpha or checkpoint wrong).
- `torch.cuda.is_available()` printed `False`, or minutes per step.

**Partial pass guidance**

- A run stopped early still has its per-cell JSON; draw the sheets with red boxes and read what
  exists, marking the verdict as read at that seed count.

**When you get results, answer**
[the review file](../../review/13-feynman-kac-steering-on-top-of-the-rank-32-correction.md).

## Figure Catalog

⬅️ [Previous](#the-check-before-moving-on) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

The standard every figure in this scope is held to is
[in the scope's MASTER_PLAN](../../MASTER_PLAN.md#the-figure-bar-every-plan-here-is-held-to).

### Pending: to be generated from prompts

None. This scope carries no `diagram-prompts.md`.

### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|---|---|---|---|---|---|
| `<run>/fk_steering_<pair>_lam<λ>_sheet.png` | — | rows seeds 9 to 16; columns Mono (`eta` 0), plain PoE (`eta` 0), adapter alone at λ (`eta` 0), control particle 0 (K 16, `eta` 1, adapter at λ), best of 16 unweighted, FK on the adapter at K 16; the detector's count on every tile | task 2.1 | ⏳ | one per (pair, λ), logged to W&B as `sheets/<pair>_lam<λ>` |
| `<run>/fk_steering_<pair>_lam<λ>_sheet.json` | — | per tile: source path, count, verdict; per column: compose rate; the adapter's attach report | task 2.1 | ⏳ | sidecar |
| `<run>/summary.json`, `<run>/verdict.json` | — | compose rates per λ, the adapter-alone rate, best of 16, agreement per read step, the verdict at λ 0.5 with its bars | task 2.1 | ⏳ | quoted in the review file |
| `<run>/<pair>/seed_N/fk_K16_lam<λ>/xhat/step_SS/p{k}.png` | — | every x0-hat the reward read | task 2.1 | ⏳ | **Supplementary** |

### Organization workflow

1. Everything under the run dir on `/datasets`.
2. The three sheets and `summary.json` promoted to
   `artifacts/results/is-the-gap-the-samplers-or-the-models/` as
   `fk-steering-on-adapter-cat-dog-lambda-0.5-sheet.png`,
   `fk-steering-on-adapter-cat-dog-lambda-1.2-sheet.png`,
   `fk-steering-on-adapter-butterfly-meadow-lambda-1.2-sheet.png` and
   `fk-steering-on-adapter-summary.json`, with README entries.

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| What changes | Where it has to be reflected |
|---|---|
| a run launches or finishes | the review file's Runs table with its W&B id, node, device and PID |
| `verdict.json` is written | the review file's pre-registered question, and plan 10's finding in `report/` |
| the plan's status | the scope [MASTER_PLAN.md](../../MASTER_PLAN.md) and the root running order, by `sync-plan-tree` |

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/13-feynman-kac-steering-on-top-of-the-rank-32-correction.md` | **task 0.1** | Conformance reported |
| Capture patterns | `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/13-feynman-kac-steering-on-top-of-the-rank-32-correction.md` | **the close out** | Errors added to catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | Auto | Error Matrix regenerated |
| Bring the tree current | `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/13-feynman-kac-steering-on-top-of-the-rank-32-correction.md` | **the close out** | Statuses and running order match reality |

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Path | Why it is read |
|---|---|
| [poe_repair/experiments/fk_steering/steer.py](../../../../poe_repair/experiments/fk_steering/steer.py) | the mixed forward, the constants |
| [poe_repair/experiments/fk_steering/run.py](../../../../poe_repair/experiments/fk_steering/run.py) | the adapter attach, the per-λ sheets, the verdict |
| [poe_repair/methods/_sampling.py](../../../../poe_repair/methods/_sampling.py) | `run_lora_residual_inject`, the composition this plan copies |
| [poe_repair/experiments/one_pair_one_seed/trainer.py](../../../../poe_repair/experiments/one_pair_one_seed/trainer.py) | `attach_lora` and `load_lora_state` |

## Next step

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

A support puts a sentence in the paper's corrector section: selection on top of a half-dose
correction reaches what the full dose reaches. A null says the correction does the work and
selection adds nothing at any dose, which is the simpler story and goes to the same section in one
sentence. Either way the λ 1.2 row reads with its headroom stated.

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

<details>
<summary>3 catalogued failures and their fixes</summary>

Auto-updated after runs via `/ingest-error-pattern` and `/sync-plan-tree`.

### From global catalog

(Patterns applicable across all projects.) None yet.

### From project catalog

#### 🔴 the adapter left enabled after a corrected run

**When it happens:** a sampler returns with the adapter on and the next reference render uses it.
**What you see:** Mono or plain PoE references that differ from the showcase renders by more than
a few grey levels.
**Why:** `enable_adapters` is sticky on the UNet.
**How to fix:** the mixed forward disables the adapter after its second pass; the attach helper
leaves it disabled; check the reference columns against the showcase grids by eye.

#### 🟡 a listed card that runs on the CPU

**When it happens:** the pinned device is in the fault state of `poe-launch-002`.
**What you see:** `torch.cuda.is_available()` is `False` while `nvidia-smi` lists the card.
**Why:** the driver reports the card but torch cannot open it.
**How to fix:** the launcher checks torch before loading models and aborts; pick another device.

#### 🟡 the co3 environment on a Blackwell node

**When it happens:** the launcher run on mscluster110 to 112 with `co3`.
**What you see:** no CUDA output at all rather than an error.
**Why:** `co3`'s torch has no `sm_120` kernels.
**How to fix:** the launcher picks `co3_bw` by hostname; keep that switch.

---

**Auto-update note:** regenerated by `/sync-plan-tree` after new errors are added to the catalogs.
Do not edit manually.

</details>
