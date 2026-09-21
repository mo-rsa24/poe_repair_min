# Improve the Rank-32 Pooled Adapter

> **This scope's plans sit in the one running order, not in a sequence of their own.** They take
> step numbers in the `## Running order` table of the
> [repo root MASTER_PLAN.md](../../MASTER_PLAN.md), interleaved with every other scope's plans.
> The `## Plans` list below is this folder's contents and its dependencies, never a list to work
> down top to bottom.

## Table of contents

- [The overall claim](#the-overall-claim)
- [What is this plan](#what-is-this-plan)
- [Why this plan exists](#why-this-plan-exists)
- [What this scope actually does (visual)](#what-this-scope-actually-does-visual)
- [High-level overview](#high-level-overview)
- [Purpose and goals](#purpose-and-goals)
- [Mission](#mission)
- [Objectives](#objectives)
- [Goals](#goals)
- [Expected Outcome](#expected-outcome)
- [Definition of Done](#definition-of-done)
- [Not in this scope](#not-in-this-scope)
- [Sub-Scopes](#sub-scopes)
- [Plans](#plans)
- [Environment Context](#environment-context)
- [Learning Coverage](#learning-coverage)
- [Diagram Prompts](#diagram-prompts)
- [Process Diagram](#process-diagram)

## The overall claim

📋 [TOC](#table-of-contents) | [Next](#what-is-this-plan) ➡️

**A rank-32 adapter trained on cleaner cells, the early steps only, and the part of the correction
the experts cannot supply, rendered with the frozen model finishing the run, gives more clean
cat-and-dog seeds than checkpoint 30050 and loses none it had.**

## What is this plan

⬅️ [Previous](#the-overall-claim) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

Checkpoint 30050 composes on most held-out cat x dog seeds but not cleanly: blended features, odd
limbs, an animal that is neither. This scope trains one parent run with five changes on at once and
four children that each switch one change back, renders all of them and 30050 under two samplers,
and judges every strip by eye, blind. It ends with the best checkpoint named and one comparison
sheet against 30050.

## Why this plan exists

⬅️ [Previous](#what-is-this-plan) | 📋 [TOC](#table-of-contents) | [Next](#what-this-scope-actually-does-visual) ➡️

Six of eight held-out cat x dog seeds from 30050 are in the problem table of
[the run design](../../artifacts/ideas/improving-the-pooled-lora-run/run-design-parent-and-children.md),
and the count scorer calls most of them composed, so no number in the tree currently sees the
defect. Earlier single-axis runs (weight decay, EMA, the clean-estimate loss, the energy price,
three inference-time fixes) each moved one thing and none removed the softness. The design puts the
five remaining data-side and loss-side changes on together, then removes them one at a time, so
each child says whether its switch mattered. The read is by eye because the scorer is blind to the
defect, and it is written before launch so no outcome is negotiable afterwards.

The run design file is this scope's spine: the problem catalogue, the run set, the read, the
pre-launch steps and the launch line all live there, and every plan quotes it rather than
re-deriving it. The walk behind it, `IDEA_MAP.md` beside that file, is background and never plan
content.

## What this scope actually does (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#high-level-overview) ➡️

```
 ┌───────────┐  sieve   ┌───────────────┐         ┌──────────────────────────────┐
 │ cache     │──▶ ⧗ ──▶ │ TRAINER       │         │ TWO SAMPLERS                 │
 │ 14 pairs  │ dropped  │  P: 5 changes │  ckpt   │ shipped: adapter all 50 steps│
 │ 108 cells │ cells ○  │  C1..C4: one  │────────▶│ stacked: adapter to step 24, │
 └───────────┘          │  switched off │         │   frozen model finishes,     │
                        └───────────────┘         │   guidance [5,35), eta 1     │
   B = 30050 ───────────────────────────────────▶ └───────────────┬──────────────┘
   (no training)                                                  ▼
                                            ┌──────────────────────────────────┐
                                            │ STRIP per seed (17 cat x dog,    │
                                            │ 8 elephant x penguin)            │
                                            │ [joint][plain PoE][30050][run]   │
                                            │ names hidden, order shuffled     │
                                            └───────────────┬──────────────────┘
        ┌────────────────┐                                  ▼
        │ scorer:        │  orders only     ┌──────────────────────────────┐
        │ count, both-   │─────────────────▶│ BLIND READ, by eye, per seed │
        │ names          │                  │ clean · unclear · not two    │
        └────────────────┘                  └───────────────┬──────────────┘
                                                            ▼
                                   ┌───────────────────────────────────────────┐
                                   │ verdicts: P against B, each child against │
                                   │ P; THE SHEET (best run, both samplers)    │
                                   │ filed under artifacts/results/ with card  │
                                   └───────────────────────────────────────────┘
```

Data flows left to right and down. The blind read is the only place a person appears. The scorer
enters from the side on an arrow that orders the strips and never decides them. The baseline
reaches the samplers without passing through the trainer, which is what makes the stacked
sampler's own contribution readable.

## High-level overview

⬅️ [Previous](#what-this-scope-actually-does-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goals) ➡️

Five parts.

**The cell pool.** 14 pairs from the cache: the 11 train pairs plus lion x horse, wolf x horse and
bear x salmon, which are already cached and contain neither cat nor dog, adding 20 cells. Train
cells whose joint target fails the both-names read are dropped, from
`/datasets/mmolefe/poe_repair_min/outputs/showcase/target_quality/target_quality.json`.

**The trainer.** Three new switches in
`poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py`: `--exclude-cells`,
`--train-step-range`, `--orth-weight`. They sit beside the existing `--weight-decay`,
`--ema-decay` and `--gradient-checkpointing`, which already exist.

**The two samplers.** Shipped is the adapter on all 50 steps, deterministic DDIM at eta 0,
guidance 7.5 throughout. Stacked is the adapter on steps 0 to 24 with the frozen model finishing,
stochastic DDIM at eta 1, guidance 7.5 on steps 5 to 35 and 1.0 outside. The windowed sampler in
`poe_repair/methods/_poe_langevin.py` carries `lambda_window` and eta already; the guidance
interval is new.

**The chain.** P, then C2, C1, C3, C4 on mscluster112, one launcher on `/datasets`, one W&B run
each, roughly twelve hours overnight on one device.

**The read.** Per-seed strips under both samplers, three blind labels by eye, the count and
both-names reads logged beside them for ordering only, the correction size per pair reported as a
measurement.

## Purpose and goals

⬅️ [Previous](#high-level-overview) | 📋 [TOC](#table-of-contents) | [Next](#mission) ➡️

**Purpose.** Scope 01 ships the adapter and measures the paper's figures from its output. This
scope replaces the checkpoint those figures are measured from, if a run beats 30050 by the read
fixed below.

**Goals.** The numbered list is in [Goals](#goals).

## Mission

⬅️ [Previous](#purpose-and-goals) | 📋 [TOC](#table-of-contents) | [Next](#objectives) ➡️

Produce a rank-32 checkpoint that gives more clean cat-and-dog seeds than 30050 under the same
sampler without losing any, say which of the five changes did it, and file one sheet that shows it.

## Objectives

⬅️ [Previous](#mission) | 📋 [TOC](#table-of-contents) | [Next](#goals) ➡️

1. Put the five changes on together in one parent run and remove them one at a time in four
   children, so any gap between a child and the parent is one axis and nothing else.
2. Judge every render by eye, blind, against a three-label read fixed before launch, with the
   scorer's numbers demoted to ordering.
3. Separate the sampler's contribution from training's: 30050 under the stacked sampler is read
   first, and every training run is read against it.
4. Report the correction size per pair as a measurement beside the labels, and let it claim
   nothing until labelled examples are laid against it.

## Goals

⬅️ [Previous](#objectives) | 📋 [TOC](#table-of-contents) | [Next](#expected-outcome) ➡️

1. Three trainer flags and the guidance interval exist, parse, and a dry run of the parent config
   prints its cell count (88 plus 20, minus the excluded cells).
2. The exclusion list exists with its count printed, and the 14-pair pool yaml names only cells
   present in the cache.
3. 30050 under the stacked sampler is rendered on the 17 cat x dog seeds and read before any
   training run starts.
4. A 200-step smoke of P on mscluster112 shows the W&B run with its panels, a checkpoint on
   `/datasets`, and a step time near 0.24 s.
5. Five runs (P, C2, C1, C3, C4) reach 30k steps, each with its own W&B run id.
6. Every run and the baseline has its strips under both samplers on W&B, over 17 cat x dog and 8
   elephant x penguin seeds: joint target, plain PoE, 30050, this run.
7. A blind-label table exists per run and per sampler, with the count and both-names reads beside
   it.
8. One verdict per child against P, and one for P against the baseline, each by the fixed rule.
9. The sheet is filed under `artifacts/results/` with its card.

## Expected Outcome

⬅️ [Previous](#goals) | 📋 [TOC](#table-of-contents) | [Next](#definition-of-done) ➡️

Under the stacked sampler at least one run has more clean cat x dog seeds than 30050 and loses
none, the children say which switches carried it, and elephant x penguin holds. The scope may
instead end with 30050 under the stacked sampler already better than 30050 shipped, which is
reported before any training run is read, and the training runs are then read against it.

What would surprise: C2 better than P. That says training on the late steps helps even though
running the adapter there hurts.

## Definition of Done

⬅️ [Previous](#expected-outcome) | 📋 [TOC](#table-of-contents) | [Next](#not-in-this-scope) ➡️

1. ⚠️ `--exclude-cells`, `--train-step-range` and `--orth-weight` in `train_pooled.py`, the
   guidance interval in the windowed sampler, and the 14-pair pool yaml under
   `artifacts/_shared/cross_pair_pool_configs/`, each smoke-tested [inferred from the run design's
   pre-launch steps]
2. ⚠️ `exclude_cells.json` written from `target_quality.json` with its count printed beside it
   [inferred]
3. The baseline under the stacked sampler is rendered and read before any training run starts
4. The chain has run: five runs at 30k steps on mscluster112, W&B run ids recorded
5. Every run has its strips under both samplers on W&B, over 17 cat x dog and 8 elephant x penguin
   seeds
6. The blind-label table exists per run and per sampler, with the shuffled order and the hidden
   names recorded so the blinding can be checked afterwards
7. ⚠️ The labels are read off the W&B strips in the browser and confirmed or vetoed per seed, the
   veto recorded beside the label [inferred, owner: human veto-after]
8. The verdict per child and for P against the baseline is written by the fixed read, in each
   plan's review file
9. The correction size per pair is reported beside the labels, with clean, unclear and not-two
   examples laid against it
10. The sheet (joint target, plain PoE, 30050, best run, under both samplers, over the 17 cat x dog
    and 8 elephant x penguin seeds) is filed under `artifacts/results/` with its card, and the
    register slot in `paper/iclr/figures.md` it serves is named
11. The scope has a recall gallery: run
    `/recap-plan-tree @plans/08-improve-the-rank-32-pooled-adapter/MASTER_PLAN.md` once every plan
    above is ✅, and record the Artifact URL it publishes

## Not in this scope

⬅️ [Previous](#definition-of-done) | 📋 [TOC](#table-of-contents) | [Next](#sub-scopes) ➡️

A better teacher than the joint prompt, which needs a new cache and a validation that the guided
sampler composes where the plain one does not. An automatic crispness score. The mechanism behind
the late softening.

## Sub-Scopes

⬅️ [Previous](#not-in-this-scope) | 📋 [TOC](#table-of-contents) | [Next](#plans) ➡️

None.

## Plans

⬅️ [Previous](#sub-scopes) | 📋 [TOC](#table-of-contents) | [Next](#environment-context) ➡️

Five plans, in the order they get done. The step numbers are their places in the
[root running order](../../MASTER_PLAN.md), which is the one order across every scope.

| Step | Plan | What it does | Status | Waits on |
|---|---|---|---|---|
| 66 | [01: the three switches and the guidance interval](plans/tools/01-the-three-switches-and-the-guidance-interval.md) | the two trainer flags and the one loss switch, the sampler's guidance interval, the 14-pair pool and the exclusion list, each proved to select something | ⚠️ | |
| 67 | [02: the blind label pass](plans/tools/02-the-blind-label-pass.md) | the read: strips, hidden names, three labels, and the two automatic reads that order but never judge | ⚠️ | 66 |
| 68 | [03: what the stacked sampler alone does](plans/baselines/03-what-the-stacked-sampler-alone-does.md) | checkpoint 30050 under both samplers on 25 seeds, judged blind, fixing which column the training runs are read against | ⚠️ | 66, 67 |
| 69 | [04: the parent and the four children](plans/hypothesis/04-the-parent-and-the-four-children.md) | five trainings in a chain overnight, one axis apart, judged blind under both samplers | ⚠️ | 66, 67 to launch; 68 to read |
| 70 | [05: the comparison sheet](plans/figures/05-the-comparison-sheet.md) | the two sheets filed under `artifacts/results/` with their card, and the correction's size per pair beside them | ⚠️ | 68, 69 |

Each plan is paired with a review file of the same number under `review/`, except plan 05, which
draws settled results and has no question of its own. Two tasks are sequences rather than single
units and have their own procedure files under `procedures/`: launching the chain, and taking the
blind labels.

## Environment Context

⬅️ [Previous](#plans) | 📋 [TOC](#table-of-contents) | [Next](#learning-coverage) ➡️

Start at [the environment index](../../environment/00-INDEX.md). This scope depends on:

- [the node facts](../../environment/hpc/nodes.md): one free Blackwell, mscluster112, verified
  2026-09-08; mscluster110 is carrying another user's job and mscluster111 is hardware-faulted.
  `biggpu` allows one job per user.
- [the execution protocol](../../environment/hpc/execution-protocol.md): shared-device launch over
  SSH with `nohup`, every path on the launch line absolute, the launcher copied under the run's
  output root on `/datasets` because `/tmp` is node-local, a device guard refusing a GPU over 1 GB
  in use or reading `[N/A]`, and harvest over SSH on 112 rather than from the session node.
- [the throughput facts](../../environment/hpc/throughput.md): rank 32 at about 0.24 s per step on
  the Blackwell, about 2 hours per 30k run, five in a chain overnight.
- [the storage facts](../../environment/storage.md): large artifacts to `/datasets` only, and a
  disk guard that checks the filesystem the job actually writes to.

`co3_bw` is the python on the Blackwell, never `co3`: a `co3` CUDA operation there produces no
output rather than an error.

The tracker is W&B, project `prime_lab/poe-repair-animals-compose`. The trainer already logs
`eval/compose_rate`, the tracking thumbnails and `train/lora_weight_norm`. A probe pass per
checkpoint adds the stacked strips, `eval/both_names` and the blind-label table into the same run
by id.

## Learning Coverage

⬅️ [Previous](#environment-context) | 📋 [TOC](#table-of-contents) | [Next](#diagram-prompts) ➡️

Seeded at scope birth and refreshed by `learning-pulse`, never by hand afterwards.

The stacked sampler's stochastic tail leans on what
`deep-learning/diffusion-models/sampler-correctors-for-composition` teaches about why a corrector
step changes what a frozen model finishes. The span projection behind `--orth-weight` reuses the
direction machinery `deep-learning/diffusion-models/spectral-structure-of-the-correction` builds.
No concept this scope names is uncovered by the learning tree.

## Diagram Prompts

⬅️ [Previous](#learning-coverage) | 📋 [TOC](#table-of-contents) | [Next](#process-diagram) ➡️

See [this scope's illustrated map](diagram-prompts.md): the system as connected image prompts,
subject and process lanes. Process history lives in `diagrams/process-versions/`.

## Process Diagram

⬅️ [Previous](#diagram-prompts) | 📋 [TOC](#table-of-contents)

Not authored yet. There are no plans to draw; `populate-plans` writes the process lane and its
first snapshot.
