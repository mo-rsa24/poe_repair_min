# Designing the Correction Loss

> **This scope's plans sit in the one running order, not in a sequence of their own.** They take
> step numbers in the `## Running order` table of the
> [repo root MASTER_PLAN.md](../../MASTER_PLAN.md), interleaved with every other scope's plans.
> The `## Plans` list below is this folder's contents and its dependencies, never a list to work
> down top to bottom.

## Do this next

Three experiments, in this order. Each is run with `drip-execute-plan`, which walks the plan one
step at a time and writes each tick back as it happens.

**First, the free check that can stop experiment 3 before it starts.** How much of the true
correction the adapter actually applies, per denoising step, on the V1 checkpoint. The tensors are
already in the cache.

```
/drip-execute-plan @plans/09-designing-the-correction-loss/plans/hypothesis/06-fitting-the-whole-path-and-handing-back-the-tail.md --only 1
```

**Then V0a and V4 together**, on the two GPUs of `mscluster109`. About 7 hours.

```
/drip-execute-plan @plans/09-designing-the-correction-loss/plans/hypothesis/04-freezing-the-empty-branch.md --from 2
```

**Then the hand-off sweep while those train**, on `mscluster106`, no training. About an hour.

```
/drip-execute-plan @plans/09-designing-the-correction-loss/plans/hypothesis/06-fitting-the-whole-path-and-handing-back-the-tail.md --from 2
```

## Table of contents

- [The overall claim](#the-overall-claim)
- [What is this plan](#what-is-this-plan)
- [Why this plan exists](#why-this-plan-exists)
- [What this scope actually does (visual)](#what-this-scope-actually-does-visual)
- [The nine variations](#the-nine-variations)
- [The pool, the rank and the card every run in this scope uses](#the-pool-the-rank-and-the-card-every-run-in-this-scope-uses)
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

**Freezing the empty branch reaches the same composed predictions as the loss that runs today,
while closing exactly the direction that loss cannot see, so one adapter works at every guidance
setting and training costs a third less.**

## What is this plan

⬅️ [Previous](#the-overall-claim) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

The adapter's loss compares the composed prediction against the model's own joint-prompt
prediction. Both halves of that, what is compared and how the three branches are combined, are
design choices. Nine of them are written out as equations. This scope runs them in order, files
each one under the same number and name in every pillar, and ends with one named as the objective
to train on.

## Why this plan exists

⬅️ [Previous](#what-is-this-plan) | 📋 [TOC](#table-of-contents) | [Next](#what-this-scope-actually-does-visual) ➡️

The loss only ever sees the sum of the three branches. Add the same tensor to the cat branch and to
the empty branch and the loss does not move at all, while sampling reads the empty branch a second
time at `-(w-1) = -6.5` and the picture moves by six and a half times that tensor.

So there is a direction training is blind to and the sampler amplifies. Two objectives have already
been trained without anyone measuring whether that direction was used.

The scope's spine is
[the variations note](../../artifacts/ideas/designing-the-correction-loss/maths/objectives-and-variations.tex),
which writes every objective out in full with its code change, what to expect and its tradeoff.
Every plan here quotes it rather than re-deriving it. The planning walk beside it,
`PLAN_WALK.md`, carries the names, the verb set and the ledger behind every decision in this file;
it is background and never plan content.

## What this scope actually does (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#the-nine-variations) ➡️

```
        ┌─────────────────────── WRITTEN ONCE, FROZEN ───────────────────────┐
        │  the cache            x_t   eps_a   eps_b   eps_uncond   eps_J     │
        │  one entry per (pair, seed, denoising step)                        │
        └──────────┬────────────────────────────────────┬───────────────────┘
                   │ x_t, K entries                     │ eps_J, eps_uncond
                   ▼                                    ▼
        ┌────────────────────────────┐        ┌──────────────────────┐
        │  ONE FORWARD PASS          │        │  THE TARGET          │
        │  3K wide, adapter attached │        │  guidance at w=7.5   │
        │  210 LoRA sites on attn2   │        │  on cached tensors   │
        │      ↓      ↓      ↓       │        └──────────┬───────────┘
        │      a      b      u       │                   │
        └──────┬──────┬──────┬───────┘                   │
               └──────┴──────┘                           │
                      │ the composition                  │
                      ▼                                  ▼
              ┌───────────────────────────────────────────────┐
              │  THE COMPARISON          subtract, square,    │
              │                          average → one number │
              └───────────────────┬───────────────────────────┘
                                  │
        ┌─────────────────────────┴──────────────────────────┐
        │  THE SWITCH: where the nine objectives differ      │
        │                                                    │
        │  which branches are adapted    ·  01, 03           │
        │  what the target is            ·  05, 06           │
        │  the guidance weight           ·  04               │
        │  an extra penalty, and on what ·  00b, 02, 07      │
        └─────────────────────────┬──────────────────────────┘
                                  │
                                  ▼
        ┌────────────────────────────────────────────────────┐
        │  WHAT COMES OUT, every 2,500 steps                  │
        │  a checkpoint  +  a strip: [target][PoE][adapter]   │
        │              ↓                                      │
        │  one page per variation, same shape every time      │
        └────────────────────────────────────────────────────┘
```

<a href="diagrams/corrloss-capstone-cache-to-one-page-per-objective.png"><img src="diagrams/corrloss-capstone-cache-to-one-page-per-objective.png" width="800" alt="A case of five coloured cubes labelled 'written once, frozen'. Three of them feed one U-Net labelled 'one U-Net, one pass', with 'adapter here, 210 sites' bracketed under its middle. Its three outputs, a red, a green and a purple cube, go into two boxes each headed 'guidance, w = 7.5', one subtracting and one summing, which meet at a minus sign and a box reading 'one number'. Top right, 'same machine, four settings' holds four panels wired by coloured leads to a three-by-three grid headed 'nine objectives, two built', with two green 'built' tiles and seven purple 'planned' ones. Along the bottom, 'every 2,500 steps' runs a row of paired checkpoint-and-strip icons into a stack of pages headed 'one page per objective'."></a>

*The same machinery as the block above: the frozen cache, the single pass with the adapter at 210
sites, the two guidance legs meeting at one number, the four-setting switch over nine objectives,
and the checkpoint strip ending in one page per objective.*

Everything above the switch is machinery shared by all nine objectives. Everything below it is the
filing. The switch is the scope.

The same machinery is walkable in [the correction-loss scene](https://claude.ai/code/artifact/ae96d471-01cd-4764-b79d-41da5053230c),
which carries what this diagram does not: the two legs drawn with every symbol bound to its box both
ways, the frozen and trainable modules coloured apart on the real counts (210 modules, 9,912,320 of
2,577,376,004 parameters), and all nine variations placed side by side with any two pickable for
comparison.

## The nine variations

⬅️ [Previous](#what-this-scope-actually-does-visual) | 📋 [TOC](#table-of-contents) | [Next](#high-level-overview) ➡️

One table, and it is the only place the set is listed. Equations, code changes and tradeoffs are in
[the variations note](../../artifacts/ideas/designing-the-correction-loss/maths/objectives-and-variations.tex).

| # | Name | What changes | State |
|---|---|---|---|
| 00 | matching-the-composition-to-the-joint-prompt | nothing, this is what runs today | ran: five runs at rank 8, 16 and 32, on 88 cells over 11 look-alike animal pairs |
| 00a | matching-the-composition-to-the-joint-prompt, on the scope pool | nothing; `00`'s objective moved onto `cells_v57` | to run: rank 16 at steps 0 to 25, the baseline every later variation is read against. No run of it exists |
| 00b | penalising-the-empty-branch-for-moving | `+ μ‖u − ū‖²`, μ = 10 | ran at rank 16 on `cells_v54`, 2026-09-12, unread; re-runs at rank 32 on the scope pool |
| 01 | freezing-the-empty-branch | `u → ū` everywhere | ran: rank 16 to 30,000 steps on `cells_v57`, W&B `kdzx03ji`, unread. A rank-32 attempt died at 7,500 |
| 02 | anchoring-the-empty-branch-to-two-animals | the penalty points at `ε̄("two animals")` | plan file owed, after 01 |
| 03 | adapting-only-the-empty-branch | bars on the concept branches | ran: rank 16 to 30,000 steps, W&B `qbvj0wsm`, unread. A rank-32 attempt died at 7,500 |
| 04 | training-at-guidance-weight-one | `w = 1` on both sides, on top of `01`'s frozen null | runs beside `01` at step 74 as the check that the cancellation is exact. Needs the `--no-dial` fix first: the branch at `trainer.py:532` never reads `freeze_null` |
| 05 | training-on-real-photographs | the target becomes real data | **dead**: the corpus is reachable and unusable |
| 06 | training-on-the-renders-that-composed | the target becomes filtered renders | **adopted**, queued last as [plan 05](plans/hypothesis/05-training-on-the-renders-that-composed.md) |
| 07 | keeping-single-concept-renders-unchanged | `+ ν` on the concept branches at single-prompt states | row only; the read it waits on is unblocked, scheduled once `01` returns |
| 08 | fitting-the-whole-path-and-handing-back-the-tail | `01` plus every denoising step in training and the loss scored in x0 space; the tail handed to frozen SDXL at render | plan owed as [plan 06](plans/hypothesis/06-fitting-the-whole-path-and-handing-back-the-tail.md), step 76. Aimed at fidelity, not composition |

### The pool, the rank and the card every run in this scope uses

**The pool is `cells_v57.json`**, 43 cells over 29 pairs, at
`artifacts/_shared/cross_pair_pool_configs/`. Roughly one seed per pair, which buys pairs at the
cost of seeds per pair, and that is the right trade for a claim about composing two concepts
rather than two animals. It is the only pool this scope trains on. `00`'s five runs used 88 cells
over 11 look-alike [animal pairs](../../context/world/animal-pair.md#what-an-animal-pair-is) and
are read as history, never as a comparison.

**Eight of its 43 cells need [task 1.5](plans/tools/01-the-three-instrument-fixes.md) first.**
`a_typewriter__x__a_cactus` seeds 1 to 8 sit in both the `train/` and `heldout/` cache
directories, and the cell lookup takes the held-out copy when no split is named. The two copies
are byte-identical so nothing is contaminated in substance, but until the trainer names its split
a run records 8 training cells as held-out. Full entry:
[the held-out-first cache lookup](../../environment/known-failures.md#entry-id-poe-data-001).

**Rank 16 carries the comparison between variations, at steps 0 to 25.** That is the lineage on
disk and the only one where runs finish. `01`, `03`, `06` and `06a` all reached 30,000 steps, while
three rank-32 attempts died within five minutes of each other on 2026-09-17 with no diagnosis.

**Rank 32 is the confirmation run for whichever variation wins**, and it is owed that diagnosis
first. At 49 GB it runs without gradient checkpointing, which a 24 GB 3090 cannot do.

**One device model per comparison.** `mscluster106` and `mscluster108` carry Quadro RTX 8000,
`mscluster109` carries RTX A6000, both 49 GB. `mscluster110` to `112` are RTX PRO 6000 Blackwell,
need `co3_bw`, and read the same uncorrected cell differently, so a matched pair never straddles
them. Rank 32 measured about 2,750 steps an hour, so 30,000 steps is roughly 11 hours.

**What promotes a row to a plan.** `03` becomes a plan if `01` fails, since it is the other way of
splitting the same predictions. `04` is promoted already: it runs beside `01` rather than after it,
because the table's own claim that it equals `01` once `u` is frozen is exactly what it tests.

`05` is dead. Retrieval reaches every pair in the pool, so the structural objection was false, but
what comes back cannot be trained on: 7 of 16 accepted cat x dog images are genuine photographs of
both animals and 0 of 11 turtle x tortoise are, the rest being watermarked stock and merchandise.
The filter that would clean it at scale reports both species present on single-species images at
0.323 for turtle x tortoise against a 0.10 bar, and the pool is look-alike species throughout. The
verdict is filed as
[can real photographs train the composition](../../report/designing-the-correction-loss/05-training-on-real-photographs.md).

`06`'s rule is met for the cells a person picked by eye. The scope trains on `cells_v57.json`, 43
cells over 29 pairs; `cells_v56.json` holds a wider 72 over 30 and is a second experiment, not a
longer version of the first. The rule is not met for growing past either, because no automatic
filter works on look-alike pairs, so the corpus is fixed until someone selects more by hand, which
is why the sizing question gates it rather than the filter question.

`07` waits on a read of whether the concept branches move differently at pair states and at
single-concept states. Nothing blocks that read; it is scheduled once `01` returns, because `07` is
a penalty on top of whichever base objective `01` settles.

## High-level overview

⬅️ [Previous](#the-nine-variations) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goals) ➡️

Four parts.

**The measurement that comes first.** Log `‖(a + b − u) − ε̄_J‖²` next to the loss. One line, no
forward pass. It says whether the invisible direction was ever used, and everything after it is
argued on a derivation until it exists.

**The foundation.** `00` has five runs and 136 checkpoints on disk and no page that reads them.
Building that page fixes the shape every later variation copies: an equation, four panels of
thumbnails, a rank table, and links to the dense detail rather than a copy of it.

**The two that already ran.** Four matched experiments from 2026-09-12 sit unrecorded. Reading them
costs no training, and it settles `00b` and the two prompt-rewrite questions at once.

**The two that have not.** `01` then `02`, each 50,000 steps, judged against bars fixed before the
run. `01` carries the risk the note names: whatever is frozen in training must be frozen in
sampling.

## Purpose and goals

⬅️ [Previous](#high-level-overview) | 📋 [TOC](#table-of-contents) | [Next](#mission) ➡️

**Purpose.** Scope 08 improves the adapter by changing its hyperparameters and its training data.
This scope changes what the loss compares, which is the one axis scope 08 holds fixed.

**Goals.** The numbered list is in [Goals](#goals).

## Mission

⬅️ [Previous](#purpose-and-goals) | 📋 [TOC](#table-of-contents) | [Next](#objectives) ➡️

Find the correction-loss objective that closes the direction the fit loss cannot see, and file
every variation under one number and one name so a result stays findable years later.

## Objectives

⬅️ [Previous](#mission) | 📋 [TOC](#table-of-contents) | [Next](#goals) ➡️

1. Measure whether the invisible direction is actually used, before changing any objective on the
   strength of a derivation.
2. Give every variation one number and one name, the same in every pillar, so a result stays
   findable.
3. Read the two objectives that already ran rather than re-running them.
4. Test the rest in the order the note argues for, one axis at a time.
5. End with one objective named as the one to train on.

## Goals

⬅️ [Previous](#objectives) | 📋 [TOC](#table-of-contents) | [Next](#expected-outcome) ➡️

1. `‖(a + b − u) − ε̄_J‖²` is logged beside the loss, and its size relative to the fit loss is
   recorded for at least one existing run.
2. `00`'s page exists: three ranks tabulated, the comparison strip embedded, and the rank-ablation
   figure drawn against a seed-noise band that was actually computed.
3. `00b`'s page exists with a verdict against both bars, fixed before the read.
4. `01` trains to 50,000 steps and is read against `00` on the same pool and the same seeds.
5. `02` trains and is read against `01`.
6. One objective is named, with the number that says why.

## Expected Outcome

⬅️ [Previous](#goals) | 📋 [TOC](#table-of-contents) | [Next](#definition-of-done) ➡️

`01` reaches the same composed predictions as `00` while closing the invisible direction exactly
rather than approximately, so one adapter works at every guidance setting and training costs a
third less.

What would surprise: `01` composing worse than `00`. The note claims freeing the empty branch buys
no predictions, so anything it was doing the two concept branches can do instead. `01` is the
experiment that can falsify that, and a falsification is the more interesting result.

## Definition of Done

⬅️ [Previous](#expected-outcome) | 📋 [TOC](#table-of-contents) | [Next](#not-in-this-scope) ➡️

1. The undialled error is logged in the trainer, and its size against the fit loss is recorded for
   at least one existing run
2. ⚠️ The six pillar files exist: the shared context file, the rank note, `00`'s report page, the
   cold-start recipe, the memory-per-rank environment row, the artifact card [inferred from the
   pillars having no entry for this scope's terms]
3. `00`'s page carries the three ranks, the comparison strip, and the rank-ablation figure with its
   computed seed-noise band
4. `00b`'s page carries a verdict against both pre-registered bars
5. ⚠️ A script in `scripts/` regenerates all eleven of `00`'s figures from the collected numbers,
   because the script that drew the existing eight was kept in a session scratchpad and is not in
   the repo [inferred]
6. ⚠️ `config.json` records every flag that changes the objective, `train_step_range` included
   [inferred: it is absent today, so a saved config cannot identify its own objective]
7. The blind read of `01`'s strips is done and recorded, names hidden and order shuffled, with the
   shuffle kept so the blinding can be checked afterwards
8. ⚠️ `01`'s training curves are opened in the W&B browser and the flattening judged by eye, since
   the panels need a logged-in session no script has [inferred, owner: human]
9. One objective is named as the one to train on, and each row-only variation carries the sentence
   that would promote it
10. The scope has a recall gallery: run
    `/recap-plan-tree @plans/09-designing-the-correction-loss/MASTER_PLAN.md` once every plan above
    is ✅, and record the Artifact URL it publishes

## Not in this scope

⬅️ [Previous](#definition-of-done) | 📋 [TOC](#table-of-contents) | [Next](#sub-scopes) ➡️

The adapter's hyperparameters. Rank, weight decay, EMA, the schedule and which cells it trains on
belong to [scope 08](../08-improve-the-rank-32-pooled-adapter/MASTER_PLAN.md). This scope changes
only what the loss compares.

Inference-time fixes. Samplers, correctors and noise search belong to
[scope 06](../06-is-the-gap-the-samplers-or-the-models/MASTER_PLAN.md).

The corpus question behind `05`. Routed to `/drip-idea` on claim 6 of
[the idea map](../../artifacts/ideas/designing-the-correction-loss/IDEA_MAP.md); it returns here as
a verdict, not as work done here.

## Sub-Scopes

⬅️ [Previous](#not-in-this-scope) | 📋 [TOC](#table-of-contents) | [Next](#plans) ➡️

None.

## Plans

⬅️ [Previous](#sub-scopes) | 📋 [TOC](#table-of-contents) | [Next](#environment-context) ➡️

Six plans, in the order they get done. The step numbers are their places in the
[root running order](../../MASTER_PLAN.md), which is the one order across every scope.

| Step | Plan | What it does | Status | Waits on |
|---|---|---|---|---|
| 71 | [01: the three instrument fixes](plans/tools/01-the-three-instrument-fixes.md) | the undialled error logged beside the loss, a checkpoint written at every render, every objective flag in the run's config; each proved to do something by a two-epoch smoke | ⚠️ | |
| 72 | [02: the V0 foundation](plans/reading/02-the-v0-foundation.md) | six pillar files, the scattered per-checkpoint numbers collected, a chart script that can redraw all eleven figures, the seed-noise band computed and the rank-ablation figure drawn, and `00`'s page: the shape every later variation copies | ⚠️ | 71 |
| 73 | [03: the V0a read](plans/hypothesis/03-the-v0a-read.md) | the four rank-16 experiments from 2026-09-12 scored and read blind against both bars, `00b`'s page and the prompt-rewrite finding, then `00a` and `00b` launched at rank 32 on the scope pool | ⚠️ | 72 for the page shape |
| 74 | [04: freezing the empty branch](plans/hypothesis/04-freezing-the-empty-branch.md) | the check that training and sampling agree, the `--no-dial` fix, then `00a` and `04` launched at rank 16 on `mscluster109` and read against the `01` that already ran (W&B `kdzx03ji`) | ⚠️ | 71 for the number, 73 for the approximate fix it is read against |
| 75 | [05: training on the renders that composed](plans/hypothesis/05-training-on-the-renders-that-composed.md) | variation `06`: a second data path that encodes a render, noises it, and trains against the noise that was added; gated by a scaling curve over 10, 20, 30 and 43 cells | ⚠️ | 74, because it moves the target and the empty branch against `00` at once |
| 76 | [06: fitting the whole path and handing back the tail](plans/hypothesis/06-fitting-the-whole-path-and-handing-back-the-tail.md) | variation `08`, the only plan aimed at fidelity rather than composition: the shrinkage check, then the hand-off sweep on checkpoints that already exist, then one training run only if the sweep falls short | ⚠️ | nothing; its first two stages read existing checkpoints |

Each plan is paired with a review file of the same number under `review/`. No procedure files yet:
no task here is a sequence long enough to earn one.

**What is not planned, and why.** `02`, anchoring the empty branch to two animals, is the only
other variation the note says to plan soon, and its design depends on whether `01` closed the
direction. Plan 04's close-out task C.3 authors it. Until then the scope's definition of done
cannot be met, which is stated rather than hidden.

## Environment Context

⬅️ [Previous](#plans) | 📋 [TOC](#table-of-contents) | [Next](#learning-coverage) ➡️

Start at [the environment index](../../environment/00-INDEX.md). This scope depends on:

- [the node facts](../../environment/hpc/nodes.md): `biggpu` allows one job per user, so a set of
  four experiments at once goes to `bigbatch` and its RTX 3090s. A Blackwell node is the right
  choice for a single long follow-up and needs `co3_bw`, never `co3`, because a `co3` CUDA
  operation there produces no output rather than an error.
- [the throughput facts](../../environment/hpc/throughput.md): rank 16 at about 1.2 s per step on an
  RTX 8000 and about 11 to 13.5 hours for 30,000 steps on a 3090, so 50,000 steps is roughly 20
  hours. Memory per rank is the column this folder does not yet have, and this scope adds it.
- [the storage facts](../../environment/storage.md): checkpoints to `/datasets` only, and a disk
  guard that checks the filesystem the job actually writes to.
- [the overview](../../environment/overview.md): cached tensors are float16 and any analysis
  stacking many of them upcasts to float32 first, which is also why the trainer casts its forward
  pass to fp32 before the branches are combined.

The tracker is W&B, project `prime_lab/poe-repair-animals-compose`.

## Learning Coverage

⬅️ [Previous](#environment-context) | 📋 [TOC](#table-of-contents) | [Next](#diagram-prompts) ➡️

Seeded at scope birth and refreshed by `learning-pulse`, never by hand afterwards.

The LoRA-on-`attn2` machinery every variation shares is what
`deep-learning/diffusion-models/latent-diffusion-architecture-rebuilt` rebuilds from an empty folder
at toy scale. Why the product of two experts composes as it does, and why the empty branch is
divided out at all, is what `deep-learning/diffusion-models/01-poe-derivation-foundations` derives.
No concept this scope names is uncovered by the learning tree.

## Diagram Prompts

⬅️ [Previous](#learning-coverage) | 📋 [TOC](#table-of-contents) | [Next](#process-diagram) ➡️

See [this scope's illustrated map](diagram-prompts.md): the system as connected image prompts, both
lanes. Ten prompts, all ten rendered into `diagrams/`. Process history lives in
`diagrams/process-versions/`.

## Process Diagram

⬅️ [Previous](#diagram-prompts) | 📋 [TOC](#table-of-contents)

Version 01, written 2026-09-16 from the four plans above:
[`diagrams/process-versions/01-2026-09-16.md`](diagrams/process-versions/01-2026-09-16.md). Three
stages and a route map. Regenerated whole by each `sync-plan-tree` pass that finds the plan set
changed, never patched.
