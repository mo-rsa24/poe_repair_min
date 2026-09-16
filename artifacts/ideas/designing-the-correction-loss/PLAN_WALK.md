# 🗂️ Turning the loss variations into a plan tree

## Position in the walk

| Group | Units | Dispositioned | State |
|---|---|---|---|
| 1. the naming spine across the pillars | 6 | 6 | settled: parent `designing-the-correction-loss`, V0 → `00`, V0a → `00a` |
| 2. V0's evidence already on disk | 6 | 6 | settled: five gathered, the chart script rewritten to cover all eleven figures |
| 3. V0's write-up across the pillars | 6 | 6 | settled: authored by scope 09's tasks, the pulses audit afterwards |
| 4. the V0a design plan and its illustration | 4 | 4 | settled: a read plan, not a training plan. Both bars fixed, art direction fixed |
| 5. what gets run, and how results are kept over time | 5 | 5 | settled: 50,000 steps, a checkpoint at every render, every objective flag in the config |
| 6. V1 to V7, ordered | 7 | 7 | settled: three get plan files, four get a table row, 05 routed out |

Progress: 37 of 37 dispositioned. The walk is complete.

## Table of contents

- [Position in the walk](#position-in-the-walk)
- [Quick context: where you are](#quick-context-where-you-are)
- [What this walk is for](#what-this-walk-is-for)
- [The verb set](#the-verb-set)
- [The groups](#the-groups)
- [The disposition ledger](#the-disposition-ledger)
- [Held pieces](#held-pieces)
- [Surveys](#surveys)
- [Drafted task lines](#drafted-task-lines)
- [Illustrations owed](#illustrations-owed)
- [Reversals](#reversals)
- [The chain so far](#the-chain-so-far)
- [Next step](#next-step)

## Quick context: where you are

Navigation: ⬅️ [Position](#position-in-the-walk) | 📋 [TOC](#table-of-contents) | [Next](#what-this-walk-is-for) ➡️

**What is being sorted out**

The eight correction-loss objectives written out in
[the objectives and variations note](maths/objectives-and-variations.tex), turned into a numbered
set of plans and pillar entries, starting by giving the one that already ran (V0) a clean home.

**Where the walk is**

Group 5: what the cluster runs now, and at what cadence every variation keeps its results.

**What compile would emit today**

A usable chain for groups 1 to 4: mint scope 09, write the six pillar files, build the V0
foundation page, and read the four V0a experiments that already ran. Group 6, the ordering of V1
to V7, is still undispositioned.

## What this walk is for

Navigation: ⬅️ [Quick context](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-verb-set) ➡️

Eight objectives are written out as equations. One of them (V0) has already been trained at three
LoRA ranks and its results are scattered across review files, results folders and W&B. One (V0a)
has code and a run behind it but no home. Six have never been run.

What has to be true at the end: every variation has one number and one plain name, the same number
and name in every pillar, so a reader six months from now finds `00-matching-the-composition-to-the-joint-prompt`
in `report/` and knows exactly which equation it is, what it produced at rank 8, 16 and 32, how to
re-run it, and what it costs on this cluster. V0 is built first and every later variation copies its
shape.

## The standing constraint on every file this walk emits

Navigation: ⬅️ [What this is for](#what-this-walk-is-for) | 📋 [TOC](#table-of-contents) | [Next](#the-verb-set) ➡️

Collecting a lot is fine. Writing a lot into a pillar is not, and `report/` and `plans/` bloat
fastest.

Every page this walk produces shows only what is critical on its face. Rendered thumbnails rather
than described figures. Hyperlinks and cross-references rather than restated content. Rendered
equations where the equation is the clearest form.

Short sentences with blank lines between them, never multi-line paragraphs where two sentences do
the work.

A disposition that would produce a long page is the wrong disposition. The detail goes in the file
the link points at.

## The verb set

Navigation: ⬅️ [What this is for](#what-this-walk-is-for) | 📋 [TOC](#table-of-contents) | [Next](#the-groups) ➡️

| Verb | What it means here | Reversible |
|---|---|---|
| write | a new file in a pillar or a new plan file | yes |
| extend | the file exists, the variation gets a section in it | yes |
| gather | evidence exists on disk, tabulate it, no new run | yes |
| run | needs a cluster run before anything can be written | yes |
| illustrate | needs a rendered picture before the plan is readable | yes |
| placeholder | one row in the table now, a real plan later | yes |
| look first | cannot be dispositioned until someone opens it | n/a |

No `delete` in this walk. Nothing here is being removed.

## The groups

Navigation: ⬅️ [The verb set](#the-verb-set) | 📋 [TOC](#table-of-contents) | [Next](#the-disposition-ledger) ➡️

### 1. 🔢 The naming spine across the pillars  ✅ settled

The question: does each variation get a folder in every pillar, or one file, and what carries the number.

- [x] 1.1 `report/designing-the-correction-loss/`, one file per variation
- [x] 1.2 `context/world/the-correction-loss.md`, one shared file
- [x] 1.3 `runbook/running-things-on-the-cluster/training-a-correction-adapter.md`, one recipe, a row per flag
- [x] 1.4 `environment/hpc/throughput.md`, one added memory-per-rank row, no new file
- [x] 1.5 `artifacts/results/designing-the-correction-loss/<NN-name>/`, folder per variation, new figures only
- [x] 1.6 `plans/09-designing-the-correction-loss/`, one scope, a plan file per variation

**The numbers and names, fixed here.**

| # | Variation | Name |
|---|---|---|
| 00 | V0 | matching-the-composition-to-the-joint-prompt |
| 00a | V0a | penalising-the-empty-branch-for-moving |
| 01 | V1 | freezing-the-empty-branch |
| 02 | V2 | anchoring-the-empty-branch-to-two-animals |
| 03 | V3 | adapting-only-the-empty-branch |
| 04 | V4 | training-at-guidance-weight-one |
| 05 | V5 | training-on-real-photographs |
| 06 | V6 | training-on-the-renders-that-composed |
| 07 | V7 | keeping-single-concept-renders-unchanged |

**The figure seam, accepted deliberately.** The eight existing V0 figures stay in
`artifacts/results/does-training-longer-help-the-pooled-lora/`. Only new figures go to the new
folder. Moving the eight would break two report files, a runbook anchor and an artifact card, and
`00`'s page links both homes.

### 2. 📦 V0's evidence already on disk  ✅ settled

The question: which checkpoints and runs are actually V0, and which are a different objective wearing a rank label. Survey 1 answered it by config key. What is left is what to do with the evidence.

- [x] 2.1 the five runs: gather into `00`'s disclosure table
- [x] 2.2 the eight already-rendered figures: keep in place, link from `00`
- [x] 2.3 the comparison strips: **already rendered**, per checkpoint, every rank. Assemble a selection, do not re-render
- [x] 2.4 the rank-ablation figure: run, and compute the seed-noise band underneath it first
- [x] 2.5 `make_closeout_figures.py`: rewrite in `scripts/` to cover all eleven figures, so every later variation reuses it
- [x] 2.6 the per-checkpoint numbers: collect the scattered probe json into one `00-per-checkpoint.json`, LoRA norms included

**What the saved strips are.** Every run writes four labelled three-way comparisons per checkpoint
under `samples/per_epoch/`: two in-sample (wolf x husky, seeds 1 and 2) and two held-out
(cat x dog, seeds 9 and 10), each showing the Mono target, plain PoE and the adapter's render.
Ten checkpoints per run, twenty-five for the 450k run. `phase1_r8_100k` predates the feature and
has none.

**The strip's layout, fixed here.** One row of five, `[Mono][plain PoE][LoRA 10k][LoRA 30k][LoRA
100k]`, not three rows of three. The saved strips repeat the Mono and PoE columns at every
checkpoint, so two thirds of the space carries no new information.

**These strips illustrate, the 8-seed grid scores.** The saved seeds are 9 and 10 only, not the
eight the reports measure.

### 3. 📚 V0's write-up across the pillars  ✅ settled

The question: per pillar, does a file already exist, and does it need writing, extending, or nothing.

- [x] 3.1 `context/world/the-correction-loss.md`, write
- [x] 3.2 `context/world/lora-corrector.md`, extend: rank is a dial, not the adjective "rank-8"
- [x] 3.3 `report/designing-the-correction-loss/00-...md`, write
- [x] 3.4 `runbook/.../training-a-correction-adapter.md`, write, the cold-start recipe
- [x] 3.5 `environment/hpc/throughput.md`, extend: the memory-per-rank row, gathered from the review files
- [x] 3.6 the artifact card for the new results folder, write

**Who writes them.** Scope 09's own numbered tasks, each naming its file, what goes on the face and
what becomes a link. The four pulse skills run afterwards as an audit, never as the author, because
each one repairs its whole folder and would write more than these six files.
`report-pulse`'s format is read before `00`'s page is written, since the page has to carry the
claim, figure and statistic triple.

### 4. ✏️ The V0a design plan and its illustration  ✅ settled

The question: what does V0a have to beat, measured how. It already ran, so the question is now what
the read is rather than what the launch is.

- [x] 4.1 the four experiments: gather. The V0a plan is a read plan and launches nothing
- [x] 4.2 the bars, fixed before the read
- [x] 4.3 the illustration: prompt written, Empty, level 2, subject lane
- [x] 4.4 attaches at `plans/09-designing-the-correction-loss/plans/hypothesis/00a-penalise-the-empty-branch.md`

**The two bars, fixed here.**

*Mechanism.* Does the empty branch stop drifting? Already answered off W&B: the drift term is
1.20e-4 rising to 4.00e-4 in the control and 1.98e-5 falling to 9.84e-6 under the penalty, so
41 times smaller at 25k to 30k, against a fit loss 23% higher.

*Outcome, and this one decides.* The anchored experiment composes at least as many of the eight held-out
cat x dog seeds as `plain`, and wins a blind by-eye read on at least 5 of 8. Below either, V0a is a
null and stops.

**Three caveats recorded, because they decide what these experiments can be compared to.**

The experiments train on denoising steps 0 to 25 of 50 (`--train-step-range 0 25`), where phase1 used all
50. The flag is not recorded in `config.json`, so the saved config cannot identify the objective.

All four experiments carry `ema_decay 0.999`; phase1 V0 carried none. The anchor-against-plain comparison
stays clean because all four share it, but none of these four is the phase1 V0.

The pool is v54 with different in-sample pairs, and two of the four held-out cells have a broken
joint-prompt reference the grid script already knows about.

**Training length.** The fit loss fell fourfold over the first 10k and 3 to 8 percent over the last
5k, so it has flattened and more steps will not buy a better fit. That does not locate the best
checkpoint: in phase1 the loss also flattened while the render kept getting worse, and only three
checkpoints were saved here.

**One correction owed to the scene artifact.** Its panel says the fine "cut it 39x, and the fit
error nearly doubled". The 39x holds (41 on medians, 33 on means). The fit error rose 23%, not
nearly double.

**What is on disk.** `/datasets/mmolefe/poe_repair_min/outputs/correction_loss_variants/` holds four
matched rank-16 experiments, 30,000 steps each, launched within seventeen seconds of each other on
2026-09-12, with twelve checkpoint sample sets apiece.

| Arm | `null_anchor` | `branch_prompt_style` | Variation | W&B |
|---|---|---|---|---|
| `r16_s0_25_plain` | 0 | plain | 00, the matched control | `tqs7qf95` |
| `r16_s0_25_anchor` | 10 | plain | **00a** | `72as6yk3` |
| `r16_s0_25_plurality` | 0 | plurality | a prompt-rewrite experiment | `mw1cqrpk` |
| `r16_s0_25_connective` | 0 | connective | a prompt-rewrite experiment | `pckb2za7` |

`grep -rl r16_s0_25 --include=*.md .` returns nothing. No plan, no review file, no report mentions
them.

Their pool is not the phase1 pool: in-sample here is giraffe x lion and lion x meerkat, and
held-out adds elephant x penguin. So these four compare only against each other, never against the
phase1 rank-16 run.

### 5. 🚀 What gets run, and how results are kept over time  ✅ settled

The question: what the cluster actually does now, and at what cadence every variation saves, so a
result six months old can still be re-read.

- [x] 5.1 score the four experiments' checkpoints on the eight held-out cat x dog seeds
- [x] 5.2 compute the seed-noise band from the control's spread
- [x] 5.3 read the strips by eye, blind
- [x] 5.4 train to 50,000 steps up front rather than 30,000 with a conditional resume
- [x] 5.5 one cadence: a checkpoint at every render, every 2,500 steps

**The cadence, and why it was wrong.** `--ckpt-every-epochs` defaults to 200 at
`train_pooled.py:222`, which is a save every 10,000 steps, and the launcher never passes it.
Renders run on `--sample-every-epochs`, set to 50, which is every 2,500 steps. Nine of the twelve
render sets have no weights behind them. The fix is one line in
`scripts/showcase/train_pool_run.sh`: pass `--ckpt-every-epochs "$SAMPLEEVERY"`.

**What that costs.** 20 checkpoints per experiment at 50,000 steps, 2.3 GB each experiment, 9.2 GB
for a set of four. `/datasets` has 233 TB free. No extra compute, because the render pass already
halts training at those epochs.

**The window: 50,000 steps.** About 20 hours per experiment on an RTX 3090, against 11 to 13.5 for
30,000. The one rank-16 run this project has done peaked at 50,018, so 30,000 stops short, and
buying the whole curve up front costs machine time overnight rather than a decision mid-week.

**Every objective flag goes into `config.json`.** Today `train_step_range` is absent, so the file
meant to identify a run cannot say whether it trained on half the schedule or all of it.

**Why `bigbatch` and not `biggpu`.** Four experiments at once need four cards and `biggpu` allows
one job per user. A Blackwell node is about three times faster per step (rank 8 at 0.236 s/step on
`mscluster110` against 0.706 on an A6000), but four experiments run one after another there comes to
the same wall clock as four in parallel on 3090s. A Blackwell is the right choice for a single long
follow-up, and it needs `PY=` pointed at `co3_bw`, because a `co3` CUDA operation on that card
produces no output and no error.

### 6. 🗺️ V1 to V7, ordered  ✅ settled

The question: which earn a plan file now and which are one table row until V0a returns.

**The order is the note's own, and it does not follow the numbers.** It opens with something that
is not a variation: log the undialled error next to the loss. One extra line, no forward pass, and
it says whether the empty branch drifted in a direction the loss could not see.

- [x] 6.0 the logging line: write, and first
- [x] 6.1 `01` freezing the empty branch: plan file. A third cheaper, exact where the penalty is approximate, and it removes the guidance weight from the problem
- [x] 6.2 `02` anchoring the empty branch to two animals: plan file, after `01`. The only variation that changes what the composition divides by
- [x] 6.3 `03` adapting only the empty branch: table row. Cheap enough for the record, expected to fail
- [x] 6.4 `04` training at guidance weight one: table row. Equals `01` when the empty branch is frozen
- [x] 6.5 `05` training on real photographs: routed to `/drip-idea` on claim 6 of this folder's idea map. Not blocked, waiting on a named check
- [x] 6.6 `06` training on the renders that composed: table row, last, because it changes the target
- [x] 6.7 `07` keeping single-concept renders unchanged: table row, conditional on a no-training read

**What `01` must carry.** Whatever is frozen in training must be frozen in sampling. On 2026-09-05
a windowed sampler left the adapter attached while rendering what were meant to be plain
references, and it cost a re-render. So `01` changes `trainer.py:450` **and** the sampler's
empty-prompt pass, and its first task is a check that proves the two agree before any training
starts.

**Two of the seven are already answered.** `plurality` and `connective` ran on 2026-09-12 and their
fit losses came out within 20% of each other, which is what the theory predicts: renaming a prompt
renames a free tensor. They need a recorded finding, not a plan.

## The disposition ledger

Navigation: ⬅️ [The groups](#the-groups) | 📋 [TOC](#table-of-contents) | [Next](#held-pieces) ➡️

| Units | Verb | Attaches to | Skill that does it | Runbook step |
|---|---|---|---|---|
| group 1, all 6 placements | write | the six pillar paths named above | `/init-master-plan` for the scope, the pulses for the pillar files | 1 |
| group 2, units 2.1, 2.2, 2.3, 2.6 | gather | `report/designing-the-correction-loss/00-...md` and its results folder | scope 09's foundation plan | 3 |
| group 2, unit 2.4 | run | the same page's rank table | scope 09's foundation plan, `/design-figure` for the band | 3 |
| group 2, unit 2.5 | write | `scripts/` | scope 09's foundation plan | 3 |
| group 3, all 6 files | write, extend | the six pillar paths | scope 09's foundation plan, pulses as audit | 2, 4 |
| group 4, all 4 units | gather | `plans/09/.../00a-penalise-the-empty-branch.md` and `report/.../00a-...md` | scope 09's hypothesis plan | 5 |
| group 5, units 5.1 to 5.4 | gather, run | `report/.../00a-...md` | scope 09's run plan | 5 |
| group 5, unit 5.5 | write | `scripts/showcase/train_pool_run.sh`, `train_pooled.py` config dump | scope 09's tools plan | 6 |
| group 6, the logging line and `01`, `02` | write | three plan files in scope 09 | `/populate-plans`, then `/frame-hypothesis` on `01` | 7, 8 |
| group 6, `03`, `04`, `06`, `07` | placeholder | one row each in scope 09's master plan | `/init-master-plan` | 7 |
| group 6, `05` | route | back into this walk via `integrate` | `/drip-idea` on claim 6 | out of band |

## Held pieces

Nothing held.

## Surveys

| # | Anchor | State | Headline |
|---|---|---|---|
| 1 | group 2 | landed | five V0 runs on disk, identified by their `config.json` carrying none of the later objective switches: `phase1_r8_100k`, `r8_200k`, `r8_450k`, `r16_100k`, `r32_100k`. The loss is `one_pair_one_seed/trainer.py:320-336` (composition), `:376-414` (target), `:453` (error), `:496-503` (the anchor V0a turns on). Sixteen CLI flags change the objective. The two review files that should carry these trainings' verdicts both still read "Nothing has run yet". |
| 3 | group 2 | landed | two report files already carry the V0 numbers with eight rendered figures, including the two same-seed-across-checkpoint strips. Neither report answers its review file's pre-registered bar (the seed-noise band was never computed), report A explicitly disclaims the matched-step rank comparison that review 09 asks for, the rank-ablation figure plan 09 owes does not exist, no aggregated per-checkpoint json exists (the LoRA norms live only as prose), and `make_closeout_figures.py` is not in the repo so nothing can be redrawn. Plans 08 and 09 still read `written` in the scope's MASTER_PLAN. |
| 2 | group 3 | landed | `context/` is the biggest hole: the objective, the branches a/b/u, the dial w, eps_J, rank as a dial and checkpoint are all absent. `report/` has the rank 8/16/32 numbers but files them as checkpoint selection with no objective named. `runbook/` cannot cold-start a pooled training. `environment/` has rank-32 timing but no memory row. Three training groupings under `artifacts/results/` have no README and one is empty. |

## Drafted task lines

Nothing drafted.

## Illustrations owed

**Art direction, fixed for the whole nine-image set.** Empty, level 2 (one mechanism opened),
subject lane. Chosen 2026-09-16. The first accepted render is passed as a reference image to the
remaining eight, because Empty drifts across a multi-image technical set otherwise.

The V0a architecture picture, tailored from the interactive loss diagram's "the loss" frame.
Prompt written, held here until scope 09 exists and its `diagram-prompts.md` can own it.

The rank-ablation figure plan 09 owes: held-out compose rate against rank at matched steps, with the seed-noise band drawn. Catalogued, never made.

## Reversals

None.

## The chain so far

**Out of band**, running beside everything below: `/drip-idea` on claim 6 of this folder's
`IDEA_MAP.md`, emitted 2026-09-16, to settle whether a corpus of real photographs holding both
animals as separate objects is reachable. Its verdict decides whether `05` becomes a plan file or a
recorded finding. Returns here via `integrate`.

**The chain**, spliced from flow 20 in `~/.claude/skills/WORKFLOWS.md`. Flow 20 assumes nothing has
run; here `00` and `00a` already have, so the hypothesis steps sit after the write-up steps instead
of before them.

1. `/init-master-plan` mints `plans/09-designing-the-correction-loss/`.
2. `/populate-plans` writes the plan files.
3. Do the foundation plan: the six pillar files, the chart script, the `00` page.
4. `/render-diagrams` drains the scope's map, starting with the `00a` picture.
5. Do the `00a` read plan: score the four experiments, compute the band, read blind, write the page.
6. `/frame-hypothesis` on `01`, whose first task is the training-and-sampling agreement check.
7. Launch `01` at 50,000 steps.
8. `/sync-plan-tree` as verdicts land.

## Next step

The walk is complete. Run the chain in `## The chain so far`, one step at a time.
