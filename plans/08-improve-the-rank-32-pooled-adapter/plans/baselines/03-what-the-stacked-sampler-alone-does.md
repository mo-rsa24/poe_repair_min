# 🧪 What the stacked sampler alone does to checkpoint 30050

**Before any training run is launched, checkpoint 30050 is rendered under the stacked sampler and judged against its own shipped render. If changing how the checkpoint is run already produces more clean seeds, that is the finding, it is reported first, and every training run in this scope is then read against it rather than against the shipped render.**

**Step 68 in the root running order. Waits on steps 66 and 67. Gates the reading of step 69, though not its launch: the chain may cook while this is judged.**

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/08-improve-the-rank-32-pooled-adapter/plans/baselines/03-what-the-stacked-sampler-alone-does.md — the baseline rendered both ways, the blind labels taken, and the verdict on whether the sampler alone already beats the shipped render, in the review file
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 67 (previous) | [02: the blind label pass](../tools/02-the-blind-label-pass.md) | the read this plan is the first user of |
| **68 (current)** | **03: what the stacked sampler alone does** | checkpoint 30050 under both samplers, judged blind |
| 69 (next) | [04: the parent and the four children](../hypothesis/04-the-parent-and-the-four-children.md) | the five training runs, read against whichever baseline column wins here |

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

- **The baseline, B**: checkpoint 30050 of the rank-32 pooled adapter, at `/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/checkpoints/lora_step_030050.pt`. No training happens in this plan.
- **Shipped**: the adapter on all 50 denoising steps, deterministic sampling, guidance 7.5 throughout. This is how the checkpoint is run today and how its existing results were produced.
- **Stacked**: the adapter on steps 0 to 24, the frozen model finishing the remaining 26, fresh randomness added at every step, and guidance 7.5 only on steps 5 to 35 with 1.0 outside.
- **A clean seed**: one whose render is labelled **clean** in the blind pass, meaning both named animals present, each clearly itself, nothing unnatural on inspection.
- **Better, under the same sampler**: more seeds labelled clean, and no seed the other side had clean falling out of that label. Both halves are required; a run that gains two and loses one is not better.

---

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The experiment.** One checkpoint, two ways of running it, 25 seeds each, judged blind.

**The hypothesis.** Part of what looks like a defect in the trained adapter is a property of how it is run. Holding strong guidance to a middle span, letting the frozen model finish, and adding fresh randomness at every step may recover clean seeds that the shipped sampler loses, with no training at all.

**If true.** The scope's baseline moves: every training run is read against the stacked column of B rather than the shipped one, and the bar rises before a single GPU-hour is spent on training. The finding also stands on its own, because it says the defect is partly in the sampler.

**If false.** The shipped column stays the baseline, and the stacked sampler is carried forward only as the second way every run is rendered.

**If it goes the other way.** The stacked sampler may lose clean seeds. That is a real answer too and it is reported: it would say the frozen tail or the weak late guidance costs composition, and it would make the stacked column the secondary read for the rest of the scope rather than the primary one.

**Dataset.** All 17 cached cat × dog seeds and all 8 elephant × penguin seeds. Cat × dog is the pair the defect was seen on; elephant × penguin is the second pair and is carried so a change that helps one pair and breaks the other is visible.

**Associated materials.**
- Review questions: [the review file](../../review/03-what-the-stacked-sampler-alone-does.md)
- The labelling procedure: [run the blind label pass](../../procedures/tools-02-run-the-blind-label-pass.md)
- The design: [the run design](../../../../artifacts/ideas/improving-the-pooled-lora-run/run-design-parent-and-children.md), which names B under the stacked sampler as the thing rendered first

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Cost.** 25 seeds × 2 samplers = 50 renders of the checkpoint, plus whatever references plan 02 did not already gather. About one GPU-hour on any free card, plus twenty minutes of labelling per sampler.

**Buys.** The baseline the whole scope is measured against, and a finding that stands whether it comes out positive or negative. It costs a fiftieth of the training chain and can change what the chain is compared to, which is why it runs first.

**Prerequisites.** Steps 66 and 67 complete: the guidance interval exists and the blind read works, including its known-example smoke.

**Output root.** `/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/baseline/`.

**W&B project.** `prime_lab/poe-repair-animals-compose`, logged as its own run tagged `baseline` by task 2.3 and opened by instruction 3.0, so the baseline's strips sit where every other run's do rather than on one machine's disk.

**This run is frozen the moment it lands.** It is a competitor to beat. No re-render with a better setting after a training run's number is known, and no tuning of the stacked sampler's span or eta once the labels are in. If the stacked settings turn out to be wrong, that is a new plan, not an edit to this one.

**Known issues:** see the [Error Matrix](#error-matrix).

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- **Inference only, so it does not need the Blackwell.** Rank-32 inference fits comfortably on the session node's card, which keeps `mscluster112` free for the training chain ([throughput](../../../../environment/hpc/throughput.md)).
- **Renders go under `/datasets`** ([storage](../../../../environment/storage.md)).
- **References are rendered with no adapter attached**, in a separate process, because the windowed sampler leaves the adapter enabled outside its window.
- **`biggpu` allows one job per user**, so if this is run on a cluster device rather than the session node, it competes with the chain and should be sequenced around it ([execution protocol](../../../../environment/hpc/execution-protocol.md)).

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**Checkpoint 30050's clean-seed count under the stacked sampler is known, beside its count under the shipped sampler, on 25 blind-labelled seeds, before any training run is read.**

**Why this matters right now:** if the sampler already carries part of the improvement, then a training run judged against the shipped render gets credit for something training did not do.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** The scope is about to spend twelve hours of one device training five adapters, and the thing they will be compared against has only ever been rendered one way. Two axes are about to move at once: what the adapter learned, and how the adapter is run.

**The approach.** Separate them, cheaply, first. Render the existing checkpoint both ways and judge both blind. Whichever column has more clean seeds becomes the number the training runs must beat.

**Key insights.**

1. **Two things changed between the shipped render and the stacked one, and this plan does not try to separate them.** The stacked sampler moves the adapter's window, the randomness and the guidance span together. That is deliberate: the question here is whether the packaged alternative is better, not which of its three parts did it. Splitting them is a later plan if the answer is yes.
2. **A cheap read that can change the comparison runs before the expensive one.** One GPU-hour that can reset the baseline is worth spending before twelve.
3. **A negative answer is as useful as a positive one.** If the stacked sampler loses clean seeds, the scope learns that before it renders every training run that way and wonders why they all look worse.

---

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
 checkpoint 30050 ─┬─▶ shipped:  adapter ████████████████████████████████████  eta 0, guidance 7.5
                   │             steps 0 ─────────────────────────────── 49
                   │
                   └─▶ stacked:  adapter ████████████████░░░░░░░░░░░░░░░░░░░░  eta 1
                                 guidance     ▁▁▁▁▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▁▁▁▁▁▁▁▁▁▁
                                 steps 0    5 ──────────────── 35        49
                                          ( ████ adapter · ░░░░ frozen model )

   25 seeds × 2 samplers ──▶ strips ──▶ shuffled, names hidden ──▶ blind labels
                                                                      │
                            clean count, shipped  ◀───── join ────────▶  clean count, stacked
                                          └──────── which is the baseline? ────────┘
```

---

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The render pass**: checkpoint 30050 on all 17 cat × dog and all 8 elephant × penguin seeds, under both samplers, from the cached initial noise per seed so both columns start from the same place.
2. **The strips**: built by plan 02's builder, four tiles per seed per sampler, ids only.
3. **The blind labels**: one label per tile, taken through the labelling procedure, under each sampler separately.
4. **The join and the verdict**: the clean count per sampler, the per-seed comparison, and the branch by the bar in source.
5. **The W&B run**: both columns' strips, the label table, and the two secondary reads, in one run tagged `baseline`.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

**Purpose.** Serves the scope's objective 3: separate the sampler's contribution from training's, by reading the baseline under both samplers before any training run is read.

**Goals.**

1. Fifty renders of checkpoint 30050 on disk, 25 seeds under each sampler, from the cached initial noise.
2. A blind label for every tile under each sampler.
3. The clean count per sampler, and the per-seed list of which seeds changed label between them.
4. A verdict by the bar in source, recorded in the review file, before any training run is read.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/08-improve-the-rank-32-pooled-adapter/plans/baselines/03-what-the-stacked-sampler-alone-does.md`
  - Done when: the report comes back clean, or its proposals have been applied.
- [ ] **0.2** Cross-reference this plan's terms against `context/`, `environment/`, `runbook/` and `report/`.
  - Paste: `/xref-pillar @plans/08-improve-the-rank-32-pooled-adapter/plans/baselines/03-what-the-stacked-sampler-alone-does.md`
  - Done when: the scan returns no candidates, or its proposed links have been applied.

▶ **Next: [task 1.1](#1--fix-the-bar-in-source-before-rendering)**.

### 1. 🔬 Fix the bar in source, before rendering

◀ **Needs: [task 0.1](#0--check-this-plan-before-working-from-it)**.

- [ ] **1.1 Write the verdict rule as constants in `scripts/showcase/blind_label.py`**, the join script [plan 02 task 2.3](../tools/02-the-blind-label-pass.md#2--build-the-strips-the-blinding-and-the-two-secondary-reads) writes, not in prose, so it cannot be adjusted after the labels are in without showing up in a diff.
  - `MIN_CLEAN_GAIN = 1` (the stacked column needs at least one more clean seed than the shipped column) and `MAX_CLEAN_LOST = 0` (it may lose none the shipped column had clean).
  - This is the single home for both constants. Plan 04 reuses them from here rather than defining its own, so the parent, the children and the baseline are all judged by one rule.
  - **Done when:** the constants exist in the script and the branch is computed from them rather than written by hand.

▶ **Next: [task 2.1](#2--render-the-baseline-both-ways)**.

### 2. 🚀 Render the baseline both ways

◀ **Needs: [task 1.1](#1--fix-the-bar-in-source-before-rendering)** done, so the rule is fixed before any picture exists.

- [ ] **2.1 Render checkpoint 30050 on all 25 seeds under both samplers**, from the cached initial noise per seed.
  - Checkpoint: `/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/checkpoints/lora_step_030050.pt`
  - Output: `/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/baseline/renders/<sampler>/<pair>/seed_<n>.png`
  - **Done when:** 50 files exist and the log names the checkpoint path and both sampler settings, so the render can be attributed afterwards.
- [ ] **2.2 Build the strips and run the two secondary reads** through plan 02's builder, one set per sampler.
  - ◀ Needs [plan 02 task 2.3](../tools/02-the-blind-label-pass.md#2--build-the-strips-the-blinding-and-the-two-secondary-reads), which writes the builder this calls.
  - **Done when:** `tiles.json` holds 100 tiles carrying ids, counts and both-names outcomes, and no condition name.
- [ ] **2.3 Log the baseline set to W&B** as one run tagged `baseline`: both columns' strips, the two secondary reads per tile, and the checkpoint path in the run config.
  - **Done when:** the run exists in `prime_lab/poe-repair-animals-compose` and its media panel holds 50 strips.

▶ **Next: [instruction 3.1](#3--label-both-samplers-blind)**.

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

**For you to follow manually.** Do these yourself.

### 3. 👁️ Label both samplers, blind

◀ **Needs: [tasks 2.2 and 2.3](#2--render-the-baseline-both-ways)** done, so the strips exist with their names hidden and the run is on W&B.

3.0 **Open the baseline run in W&B** at `prime_lab/poe-repair-animals-compose`, filter to the `baseline` tag, and open the newest run.
   - ✅ The media panel holds 50 strips, 25 per sampler, and the run config names the 30050 checkpoint path.
   - ❌ The panel is empty or holds one sampler only: task 2.3 did not finish. Do not start labelling from a partial set.

3.1 **Read [the labelling procedure](../../procedures/tools-02-run-the-blind-label-pass.md) to completion and label the shipped set.**
   - 25 seeds, four tiles each.
   - ✅ Every tile carries one of the three labels and `labels.json` has 100 entries for this set.
   - ❌ You find yourself unable to decide between **unclear** and **not two** on several tiles: record which, and say so in the review file. The boundary between those two labels is the one most likely to move a verdict.

3.2 **Label the stacked set, in a separate sitting.**
   - Same procedure. A separate sitting matters because labelling the same seeds twice in a row invites remembering the first answer rather than judging the second picture.
   - ✅ 100 more entries in `labels.json`.

3.3 **Run the join and read the two clean counts.**
   - The join resolves ids to conditions and prints the clean count per sampler plus the per-seed changes.
   - ✅ The counts are printed with the branch the constants in task 1.1 imply.
   - ❌ The join reports an id with no label or a label with no id: the sets are out of step and the counts cannot be trusted.

3.4 **Write the verdict into the [review file](../../review/03-what-the-stacked-sampler-alone-does.md)** before opening plan 04's results.
   - Record which sampler's column is the baseline for the rest of the scope, in one sentence.

▶ **Next: [the close out](#close-out--record-what-this-plan-taught)**.

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted.

- [ ] **Capture the failures this plan hit.**
  - Paste: `/ingest-error-pattern --from-run-log @plans/08-improve-the-rank-32-pooled-adapter/plans/baselines/03-what-the-stacked-sampler-alone-does.md`
  - Done when: each failure has a catalog entry, or there were none.
- [ ] **Bring the tree current.**
  - Paste: `/sync-plan-tree @plans/08-improve-the-rank-32-pooled-adapter/plans/baselines/03-what-the-stacked-sampler-alone-does.md — the baseline rendered both ways, labelled blind, and which column is now the baseline`
  - Done when: statuses, the running order and the Error Matrix match reality.

▶ **Next: [what has to pass before this runs](#what-has-to-pass-before-this-runs)**.

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **Why this checkpoint matters:** this sets the number every training run in the scope is compared against. Getting it wrong means five runs are judged against the wrong thing.

**Pass criteria:**
- The bar's constants exist in source before any render.
- Fifty renders on disk, both samplers, from the cached initial noise.
- Every tile labelled, the join reporting no orphan ids or orphan labels.
- The verdict written in the review file before plan 04's results are opened.

**Fail criteria (STOP):**
- The two samplers were rendered from different initial noise: the comparison is mixed and neither column means what it says.
- The labels were taken with the mapping visible: the read is not blind and the counts cannot stand.

**Partial pass:** if only cat × dog is labelled and elephant × penguin is not, the verdict may be recorded for cat × dog alone, stated as covering one pair, and plan 04 may be read against it. Both pairs are required before the sheet in plan 05.

**When you get results, answer the questions in the [review file](../../review/03-what-the-stacked-sampler-alone-does.md).**

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

### Pending: to be generated from prompts

| Item | Lane | Prompt file | What it shows | Save to |
|------|------|-------------|---------------|---------|
| The two ways every checkpoint is rendered | subject | [diagram-prompts.md](../../diagram-prompts.md#prompt-3-subject-the-two-ways-every-checkpoint-is-rendered) | the shipped and stacked samplers step for step, and the three ways they differ | `diagrams/improve-r32-03-two-ways-every-checkpoint-is-rendered.png` |

### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|------|------|-------------|--------------|--------|---------|
| the baseline strips | — | 25 seeds under each sampler, four tiles per row | task 2.2 | ⏳ generated during run | `baseline/strips/` |
| the two clean counts | — | clean, unclear and not-two counts per sampler, with the per-seed changes | instruction 3.3 | ⏳ generated during run | `baseline/labels_joined.csv` |

### Organization workflow

1. Everything stays under the output root; this plan files nothing to `artifacts/results/`.
2. The baseline columns reach `artifacts/results/` only as part of [plan 05's sheet](../figures/05-the-comparison-sheet.md).

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @...baselines/03-what-the-stacked-sampler-alone-does.md` | **task 0.1**, before any work | conformance and thin instructions reported |
| Cross-reference the plan | `/xref-pillar @...baselines/03-what-the-stacked-sampler-alone-does.md` | **task 0.2**, before any work | terms already documented elsewhere linked |
| Capture patterns | `/ingest-error-pattern --from-run-log` | **the close out**, after any red run | errors added to the catalogs |
| Bring the tree current | `/sync-plan-tree @...baselines/03-what-the-stacked-sampler-alone-does.md` | **the close out** | statuses, running order and Error Matrix match reality |

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**File:** `poe_repair/methods/_poe_langevin.py`
**Relevant section:** the windowed sampler, carrying `lambda_window`, the eta setting, and the guidance interval added by plan 01. Both of this plan's columns come from it, which is what makes them differ on the sampler and nothing else.

**File:** the strip builder written by [plan 02](../tools/02-the-blind-label-pass.md)
**Relevant section:** `--build`, `--score` and `--join`. This plan is its first caller.

**Checkpoint:** `/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/checkpoints/lora_step_030050.pt`

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

— custom; no skill fits. The render is repo-specific and the read is plan 02's tool.

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents)

[04: the parent and the four children](../hypothesis/04-the-parent-and-the-four-children.md) launches the five training runs and reads them against whichever baseline column wins here.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

<details>
<summary>No failures catalogued yet</summary>

**Purpose**: known issues and their fixes, regenerated by `/ingest-error-pattern` and `/sync-plan-tree`.

#### From global catalog

#### From project catalog

---

**Auto-update note:** regenerated by `/sync-plan-tree`. Do not edit by hand.

</details>
