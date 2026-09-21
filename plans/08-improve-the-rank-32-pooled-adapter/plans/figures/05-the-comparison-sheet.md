# 🎨 The comparison sheet, and what the correction's size turns out to mean

**One sheet: the joint target, plain product of experts, checkpoint 30050 and the best run, under both samplers, over all 17 cat × dog and 8 elephant × penguin seeds. Filed under `artifacts/results/` with its card, with the correction's size per pair reported beside it and claiming nothing until labelled examples have been laid against it.**

**Step 70 in the root running order, and the last plan in this scope. Waits on steps 68 and 69.**

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/08-improve-the-rank-32-pooled-adapter/plans/figures/05-the-comparison-sheet.md — the sheet filed with its card, the correction-size table beside it, and the register slot it serves named
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 68 | [03: what the stacked sampler alone does](../baselines/03-what-the-stacked-sampler-alone-does.md) | the baseline columns the sheet carries |
| 69 (previous) | [04: the parent and the four children](../hypothesis/04-the-parent-and-the-four-children.md) | the run the sheet's last column comes from |
| **70 (current)** | **05: the comparison sheet** | the filed evidence, and the scope's close |

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

- **The sheet**: one image per sampler. Rows are seeds; columns are the joint target, plain product of experts, checkpoint 30050, and the best run. The labels sit under each tile.
- **The best run**: whichever of the five trainings won by [plan 04's rule](../hypothesis/04-the-parent-and-the-four-children.md#words-this-plan-uses), or, when none beat the baseline, the parent, shown as the run that did not win. A sheet with no winner is still filed; it is what a null looks like.
- **The correction's size, per pair**: the norm of the correction the adapter produces, averaged over the pair's cells and denoising steps, expressed in the units the size measure this repository already fixed uses.
- **The card**: the entry in the results folder's `README.md` saying what the file shows, how it was made, and where its numbers came from.
- **A register slot**: a reserved place in the paper's figure register carrying, written before the experiment ran, the sentence its caption will be allowed to make.

---

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The system.** Two sheet images, one table, one card entry, one register slot named.

**What it does.** It turns ten label tables and five checkpoints into the one thing a person outside this scope can look at and judge. Everything else this scope produced lives in W&B runs and output directories that will not be opened again.

**What it may not do.** This is a figure plan, so it draws settled results and may change nothing. The caption may claim no more than the verdicts already written in the review files, and no number here may be recomputed in a way that could revise one.

**The correction-size question is a measurement, not a claim.** The size per pair is reported because it is cheap and it may explain which pairs improved. It says nothing on its own: a pair whose correction is large may improve because the adapter had more to learn, or fail because it had further to go. The rule is that no sentence about size is written until clean, unclear and not-two examples have been laid beside the numbers.

**Associated materials.**
- No review file. This plan draws settled results and has no question of its own to answer.
- The verdicts it draws: [plan 03's review](../../review/03-what-the-stacked-sampler-alone-does.md) and [plan 04's review](../../review/04-the-parent-and-the-four-children.md)
- The filing convention: `~/.claude/ARTIFACT_TREE_FORMAT.md`

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Cost.** No GPU for the sheet itself; every render already exists. The correction-size read is a cache pass of about one GPU-hour. Assembly and filing is an afternoon.

**Buys.** The scope's only filed artifact, and the row that lets a later reader compare 30050 against its successor without re-rendering anything.

**Prerequisites.** Plans 03 and 04 complete, their verdicts written. If plan 04 landed only some of its runs, this plan still runs and the sheet says which runs are missing.

**Where it lands.** `artifacts/results/does-a-better-trained-adapter-give-a-clean-cat-and-dog/`, with its `README.md` card entry.

**Sizing.** Twenty-five rows by four columns is too tall to read at one page width. It goes as two images, one per sampler, each landscape, with the cat × dog seeds and the elephant × penguin seeds in separate blocks within the image and the block labelled on the image rather than in a legend.

**Known issues:** see the [Error Matrix](#error-matrix).

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- **The renders live on `/datasets`** and the sheet is assembled from there; the assembled image is small enough to live in the repository under `artifacts/results/` ([storage](../../../../environment/storage.md)).
- **No system LaTeX exists here** ([paper](../../../../environment/paper.md)), so the sheet is built as an image rather than as a typeset figure, and any caption text in it is drawn by the plotting code.
- **The correction-size read needs a card for one cache pass**, inference only, so it does not compete with anything else running.

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**One sheet per sampler, filed with its card, showing every seed this scope judged and the label each render received, with the correction's size per pair beside it and no claim attached to that size beyond what the examples support.**

**Why this matters right now:** the scope's conclusion currently exists as ten CSV files and five W&B runs. A reader who was not here cannot reach any of it.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** Evidence that lives only in an experiment tracker is evidence nobody revisits. The scope's whole argument is visual, and it has produced no filed picture.

**The approach.** Assemble the sheet from renders that already exist, put the blind label under each tile so the picture and its judgement are in one place, and file it where the repository's other results live, with a card saying how it was made.

**Key insights.**

1. **The label belongs under the tile.** A sheet that shows the pictures and puts the labels in a separate table charges the reader with a lookup, and the lookup is exactly where a reader stops.
2. **The failures stay on the sheet.** A sheet showing only the seeds that improved is a claim by omission. Every seed judged appears, including the ones the best run got wrong.
3. **The size measurement is filed with its caveat attached, or not at all.** A number reported without the examples beside it becomes a claim the moment somebody quotes it.

---

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
 sheet, one per sampler:

           joint target   plain PoE      30050          best run
 seed 9    [   ]          [   ]          [   ]          [   ]
           clean          not two        unclear        clean
 seed 10   [   ]          [   ]          [   ]          [   ]
           not two *      not two        unclear        unclear
 ...
 seed 25   [   ]          [   ]          [   ]          [   ]

 * a joint target labelled "not two" is a seed whose own reference is wrong.
   It stays on the sheet and is marked, because the adapter was trained
   toward it and a reader has to be able to see that.

 beside it, the size table:
   pair            mean correction size    clean seeds, best run
   cat x dog             ...                      ...
   elephant x penguin    ...                      ...
   ... one row per training pair ...
```

---

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The two sheets**: one per sampler, four columns, 25 rows split into a cat × dog block and an elephant × penguin block, each tile carrying its blind label underneath and each joint-target tile whose own reference is wrong marked as such.
2. **The sidecar**: a JSON beside each sheet naming every source render's path, the run it came from, its W&B run id, the label it received, and the two secondary reads, so the sheet can be regenerated and audited without opening the labelling tool.
3. **The correction-size table**: mean correction size per pair over that pair's cells and steps, in the units the repository's fixed size measure uses, with the best run's clean-seed count beside it.
4. **The card entry** in `artifacts/results/does-a-better-trained-adapter-give-a-clean-cat-and-dog/README.md`: what each file shows, how it was made, and which review file its verdicts come from.
5. **The register slot**, named: which slot in `paper/iclr/figures.md` this sheet serves, or an explicit statement that it serves none and is scope evidence only.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

**Purpose.** Serves the scope's objective 4 and closes it: the sheet is what the scope hands to anyone who was not here.

**Goals.**

1. Two sheets exist under `artifacts/results/does-a-better-trained-adapter-give-a-clean-cat-and-dog/`, one per sampler.
2. Each has a sidecar naming every source render, its run, its W&B id and its label.
3. The correction-size table exists with one row per training pair.
4. The card entry exists and names the review files the verdicts come from.
5. The register slot this sheet serves is named, or its absence is stated.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/08-improve-the-rank-32-pooled-adapter/plans/figures/05-the-comparison-sheet.md`
  - Done when: the report comes back clean, or its proposals have been applied.
- [ ] **0.2** Cross-reference this plan's terms against `context/`, `environment/`, `runbook/` and `report/`.
  - Paste: `/xref-pillar @plans/08-improve-the-rank-32-pooled-adapter/plans/figures/05-the-comparison-sheet.md`
  - Done when: the scan returns no candidates, or its proposed links have been applied.

▶ **Next: [task 1.1](#1--assemble-the-two-sheets)**.

### 1. 🎨 Assemble the two sheets

◀ **Needs: [plan 04 instruction 5.4](../hypothesis/04-the-parent-and-the-four-children.md#5--label-every-run-blind-and-write-the-verdicts)** for the winning run and its labels, and [plan 03 instruction 3.4](../baselines/03-what-the-stacked-sampler-alone-does.md#3--label-both-samplers-blind) for the baseline columns the sheet carries and the verdict its caption may not exceed.

- [ ] **1.1 Build one sheet per sampler** from the existing renders, four columns, 25 rows in two labelled blocks, each tile carrying its blind label underneath.
  - Mark every joint-target tile whose own reference shows the wrong animals, so a reader can see which seeds the adapter was trained toward incorrectly.
  - Output: `artifacts/results/does-a-better-trained-adapter-give-a-clean-cat-and-dog/`
  - **Done when:** both PNGs exist, every seed judged appears on them, and the seeds the best run got wrong are present rather than dropped.
- [ ] **1.2 Write the sidecar** beside each sheet: per tile, the source render's path, its run, its W&B run id, its label and the two secondary reads.
  - **Done when:** the sidecar's tile count matches the sheet's, and every path in it resolves.

▶ **Next: [task 2.1](#2--measure-the-corrections-size-per-pair)**.

### 2. 📊 Measure the correction's size per pair

◀ **Needs: [task 1.1](#1--assemble-the-two-sheets)** done, so the sizes can be put beside a settled clean-seed count.

- [ ] **2.1 Read the mean correction size per pair** from the cache for the best run's adapter, over that pair's cells and denoising steps, in the units the repository's fixed size measure uses.
  - **Done when:** one row per training pair exists with its size and the best run's clean-seed count on that pair, written to the sheet's folder as a table with its own sidecar.
  - The table is filed with no interpretation attached. Instruction 3.2 is where any sentence about it gets written, and only after the examples have been looked at.

▶ **Next: [instruction 3.1](#3--check-the-sheet-reads-and-then-file-it)**.

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

**For you to follow manually.** Do these yourself.

### 3. 👁️ Check the sheet reads, and then file it

◀ **Needs: [task 2.1](#2--measure-the-corrections-size-per-pair)** done.

3.1 **Open both sheets at full size and read them cold.**
   - Ask one question: without reading any caption, can you see which column is better?
   - ✅ You can. The sheet is doing its job.
   - ❌ You cannot: either the improvement is not visible at sheet size, which is itself worth recording, or the tiles are too small. Try one sampler's sheet split into two images before concluding the former.

3.2 **Lay the size table against the examples before writing any sentence about it.**
   - Take the pair with the largest correction size and the pair with the smallest, and look at a clean, an unclear and a not-two example from each on the sheet.
   - ✅ A pattern survives that comparison: write it, in one sentence, in the card.
   - ❌ No pattern survives, or the examples contradict the ordering: write that the size was measured and explains nothing here. That is the honest outcome and it is more useful than silence, because the next person will otherwise measure it again.

3.3 **Write the card entry** in the results folder's `README.md`: what each file shows, how it was made, which review files the verdicts come from, and the caveat that the labels are one person's blind read.

3.4 **Name the register slot.**
   - Open `paper/iclr/figures.md` and find the slot this sheet serves.
   - ✅ A slot exists: record its identifier in the card, and check the sheet claims no more than that slot's sentence allows.
   - ❌ No slot fits: write in the card that this sheet is scope evidence and serves no register slot, rather than inventing one.

▶ **Next: [the close out](#close-out--record-what-this-plan-taught)**.

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted.

- [ ] **Capture the failures this plan hit.**
  - Paste: `/ingest-error-pattern --from-run-log @plans/08-improve-the-rank-32-pooled-adapter/plans/figures/05-the-comparison-sheet.md`
  - Done when: each failure has a catalog entry, or there were none.
- [ ] **Bring the tree current.**
  - Paste: `/sync-plan-tree @plans/08-improve-the-rank-32-pooled-adapter/plans/figures/05-the-comparison-sheet.md — the two sheets filed with their card, the size table, and the register slot named`
  - Done when: statuses, the running order and the Error Matrix match reality.
- [ ] **Close the scope with its recall gallery**, once every plan above is ✅.
  - Paste: `/recap-plan-tree @plans/08-improve-the-rank-32-pooled-adapter/MASTER_PLAN.md`
  - Done when: the Artifact URL it publishes is recorded in the scope's [MASTER_PLAN](../../MASTER_PLAN.md).

▶ **Next: [what has to pass before this runs](#what-has-to-pass-before-this-runs)**.

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **Why this checkpoint matters:** this is the only thing the scope files. A sheet that overstates its verdict outlives every correction.

**Pass criteria:**
- Every seed judged appears on a sheet, including those the best run got wrong.
- Each sheet's sidecar resolves to real renders and matches the sheet's tile count.
- The card names the review files the verdicts come from.
- No sentence about the correction's size exists unless instruction 3.2's comparison supports it.

**Fail criteria (STOP):**
- The sheet's caption claims more than the verdicts in the review files: a figure plan may draw a settled result and may not extend one.
- A seed was dropped from the sheet because its render was poor: that is the claim by omission this plan exists to avoid.

**No review file.** This plan draws settled results and has no question of its own, per the run-kind rules in `~/.claude/EXPERIMENT_CONVENTIONS.md`.

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

### Pending: to be generated from prompts

| Item | Lane | Prompt file | What it shows | Save to |
|------|------|-------------|---------------|---------|
| From cached cells to one comparison sheet | subject | [diagram-prompts.md](../../diagram-prompts.md#subject-capstone-from-cached-cells-to-one-comparison-sheet) | the whole scope on one page, ending at this sheet | `diagrams/improve-r32-06-capstone-cells-to-comparison-sheet.png` |

### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|------|------|-------------|--------------|--------|---------|
| the shipped-sampler sheet | — | 25 rows, four columns, blind label under every tile | task 1.1 | ⏳ generated during run | `artifacts/results/does-a-better-trained-adapter-give-a-clean-cat-and-dog/shipped-sampler-sheet.png` |
| the stacked-sampler sheet | — | the same 25 rows under the stacked sampler | task 1.1 | ⏳ generated during run | `.../stacked-sampler-sheet.png` |
| the correction-size table | — | mean correction size per training pair beside the best run's clean-seed count on that pair | task 2.1 | ⏳ generated during run | `.../correction-size-per-pair.png`, `.json` |

### Organization workflow

1. Both sheets, the size table and their sidecars land in one results grouping folder.
2. The card entry in that folder's `README.md` is written in the same act as the filing, never later.
3. The register slot is named in the card, or its absence is stated there.

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @...figures/05-the-comparison-sheet.md` | **task 0.1**, before any work | conformance and thin instructions reported |
| Cross-reference the plan | `/xref-pillar @...figures/05-the-comparison-sheet.md` | **task 0.2**, before any work | terms already documented elsewhere linked |
| Capture patterns | `/ingest-error-pattern --from-run-log` | **the close out**, after any red run | errors added to the catalogs |
| Bring the tree current | `/sync-plan-tree @...figures/05-the-comparison-sheet.md` | **the close out** | statuses, running order and Error Matrix match reality |
| Close the scope | `/recap-plan-tree @...08-improve-the-rank-32-pooled-adapter/MASTER_PLAN.md` | **the close out**, once every plan is ✅ | the scope's recall gallery, its URL recorded in the master plan |

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**File:** the strip builder from [plan 02](../tools/02-the-blind-label-pass.md)
**Relevant section:** the join, which already produces the label per tile. The sheet is a different layout over the same table, not a second labelling.

**File:** the repository's fixed correction-size measure
**Relevant section:** the size expression settled before any result was read, which this plan reuses rather than choosing a new one. A different size measure here would not be comparable with anything else in the tree.

**Convention:** `~/.claude/ARTIFACT_TREE_FORMAT.md` for the results grouping, the filename shape and the card entry.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

▶ `/design-figure` ✅ on the sheet layout, before task 1.1, since twenty-five rows by four columns needs a reachable axis and a decision about what goes in the appendix.

alt: `/pair-figure` ✅ for the size table, which is exactly a qualitative view beside its number and is the shape instruction 3.2 asks for.

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents)

None in this scope; this is its last plan. The close out's `/recap-plan-tree` publishes the scope's recall gallery, and whichever checkpoint won here becomes the one [scope 01's showcase figures](../../../01-showcase-the-trained-lora/MASTER_PLAN.md) are measured from.

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
