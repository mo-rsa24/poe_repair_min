# 📚 Review: at matched steps, does rank change what the adapter reaches?

Nothing has run. This file judges [the design](../plans/reading/02-the-v0-foundation.md), which writes the six pillar files, the chart script and `00`'s page. Its bar was pre-registered elsewhere and never answered: `plans/01-showcase-the-trained-lora/review/09-experiment-b-rank-16-32.md` fixed the seed-noise band as the threshold before the rank 16 and rank 32 trainings ran, and the report published from those runs explicitly declined to answer it.

## Recommended prompt (when the run lands)

```
/report-pulse @report/designing-the-correction-loss/
```
(This plan produces no W&B run. For a failure worth keeping: `/ingest-error-pattern --from-run-log`.)

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/reading/02-the-v0-foundation.md) | the six pillar files, the collected numbers, the chart script, the page |
| **this file** | **the verdict: whether rank 16 and rank 32 sit inside rank 8's band at matched steps, and whether the eight existing figures reproduce** |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

- **The seed-noise band**: the spread of held-out compose rate across the eight held-out cat × dog seeds at one reference checkpoint. Its definition is fixed in `plans/01-showcase-the-trained-lora/review/09-experiment-b-rank-16-32.md` and is used as written.
- **Matched steps**: two ranks compared at the same optimizer step. Rank 8 at 30k against rank 16 at 50k is two axes and says nothing about rank.
- **Drift**: the DINOv2 distance from the corrected render to the joint-prompt render minus its distance to the plain product render. More negative is closer to the target.
- **Reproducible**: a figure that one command redraws from a file in the repository. The eight existing figures are not, because the script that drew them was kept in a session scratchpad.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Ablation over the rank axis** (may not change the claim; can simplify the method or bound it), on runs that already exist, plus a **reproduction** of eight figures whose provenance is currently unrecoverable. The rank question's threshold predates its runs, so it is a real pre-registration rather than one written with the answer in hand.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| collect the thirteen probe json and recompute the LoRA norms | reproduces | | | `00-per-checkpoint.json` | not started |
| compute the seed-noise band from the existing renders | ablation | | | the band, into the same file | not started |
| redraw the eight existing figures from the collected numbers | reproduces | | | eight png, compared against the originals | not started |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ **At matched 100k steps, do rank 16 and rank 32 sit inside rank 8's held-out seed-noise band?**
  - **The threshold, fixed at launch of those runs in `review/09-experiment-b-rank-16-32.md` of scope 01:** both inside is a capacity null. Either outside by more than the band reopens capacity as the lever.
  - **This is the only question here whose failure may move a plan.** A capacity null closes rank as a knob for this scope and for scope 08's rank choice. Capacity reopened would make rank a variable every variation has to control for, which changes what a fair comparison between objectives looks like.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ **Do the eight existing figures redraw from the collected numbers, matching what the two report files embed?**
  - A figure that does not reproduce is not silently replaced. Both versions are recorded and the discrepancy named, because the existing one is already cited.
- [ ] ⚠️ **Do the recomputed LoRA Frobenius norms match the prose in `does-training-longer-keep-improving-the-held-out-fix.md` lines 65 to 67?**
  - Those numbers exist nowhere else. If they do not match, one of the two is wrong and the prose is the one with no provenance.
- [ ] ⚠️ **Does the assembled five-across strip show the same renders as the saved three-panel strips it is built from?**
  - It drops repeated columns; it must not drop or reorder a render.
- [ ] ⚠️ **Does `00`'s page carry anything that a file it links to already says in full?**
  - The page is the template eight more pages copy. Restated content propagates.

## Asked after the result

Navigation: ⬅️ [Written before the run, answered after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Nothing yet.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?**
  - The rank comparison is at matched optimizer steps, on the same pool, the same held-out seeds and the same λ. The three "best" checkpoints the existing report names sit at three different steps and must not be used for this.
- [ ] ⚠️ **Was the measuring tool sound?**
  - The same scorer, the same DINOv2 weights and the same 50-step DDIM at eta 0 for every rank. Renders compared across GPU generations differ by about two of 255 in mean pixel value, which is below anything this question reads.
- [ ] ⚠️ **Did the run respect the environment?**
  - Cached tensors upcast to float32 before any analysis stacks them. The scoring pass ran on a node pinned from a live idle probe. Nothing was written to `/home-mscluster`.
- [ ] ⚠️ **Is the band computed from the definition already written, or a new one?**
  - Recomputing a threshold after seeing the data is how a null becomes a result. The definition in scope 01's review file is used verbatim.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#what-the-run-cost-and-what-it-bought) ➡️

`00`'s page owes the rank verdict in one line beside its rank table, the band drawn on the figure rather than described, and a sentence naming which of the eight existing figures reproduce and which do not.

`report/00-INDEX.md` owes two new rows.

## What the run cost, and what it bought

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

Not yet run.

## Still open

Navigation: ⬅️ [What the run cost](#what-the-run-cost-and-what-it-bought) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

Nothing open.

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

[03: the V0a read](../plans/hypothesis/03-the-v0a-read.md).
