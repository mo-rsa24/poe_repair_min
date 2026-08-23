# 📊 Review: do the transfer figures say what the evidence supports?

**Two figures are built and unjudged; four have not started.** This file holds the questions,
written before the figures are read. It judges
[../plans/figure-01-the-transfer-figures.md](../plans/figure-01-the-transfer-figures.md), whose
figures carry the transfer argument into the paper.

A figure plan is judged differently from a run. There is no metric to clear here: the question is
whether each figure reads cold, and whether what it claims sits at or below its row in
[../../../../../paper/iclr/figures.md](../../../../../paper/iclr/figures.md). A figure that
overclaims is worse than a missing one, because it survives review until someone checks.

## Recommended prompt (to judge a built figure)

```
/analyze-figure paper/iclr/figures/F8a-one-adapter-transfers.png
```
(For a figure that fails its read: `/design-figure` to redesign it, never a caption edit.)

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/figure-01-the-transfer-figures.md) | the six figures, their panels, and what each is for |
| **this file** | **the verdict: does each figure read cold and stay inside its register row** |
| [the register](../../../../../paper/iclr/figures.md) | the claim ceiling each caption may not exceed |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The pre-registered bar](#the-pre-registered-bar)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

- **Reads cold**: a person who was not in the room opens the PDF, reads only the figure and its
  axis labels, and can say in one sentence what it claims. If they need the caption to say it, the
  figure has not done its job.
- **Capped by the register**: each figure has a row in `paper/iclr/figures.md` holding one claim
  sentence. The caption may claim that or less, never more. Where the result supports more, the
  row gets widened first, on evidence.
- **The caption caps**: the sentences on a register row naming what the figure cannot be read to
  say. F8a and F8b each carry four.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Figure run.** Presents settled results only. It may not create a claim, and a failure here sends
a figure back to `/design-figure` rather than closing the plan.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-pre-registered-bar) ➡️

| Figure | Kind | Built | Cost | Output | State |
|---|---|---|---|---|---|
| F8a one adapter transfers | Figure run | commit 252441b | no GPU, drawn from scored results | `paper/iclr/figures/F8a-one-adapter-transfers.{png,pdf,json}` | built, unjudged |
| F8b adapter against the oracle | Figure run | commit 252441b | no GPU, drawn from scored results | `paper/iclr/figures/F8b-adapter-against-the-oracle.{png,pdf,json}` | built, unjudged |
| A2 delivery-live | Figure run | | | | not started |
| A3 transfer | Figure run | | | | blocked on step 11 |
| A4 delivery against transfer | Figure run | | | | blocked on step 11 |
| A5 pool contrast | Figure run | | | | blocked on step 12 |

## The pre-registered bar

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ Does every built figure read cold, and does its claim sit at or below its register row?
      This is the bar because the figures are what the paper argues from. A figure whose claim
      outruns its row puts a sentence in the manuscript the evidence does not carry, and the
      register is the only place that ceiling is written down.

## Written before the run, answered after

Navigation: ⬅️ [The pre-registered bar](#the-pre-registered-bar) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

**The run here is the reading of the figures, not their building.** F8a and F8b were already built
when these questions were written, so this section pre-registers the read and not the build. The
last question makes that gap explicit rather than hiding it.

- [ ] ⚠️ **F8a**: does the two-panel figure show that the aggregate is not carried by one easy
      pair? The unpooled second panel exists for exactly this, so if the per-pair curves do not
      make it visible, the panel has failed and the aggregate claim is unsupported.

- [ ] ⚠️ **F8a**: is the step-60000 read stated on the figure itself, not only in the caption?
      Training ran to 100000 and only checkpoints to 60000 were scored. A reader who takes the
      figure as the final word is reading something the data does not say.

- [ ] ⚠️ **F8b**: is the λ=1 marker legible as the target rather than as a competing method? It
      reproduces the joint render by construction, so a reader who mistakes it for a baseline
      draws the wrong conclusion about what the adapter is beating.

- [ ] ⚠️ **F8b**: does the figure make its pair-matched, not cell-matched, sampling visible? The
      oracle arm and the adapter arm ran different numbers of cells per pair, so the pair is the
      sampling unit. If the figure implies otherwise, the comparison looks tighter than it is.

- [ ] ⚠️ **F8a and F8b**: is the control pair marked on the figure? Elephant × penguin composes
      without any intervention, so its row is not evidence of transfer, and a reader who counts it
      as one overcounts the result.

- [ ] ⚠️ **A2**: does the plateau reference line read as an observation rather than a target? It
      is where corrections have tended to stall, not where they are supposed to land.

- [ ] ⚠️ **A3**: is the do-no-harm baseline drawn as a band rather than a line? A single line
      claims a precision the control pairs do not have.

- [ ] ⚠️ **A4**: can a reader see delivery and transfer come apart? The twin panels exist so a
      floor compose-rate can be split into "the correction never arrived" and "it arrived pointing
      wrong". If the panels do not make that split visible, the figure has not earned its place.

- [ ] ⚠️ **A5**: is the same-pair pairing drawn explicitly rather than implied by bar position?
      The contrast is paired, and a reader who reads it as two independent means reads a weaker
      and different result.

- [ ] ⚠️ **All six**: was each designed through `/design-figure` before it was built, with a design
      note to show for it? F8a and F8b were built before this plan carried the requirement, so the
      honest answer for those two may be no; record it either way rather than backfilling a note.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions the figures themselves raised once read. **Nothing here may ever become a bar**, because
it was written with the answer already visible. Nothing yet: neither built figure has been read.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

For a figure run these ask whether the picture could mislead even when the underlying result is
sound. That is the failure mode a figure has and a run does not.

- [ ] ⚠️ **Was the comparison fair?** Do the arms drawn side by side differ in one thing only, and
      does the figure show the sampling unit it actually used? F8b is the live risk: pair-matched
      arms drawn as though cell-matched read tighter than they are.
- [ ] ⚠️ **Was the instrument sound?** Does each figure's `.json` sidecar carry the numbers the
      panel draws, from the scored run and not a stale copy? A figure regenerated from an old
      sidecar is the quiet version of this failure.
- [ ] ⚠️ **Did the run respect the environment?** Both `.png` and `.pdf` present at the register
      path, and the build reproducible without a system LaTeX, which this cluster does not have.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| F8a: one adapter transfers | the step the read came from (60000), on the figure and not only in the caption |
| F8a and F8b | that elephant × penguin composes without intervention, so its row is not evidence of transfer |
| F8b: the adapter against the oracle | that λ=1 reproduces the joint render by construction, so it is the target and not a competing method |
| F8b | that the sampling unit is the pair, not the cell |
| all six | whether each went through `/design-figure` first. For F8a and F8b the honest answer may be no, and it is recorded rather than backfilled |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| whether F8a and F8b read cold and stay inside their register rows | reading both against `paper/iclr/figures.md` | the transfer paragraphs in the manuscript, which argue from these two |
| A3 and A4 | step 11 landing | the delivery-against-transfer argument |
| A5 | step 12 landing | the pool-contrast argument, which is what [baseline-01](baseline-01-the-size-matched-control-pool.md) supplies |
| A2 | nothing external; it has not been started | the delivery-live figure |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Read F8a and F8b against their register rows and answer the bar. Both are built and neither has
been judged, so this is the cheapest unblocked work in the file.
