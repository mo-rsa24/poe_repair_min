# 🔬 Review: can the read fail, and is the blinding a property rather than a promise?

Nothing has run. This file judges [the design](../plans/tools/02-the-blind-label-pass.md), which builds the strips, the blinding, the three-label collector, the join, and the two automatic reads that sit beside the labels. Its answers decide whether any verdict in this scope can be believed, because every verdict here is a person's judgement of a picture.

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```
(For a failure worth keeping: `/ingest-error-pattern --from-run-log`.)

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/tools/02-the-blind-label-pass.md) | the strips, the blinding, the three labels, the two secondary reads |
| **this file** | **the verdict: whether the tool separates settled cases, and whether a labeller can tell which tile is which** |
| [procedure](../procedures/tools-02-run-the-blind-label-pass.md) | the steps a person follows to take the labels |

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

- **The three labels**: **clean** is both named animals present, each clearly itself, nothing unnatural on inspection. **unclear** is two animals but not clearly the two named ones, or clearly both with unnatural details. **not two** is one animal, a single blended creature, or the same animal twice.
- **Structural blinding**: the labelling step can only see tile ids, and the file mapping ids to conditions is written elsewhere and opened only by a later join. The alternative, a person deciding not to look, is not a control.
- **The known-example set**: renders whose label is already settled, drawn from existing outputs, with their expected labels written down before the smoke runs.
- **The two secondary reads**: the validated detector's count of animal-shaped regions, and the forced-choice check of which of the pair's two names each region matches. Both order the strips and neither decides a label.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Builds a measuring tool.** Judged by whether it can fail. A failed bar stops the scope: every verdict downstream is a label taken with this tool, so a tool that cannot separate settled cases makes those verdicts unreadable rather than merely uncertain.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| the known-example smoke | builds a measuring tool | | | `read/smoke/` | not started |
| one seed labelled by hand, to check the blinding | builds a measuring tool | | | `read/labels.json` | not started |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ **Does the read return the already-settled label on every known example?**
      This is the bar because an instrument that cannot fail cannot be trusted when it succeeds. The set holds at minimum one blended single creature (expected **not two**), one render showing the same animal twice (expected **not two**), and one joint-target render of a seed whose target is right (expected **clean**), each expectation written down before the smoke runs. **Pass** if every expected label is matched. **Fail** if any is not: the three labels as written do not separate the cases they were written to separate, and their definitions are fixed before any run is judged rather than after.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ **Can a labeller identify a tile's condition without the mapping?**
      Answered by labelling one seed by hand and saying so plainly. The ways it fails are a filename surviving into a caption, a condition sitting in the same position on every row, and a visible watermark. Any of them means the blinding is a promise rather than a property, and no label taken with it counts.

- [ ] ⚠️ **Was the mapping file untouched during labelling?**
      Its modification time against the labelling session's start. This is what makes the blinding checkable afterwards by someone who was not in the room, which is the whole reason the mapping is a separate file.

- [ ] ⚠️ **How often is the boundary between unclear and not two hard to call?**
      Recorded as a count of tiles the labeller hesitated on, out of the seed labelled. That boundary is the one most likely to move a verdict, because a run gains no credit for turning a **not two** into an **unclear**. A high hesitation rate is not a failure; an unrecorded one leaves a later reader unable to weigh the verdicts.

- [ ] ⚠️ **Does either secondary read predict the label closely?**
      Report either way. A read that tracks the labels is one a future reader will be tempted to substitute for them, which would reintroduce exactly the blindness this scope is routing around. If it does track them, the fact is written into the card of anything built on these labels.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

*(none yet: nothing has run)*

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** The known examples must be labelled through the same path a real tile takes: same strip layout, same shuffle, same collector. An example judged in isolation proves nothing about the pass.
- [ ] ⚠️ **Was the measuring tool sound?** The count and both-names reads must run over the tiles this pass wrote and no others, and the expected labels must have been recorded before the smoke, provable from the file's own history.
- [ ] ⚠️ **Did the run respect the environment?** References were rendered with no adapter attached, since the windowed sampler leaves it enabled outside its window; the strips landed on `/datasets`; the CPU DINOv2 path ran with its required setting rather than silently falling back.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| any clean-seed count | that the labels are one person's blind read, and the hesitation rate on the unclear-against-not-two boundary |
| any comparison of two runs by clean seeds | that the count scorer disagrees, and calls most of the unclean renders composed |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| whether one labeller is enough, or whether a second reader would disagree often enough to matter | a second person labelling one run's set and the two being compared | nothing yet; it becomes real if a verdict turns on one or two seeds |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Run [the design's task 1.1](../plans/tools/02-the-blind-label-pass.md#1--gather-the-references-before-any-adapter-is-attached), the reference inventory.
