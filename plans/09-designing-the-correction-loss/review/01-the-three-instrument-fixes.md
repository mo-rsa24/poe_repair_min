# 🔬 Review: is the direction the loss cannot see actually carrying size?

Nothing has run. This file judges [the design](../plans/tools/01-the-three-instrument-fixes.md), which adds the undialled error to the trainer's log, pairs checkpoints with renders, and writes the objective flags into the run's config. Its first question is the cheapest falsification this scope can buy: if the invisible direction is negligible in a trained run, the case for plan 04 weakens before twenty hours are spent on it.

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```
(The smoke produces one short W&B run. For a failure worth keeping: `/ingest-error-pattern --from-run-log`.)

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/tools/01-the-three-instrument-fixes.md) | the three changes and the smoke that proves each one |
| **this file** | **the verdict: whether the undialled error carries real size, and whether each change does what its name claims** |

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

- **The dialled error**: what the loss minimises today, the guided composition against the guided cached target. `train/loss_fit`.
- **The undialled error**: the same comparison at guidance weight 1, so `a + b − u` against the raw cached joint prediction. New here.
- **The invisible direction**: a perturbation added to both the cat branch and the empty branch. The loss does not move; the picture moves by 6.5 times its size.
- **A silent no-op**: a change that runs, reports a plausible number, and did nothing. A `loss_undialled` that is identically zero looks like a result and is a bug.
- **The pairing**: a checkpoint written at the same epoch as a render set, so every picture in the repo has the weights that made it.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Builds a measuring tool** (an instrument: no result lands on any claim). It is judged by whether it can fail, not by what it found. The one exception is the ratio recorded in task 2.2, which is a measurement and is allowed to weaken the motivation for plan 04.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| the two-epoch smoke over all three changes | builds a measuring tool | | | a W&B run, a checkpoint directory, a `config.json` | not started |
| the offline recompute on `phase1_r32_100k` at 30,050 | builds a measuring tool | | | `undialled-vs-fit.json` | not started |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ **Is the undialled error a meaningful fraction of the fit loss on a trained run, or is it negligible?**
  - **The threshold, fixed before the measurement:** meaningful means the undialled error is at least 10% of the fit loss at the median denoising step of `phase1_r32_100k` at checkpoint 30,050. Below 10%, the invisible direction is small in practice whatever the algebra says, and plan 04's motivation is written down as weakened rather than quietly carried forward.
  - **This is the only question here whose failure may move a plan.** It cannot move plan 04's design, only the strength of the reason for running it.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ **Does `train/loss_undialled` appear on a live run, non-null and varying between steps?**
  - Identically zero is a bug, not a finding. The most likely cause is subtracting the guided target instead of the raw one.
- [ ] ⚠️ **Does a checkpoint directory exist beside every render directory the smoke writes?**
  - The whole point of the pairing is that a picture can be rescored later. One missing pair means the flag did not take.
- [ ] ⚠️ **Does the smoke's `config.json` name `train_step_range`?**
  - And `orth_weight` and `exclude_cells`. Three keys, all three present or the change is partial.
- [ ] ⚠️ **Does the undialled error change the gradient?**
  - It must not. It is computed under `no_grad` and never enters `loss`. A diagnostic that changes what the run learns is not a diagnostic.

## Asked after the result

Navigation: ⬅️ [Written before the run, answered after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Nothing yet.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?**
  - The undialled and dialled errors must be computed from the same forward pass on the same batch. Computing them on different steps would make their ratio meaningless.
- [ ] ⚠️ **Was the measuring tool sound?**
  - Both quantities are computed in fp32, per the environment's upcast rule. An fp16 path in either one destroys the subtraction before it is measured.
- [ ] ⚠️ **Did the run respect the environment?**
  - The smoke ran on a node pinned from a live idle probe, not from `sinfo`'s idle list; its checkpoints landed on `/datasets`; the disk guard named the filesystem it actually wrote to.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#what-the-run-cost-and-what-it-bought) ➡️

The ratio, with its units and its source file, goes into `00`'s report page as one line under the equation, because it is the number that says whether the whole variation programme is about something real.

## What the run cost, and what it bought

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

Not yet run.

## Still open

Navigation: ⬅️ [What the run cost](#what-the-run-cost-and-what-it-bought) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

Nothing open.

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

[02: the V0 foundation](../plans/reading/02-the-v0-foundation.md).
