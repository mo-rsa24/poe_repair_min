# 🧪 Review: does doubling the training length move anything?

Nothing has run yet. This file judges [the design](../plans/08-experiment-a-resume-to-200k.md).
Run kind: hypothesis (an experiment on the length axis with a null threshold fixed in advance).

> A null threshold is the number the result has to stay under to count as no effect: a held-out
> compose rate at 200k inside the seed-noise band means the numbers with the extra training look
> the same as the numbers without it.

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/08-experiment-a-resume-to-200k.md) | the resume, the frozen tracking set, the null threshold |
| this file | the verdict against that threshold |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [Still open](#still-open)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

- **The seed-noise band**: the spread of held-out [compose rate](../../../context/world/compose-rate.md)
  over the 8 held-out seeds at the 100k checkpoint; a 200k value inside it is a null.
- **The frozen tracking set**: plan 06's manifest; its crispness reads are the second half of
  the threshold.

> Held-out means the pairs were never shown during training, so the number says how well the
> adapter does on animals it has not seen.

## Runs

Navigation: ⬅️ [Words](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Date | Job id | Run id | What ran | Wall time | Outcome |
|---|---|---|---|---|---|
| | | | | | |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

**This is the one question whose failure moves the plan.**

- [ ] ⚠️ **At 200k, is held-out compose rate inside the 100k seed-noise band AND is every
  frozen crispness read unchanged?** The threshold, fixed at launch: both inside means length is
  not the knob and "train longer" closes with a number. Either outside means plan 01's read of
  where the curve stops rising gets revisited, and B's design may change before it launches.

## Written before the run, answered after

Navigation: ⬅️ [The bar](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

- [ ] ⚠️ Did the train loss keep its 6%-per-decade creep, and did anything downstream of it move?
- [ ] ⚠️ Do the tracking set's learned-vs-actual cosines change shape between 100k and 200k
  (which part of the trajectory keeps learning, if any)?

## Could the answer be an artefact

Navigation: ⬅️ [Before/after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- [ ] ⚠️ **Was the comparison fair?** Resume only; no other flag differs from the 100k config
  (diff the two W&B configs and say so).
- [ ] ⚠️ **Was the measuring tool sound?** The manifest hash identical at every logged step.
- [ ] ⚠️ **Did the run respect the environment?** sbatch on biggpu; checkpoints on `/datasets`;
  the disk guard's filesystem named in the log.

## Still open

Navigation: ⬅️ [Artefact checks](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents)

Nothing open.
