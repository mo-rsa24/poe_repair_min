# 🧪 Review: does doubling the training length move anything?

Nothing has run yet. This file judges [the design](../plans/08-experiment-a-resume-to-200k.md).
Run kind: hypothesis (a null-bar experiment on the length axis).

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/08-experiment-a-resume-to-200k.md) | the resume, the frozen instrument, the null bar |
| this file | the verdict against that bar |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Runs](#runs)
- [The pre-registered bar](#the-pre-registered-bar)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [Still open](#still-open)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

- **The seed-noise band**: the spread of held-out compose rate over the 8 held-out seeds at the
  100k checkpoint; a 200k value inside it is a null.
- **The frozen tracking set**: plan 06's manifest; its crispness reads are the second half of
  the bar.

## Runs

Navigation: ⬅️ [Words](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#the-pre-registered-bar) ➡️

| Date | Job id | Run id | What ran | Wall time | Outcome |
|---|---|---|---|---|---|
| | | | | | |

## The pre-registered bar

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

**This is the one question whose failure moves the plan.**

- [ ] ⚠️ **At 200k, is held-out compose rate inside the 100k seed-noise band AND is every
  frozen crispness read unchanged?** Bar fixed at launch: both inside means length is not the
  knob and "train longer" closes with a number. Either outside means the plateau read (plan 01)
  is revisited and B's design may change before it launches.

## Written before the run, answered after

Navigation: ⬅️ [The bar](#the-pre-registered-bar) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

- [ ] ⚠️ Did the train loss keep its 6%-per-decade creep, and did anything downstream of it move?
- [ ] ⚠️ Do the tracking set's learned-vs-actual cosines change shape between 100k and 200k
  (which part of the trajectory keeps learning, if any)?

## Could the answer be an artefact

Navigation: ⬅️ [Before/after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- [ ] ⚠️ **Was the comparison fair?** Resume only; no other flag differs from the 100k config
  (diff the two W&B configs and say so).
- [ ] ⚠️ **Was the instrument sound?** The manifest hash identical at every logged step.
- [ ] ⚠️ **Did the run respect the environment?** sbatch on biggpu; checkpoints on `/datasets`;
  the disk guard's filesystem named in the log.

## Still open

Navigation: ⬅️ [Artefact checks](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents)

Nothing open.
