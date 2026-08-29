# 🔬 Review: does the extended tracking set log everything, before anything launches?

Nothing has run yet. This file judges [the design](../plans/06-extend-the-tracking-set.md). Run
kind: instrument (a smoke proving the wiring; no science question of its own).

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/06-extend-the-tracking-set.md) | the four reads, the admission rule, the freeze |
| this file | the smoke verdict and the manifest hash |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Runs](#runs)
- [The pre-registered bar](#the-pre-registered-bar)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [Still open](#still-open)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

- **The tracking set**: the frozen list of cells and metrics rendered at every checkpoint of
  experiments A and B; frozen means the manifest hash in the W&B config never changes mid-run.
- **Teacher-forced**: computed on cached PoE states, immune to fp16 closed-loop drift.

## Runs

Navigation: ⬅️ [Words](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#the-pre-registered-bar) ➡️

| Date | Run id | What ran | Wall time | Outcome |
|---|---|---|---|---|
| | | | | |

## The pre-registered bar

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

**This is the one question whose failure blocks A and B.**

- [ ] ⚠️ **Do all seven metric families log at least one non-null value per eval step on the
  smoke?** Bar: yes for all seven (three inherited, four new), else no launch.

## Written before the run, answered after

Navigation: ⬅️ [The bar](#the-pre-registered-bar) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

- [ ] ⚠️ Did the eval pass's wall time grow, and by how much, against instrument-02's measured
  smoke band (the admission rule promises roughly zero growth)?
- [ ] ⚠️ Is the manifest hash visible in the run config, and does re-running startup reproduce it?

## Could the answer be an artefact

Navigation: ⬅️ [Before/after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- [ ] ⚠️ **Was the comparison fair?** The smoke used the frozen manifest, not a subset.
- [ ] ⚠️ **Was the instrument sound?** The inherited three curves still match instrument-02's
  healthy shapes.
- [ ] ⚠️ **Did the run respect the environment?** Free-device launch per protocol; outputs on
  `/datasets`.

## Still open

Navigation: ⬅️ [Artefact checks](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents)

Nothing open.
