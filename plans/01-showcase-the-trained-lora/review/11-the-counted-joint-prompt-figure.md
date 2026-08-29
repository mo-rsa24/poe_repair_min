# 📊 Review: what is the joint prompt's own compose rate?

Nothing has run yet. This file judges [the design](../plans/11-the-counted-joint-prompt-figure.md).
Run kind: measurement (scoring renders that already exist).

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/11-the-counted-joint-prompt-figure.md) | the scoring pass and the three-bar figure |
| this file | the measured baseline and the spot-check verdict |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Runs](#runs)
- [The pre-registered bar](#the-pre-registered-bar)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [Still open](#still-open)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

- **Mono / the joint prompt**: the model asked for "a cat and a dog" in one prompt; the target
  the LoRA is trained toward, itself imperfect.
- **The pool**: the 8 held-out pairs × 8 seeds whose `mono.png` renders sit in the training cache.

## Runs

Navigation: ⬅️ [Words](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#the-pre-registered-bar) ➡️

| Date | What ran | Cells scored | Wall time | Outcome |
|---|---|---|---|---|
| | | | | |

## The pre-registered bar

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

**The question this measurement exists to answer** (a measurement's bar is completeness, not a
threshold):

- [ ] ⚠️ **Is every cell of the pool scored, and does the eye agree with the scorer on the
  ten-render spot-check?** The per-pair baseline table goes here.

## Written before the run, answered after

Navigation: ⬅️ [The bar](#the-pre-registered-bar) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

- [ ] ⚠️ On which pairs does the joint prompt itself fail, and does the LoRA restore any of
  those exact cells (the "restores what the target loses" sentence needs named cells)?
- [ ] ⚠️ Does `an_elephant__x__a_penguin` compose by default? Carry the answer to the
  [clean-pair-pool review's open question](../../04-does-the-fix-reach-unseen-pairs/review/instrument-01-the-clean-pair-pool.md).

## Could the answer be an artefact

Navigation: ⬅️ [Before/after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- [ ] ⚠️ **Was the comparison fair?** The three bars use the same cells and seeds; any missing
  cell in one arm is named, not silently dropped.
- [ ] ⚠️ **Was the instrument sound?** The scorer is validated on two-animal scenes; mono
  renders are inside that class; the spot-check is the check.
- [ ] ⚠️ **Did the run respect the environment?** In-session; outputs to `/datasets`.

## Still open

Navigation: ⬅️ [Artefact checks](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents)

Nothing open.
