# 🔬 Review: can the scorer be trusted off animals?

Nothing has run yet. This file judges [the design](../plans/13-revalidate-the-scorer-off-animals.md).
Run kind: instrument validation (labels first, bar in code, then the reading).

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/13-revalidate-the-scorer-off-animals.md) | the definitions, the mini-set, the bar |
| this file | agreement per kind and the tier-three verdict |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Runs](#runs)
- [The pre-registered bar](#the-pre-registered-bar)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [Still open](#still-open)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

- **Kind**: object × style versus object × scene; each gets its own pass definition and its own
  verdict.
- **Tier three**: the population ladder's widest caption, "SDXL composition generally"; it opens
  per kind, only on a met bar.

## Runs

Navigation: ⬅️ [Words](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#the-pre-registered-bar) ➡️

| Date | What ran | Renders labelled | Wall time | Outcome |
|---|---|---|---|---|
| | | | | |

## The pre-registered bar

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

**This is the one question whose failure moves the plan.**

- [ ] ⚠️ **Does scorer-vs-eye agreement meet the in-code bar per kind?** The bar lives in
  `scripts/showcase/offanimal_validation.py` and is quoted here verbatim at judgment time,
  with the agreement rates and the ambiguous-case count beside it.

## Written before the run, answered after

Navigation: ⬅️ [The bar](#the-pre-registered-bar) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

- [ ] ⚠️ Which kind fails harder, object × style or object × scene, and does the failure mode
  suggest a different read (a style is not an instance; was the definition or the scorer at
  fault)?
- [ ] ⚠️ What scorer configuration was used off animals, verbatim?

## Could the answer be an artefact

Navigation: ⬅️ [Before/after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- [ ] ⚠️ **Was the comparison fair?** Labels done blind to scorer output; ambiguous renders
  excluded from the bar and counted.
- [ ] ⚠️ **Was the instrument sound?** The original two-animal validation still reproduces on a
  spot-check (the scorer itself did not drift).
- [ ] ⚠️ **Did the run respect the environment?** In-session; outputs to `/datasets`.

## Still open

Navigation: ⬅️ [Artefact checks](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents)

Nothing open.
