# 🧪 Review: does softness track lambda at a fixed checkpoint?

Nothing has run yet. This file judges [the design](../plans/07-experiment-c-lambda-window.md).
Run kind: hypothesis (an intervention on injection strength, no training). Its answer is the
reading key for experiments A and B.

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/07-experiment-c-lambda-window.md) | the grid, the softness reads, the strip |
| this file | the verdict and the per-λ numbers |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Runs](#runs)
- [The pre-registered bar](#the-pre-registered-bar)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [Still open](#still-open)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

- **λ**: the injection strength on the LoRA's predicted correction; 0 is plain PoE, 1 is the
  shipped configuration.
- **The drift read**: DINOv2/CLIP distance-to-mono minus distance-to-poe per render (plan 06's
  embedding read, computed offline here).

## Runs

Navigation: ⬅️ [Words](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#the-pre-registered-bar) ➡️

| Date | Run id | What ran | Wall time | Outcome |
|---|---|---|---|---|
| | | | | |

## The pre-registered bar

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

**This is the one question whose failure moves the plan.**

- [ ] ⚠️ **Does the sharpness proxy fall monotonically with λ on the tracked cells?** Bar,
  fixed before looking: a monotone decrease across the five λ values on the majority of cells
  supports the injection account; flat within the cells' own spread kills it. The per-λ table
  goes here.

## Written before the run, answered after

Navigation: ⬅️ [The bar](#the-pre-registered-bar) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

- [ ] ⚠️ Where along the grid does compose rate arrive (the dose story) relative to where blur
  arrives (the cost story), and do the two leave a usable middle λ?
- [ ] ⚠️ Did the λ=0 identity hold against the cached PoE render (mode-level, fp16 drift band)?

## Could the answer be an artefact

Navigation: ⬅️ [Before/after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- [ ] ⚠️ **Was the comparison fair?** Only λ varied; checkpoint, window, seeds, guidance fixed.
- [ ] ⚠️ **Was the instrument sound?** The sharpness proxy sanity-checked on a known-crisp and a
  known-soft render before use.
- [ ] ⚠️ **Did the run respect the environment?** In-session on a free device; outputs on
  `/datasets`.

## Still open

Navigation: ⬅️ [Artefact checks](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents)

Nothing open.
