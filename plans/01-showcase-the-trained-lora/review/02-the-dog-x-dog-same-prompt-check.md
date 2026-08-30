# 🧪 Review: does the LoRA leave an agreeing pair alone?

Nothing has run yet. This file judges [the design](../plans/tests/02-the-dog-x-dog-same-prompt-check.md). Run
kind: hypothesis (a pre-registered same-prompt check). Its verdict feeds the "learned a rule,
not a vector" caption used across the showcase figures.

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/tests/02-the-dog-x-dog-same-prompt-check.md) | what runs, the identity check, the outcomes |
| this file | the verdict and the counts that decided it |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [Still open](#still-open)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

- **Agreeing pair**: both PoE experts get the same concept ("a dog"), so the true
  [interaction term](../../../context/world/interaction-term.md) is near zero.
- **‖r̂‖**: the norm of the LoRA's predicted correction at a step; "small" means small against
  the cross-pair scale (`train/delta_target_norm` ≈ 29.7 in the training history).
- **The identity check**: with the LoRA off, PoE(A,A) must reduce to Mono(A) within fp16
  drift, or the runner itself is broken.

## Runs

Navigation: ⬅️ [Words](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Date | Run id | What ran | Wall time | Outcome |
|---|---|---|---|---|
| | | | | |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

**This is the one question whose failure moves the plan.**

- [ ] ⚠️ **Did any dog × dog run render two dogs?** The threshold, fixed before looking: zero
  runs with instance count ≥ 2 supports the rule story; any run with two dogs falsifies "learned
  a rule" as stated and the caption is rewritten, not defended. One dog with per-step ‖r̂‖ large
  on the 29.7 scale is 🟡 inconclusive, recorded as such.

## Written before the run, answered after

Navigation: ⬅️ [The bar](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

- [ ] ⚠️ Did the identity check hold (max-abs pixel difference within the fp16 drift band of the
  λ=0 check that must pass before anything runs)?
- [ ] ⚠️ Is ‖r̂‖ on the agreeing pair small at every step of the window, and what fraction of
  the cross-pair scale is its maximum?
- [ ] ⚠️ Does language space agree (the L1 additivity gap for "a dog"+"a dog" near zero), as the
  ledger predicts?

## Could the answer be an artefact

Navigation: ⬅️ [Before/after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- [ ] ⚠️ **Was the comparison fair?** Same seeds, same sampler settings, same window as the
  cross-pair runs; only the concepts changed.
- [ ] ⚠️ **Was the measuring tool sound?** The scorer is validated for two-animal scenes; a
  same-species pair is inside its validation only if the count read holds; spot-check five
  renders by eye against the counts.
- [ ] ⚠️ **Did the run respect the environment?** Outputs on `/datasets`; the disk guard checked
  the filesystem written to; guidance stayed 7.5.

## Still open

Navigation: ⬅️ [Artefact checks](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents)

Nothing open.
