# 🧪 Review: does more of the LoRA's own output give more composition?

Nothing has run yet. This file judges [the design](../plans/tests/03-the-correction-amount-series.md). Run
kind: hypothesis (more correction, more composition, with matched controls). Its figure is what
the showcase's causal claim rests on.

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/tests/03-the-correction-amount-series.md) | the grid, the controls, the schema to mirror |
| this file | the verdict, AUC per condition, launch mode, wall time |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [Still open](#still-open)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

- **r̂**: the LoRA's predicted correction, injected at strength λ over steps 0-10.
- **AUC**: area under the curve of [compose rate](../../../context/world/compose-rate.md)
  against λ, on a 0-to-1 scale; 1.0 would mean every generation composed at every λ. The cached
  true correction's reference values are 0.387 (real correction) against 0.023 (random), from
  `dose_curves.json`.
- **Wrong-seed / shuffled controls**: r̂ taken from another seed's states, and r̂ with its
  step-assignment shuffled; both are size-matched and should do nothing.

## Runs

Navigation: ⬅️ [Words](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Date | Run id | What ran | Launch mode | Wall time | Outcome |
|---|---|---|---|---|---|
| | | | | | |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

**This is the one question whose failure moves the plan.**

- [ ] ⚠️ **Does compose rate rise with λ on r̂ while both controls stay at what you would get by
  luck?** The
  threshold, fixed before looking: the real-r̂ AUC exceeds both control AUCs by more than the
  spread implied by the run counts per condition (binomial), and no control's curve rises
  monotonically. The AUCs and counts per condition go here.

> What you would get by luck is what compose rate reads when nothing real is being injected.

## Written before the run, answered after

Navigation: ⬅️ [The bar](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

- [ ] ⚠️ How does the LoRA's AUC sit beside the cached true correction's 0.387, on the same axes and the same runs?
- [ ] ⚠️ Does softness track λ at the fixed checkpoint (experiment C's question)? If yes, the
  blur is the injection, not undertraining; this answer is plan 06's interpretation input.
- [ ] ⚠️ Was the dog × dog zero-interaction run (plan 02) consistent with the λ=0 end of this curve?

## Could the answer be an artefact

Navigation: ⬅️ [Before/after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- [ ] ⚠️ **Was the comparison fair?** Exactly one thing differs between conditions: same pairs,
  seeds, window and guidance throughout; the printed run counts per condition were non-zero and
  equal where intended.
- [ ] ⚠️ **Was the measuring tool sound?** Scorer validated (`scorer_validated.json`); spot-check
  renders against counts at λ=0 and λ=1.
- [ ] ⚠️ **Did the run respect the environment?** Outputs on `/datasets`; disk guard on the
  written filesystem; launch mode recorded (Slurm vs nohup, since `squeue` is blind to nohup).

## Still open

Navigation: ⬅️ [Artefact checks](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents)

Nothing open.
