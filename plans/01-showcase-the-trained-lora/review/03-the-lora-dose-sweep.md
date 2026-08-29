# 🧪 Review: does the LoRA's own output cause composition, dose for dose?

Nothing has run yet. This file judges [the design](../plans/03-the-lora-dose-sweep.md). Run
kind: hypothesis (a dose-response with matched controls). Its figure is the causal spine of the
showcase set.

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/03-the-lora-dose-sweep.md) | the grid, the controls, the schema to mirror |
| this file | the verdict, per-arm AUC, launch mode, wall time |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Runs](#runs)
- [The pre-registered bar](#the-pre-registered-bar)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [Still open](#still-open)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

- **r̂**: the LoRA's predicted correction, injected at strength λ over steps 0-10.
- **AUC**: area under the compose-rate-vs-λ curve, 0 to 1; the oracle's reference values are
  0.387 (real correction) against 0.023 (random), from `dose_curves.json`.
- **Wrong-seed / shuffled controls**: r̂ taken from another seed's states, and r̂ with its
  step-assignment shuffled; both are size-matched and should do nothing.

## Runs

Navigation: ⬅️ [Words](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#the-pre-registered-bar) ➡️

| Date | Run id | What ran | Launch mode | Wall time | Outcome |
|---|---|---|---|---|---|
| | | | | | |

## The pre-registered bar

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

**This is the one question whose failure moves the plan.**

- [ ] ⚠️ **Does compose rate rise with λ on r̂ while both controls stay at the floor?** Bar,
  fixed before looking: the real-r̂ AUC exceeds both control AUCs by more than the spread
  implied by per-arm cell counts (binomial), and no control's curve rises monotonically. The
  per-arm AUCs and counts go here.

## Written before the run, answered after

Navigation: ⬅️ [The bar](#the-pre-registered-bar) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

- [ ] ⚠️ How does the LoRA's AUC sit beside the oracle's 0.387, on the same axes and cells?
- [ ] ⚠️ Does softness track λ at the fixed checkpoint (experiment C's question)? If yes, the
  blur is the injection, not undertraining; this answer is plan 06's interpretation input.
- [ ] ⚠️ Was the dog × dog zero cell (plan 02) consistent with the λ-0 end of this curve?

## Could the answer be an artefact

Navigation: ⬅️ [Before/after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- [ ] ⚠️ **Was the comparison fair?** One axis differs per arm: same pairs, seeds, window,
  guidance across arms; the printed per-arm counts were non-zero and equal where intended.
- [ ] ⚠️ **Was the instrument sound?** Scorer validated (`scorer_validated.json`); spot-check
  renders against counts at λ=0 and λ=1.
- [ ] ⚠️ **Did the run respect the environment?** Outputs on `/datasets`; disk guard on the
  written filesystem; launch mode recorded (Slurm vs nohup, since `squeue` is blind to nohup).

## Still open

Navigation: ⬅️ [Artefact checks](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents)

Nothing open.
