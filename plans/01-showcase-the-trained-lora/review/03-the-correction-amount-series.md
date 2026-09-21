# 🧪 Review: does more of the LoRA's own output give more composition?

**The threshold question passed.** This file judges [the design](../plans/tests/03-the-correction-amount-series.md). Run
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
| 2026-09-01 | slurm job 48596, mscluster72 | first attempt: capture deltas (8 seeds) → sweep → score → softness. Launched at commit `32d1973`. | slurm | ~48 min | **invalidated**: `wrong_seed`/`shuffled` were routed through `run_constant_residual_inject`, which runs full PoE off-window every step, while `real` ran through the masked sampler, whose off-window steps are unconditional-only. The two conditions differed on two axes (injection source *and* off-window sampling), not one. AUC came back real 0.203, wrong_seed 0.734, shuffled 0.875 — controls trivially outcomposing real, tripping the fail criterion, but as a comparison-fairness artefact, not a result. |
| 2026-09-01 | slurm job 48619, mscluster72 | corrected rerun: same captured deltas, `wrong_seed`/`shuffled` now routed through `run_lora_residual_inject_masked`'s new `external_delta_by_step` parameter, so all three conditions share the identical masked sampler and differ only in where Δ̂ comes from. Sweep → score → softness. Launched at commit `c6a470b`. | slurm | ~29 min | **valid**: real AUC 0.203, wrong_seed AUC 0.031, shuffled AUC 0.0; see below. |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

**This is the one question whose failure moves the plan.**

- [x] ✅ **Does compose rate rise with λ on r̂ while both controls stay at what you would get by
  luck?** Yes. Per-λ compose rate (n=8 seeds per cell, λ = 0, 0.25, 0.5, 0.75, 1.0):
  - real: 0.0, 0.0, 0.125, 0.375, 0.625 — rises monotonically once λ exceeds 0.25.
  - wrong_seed: 0.0, 0.0, 0.125, 0.0, 0.0 — one isolated blip at λ=0.5 (1/8 seeds), no trend.
  - shuffled: 0.0, 0.0, 0.0, 0.0, 0.0 — flat at the floor throughout.

  AUC (0-1 scale): real 0.203125, wrong_seed 0.03125, shuffled 0.0. Real clears both controls
  by roughly 0.17-0.20 of AUC; neither control rises monotonically. Threshold met.

> What you would get by luck is what compose rate reads when nothing real is being injected.

## Written before the run, answered after

Navigation: ⬅️ [The bar](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

- [x] ✅ How does the LoRA's AUC sit beside the cached true correction's 0.387, on the same axes and the same runs? The LoRA's real-condition AUC (0.203) is about half the cached true correction's (0.387), on the same 0-to-1 scale but not the same runs (different checkpoints and injection sources): the deployed artifact composes, just less reliably than the correction it was trained to imitate.
- [x] ✅ Does softness track λ at the fixed checkpoint (experiment C's question)? Laplacian-variance sharpness (real condition, mean over 8 seeds per λ): 80.6, 70.5, 64.8, 27.6, 35.2 for λ = 0, 0.25, 0.5, 0.75, 1.0. Mostly decreasing (blurrier) as λ rises from 0 to 0.75, with a small uptick at λ=1.0. Descriptive only (labelled so in `softness_vs_lambda.json`), not a scorer contract, but the direction is consistent with "the blur is the injection, not undertraining" — this is plan 06's interpretation input, not settled here.
- [x] ✅ Was the dog × dog zero-interaction run (plan 02) consistent with the λ=0 end of this curve? Plan 02 is the interaction-term-zero pair (both experts prompted "a dog"), always run at λ=1 (`dog_x_dog_probe.py` has no λ parameter); it isn't a λ=0 point. What it does check: plan 02's own λ=1 result (5/8 seeds composed) exactly reproduces this series' real condition at λ=1.0 (5/8, compose rate 0.625) — the two runs share checkpoint, window and seeds, and land on the identical count.

## Could the answer be an artefact

Navigation: ⬅️ [Before/after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- [x] ✅ **Was the comparison fair?** Not on the first attempt (job 48596): `wrong_seed`/`shuffled`
  ran through a different sampler with different off-window behaviour than `real`, so the
  comparison mixed two axes. Fixed for job 48619 by routing all three conditions through the
  same masked sampler (`run_lora_residual_inject_masked`), differing only in the source of Δ̂
  (`external_delta_by_step`). Same pairs, seeds, window and guidance throughout; printed run
  counts per condition were non-zero and equal (40 each) in both attempts.
- [x] ✅ **Was the measuring tool sound?** Scorer validated (`scorer_validated.json`), same
  instance-count contract plan 02 already used. Not separately spot-checked here beyond the
  plan-02 cross-check above (identical λ=1 counts), which doubles as a scorer consistency check.
- [x] ✅ **Did the run respect the environment?** Outputs on `/datasets/mmolefe/poe_repair_min/outputs/showcase/lora_dose/`;
  disk guard checked that filesystem before both launches; launch mode recorded as Slurm (jobs
  48596, 48619) in the Runs table above.

## Still open

Navigation: ⬅️ [Artefact checks](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents)

Nothing open.
