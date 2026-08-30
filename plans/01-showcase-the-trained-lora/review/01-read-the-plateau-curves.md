# 📊 Review: is the LoRA at a ceiling, or was it still climbing at 100k?

Nothing has run yet. This file judges [the design](../plans/01-read-the-plateau-curves.md); its
answer is the interpretation key for experiments A and B in
[08-experiment-a-resume-to-200k](../plans/08-experiment-a-resume-to-200k.md) and
[09-experiment-b-rank-16-32](../plans/09-experiment-b-rank-16-32.md). Run kind:
measurement (a read of already-logged curves; no new training).

## Recommended prompt (when the read lands)

```
/analyze-run <run id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/01-read-the-plateau-curves.md) | what to read, where, and the two verdicts |
| this file | the verdict and the numbers that decided it |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [Still open](#still-open)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

- **`eval/frac_distance_reached`**: how much of the gap between the plain PoE prediction and the
  joint-prompt prediction the LoRA's correction closes, 0 to 1, logged during training.
- **Ceiling**: the curve is flat over the last half of training; more steps buy nothing.
- **Waypoint**: the curve is still rising at 100k; more steps plausibly buy more.

## Runs

Navigation: ⬅️ [Words](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Date | Run id | What ran | Wall time | Outcome |
|---|---|---|---|---|
| | | | | |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

**This is the one question whose answer frames plan 06.**

- [ ] ⚠️ **Is the last-half slope of `eval/frac_distance_reached` consistent with zero?** The
  threshold, fixed before looking: over steps 50k to 100k, a linear fit whose slope is smaller in magnitude
  than the curve's own per-eval-point noise (the standard deviation of residuals) means ceiling;
  a slope larger than that noise, and positive, means waypoint. The slope, the noise, and the
  verdict go here.

## Written before the run, answered after

Navigation: ⬅️ [The bar](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

- [ ] ⚠️ Do the W&B curve and the local `history.json` extraction agree at every checkpoint, to
  within logging precision?
- [ ] ⚠️ Does held-out [compose rate](../../../context/world/compose-rate.md) (0.961 at 50k and
  60k, unscored beyond) tell the same story as `frac_distance_reached`, or do the two metrics
  diverge late?

> Held-out means the pairs were never shown during training, so the number says how well the
> adapter does on animals it has not seen.

## Could the answer be an artefact

Navigation: ⬅️ [Before/after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- [ ] ⚠️ **Was the comparison fair?** The metric averages over the eval runs; confirm the set of
  runs did not change across training (same pairs, same seeds at every eval step).
- [ ] ⚠️ **Was the measuring tool sound?** `eval/frac_distance_reached` came from the wiring built in
  `instrument-02`; the verdict on its first short run is in
  [that review file](../../04-does-the-fix-reach-unseen-pairs/review/instrument-02-three-live-curves-while-training.md).
- [ ] ⚠️ **Did the run respect the environment?** Not applicable: nothing runs; the read is of
  logs already on disk.

## Still open

Navigation: ⬅️ [Artefact checks](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents)

Nothing open.
