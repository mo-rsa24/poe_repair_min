# 🧪 Review: does the posterior mean settle at or before the paths visibly separate?

**Nothing has run yet.** This file judges [the design](../plans/tests/02-the-free-test.md); answers land
here and nowhere else. Questions below were written at design time, before any number existed.

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```
(For a run that failed and whose failure is worth keeping: `/ingest-error-pattern --from-run-log`.)

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/tests/02-the-free-test.md) | the hypothesis, the thresholds, the code to write |
| **this file** | **the verdict: what the runs answered, and what they could not** |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

- **posterior-mean drift**: how far Tweedie's running estimate of the final image moves
  between consecutive steps; it settles when the posterior mass has concentrated on one basin.
- **settling step**: the first step after which drift stays under its threshold for all
  remaining steps.
- **divergence step**: where this pair-and-seed run's paths visibly separate, read from the
  existing trajectory-divergence analyses (18 to 36 across runs).

> **Tweedie** is the formula that turns a noisy state and the model's noise prediction into a
> running guess at the final clean image. A **basin** is the set of states that all flow to the
> same ending. **Speciation** is the field's word for the step at which the outcome stops being
> undecided.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim.** A missed threshold does not close the scope; it kills the reading that the run decides early and then only descends, before plan 05 spends GPU on it, which is this run's whole value.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| | | | | | not started |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ In every clean pair-and-seed run (drift defined and settling under
      `DRIFT_SETTLED_MAX`), does the settling step land at or before that run's divergence step
      (`ORDERING_HOLDS_FRAC_MIN = 1.0` in `scripts/commitment/posterior_drift.py`)?
      This is the deciding question because the ledger pre-registered the ordering as the
      story's first falsifiable consequence.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ Are per-step epsilons cached? Answer with the tensors found in one pair-and-seed run,
      and the measured recompute time for that one run if they are not.
- [ ] ⚠️ Where do the settling steps cluster: near 10, or inside 18 to 36? Informative here;
      it is plan 05's deciding question.
- [ ] ⚠️ How many pair-and-seed runs are 🟡 (drift never settles under the threshold)? Each is
      listed, none is forced into a verdict.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions the result itself raised. **Nothing here may ever become the question above**, because it was
written with the answer already visible.

(none yet)

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

Three fixed checks. Each is answered or explicitly marked not applicable; none is dropped.

- [ ] ⚠️ **Was the comparison fair?** Did exactly one axis differ between the sides being
      compared, and did the counts confirm it rather than the config claiming it?
- [ ] ⚠️ **Was the measuring tool sound?** Did the thing doing the measuring measure what its name
      says, over the data this run wrote and no other?
- [ ] ⚠️ **Did the run respect the environment?** Did every flag select a non-empty group, did
      the output land where the plan said, and did nothing silently fall back?

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| settling reads the decision, divergence reads the display | the two are cousins, not the same theorem; the lag column is reported per pair-and-seed run, never averaged away |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

**Required, even when empty.** Nothing open.

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| | | |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Run [the design's tasks](../plans/tests/02-the-free-test.md#tasks), then answer the questions above.
