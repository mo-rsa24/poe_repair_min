# 🧪 Review: do basins and a ridge exist for the composed flow?

**Nothing has run yet.** This file judges [the design](../plans/01-basins-by-hand.md); answers land
here and nowhere else. Questions below were written at design time, before any number existed.

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```
(For a run that failed and whose failure is worth keeping: `/ingest-error-pattern --from-run-log`.)

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/01-basins-by-hand.md) | the hypothesis, the thresholds, the code to write |
| **this file** | **the verdict: what the runs answered, and what they could not** |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The pre-registered bar](#the-pre-registered-bar)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

- **ending**: the final latent (and its decoded image) after finishing the remaining DDIM
  steps from a given state with the same composed epsilon the cache was made with.
- **nudge**: a random-direction perturbation of 1% of the state's norm.
- **the ridge**: where the nearest ending stops being unique, so a nudge decides the side.

> A **basin** is the field's term for the set of states that all flow to the same ending, and
> the ridge is the boundary between two of them.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Builds an instrument.** A missed threshold blocks every plan downstream: the scope's premise is that commitment is a measurable event, and this run is the premise's test.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-pre-registered-bar) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| | | | | | not started |

## The pre-registered bar

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ At step 40, do both nudged endings agree with the unnudged one, with both relative
      latent distances under `REL_ENDING_DIST_MAX` (the constant in
      `scripts/commitment/perturb_finish.py`), while the step-5 endings were free to differ?
      This is the deciding question because agreement late plus freedom early is exactly what
      basins with a ridge predict, and its failure kills the scope at the cost of one afternoon.

## Written before the run, answered after

Navigation: ⬅️ [The pre-registered bar](#the-pre-registered-bar) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ At step 25 (inside the measured divergence window, 18 to 36), do the three endings
      agree, differ, or split 2-to-1? Any of the three is informative: agreement says this
      pair-and-seed run decided before 25, a split says 25 sits near the ridge. Report it
      either way.
- [ ] ⚠️ How large are the step-5 ending differences relative to typical inter-mode distance?
      A tiny spread even at step 5 would say the flow is more contractive than the divergence
      measurement suggests, a tension worth recording rather than resolving here.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions the result itself raised. **Nothing here may ever become a bar**, because it was
written with the answer already visible.

(none yet)

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

Three fixed checks. Each is answered or explicitly marked not applicable; none is dropped.

- [ ] ⚠️ **Was the comparison fair?** Did exactly one axis differ between the sides being
      compared, and did the counts confirm it rather than the config claiming it?
- [ ] ⚠️ **Was the instrument sound?** Did the thing doing the measuring measure what its name
      says, over the data this run wrote and no other?
- [ ] ⚠️ **Did the run respect the environment?** Did every flag select a non-empty group, did
      the output land where the plan said, and did nothing silently fall back?

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| the composed flow has basins with an unstable ridge | one pair-and-seed run, three steps: the claim's population is that single run until plan 05 widens it |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

**Required, even when empty.** Nothing open.

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| | | |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Run [the design's tasks](../plans/01-basins-by-hand.md#tasks), then answer the questions above.
