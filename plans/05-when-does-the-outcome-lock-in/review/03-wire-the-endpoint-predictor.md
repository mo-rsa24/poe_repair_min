# 🧪 Review: does the wired endpoint predictor reproduce one teacher ending?

**Nothing has run yet.** This file judges [the design](../plans/tools/03-wire-the-endpoint-predictor.md); answers land
here and nowhere else. Questions below were written at design time, before any number existed.

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```
(For a run that failed and whose failure is worth keeping: `/ingest-error-pattern --from-run-log`.)

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/tools/03-wire-the-endpoint-predictor.md) | the hypothesis, the thresholds, the code to write |
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

- **the endpoint predictor**: LCM-SDXL behind the adapter, jumping a cached state to its
  settled frame.
- **the teacher**: SDXL itself finishing the run from the same state, the ground truth.
- **the three asserts**: guidance enters as an embedding (`time_cond_proj_dim` present), no
  negative-prompt double pass, exact timestep mapping printed once.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Builds a measuring tool.** A missed threshold blocks plans 04 and 05: a failed check means the adapter, not the science, is wrong.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| | | | | | not started |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ On the one state checked (plan 01's chosen pair-and-seed run, step 25, joint prompt),
      does the predictor's frame match the teacher's ending on scorer verdict
      (`SMOKE_SCORER_MATCH`) with DINOv2 distance under `SMOKE_DINO_DIST_MAX` (both in
      `scripts/commitment/lcm_smoke.py`), with all three adapter asserts holding? This is the
      deciding question because a parameterisation mistake here fails silently as plausible
      frames everywhere downstream.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ Where did the checkpoint land and how big is it? Path, filesystem, size on disk.
- [ ] ⚠️ What does the predictor's 4-step sharpness look like against the teacher's 25-step
      ending, to the eye? Blur alone is not failure; a different subject is.

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
| the endpoint predictor is wired and checked once | one state, one prompt: adoption is plan 04's verdict, never this check's |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

**Required, even when empty.** Nothing open.

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| | | |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Run [the design's tasks](../plans/tools/03-wire-the-endpoint-predictor.md#tasks), then answer the questions above.
