# 🧪 Review: does the student track the teacher, comparably across families?

**Nothing has run yet.** This file judges [the design](../plans/04-calibrate-the-instrument.md); answers land
here and nowhere else. Questions below were written at design time, before any number existed.

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```
(For a run that failed and whose failure is worth keeping: `/ingest-error-pattern --from-run-log`.)

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/04-calibrate-the-instrument.md) | the hypothesis, the bars, the code to write |
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

- **agreement**: scorer-verdict match plus DINOv2 distance between the oracle's frame and the
  teacher's ending from the same state, at guidance 7.5.
- **the families**: joint prompting, plain PoE, and LoRA-corrected, whose cached states sit at
  different distances from the tube the student was distilled on.
- **the verdict**: adopt, shrink (families and steps that pass), or fall back (teacher
  everywhere).

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Builds an instrument.** A missed bar does not stop the scope: it forks plan 05 onto its named fallback (shrink, or finish-the-run everywhere), and the verdict is the deliverable either way.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-pre-registered-bar) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| | | | | | not started |

## The pre-registered bar

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ Does every family's agreement sit at or above `AGREEMENT_FLOOR`, with no family
      more than `FAMILY_GAP_MAX` below another (both in
      `scripts/commitment/calibrate_oracle.py`)? This is the bar because a per-family bias
      is the confound that corrupts every cross-family figure even when average agreement
      looks fine.

## Written before the run, answered after

Navigation: ⬅️ [The pre-registered bar](#the-pre-registered-bar) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ Which family agrees worst, and is it PoE as the off-distribution reasoning predicts?
      Report the ordering either way.
- [ ] ⚠️ Does agreement vary with step (worse at high noise)? If so, the shrink option is a
      step range, not only a family list.
- [ ] ⚠️ What do the five worst disagreements look like by eye: blur, different subject, or
      different composition?

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
| the oracle is calibrated at guidance 7.5 | the checkpoint's trained guidance range is undocumented (its card's Training section is TODO); calibration at 7.5 covers our use, not the range |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

**Required, even when empty.** Nothing open.

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| | | |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Run [the design's tasks](../plans/04-calibrate-the-instrument.md#tasks), then answer the questions above.
