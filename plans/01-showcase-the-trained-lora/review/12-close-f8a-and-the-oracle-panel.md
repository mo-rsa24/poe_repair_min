# 📊 Review: did anything move after 60k, and is the best case crisp?

Nothing has run yet. This file judges [the design](../plans/12-close-f8a-and-the-oracle-panel.md).
Run kind: measurement (scoring existing samples; a small render batch for the panel).

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/12-close-f8a-and-the-oracle-panel.md) | the tail scoring, the panel, the reconciliation duty |
| this file | both verdicts |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [Still open](#still-open)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

- **The tail**: checkpoints 70k to 100k, sampled every 10k, whose renders exist and were never
  scored.
- **The best-case panel**: LoRA-corrected renders beside true-r_t-injected renders on the same
  pairs and seeds; qualitative, because nothing here measures crispness with a number.

> The one number the rows report is [compose rate](../../../context/world/compose-rate.md).

## Runs

Navigation: ⬅️ [Words](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Date | What ran | Runs scored | Wall time | Outcome |
|---|---|---|---|---|
| | | | | |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

**This is the one question whose failure moves other plans.**

- [ ] ⚠️ **Does out_out compose rate at 70k-100k stay inside the band the 10k-60k points draw?**
  The threshold, fixed before scoring: inside means saturation confirmed and plan 08's framing
  stands; a late rise beyond the existing points' spread reopens train-longer and is flagged to
  plan 08 before it launches.

## Written before the run, answered after

Navigation: ⬅️ [The bar](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

- [ ] ⚠️ Is the cached true correction visibly crisper than the LoRA on the four F9 pairs, or
  equally soft? (The eye's answer, one render at a time; this is experiment B's interpretation
  key.)
- [ ] ⚠️ Did figure-01 next door already do either piece (the reconciliation line)?

## Could the answer be an artefact

Navigation: ⬅️ [Before/after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- [ ] ⚠️ **Was the comparison fair?** The tail's samples used the same λ, window and seeds as
  the earlier eval points (checked against the sample manifest before scoring).
- [ ] ⚠️ **Was the measuring tool sound?** Same validated scorer as the earlier table.
- [ ] ⚠️ **Did the run respect the environment?** In-session scoring; the panel's few renders on
  a free device; outputs to `/datasets`.

## Still open

Navigation: ⬅️ [Artefact checks](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents)

Nothing open.
