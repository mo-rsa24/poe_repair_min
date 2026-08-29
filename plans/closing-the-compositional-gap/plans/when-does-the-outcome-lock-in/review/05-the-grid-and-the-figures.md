# 🧪 Review: where does speciation cluster, and does it explain the gap?

**Nothing has run yet.** This file judges [the design](../plans/05-the-grid-and-the-figures.md); answers land
here and nowhere else. Questions below were written at design time, before any number existed.

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```
(For a run that failed and whose failure is worth keeping: `/ingest-error-pattern --from-run-log`.)

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/05-the-grid-and-the-figures.md) | the hypothesis, the bars, the code to write |
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

- **speciation step**: the step where a cell's counterfactual finish stops changing, the
  field's word for basin entry (Biroli et al., arXiv 2402.18491).
- **counterfactual finish**: the image the plain guided model would land on if composing
  stopped at this step.
- **on-the-fence**: a read whose endpoint flips under a 1% nudge; marked, never averaged in.
- **the two sweeps**: joint prompt (does it compose from here) and expert pair (which animal
  wins).

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim.** A missed bar rewrites the scope's story (the gap stays unexplained) and the paper's mechanism section says so; the figures ship either way, captioned honestly.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-pre-registered-bar) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| | | | | | not started |

## The pre-registered bar

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ Do the per-cell speciation steps cluster at or below `SPECIATION_EARLY_MAX = 10`
      (in `scripts/commitment/grid_sweep.py`), per family and overall, rather than inside
      the divergence band (18 to 36)? This is the bar because the ledger pre-registered it as
      the decide-then-descend test: at-or-before-10 explains the window-versus-divergence gap,
      inside-the-band kills that explanation.

## Written before the run, answered after

Navigation: ⬅️ [The pre-registered bar](#the-pre-registered-bar) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ Which cells are the filmstrip cells? Name them here before the run, so the choice
      cannot chase the prettiest result.
- [ ] ⚠️ Does the oracle's speciation agree with plan 02's drift settling per cell? A
      systematic disagreement is a finding and is reported, with the scatter as its figure.
- [ ] ⚠️ Does the LoRA family's compose-rate curve rise during steps 0 to 10 and plateau by
      the window's end, as decide-then-descend predicts? Report the curve shape either way.
- [ ] ⚠️ How many reads were on-the-fence per family, and where do they sit relative to each
      cell's speciation step?

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
| speciation at/before 10 explains the gap as decide-then-descend | instrument per plan 04's verdict in every caption; state-space views are paired with outcome curves, PCA variance printed |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

**Required, even when empty.** Nothing open.

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| | | |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Run [the design's tasks](../plans/05-the-grid-and-the-figures.md#tasks), then answer the questions above.
