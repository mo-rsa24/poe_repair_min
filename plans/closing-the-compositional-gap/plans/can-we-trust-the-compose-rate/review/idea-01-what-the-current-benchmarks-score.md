# 🔍 Review: does any published metric agree with people where ours does not?

**Unanswered, and blocked until `gate-01` returns.** This file judges
[../plans/idea-01-what-the-current-benchmarks-score.md](../plans/idea-01-what-the-current-benchmarks-score.md),
the bake-off between our scorer and the published alternatives on the same labelled images.

## Recommended prompt (when the bake-off lands)

```
/analyze-run <bake-off run id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/idea-01-what-the-current-benchmarks-score.md) | the candidates, their published defaults, the bake-off script |
| **this file** | **the verdict: did any published metric beat ours on human agreement** |
| [what gates it](gate-01-is-this-hole-already-known.md) | the literature check that must return first |
| [what supplies its labels](instrument-01-the-three-state-labelled-set.md) | the judgeable-pair denominator every score is computed over |

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

- **Candidate**: any published metric or detector scored here, each named to the paper it came
  from. A method with no source paper is not a candidate.
- **Published defaults**: the thresholds the source paper reports. Candidates run at those and
  nowhere else, so a win cannot be manufactured by tuning.
- **Agreement**: how often a candidate's compose or not-compose call matches the human label,
  on judgeable pairs only.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Group 2, tries a new idea.** It may change nothing. A winner earns a row in `PARKING_LOT.md`
and the right to propose a group-1 run. It never changes a paper number directly, and a striking
agreement score here does not rewrite any hypothesis.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-pre-registered-bar) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| the bake-off, all candidates at published defaults | Tries a new idea | | one score per candidate; some candidates may need a GPU | the agreement table | not started |

## The pre-registered bar

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ Did any candidate agree with the human labels at 95% or better, on
      judgeable pairs, where the current scorer agreed at 85% or less?
      Yes: that candidate goes to `PARKING_LOT.md` with its number and its source paper, and
      `gate-02` decides whether it re-certifies or replaces `scorer_validated.json`. No: nothing
      is swapped, and the table is evidence for the limitations paragraph. Both thresholds are
      constants in the bake-off script, so neither can be adjusted after the scores are in. This
      is the only question that may promote anything.

## Written before the run, answered after

Navigation: ⬅️ [The pre-registered bar](#the-pre-registered-bar) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ Did the prediction written before running hold?
      The plan requires a prediction on record first: which candidates are presence-family and
      therefore expected to reproduce the hole. Answer it as held, broke, or partly, and say
      which candidate surprised you. A prediction nobody scores afterwards was decoration.
- [ ] ⚠️ Does any candidate ask a question neither family asks?
      If one does, name it even if it lost on agreement. A different question is worth more to
      the paper's limitations section than a marginally better score on the same question.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions the bake-off itself raised. **Nothing here may ever become a bar**, because it was
written with the answer already visible. Nothing yet: the bake-off has not run.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** Did every candidate run at published defaults?
      Assert it in the script rather than claiming it. A candidate whose thresholds were touched
      is disqualified, not asterisked, because a tuned competitor is not a comparison.
- [ ] ⚠️ **Was the instrument sound?** Were all scores computed on judgeable pairs only?
      The row count must match `instrument-01`'s judgeable denominator exactly. A candidate
      scoring well across all pairs including the unjudgeable ones is guessing, and the score is
      measuring the guess.
- [ ] ⚠️ **Did the run respect the environment?** Which candidates could not run on CPU, or
      needed a hosted model? The GPU here is shared and often full, so a candidate that only runs
      on a free GPU is one this project cannot depend on. Record it as part of the candidate's
      score, not as a footnote.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| our scorer measures what published metrics do not | the agreement table with every candidate's number and source paper, including the ones that beat us on their own question |
| any candidate's agreement score | the judgeable-pair denominator it was computed over, which is a fraction of the pool and not the whole of it |
| a candidate we did not adopt | whether it lost on agreement or was ruled out for needing a GPU this project cannot depend on. Those are different reasons and only one of them is about the metric |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| everything in this file | the bake-off | [gate-02](gate-02-promote-or-close.md), which needs the best agreement score to decide the promotion level |
| whether this plan runs at all | [gate-01](gate-01-is-this-hole-already-known.md) returning. An "already known" verdict cancels it | the bake-off itself |
| the judgeable-pair denominator every score needs | [instrument-01](instrument-01-the-three-state-labelled-set.md) landing | the scores, which cannot be computed without it |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Wait on [gate-01](gate-01-is-this-hole-already-known.md). If it says the scope continues, record
the prediction before running anything.
