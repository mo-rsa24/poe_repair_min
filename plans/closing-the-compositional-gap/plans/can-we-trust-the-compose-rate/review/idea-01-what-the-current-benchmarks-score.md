# Review: does any published metric agree with people where ours does not?

Unanswered, and blocked until `gate-01` returns. This file judges
[../plans/idea-01-what-the-current-benchmarks-score.md](../plans/idea-01-what-the-current-benchmarks-score.md),
the bake-off between our scorer and the published alternatives on the same labelled images.

## Words this file uses
- **Candidate**: any published metric or detector scored here, each named to the paper it came
  from. A method with no source paper is not a candidate.
- **Published defaults**: the thresholds the source paper reports. Candidates run at those and
  nowhere else, so a win cannot be manufactured by tuning.
- **Agreement**: how often a candidate's compose or not-compose call matches the human label,
  on judgeable pairs only.

## Run kind
**Group 2, tries a new idea.** It may change nothing. A winner earns a row in `PARKING_LOT.md`
and the right to propose a group-1 run. It never changes a paper number directly, and a striking
agreement score here does not rewrite any hypothesis.

## Runs

| Run | Kind | Launched at | Output | State |
|---|---|---|---|---|
| | | | | |

## The questions

- [ ] ⚠️ **The bar.** Did any candidate agree with the human labels at 95% or better, on
      judgeable pairs, where the current scorer agreed at 85% or less?
      Yes: that candidate goes to `PARKING_LOT.md` with its number and its source paper, and
      `gate-02` decides whether it re-certifies or replaces `scorer_validated.json`. No: nothing
      is swapped, and the table is evidence for the limitations paragraph. Both thresholds are
      constants in the bake-off script, so neither can be adjusted after the scores are in. This
      is the only question that may promote anything.
- [ ] ⚠️ Did the prediction written before running hold?
      The plan requires a prediction on record first: which candidates are presence-family and
      therefore expected to reproduce the hole. Answer it as held, broke, or partly, and say
      which candidate surprised you. A prediction nobody scores afterwards was decoration.
- [ ] ⚠️ Did every candidate run at published defaults?
      Assert it in the script rather than claiming it. A candidate whose thresholds were touched
      is disqualified, not asterisked, because a tuned competitor is not a comparison.
- [ ] ⚠️ Were all scores computed on judgeable pairs only?
      The row count must match `instrument-01`'s judgeable denominator exactly. A candidate
      scoring well across all pairs including the unjudgeable ones is guessing, and the score is
      measuring the guess.
- [ ] ⚠️ Which candidates could not run on CPU, or needed a hosted model?
      The GPU here is shared and often full, so a candidate that only runs on a free GPU is one
      this project cannot depend on. Record it as part of the candidate's score, not as a
      footnote.
- [ ] ⚠️ Does any candidate ask a question neither family asks?
      If one does, name it even if it lost on agreement. A different question is worth more to
      the paper's limitations section than a marginally better score on the same question.
