# 🖨️ Before-print checks and the mechanism section

## Description
The checks that must pass before specific claims go into the paper, plus
writing the mechanism section once plan 02's verdict exists.

## Purpose
Two printed claims rest on assertions about the literature and a published
method; each gets one /pressure-test pass. The mechanism section is written
here, not in plan 02, so all print-gated writing lands together. Serves DoD 9
and closes Goal 6's write-up.

## Goal
Both pressure-test verdicts recorded and folded in; the mechanism section (or
the negative paragraph) drafted; the three honesty caveats written into the
paper text.

## Environment Facts This Plan Depends On
- None apply (writing and literature checks, in-session).

## Tasks
- [ ] ⚠️ /pressure-test: "the interaction term's timing has not been causally
      measured (sliding-window injection of the cached PoE→joint residual)"
- [ ] ⚠️ /pressure-test: "reweighting two experts' predictions cannot
      reproduce a correction outside their span (contra SuperDiff AND on
      chimera pairs)"
- [ ] ⚠️ write the mechanism section per plan 02's verdict (full section if
      replicated, negative paragraph if not)
- [ ] ⚠️ write the three honesty caveats: the oracle uses the joint prompt
      (not Mono-free, demonstration only); mid-λ doses are off-policy
      (endpoints exact); the transfer number is cited with its checkpoint

## Success/Failure Outcomes
- **pressure-test passes**
  - Success: verdict recorded; claim kept, sharpened, or downgraded
    accordingly.
  - Failure: a verdict of "already measured" on the window claim. That
    downgrades novelty wording, not the experiment (the figure still stands
    as evidence).

## Recommended skill
▶ `/pressure-test` ✅ (the two tasks name their exact arguments).

## Engagement Instructions
```bash
ls docs/pressure_tests/          # expect two dated verdict notes
grep -l "mechanism" paper/iclr/  # section text present in the draft
```
