# 🖨️ Two claims checked against the literature before they print

## What this asks, in one line
Two sentences the paper wants to print rest on the literature rather than on our runs: that
nobody has causally measured the correction's timing, and that reweighting two experts cannot
manufacture a correction outside their span. Each gets one `/pressure-test` verdict before the
wording is allowed into the manuscript. One of them is also the recorded defence of the decision
not to run baselines (Attend-and-Excite and SuperDiff are held in reserve in `PARKING_LOT.md`
with their trigger written down).

## Description
The two literature checks that must pass before specific claims go into the
paper.

## Purpose
Two printed claims rest on assertions about the literature and a published
method; each gets one /pressure-test pass. Serves DoD 9.

## Goal
Both pressure-test verdicts recorded, and handed to `paper-iclr` for folding
into the wording.

## Environment Facts This Plan Depends On
- None apply (literature checks, in-session).
- The writing this plan used to own moved to
  `plans/closing-the-compositional-gap/plans/paper-iclr/plans/06-mechanism-and-caveats.md` on 2026-08-05, so all
  paper prose has one owner. The verdicts produced here are its input.

## Tasks
- [ ] /pressure-test: "the interaction term's timing has not been causally
      measured (sliding-window injection of the cached PoE→joint residual)"
- [ ] /pressure-test: "reweighting two experts' predictions cannot
      reproduce a correction outside their span (contra SuperDiff AND on
      chimera pairs)"

## Success/Failure Outcomes
- **pressure-test passes**
  - Success: verdict recorded; claim kept, sharpened, or downgraded
    accordingly.
  - Failure: a verdict of "already measured" on the window claim. That
    downgrades novelty wording, not the experiment (the figure still stands
    as evidence).

## Next

1. `/pressure-test` the timing-novelty claim, verbatim from task 1. Verdict to
   `docs/pressure_tests/`.
2. `/pressure-test` the span-argument claim, verbatim from task 2. Same destination.
3. Hand both verdicts to `paper-iclr/06-mechanism-and-caveats`, which owns the wording.

## Engagement Instructions
```bash
ls docs/pressure_tests/          # expect two dated verdict notes
```
The draft-text check moved with the writing, to
`plans/closing-the-compositional-gap/plans/paper-iclr/plans/06-mechanism-and-caveats.md`.
