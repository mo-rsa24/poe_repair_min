# 🖨️ Before-print checks

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
  `plans/paper-iclr/plans/06-mechanism-and-caveats.md` on 2026-08-05, so all
  paper prose has one owner. The verdicts produced here are its input.

## Tasks
- [ ] ⚠️ /pressure-test: "the interaction term's timing has not been causally
      measured (sliding-window injection of the cached PoE→joint residual)"
- [ ] ⚠️ /pressure-test: "reweighting two experts' predictions cannot
      reproduce a correction outside their span (contra SuperDiff AND on
      chimera pairs)"

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
```
The draft-text check moved with the writing, to
`plans/paper-iclr/plans/06-mechanism-and-caveats.md`.
