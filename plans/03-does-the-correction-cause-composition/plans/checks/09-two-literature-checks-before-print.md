# 🖨️ Two claims checked against the literature before they print

This plan asks whether two sentences the paper wants to print survive contact with the published
literature, and it answers that before either sentence is written.

**Step 15 of 22.** Waits on step 13. The one order is the `## Running order` table in the [repo root MASTER_PLAN.md](../../../../MASTER_PLAN.md).

| Step | Plan | Status |
|---|---|---|
| 14 | [06-the-transfer-figures](../../../04-does-the-fix-reach-unseen-pairs/plans/figures/06-the-transfer-figures.md) | ◑ F8 waits on its series of runs |
| **15** | **this plan** | **⚠️** |
| 16 | [01-make-the-template-build](../../../07-writing-the-paper/plans/writing/01-make-the-template-build.md) | ◑ title still a stub |

## What this asks, in one line
Two sentences the paper wants to print rest on what the literature does and does not already
contain, rather than on anything we ran. The first says nobody has causally measured when the
correction matters during a run. The second says that reweighting two experts' predictions cannot
manufacture a correction lying outside what those two predictions can express between them. Each
sentence gets one `/pressure-test` verdict before the wording is allowed into the manuscript. The
second verdict is also the written defence of the decision not to run baselines, since
Attend-and-Excite and SuperDiff are held in reserve as deferred tasks in this scope's own tree,
with the condition that would bring them back written down.

## Description
The two literature checks that must pass before specific claims go into the
paper.

## Purpose
Two printed claims rest on assertions about the literature and a published
method; each gets one /pressure-test pass. Serves DoD 9.

## Goal
Both pressure-test verdicts recorded, and handed to `writing-the-paper` for folding
into the wording.

## Environment Facts This Plan Depends On
- None apply. Both checks are literature reading done in-session.
- All paper prose has one owner,
  `plans/07-writing-the-paper/plans/writing-06-mechanism-and-limitations.md`. The verdicts
  produced here are its input.

## Tasks
- [ ] /pressure-test: "the [interaction term](../../../../context/world/interaction-term.md)'s
      timing has not been causally measured (sliding-window injection of the cached PoE→joint
      residual)"
- [ ] /pressure-test: "reweighting two experts' predictions cannot
      reproduce a correction outside their span (contra SuperDiff AND on
      [chimera](../../../../context/world/chimera.md) pairs)"

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
3. Hand both verdicts to `writing-the-paper/writing-06-mechanism-and-limitations`, which owns the wording.

## Engagement Instructions
```bash
ls docs/pressure_tests/          # expect two dated verdict notes
```
The check on the draft text itself belongs to the writing plan,
`plans/07-writing-the-paper/plans/writing-06-mechanism-and-limitations.md`.
