# 🔍 The mechanism section and the three honesty caveats

## Description
Write the mechanism section to match whatever verdict the re-probe returns, and
write the three caveats that keep the paper's claims honest.

## Purpose
Both were moved here from `interaction-term/plans/09-print-gates.md` on
2026-08-05 so that all paper prose has one owner. That scope keeps the two
/pressure-test gates and hands their verdicts over as input. Serves DoD 9.

## Goal
The mechanism section (or its negative paragraph) and the three caveats present
in the `.tex`, with both pressure-test verdicts reflected in the wording.

## Environment Facts This Plan Depends On
- Blocked on `interaction-term/plans/02-mechanism-reprobe.md`. Its verdict
  decides which version gets written: a full section if the value-channel
  finding replicates across held-out pairs and seeds on `lora_step_100000.pt`,
  a negative paragraph if not. Do not write it before the verdict exists.
- Blocked on `interaction-term/plans/09-print-gates.md` for the two verdicts,
  which land in `docs/pressure_tests/`.
- A negative verdict shrinks the section; it does not remove it. The honest
  paragraph is the deliverable in that case.

## Tasks
- [ ] ⚠️ write the mechanism section per plan 02's verdict (full section if
      replicated, negative paragraph if not)
- [ ] ⚠️ write the three honesty caveats: the oracle uses the joint prompt
      (not Mono-free, demonstration only); mid-λ doses are off-policy
      (endpoints exact); the transfer number is cited with its checkpoint
- [ ] ⚠️ fold both /pressure-test verdicts into the wording: a verdict of
      "already measured" on the window claim downgrades the novelty wording,
      not the experiment

## Success/Failure Outcomes
- **the mechanism section**
  - Success: what it claims matches what plan 02 actually found, at the size
    the verdict supports.
  - Failure: a full section written on an inconclusive verdict. Shrink it to
    the negative paragraph instead.
- **the caveats**
  - Success: all three present, each stated plainly rather than buried in a
    footnote.
  - Failure: the transfer number appears anywhere in the paper without its
    checkpoint. That is the specific dishonesty this project has committed to
    avoiding.

## Engagement Instructions
```bash
ls docs/pressure_tests/                              # expect two dated verdicts
grep -c "mechanism" paper/iclr/iclr2027_conference.tex   # expect the section present
```
Manual check, by eye: read the mechanism section next to plan 02's verdict note.
Does the section claim more than the verdict supports? That comparison is the
gate, and no command can make it.
