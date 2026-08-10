# 🔍 The mechanism section and the three honesty caveats

## What this asks, in one line
Write the mechanism section from the answered review questions, honest about scope: what replicated (the value-channel read, median 1.52x over 64 cells) and exactly where its boundary sits.

## Description
Write the mechanism section to match whatever verdict the re-probe returns, and
write the three caveats that keep the paper's claims honest.

## Purpose
Both were moved here from `does-the-correction-cause-composition/plans/gate-01-two-literature-checks-before-print.md` on
2026-08-05 so that all paper prose has one owner. That scope keeps the two
/pressure-test gates and hands their verdicts over as input. Serves DoD 9.

## Goal
The mechanism section (or its negative paragraph) and the three caveats present
in the `.tex`, with both pressure-test verdicts reflected in the wording.

## Environment Facts This Plan Depends On
- The section and caveats go in `paper/iclr/iclr2027_conference.tex`; build with
  Ctrl+Shift+P → Build with recipe → `tectonic`. See `docs/ENVIRONMENT.md`,
  "Paper: where the LaTeX lives and how it is built".
- Blocked on `does-the-correction-cause-composition/plans/hypothesis-01-what-the-fix-changes-inside-the-model.md`. Its verdict
  decides which version gets written: a full section if the value-channel
  finding replicates across held-out pairs and seeds on `lora_step_100000.pt`,
  a negative paragraph if not. Do not write it before the verdict exists.
- Blocked on `does-the-correction-cause-composition/plans/gate-01-two-literature-checks-before-print.md` for the two verdicts,
  which land in `docs/pressure_tests/`.
- A negative verdict shrinks the section; it does not remove it. The honest
  paragraph is the deliverable in that case.

## Tasks
- [ ] write the mechanism section per plan 02's verdict (full section if
      replicated, negative paragraph if not)
- [ ] write the three honesty caveats: the oracle uses the joint prompt
      (not Mono-free, demonstration only); mid-λ doses are off-policy
      (endpoints exact); the transfer number is cited with its checkpoint
- [ ] fold both /pressure-test verdicts into the wording: a verdict of
      "already measured" on the window claim downgrades the novelty wording,
      not the experiment

- [ ] Write the do-no-harm limitation. The pool lists `an_elephant__x__a_penguin` as a pair that
      composes fine without any correction, and the strength sweep scored it 0 of 4 at strength 0
      with four single fused creatures. So the claim currently has no demonstrated do-no-harm
      control. State that as the boundary it is, and say what would settle it: a pair the plain
      method composes reliably, verified over seeds before being called a control.

## Next

1. `/draft-section limitations`: drafted from the ❌-bounded and 🟡 review answers, a bounded
   miss written as its boundary in one sentence. The two /pressure-test verdicts from
   does-the-correction-cause-composition plan 09 are its inputs; it does not run before they exist.

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
