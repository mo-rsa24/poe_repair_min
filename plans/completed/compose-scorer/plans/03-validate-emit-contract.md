# ✅ Validate: prove the scorer, then emit the cross-scope contract

## Description
Run the built scorer on the validation set and check it against ground truth we can
see by eye: cat×dog should come out compose, wolf×husky should come out blend (that's
the case that fools the eye and is the whole reason the scorer exists). Only if both
land correctly does the scope write `scorer_validated.json`, the file the sibling
`does-the-fix-reach-unseen-pairs` scope reads as its precondition to start. If the scorer
cannot separate the wolf×husky blend in either space, the file is NOT written and the
scope halts: the downstream read would be invalid.

## Purpose
Serves Objective 2 and Definition-of-Done items 2 (validated, evidence saved), 3
(contract file emitted only on gate pass), and 4 (F1 figure).

## Goal
`scorer_validated.json` on disk (written only when the gate passes), the saved
validation evidence, and the F1 scorer-works figure.

## Tasks
- [x] Run the scorer on the cat×dog validation outputs and the wolf×husky corrected
  samples, in both spaces. Record the labels.
- [x] Check the gate: cat×dog labelled compose AND wolf×husky labelled blend. If
  the two spaces disagree, record BOTH (inconclusive per the master-plan three-way
  rule), do not pick the flattering one.
- [x] On gate pass, write `scorer_validated.json` (the contract): pass flag, which
  spaces passed, the threshold used, the validation labels, a timestamp field left for
  the caller to stamp. On gate fail, write a `scorer_validation_FAILED.json` instead
  and do NOT write the contract file.
- [x] **[/design-figure ran]** Produce the F1 scorer-works figure: each validation
  output pinned beside its three anchor thumbnails, compose/blend as the point colour.

All four ticked from the output: `outputs/compose_scorer/scorer_validated.json` has
`pass: true`, separates the hard pair both ways via `instance_count`, and both F1
figures exist (`F1_scorer_works.png` for the read that shipped, `F1_scorer_null.png`
for the embedding read that was rejected). The failed-attempt file
`scorer_validation_FAILED.embedding_attempt.json` is the proof the gate could fail.

## Engagement Instructions
GATE (unattended pass/fail, this is the load-bearing one): assert
`scorer_validated.json` exists AND its pass flag is true AND it records cat×dog→compose
and wolf×husky→blend. A downstream script reads exactly this file to decide go/no-go.
The F1 figure PNG exists and is non-empty.
STOP: if the scorer cannot separate the wolf×husky blend in either embedding space, do
NOT write `scorer_validated.json`. Halt the scope and surface the failure. The absence
of the contract file is itself the correct unattended signal to the sibling scope: it
will not start. Never write the contract file on a failed or inconclusive validation.

## Recommended skill
▶ `/design-figure` ✅: designs the F1 scorer-works figure (outputs beside anchor
   thumbnails, compose/blend as colour) before it's built. alt: `/evidence-ladder` to
   pair the qualitative thumbnails with the quantitative agreement table.
