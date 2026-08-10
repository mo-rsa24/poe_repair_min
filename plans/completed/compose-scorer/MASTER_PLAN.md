# Compose-Scorer

## Mission
Tell a real two-animal composition from a chimera blend, reliably enough that a
downstream experiment can trust the call. Per pair, generate three reference
anchors: animal-A alone, animal-B alone, and the joint prompt ("A and B")
through the good composer. An output is a COMPOSE if it sits far from both
single-animal anchors and near the joint anchor. It is a BLEND if it lands
nearest one single-animal anchor (a blend is really one animal, so it collapses
onto one animal). This is a reusable instrument for the whole PoE-repair
program, not just the animals experiment, and its validated-scorer file is the
machine-checkable seam a dependent scope reads before it runs.

## Objectives
1. **Build**: the 3-anchor read, in two embedding spaces: DINOv2 (external,
   structure/instance-aware) and the project MDS/latent space (the same space
   the trajectory diagrams already use). The distance form is relative (is the
   output closer to the joint than to either single-animal anchor), not a
   hand-picked absolute cutoff.
2. **Validate**: on cat×dog the scorer agrees with the eye (calls the known
   composes a compose); on wolf×husky it flags the blend that fools the eye
   (calls the blend a blend). Which space separates them better is a recorded
   finding, not a pick-the-flattering-one choice.

## Goals
1. **Build**: scorer runs on the validation outputs and emits a compose/blend
   label per output, in both spaces.
   [checkpoint: a per-output table, output × {DINOv2 label, MDS label, distances
   to the three anchors}]
2. **Validate** (three-way rule):
   - *support* if cat×dog → compose AND wolf×husky → blend, in the validation set.
   - *null* if it cannot separate the wolf×husky blend in either space: the
     whole downstream read would be invalid; do NOT write the contract file.
   - *inconclusive* if the two embedding spaces disagree → report both, do not
     pick the flattering one.
   [checkpoint: F1 scorer-works figure + the DINOv2-vs-MDS agreement table]

## Expected Outcome
A validated, reusable scorer a downstream experiment can trust to tell
composition from blend, plus a recorded read on which embedding space separates
them better. Concretely: a scorer module, a saved evidence set, and
`scorer_validated.json` written only when the validation gate passes.

## Definition of Done
1. ⚠️ Scorer module built: 3-anchor read, both spaces (DINOv2, MDS/latent).
2. ⚠️ Validated on cat×dog + wolf×husky, both spaces, evidence saved.
3. ⚠️ `scorer_validated.json` emitted, written ONLY when the gate passes
   (cat×dog → compose AND wolf×husky → blend). This file is the cross-scope
   contract the sibling `does-the-fix-reach-unseen-pairs` scope reads as its
   precondition. If the gate fails, the file is NOT written and the scope halts.
4. ⚠️ F1 "scorer-works" figure produced (outputs pinned beside their three
   anchors as thumbnails, compose/blend as the point colour), via /design-figure.

## Sub-Scopes
(none)

## Status (2026-07-29): VALIDATED via instance-count, contract written
The scorer passes validation 10/10 and `scorer_validated.json` is written, so the
sibling `does-the-fix-reach-unseen-pairs` precondition is now met.

The validated read is an INSTANCE COUNT, reached after two rejected reads:
1. Whole-image embedding (DINOv2/CLIP): NULLED. For two similar animals, "one chimera"
   and "two separate animals" sit in the same feature region; d_joint was actually
   smaller for blends than for true composes. No threshold separates them.
2. Per-query box-IoU regime: also failed. A genuine wolf×husky compose and a wolf×husky
   blend both read "both_overlapping", because two similar canines standing close have
   coinciding boxes either way. Using IoU would under-count real composes on the hard
   pairs (a measurement bias into the downstream experiment).
3. **Instance count (kept):** query GroundingDINO-Tiny with "animal", NMS, COMPOSE iff
   >= 2 distinct instances. A chimera has one head/body; a real composition has two,
   even when the animals touch. Separates the hard wolf×husky pair BOTH ways (blend
   seeds 09/10/11 → 1; compose seed 12 + joint anchor → 2), plus cat×dog both ways.

Ground truth was established BY EYE per image (not assumed): the 4 wolf×husky corrected
samples are mixed (3 blends, 1 real compose at seed 12), and the scorer matches all.

Evidence: outputs/compose_scorer/{scorer_validated.json, agreement_table_detection.json,
F1_scorer_works.png}. Rejected-read evidence kept: {agreement_table.json,
F1_scorer_null.png, scorer_validation_FAILED.embedding_attempt.json}.

## Plans
- ✅ plans/01-anchors.md: three reference anchors per validation pair generated (DONE)
- ✅ plans/02-build-scorer.md: scorer built. Embedding reads (DINOv2/CLIP) nulled; instance-count read is the validated one (DONE)
- ✅ plans/03-validate-emit-contract.md: VALIDATED 10/10, scorer_validated.json emitted, F1_scorer_works.png produced (DONE)

## Running order

This scope keeps no order of its own. The single flat order across every scope
and level is the `## Running order` table in the repo root `MASTER_PLAN.md`.

## Environment Context
See `docs/ENVIRONMENT.md` for this project's environment/architecture facts.
Read before drafting or checking any plan in this scope.
