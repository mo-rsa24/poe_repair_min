# Interaction-Term

## Mission
PoE fails at "a cat and a dog" because multiplying two predictions asks for an
image that is both things at once, while the sentence means two things side by
side. The gap between those readings is a concrete cached quantity, r_t.
Injected back at the right dose and time it turns the blend into two animals.
It is small, shared across pairs, and concentrated in a narrow noise band,
which is why a rank-8 LoRA can learn it once and fix pairs it never saw. This
scope proves that account and produces the paper's figures.

## Objectives
1. Verify the inherited ground: score phase1_r8_100k at step 100000, re-probe
   the mechanism beyond seed 9, pre-register the correction-size normalization.
2. Establish the causal claim: dose, direction, and timing (matched window
   pair W1/W2), corroborated in image, manifold, and language space.
3. Characterize the term: small (spectrum + held-out projection) and universal
   (log-SNR collapse; second and third model; sampler sweep; density traces).
4. Explain the reach and deliver the evidence: the composition-type scatter,
   the gated mechanism section, the figure cascade, the Inspector tabs.

## Goals
1. Causal: support if compose-rate rises with λ on most pairs while the
   norm-matched random control stays at floor; null if the oracle fails at λ=1
   or the random control matches it; inconclusive if scorer and eyeball
   disagree, then fix the instrument and rerun, never loosen the threshold.
2. Timing: the W2 sliding-window curve peaks in a band and the path-split d(t)
   elbow lands in it; W1-vs-W2 coincidence or divergence recorded either way.
3. Smallness: top-k energy of stacked cached targets beats the same-shape
   Gaussian floor; a train-fitted subspace explains most held-out energy at
   small k; slow decay narrows the claim, does not kill the paper.
4. Universality: ‖r_t‖ curves collapse in log-SNR across pairs; the window
   sits at the same noise levels across samplers; the dose result replicates
   on SD 1.5 and SD 2.1.
5. Prediction: the three pair groups order along a falling curve under the
   pre-registered normalization; if only one normalization shows it, report
   both and adopt neither.
6. Mechanism gate: the value-channel finding replicates across held-out pairs
   and seeds on lora_step_100000.pt, else the section shrinks to the negative
   paragraph.
7. Transfer honesty: the 100k number is scored (owned by
   animals-compose-transfer plan 03a) and always cited with its checkpoint.

## Expected Outcome
Either a confirmed causal account (measured, injected, timed, explained,
learned, transferred) with a two-figure theory core and a driveable demo, or
a diagnosed failure at a named goal with the evidence showing which link
broke. No landing is narrated without its instrument.

## Definition of Done
1. Normalization pre-registered in writing before any cross-type plot exists.
2. Mechanism re-probe verdict recorded (pass or negative paragraph).
3. Dose-response figure with all three control rows.
4. W2 timing curve, enhanced W1 companion, joint window figure.
5. Cache analyses delivered: SNR collapse, d(t), density climb, spectrum with
   held-out projection.
6. Three-space corroboration delivered: manifold slide, language probes,
   chimera quality control.
7. Composition-type scatter with the new attribute cells and their separate
   instrument.
8. Replication delivered: SD 1.5 and SD 2.1 dose tests, sampler sweep, SDE
   density traces.
9. Two /pressure-test passes done (window-timing novelty; SuperDiff span
   sentence) before those claims go to print.
10. Seven figures through /design-figure and built via /evidence-ladder.
11. Inspector tabs consuming only this scope's grids.

## Sub-Scopes
- ⚠️ plans/composition-type-cells/ — "the missing regime's cells + the validated instrument that can read them"
- ⚠️ plans/cross-model-replication/ — "the same story on three models and three samplers, or where it breaks"
- ⚠️ plans/inspector-interaction-term/ — "every headline figure explorable by hand, zero new generation"

## Plans
- ⚠️ 01-preregister-normalization.md
- ⚠️ 02-mechanism-reprobe.md
- ⚠️ 03-dose-response.md
- ⚠️ 04-window-pair.md
- ⚠️ 05-cache-analyses.md
- ⚠️ 06-corroborations.md
- ⚠️ 07-composition-type.md
- ⚠️ 08-replication.md
- ⚠️ 09-print-gates.md
- ⚠️ 10-figures.md
- ⚠️ 11-inspector.md

## Environment Context
See `docs/ENVIRONMENT.md` for this project's environment/architecture facts.
Read before drafting or checking any plan in this scope.
