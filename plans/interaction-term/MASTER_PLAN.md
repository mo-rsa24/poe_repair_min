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
   <!-- ⚠ CONFLICT surfaced by sync 2026-08-05, wording NOT changed here.
        The second clause cannot pass, and not because the method fails.
        Measured on the 11-train / 6-transfer split: the train-fitted subspace
        captures 6.0% of held-out energy at k=64, while the adapter trained on
        the same split composes 96.9% on those pairs (0% without it). r_t
        vectors are mutually near-orthogonal (cosine ~0.00, even train-to-
        train), so no fitted subspace can contain unseen pairs and the test
        reads low regardless of transfer.
        First clause HOLDS: 62.6% vs a 13.2% Gaussian floor, 4.8x.
        Evidence: docs/evidence/subspace-vs-transfer/QUERY.md
        Rewording direction text is /integrate-plans or /refine-plan, not
        sync-plan-tree. -->
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
1. ✅ Normalization pre-registered in writing before any cross-type plot exists.
   (relative_norm, docs/normalization_preregistration.md, 2026-08-05)
2. ✅ Mechanism re-probe verdict recorded: REPLICATES (median 1.52x over
   64 cells, 2026-08-05). Caveat: the control pair shows the effect too.
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
<!-- All three are scaffolded (MASTER_PLAN.md + Definition of Done) but hold
     ZERO plan files. 11 DoD items with no tasks behind them. Run
     /populate-plans on each before its parent plan needs it. -->
- ⚠️ plans/interaction-term/plans/composition-type-cells/ — "the missing regime's cells + the validated instrument that can read them"
- ⚠️ plans/interaction-term/plans/cross-model-replication/ — "the same story on three models and three samplers, or where it breaks"
- ⚠️ plans/interaction-term/plans/inspector-interaction-term/ — "every headline figure explorable by hand, zero new generation"

## Plans

Grouped by what the group is for. The file numbers are per folder and are not an order, so
the Paper step column carries the position from the root `MASTER_PLAN.md`. "bg" means the
plan sits in the root's background-experiments pool and does not block the paper.

**The instruments, and the choices fixed before any result could be seen**

| Plan | What it does | Paper step | Status | Owes |
|---|---|---|---|---|
| 00-build-the-instruments | the 13 measuring scripts, built and smoked | | ✅ | scripts take `--pool`, because the cache mixes experiments |
| 01-preregister-normalization | fixes how the correction's size is expressed, committed before any result was read | | ✅ | |

**Does the correction cause composition, or merely accompany it**

| Plan | What it does | Paper step | Status | Owes |
|---|---|---|---|---|
| 03-dose-response | more correction, more composition, with two flat controls. The headline | 1 | ◑ | the re-score, then the figure and the strip |
| 04-window-pair | when in the denoising run the correction matters | 3 | ⚠️ | all of it |
| 06-corroborations | the independent checks on the causal claim | 8 | ⚠️ | all of it |

**What changes inside the model when the fix is on**

| Plan | What it does | Paper step | Status | Owes |
|---|---|---|---|---|
| 02-mechanism-reprobe | the fix changes what a word paints, not where it looks. Replicates: median 1.52x over 64 cells | 19 | ◑ | one decision: per-seed points or pair-level means, via /pair-figure |

**Analyses off the cached predictions, needing no GPU and no queue**

| Plan | What it does | Paper step | Status | Owes |
|---|---|---|---|---|
| 05-cache-analyses | how few directions the correction needs, how its size tracks noise level, where two paths fork | 2 | ⚠️ | five of six tasks. The fork curve is done, elbow at step 16 |

**How far the claim reaches. Background: none of this blocks the paper**

| Plan | What it does | Paper step | Status | Owes |
|---|---|---|---|---|
| 07-composition-type | whether attribute pairs behave like object pairs | bg | ⚠️ | its cells, via the composition-type-cells sub-scope |
| 08-replication | the same result on another model and sampler | bg | ⚠️ | its runs, via the cross-model-replication sub-scope |
| 11-inspector | the correction explorable by hand, for understanding rather than the manuscript | bg | ⚠️ | its build, via the inspector-interaction-term sub-scope |

**What reaches the paper**

| Plan | What it does | Paper step | Status | Owes |
|---|---|---|---|---|
| 09-print-gates | the two /pressure-test passes run before anything is written. Defends, among other things, the decision not to run a baseline | 9 | ⚠️ | both passes |
| 10-figures | the figures this scope owes the manuscript, via /design-figure | 10 | ⚠️ | all of them |

All three sub-scopes are empty, and all three belong to background plans. That is why nobody
has populated them and why it is not urgent. Populate one when its parent plan is promoted out
of the background pool.

## Running order

This scope keeps no order of its own. The order lives in the four lists in the repo root
`MASTER_PLAN.md`, and the Paper step column above carries those positions verbatim. If a number
here disagrees with the root, the root is right and this file is stale.

## Environment Context
See `docs/ENVIRONMENT.md` for this project's environment/architecture facts.
Read before drafting or checking any plan in this scope.

## Glossary

Terms used only in this scope, one plain line each. The shared vocabulary (PoE, chimera,
Mono, the residual `r_t`, λ, seed against pair, cell, crossbar, MDS) is in the root
`MASTER_PLAN.md` and is not repeated here.

- **The correction:** plain-English name for the residual `r_t`. The step-by-step gap between
  what the model predicts from the joined prompt and what plain PoE predicts.
- **Dose-response:** the shape borrowed from pharmacology. Add more of something and measure
  more effect. A real cause gives a rising curve; a coincidence gives a flat one.
- **The three rows (`oracle`, `random`, `wrong_pair`):** what gets injected. The pair's own real
  correction, a random vector of the same size, and a different pair's correction. The last two
  are the fakes that make the first one evidence.
- **Norm-matched:** the fakes are scaled to the same length as the real correction at every step,
  so a fake that fails cannot have failed for being too weak. Only direction differs.
- **`delta_norm`:** how big the real correction is at a step. Recorded even during a fake run,
  because it describes the pair of concepts and not what we injected.
- **The PMI identity:** a relationship the real correction satisfies, recorded per step for the
  same reason as `delta_norm`. A property of the pair, not of the injection.
- **`eps_poe`, `eps_j`, `eps_a`, `eps_b`, `eps_uncond`:** the model's raw predictions at a step.
  From plain PoE, from the joined prompt, from each concept alone, and from no prompt at all.
  The four cached branches every analysis reads.
- **Canary:** a check that the harness has not disturbed the thing it is measuring against. Ours
  compares λ=0 output against the sampler's own saved plain-PoE output. It is a test that is
  shown to fail against deliberately broken code, not a formality.
- **`relative_norm`:** the fixed way of expressing the correction's size, chosen and committed
  before any result was read so the choice cannot follow the answer.
- **Compose-rate:** the fraction of pictures showing two separate animals rather than one blended
  one, decided by the validated scorer.
- **AUC:** area under the dose curve. One number summarising a whole curve, so three rows can be
  compared at a glance.
- **Fork, and the elbow:** where two denoising paths separate, and the step at which they start to.
  The elbow is the answer to "when in the process does this decision get made".
- **Sweep:** one run covering many cells in a loop, resumable, rather than one cell at a time.
- **Held-out:** a pair or seed the LoRA never trained on. The only kind that tests transfer.
- **Triptych:** the three-panel picture logged per cell, Mono beside PoE beside corrected, so a
  number always has a picture next to it.
- **SVD:** the decomposition used to ask how few directions the correction really needs. Cached
  predictions are float16 and must be upcast before it accumulates.
