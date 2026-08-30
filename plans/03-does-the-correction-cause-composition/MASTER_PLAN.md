# Interaction-Term

## Where this scope sits in the order

This scope owns **9 of the 30 steps**, 5 of them done. The steps interleave with the other scopes', so the list below is a filter on the one `## Running order` table in the [repo root MASTER_PLAN.md](../../MASTER_PLAN.md), never an order of its own.

**Next in this scope: step 6**, [when in the run the correction matters](plans/hypothesis-03-when-in-the-run-it-matters.md), driving the timing tab by hand.

| Step | Plan | What it does | Status |
|---|---|---|---|
| 1 | ~~[building the measuring scripts](plans/instrument-01-build-the-measuring-scripts.md)~~ | the thirteen measuring scripts | ✅ |
| 2 | ~~[fixing the size measure before any result](plans/instrument-02-fix-the-size-measure-before-any-result.md)~~ | how the correction's size is expressed | ✅ |
| 4 | [hypothesis-02-more-correction-more-composition](plans/hypothesis-02-more-correction-more-composition.md) | the headline causal result | ◑ 6.3GB owed off /home-mscluster |
| 5 | ~~[hypothesis-04-what-the-cached-runs-already-show](plans/hypothesis-04-what-the-cached-runs-already-show.md)~~ | the analyses needing no GPU | ✅ |
| 6 | [hypothesis-03-when-in-the-run-it-matters](plans/hypothesis-03-when-in-the-run-it-matters.md) | when the correction matters | ◑ driving the timing tab |
| 7 | ~~[hypothesis-05-the-same-story-from-three-sides](plans/hypothesis-05-the-same-story-from-three-sides.md)~~ | the independent checks | ✅ |
| 8 | ~~[hypothesis-01-what-the-fix-changes-inside-the-model](plans/hypothesis-01-what-the-fix-changes-inside-the-model.md)~~ | what the fix changes inside | ✅ |
| 13 | [figure-01-the-seven-paper-figures](plans/figure-01-the-seven-paper-figures.md) | the figures this scope owes | ◑ F6 needs a decision |
| 15 | [the two literature checks before print](plans/gate-01-two-literature-checks-before-print.md) | the two literature checks that must pass before print | ⚠️ |

**The corrector question is no longer in this scope.** Whether the correction is the sampler's
error or the model's grew into
[a scope of its own](../06-is-the-gap-the-samplers-or-the-models/MASTER_PLAN.md), steps 24 to 30,
because it builds two new composers before it measures anything and none of it has run. It still
puts this scope's timing result under threat, and step 21 of the paper cannot go ahead until that
scope's own check has produced the number.

## Mission
PoE fails at "a cat and a dog" because multiplying two predictions asks for an
image that is both things at once, while the sentence means two things side by
side. The gap between those readings is a concrete cached quantity, r_t.
Injected back in the right amount and at the right time it turns the blend into two animals.
It is small, shared across pairs, and concentrated in a narrow noise band,
which is why a rank-8 LoRA can learn it once and fix pairs it never saw. This
scope proves that account and produces the paper's figures.

## Objectives
1. Check the ground this scope inherits. Score phase1_r8_100k at step 100000,
   re-test the mechanism beyond seed 9, and commit the correction-size
   normalization in writing before any result is read.
2. Establish the causal claim through amount, direction, and timing (the matched
   window pair W1/W2), corroborated in image, manifold, and language space.
3. Say what kind of thing the term is. It is small (the spectrum, plus how much
   of a held-out pair's correction the training pairs' directions explain) and
   it behaves the same way everywhere (log-SNR collapse, a second and third
   model, the same test repeated across samplers, density traces).
4. Explain how far the claim reaches and deliver the evidence: the
   composition-type scatter, the mechanism section that is written only if its
   replication check passes, the figure cascade, and the Inspector tabs.

## Goals
1. Causal: this is supported if the [compose rate](../../context/world/compose-rate.md)
   rises with λ on most pairs while the norm-matched random control stays at what
   you would get by luck. It is a null if the pair's own real correction fails at
   λ=1, or if the random control does as well as it. It is inconclusive if the
   scorer and the eyeball disagree, and the answer to that is to fix the measuring
   tool and rerun, never to loosen the threshold.

   > A null here would read like this: the compose rate at λ=1 is the same as the
   > compose rate at λ=0, so adding the correction back changed nothing.
2. Timing: the W2 sliding-window curve peaks in a band and the path-split d(t)
   elbow lands in it; W1-vs-W2 coincidence or divergence recorded either way.
3. Smallness: the top-k energy of the stacked cached targets beats what a
   same-shape Gaussian gives, and a subspace fitted on the training pairs
   explains most of the energy of a pair it never saw at small k. If the decay
   is slow the claim narrows, and the paper survives it.
   <!-- ⚠ CONFLICT surfaced by sync 2026-08-05, wording NOT changed here.
        The second clause cannot pass, and not because the method fails.
        Measured on the 11-train / 6-transfer split: the train-fitted subspace
        captures 13.3% of held-out energy at k=64, while the adapter trained on
        the same split composes 96.9% on those pairs (0% without it). r_t
        vectors are mutually near-orthogonal (cosine ~0.00, even train-to-
        train), so no fitted subspace can contain unseen pairs and the test
        reads low regardless of transfer.
        First clause HOLDS: 63.0% against 16.4% for a same-shape Gaussian,
        3.8x, from
        scripts/spectrum.py --pool --stride 10 --max-seeds 8 (440 rows).
        Evidence: artifacts/results/which-way-the-correction-points/does-the-subspace-test-predict-transfer/QUERY.md
        Rewording direction text is /integrate-plans or /refine-plan, not
        sync-plan-tree. -->
4. Universality: ‖r_t‖ curves collapse in log-SNR across pairs; the window
   sits at the same noise levels across samplers; more correction still gives
   more composition on SD 1.5 and SD 2.1.
5. Prediction: the three pair groups order along a falling curve under the
   pre-registered normalization; if only one normalization shows it, report
   both and adopt neither.
6. Mechanism: the value-channel finding replicates across held-out pairs and
   seeds on lora_step_100000.pt. If it does not, the section shrinks to the one
   paragraph saying so.
7. Transfer honesty: the 100k number is scored (owned by
   does-the-fix-reach-unseen-pairs plan 03a) and always cited with its checkpoint.

## Expected Outcome
Either a confirmed causal account (measured, injected, timed, explained,
learned, transferred) with a two-figure theory core and a driveable demo, or
a diagnosed failure at a named goal with the evidence showing which link
broke. Nothing is claimed here without the tool that measured it.

## Definition of Done
1. ✅ Normalization pre-registered in writing before any cross-type plot exists.
   (relative_norm, report/normalization_preregistration.md, 2026-08-05)
2. ✅ Mechanism re-test verdict recorded: REPLICATES (median 1.52x over
   64 pair-and-seed runs, 2026-08-05). Caveat: the control pair shows the
   effect too.
3. The more-correction-more-composition figure with all three control rows.
4. W2 timing curve, enhanced W1 companion, joint window figure.
5. Cache analyses delivered: SNR collapse, d(t), density climb, spectrum with
   held-out projection.
6. Three-space corroboration delivered: manifold slide, language-side tests,
   [chimera](../../context/world/chimera.md) quality control.
7. Composition-type scatter with the new attribute-pair runs and the separate
   measuring tool they need.
8. Replication delivered: SD 1.5 and SD 2.1 runs across correction amounts, the
   same test repeated across samplers, SDE density traces.
9. Two /pressure-test passes done (window-timing novelty; SuperDiff span
   sentence) before those claims go to print.
10. Seven figures through /design-figure and built via /evidence-ladder.
11. Inspector tabs consuming only this scope's grids.

## Sub-Scopes
None live. The three background scaffolds (composition-type runs, cross-model replication,
the inspector) are parked whole at `artifacts/plans/parked/`. Each says what brings it back:
its parent plan moving out of the background pool.

## Plans

Grouped by what the group is for. The file numbers are per folder and are not an order, so
the Paper step column carries the position from the root `MASTER_PLAN.md`. "bg" means the
plan sits in the root's background-experiments pool and does not block the paper.

**The measuring tools, and the choices fixed before any result could be seen** (committed before any run produced a number)

| Plan | What it does | Paper step | Status | Owes |
|---|---|---|---|---|
| instrument-01-build-the-measuring-scripts | the 13 measuring scripts, built and given a first short run | | ✅ | scripts take `--pool`, because the cache mixes experiments |
| instrument-02-fix-the-size-measure-before-any-result | fixes how the correction's size is expressed, committed before any result was read | | ✅ | |

**Does the correction cause composition, or does it only come along with it** (hypothesis runs)

| Plan | What it does | Paper step | Status | Owes |
|---|---|---|---|---|
| hypothesis-02-more-correction-more-composition | more correction, more composition, with two flat controls. The headline | 1 | ◑ | 6.3GB of run outputs owed off /home-mscluster |
| hypothesis-03-when-in-the-run-it-matters | when in the denoising run the correction matters. The cliff is at the start | 3 | ◑ | driving the timing tab by hand |
| hypothesis-05-the-same-story-from-three-sides | the independent checks on the causal claim. Two image-side yes, two language-side null | 8 | ✅ | |

**What changes inside the model when the fix is on** (hypothesis runs)

| Plan | What it does | Paper step | Status | Owes |
|---|---|---|---|---|
| hypothesis-01-what-the-fix-changes-inside-the-model | the fix changes what a word paints rather than where it looks. Replicates, median 1.52x over 64 pair-and-seed runs | 19 | ✅ | |

**Analyses off the cached predictions, needing no GPU and no queue** (hypothesis runs)

| Plan | What it does | Paper step | Status | Owes |
|---|---|---|---|---|
| hypothesis-04-what-the-cached-runs-already-show | how few directions the correction needs, how its size tracks noise level, where two paths fork | 2 | ✅ | |

**How far the claim reaches** (idea and generalization runs; background, none of it blocks the paper)

| Plan | What it does | Paper step | Status | Owes |
|---|---|---|---|---|
| idea-01-does-it-hold-for-attribute-pairs | whether attribute pairs behave like object pairs | bg | ⚠️ | its runs; the scaffold waits at artifacts/plans/parked/composition-type-cells |
| generalization-01-other-models-and-samplers | the same result on another model and sampler | bg | ⚠️ | its runs; the scaffold waits at artifacts/plans/parked/cross-model-replication |
| 11-inspector | shelved with its sub-scope: tooling, not paper work; returns after submission or on promotion | | shelved | |

**What reaches the paper** (figure runs, plus the two checks that must pass before print)

| Plan | What it does | Paper step | Status | Owes |
|---|---|---|---|---|
| gate-01-two-literature-checks-before-print | the two /pressure-test passes run before anything is written. Defends, among other things, the decision not to run a baseline | 9 | ⚠️ | both passes |
| figure-01-the-seven-paper-figures | the figures this scope owes the manuscript, via /design-figure | 10 | ◑ | F6's decision, F7's remaining two bands, and eight smaller tasks. F1 to F5b, F7a and D1 to D4 are built |


## Environment Context
See `environment/00-INDEX.md` for this project's environment/architecture facts.
Read before drafting or checking any plan in this scope.

## Glossary

Terms used only in this scope, one plain line each. The shared vocabulary (PoE, chimera,
Mono, the residual `r_t`, λ, seed against pair, one run, the train-against-evaluate grid, MDS)
is in the root `MASTER_PLAN.md` and is not repeated here.

- **The correction:** plain-English name for the residual `r_t`. The step-by-step gap between
  what the model predicts from the joined prompt and what plain PoE predicts.
- **More correction, more composition:** the shape the causal claim predicts. Add more of the
  correction and the compose rate rises. A coincidence gives a flat line instead.
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
- **The check that must pass before anything else runs:** it proves the code that launches and
  collects a run has not disturbed the thing that run is measured against. Ours compares the
  λ=0 output against the sampler's own saved plain-PoE output. It is itself shown to fail
  against deliberately broken code, so it is a real test.
- **`relative_norm`:** the fixed way of expressing the correction's size, chosen and committed
  before any result was read so the choice cannot follow the answer.
- **Compose-rate:** the fraction of pictures showing two separate animals rather than one blended
  one, decided by the validated scorer.
- **AUC:** the area under the compose-rate-against-λ curve, on a 0-to-1 scale. One number
  summarising a whole curve, so the three injection rows can be compared at a glance.
- **Fork, and the elbow:** where two denoising paths separate, and the step at which they start to.
  The elbow is the answer to "when in the process does this decision get made".
- **Running the series in one go:** one job that loops over many pair-and-seed combinations and
  can be resumed, rather than launching each combination by hand.
- **Held-out:** a pair or seed the LoRA never trained on. The only kind that tests whether the
  fix reaches something new.
- **Triptych:** the three-panel picture logged for every run, Mono beside PoE beside corrected,
  so a number always has a picture next to it.
- **SVD:** the decomposition used to ask how few directions the correction really needs. Cached
  predictions are float16 and must be upcast before it accumulates.
