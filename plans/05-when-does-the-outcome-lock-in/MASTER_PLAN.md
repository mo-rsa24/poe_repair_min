# When Does the Outcome Lock In

## The overall claim
**This scope delivers a calibrated per-step commitment test for the cached trajectories, so the 8-to-26-step gap between the correction window and visible divergence stops being unexplained.**

## What is this plan
Build the commitment measuring tool the design walk settled on: two free tests first (a
perturbed finish-the-run check and posterior-mean drift), then LCM-SDXL as a jump-to-the-end
readout of cached states, adopted only if a calibration pass against the base model clears
thresholds set in code.

## Why this plan exists
The correction's effective window is steps 0 to 10 (EXP-04) and trajectories visibly [diverge](../../../../../goal-setting/learning/deep-learning/diffusion-models/diffusion-speciation-dynamics/plans/03-manifold-and-the-later-transition/MASTER_PLAN.md) at
steps 18 to 36, so the correction stops mattering 8 to 26 steps before anything shows. Nothing
built so far reads commitment per step: Tweedie estimates blur at high noise, and
divergence needs a pair of paths and reports the separation late. If commitment (the field's
[speciation step](../../../../../goal-setting/learning/deep-learning/diffusion-models/speciation-before-divergence/MASTER_PLAN.md)) lands near step 10, the gap is explained because the run [decides early, then only descends](../../../../../goal-setting/learning/deep-learning/diffusion-models/speciation-before-divergence/plans/06-decide-then-descend-overlay.md); if it lands
with divergence, that story dies. The reasoning, the five verdicts on the proposal, and every
choice that could have gone the other way live in [the decision ledger](decisions-taken-here.md);
this file does not repeat them.

> **Speciation** is the field's word for the step at which the outcome stops being undecided:
> before it the run could still finish as either animal, after it the ending is settled and the
> remaining steps only render it.

> **Tweedie** is the formula that turns a noisy state and the model's noise prediction into a
> running guess at the final clean image.

## What this scope actually does (visual)

```
  cached trajectories: JOINT | POE | LORA, one x_t per step
        |
        v                perturbation x_t +/- d
  +----------------+ <----------------------> +------------------+
  | endpoint       |                          | finish-the-run   |
  | predictor (LCM)|                          | teacher (SDXL)   |
  +----------------+                          +------------------+
        | settled frame                              | true ending
        +--------------------+-----------------------+
                             v
                    calibration: does the
                    predictor match the teacher
                    (thresholds in code)
                             | pass
                             v
                scorer lens + embeddings
                             |
                             v
  step strip: |-window 0-10-|--speciation?--|-divergence 18-36-|
  compose-rate curves per family - filmstrips - PCA overlay
                             |
                             v
                       register board
```

The rendered version is the subject capstone in [the diagram prompts](diagram-prompts.md).

## High-level overview
Five slices, each becoming one plan: basins by hand (the perturbation check with the repo's own
DDIM code), the free test (posterior-mean drift beside the divergence numbers), wire the
endpoint predictor (the LCM-SDXL adapter and its one-state check), calibrate (roughly 240 states
against the teacher, two thresholds in code), then the grid and the figures. Slices 1 and 2 need
no new model; slice 4 decides whether slice 5 runs.

> A **basin** is the field's term for the set of states that all flow to the same ending, so a
> ridge between two basins is where a small nudge switches which ending you get.

## Purpose and goals
Purpose: a standalone scope under the [root master plan](../../MASTER_PLAN.md), whose mechanism
story needs to say why the correction window ends at step 10.
Goals: the numbered list under Goals below.

## Do this next
One ordering spans the whole tree: see the root MASTER_PLAN.md's `## Do this next`.

## Running order
This scope keeps no table of its own. Its plans join the root MASTER_PLAN.md running-order table
once `populate-plans` creates them.

## Mission
Measure, for every cached pair-and-seed run, the step at which the final image is decided, with
tools cheap enough to run across all three trajectory families and honest enough that
every endpoint-predictor read carries a stability check and a calibration verdict.

## Objectives
1. Establish whether basin structure is measurable on the cached trajectories at all.
2. Measure the speciation step per pair-and-seed run with two independent tests.
3. Adopt or reject LCM-SDXL as a measuring tool by thresholds set in code before results are seen.
4. Deliver the three-timestamp figure and the per-family counterfactual [compose-rate](../../context/world/compose-rate.md) curves.

## Goals
1. The perturbation check has returned its verdict, with the saved endings and their numbers filed.
2. Speciation numbers sit beside divergence numbers for every pair-and-seed run, and the
   pre-registered ordering (speciation at or before divergence) is judged.
3. The calibration verdict is recorded as adopt, shrink, or fall back.
4. The figures are filed with sidecars, every caption naming the pass that produced it.

## Expected Outcome
The paper can state when the outcome locks in: either the gap is explained because the run
decides early and then only descends, with the speciation numbers to show it, or that story is
honestly killed and the review file says what killed it.

## Definition of Done
1. The slice-1 verdict is recorded in its review file: basins measurable, or the proposal
   re-marked.
2. Two-test speciation exists per pair-and-seed run and the ordering prediction is judged against
   its threshold in code.
3. The adapter runs with the no-double-guiding assert and passes its one-state check against a
   teacher ending.
4. Calibration thresholds live in source and the verdict is recorded. ⚠️ The calibration run's
   W&B curves read by eye and the read noted in the review file [inferred, owner: human].
5. The grid figures are built and filed with sidecars. ⚠️ Filmstrips and destination frames
   eyeballed and judged, the judgement recorded [inferred, owner: human].
6. The scope has a recall gallery: run `/recap-plan-tree` on this MASTER_PLAN.md once every plan
   above is ✅, and record the Artifact URL it publishes.

## Sub-Scopes
(none)

## Plans
- ⚠️ [plans/01-basins-by-hand.md](plans/reading/01-basins-by-hand.md) — proves basins and a ridge exist for the composed flow, one pair-and-seed run, one afternoon
- ⚠️ [plans/02-the-free-probe.md](plans/tests/02-the-free-test.md) — posterior-mean drift per pair-and-seed run, judged against the pre-registered ordering
- ⚠️ [plans/03-wire-the-oracle.md](plans/tools/03-wire-the-endpoint-predictor.md) — LCM-SDXL downloaded, adapted, checked once against one teacher ending
- ⚠️ [plans/04-calibrate-the-instrument.md](plans/tools/04-calibrate-the-measuring-tool.md) — 240 states, two thresholds in code, verdict: adopt, shrink, or fall back
- ⚠️ [plans/05-the-grid-and-the-figures.md](plans/figures/05-the-grid-and-the-figures.md) — both prompt passes, the speciation table, the scope's paper figures
- ◑ [plans/06-where-each-condition-lands.md](plans/figures/06-where-each-condition-lands.md) — five conditions as clouds in DINOv2 space, axis pictures decoded through a representation autoencoder, per-step tracks with a commit step each; cat×dog endpoint figures filed, the rest open
- ◑ [plans/07-what-the-correction-is-made-of.md](plans/tests/07-what-the-correction-is-made-of.md) — the correction's in-span and orthogonal share per step against the three predictions PoE already has, the experts' decoded estimates, the adapter's output projected the same way, and the tracks' kinetic energy and which-animal score; cache only; four rungs done, bar inconclusive at 0.374, finding filed, close-out open

## Environment Context
Start at [the environment index](../../environment/00-INDEX.md). The facts this scope
leans on: the `co3` absolute python path, a GPU node, large artifacts to `/datasets` only (a
disk guard checks the filesystem it actually writes to), and W&B project
`prime_lab/poe-repair-animals-compose`, where W&B owns the numbers and this tree owns the
verdicts.

## Diagram Prompts
The scope's illustrated map is [the diagram prompts file](diagram-prompts.md): five subject
pieces and a capstone, inheriting the parent map's art direction, palette and glyphs. The
process lane is authored by `populate-plans` once plans exist.
