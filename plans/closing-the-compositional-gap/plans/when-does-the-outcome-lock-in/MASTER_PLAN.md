# When Does the Outcome Lock In

## The overall claim
**This scope delivers a calibrated per-step commitment probe for the cached trajectories, so the 8-to-26-step gap between the correction window and visible divergence stops being unexplained.**

## What is this plan
Build the commitment instrument compiled by the basin-oracle walk: two free probes first (a
perturbed finish-the-run check and posterior-mean drift), then LCM-SDXL as a jump-to-the-end
readout of cached states, adopted only if a calibration pass against the base model clears bars
set in code.

## Why this plan exists
The correction's effective window is steps 0 to 10 (EXP-04) and trajectories visibly diverge at
steps 18 to 36, so the correction stops mattering 8 to 26 steps before anything shows. No
current instrument reads commitment per step: Tweedie estimates blur at high noise, and
divergence needs a pair of paths and reports the separation late. If commitment (the field's
speciation step) lands near step 10, the gap is explained as decide-then-descend; if it lands
with divergence, that story dies. The reasoning, the five verdicts on the proposal, and every
choice that could have gone the other way live in [the decision ledger](decisions-taken-here.md);
this file does not repeat them.

## What this scope actually does (visual)

```
  cache drum: JOINT | POE | LORA, one x_t per step
        |
        v                perturbation x_t +/- d
  +----------------+ <----------------------> +------------------+
  | endpoint       |                          | finish-the-run   |
  | spyglass (LCM) |                          | teacher (SDXL)   |
  +----------------+                          +------------------+
        | settled frame                              | true ending
        +--------------------+-----------------------+
                             v
                     calibration gauge
                       (bars in code)
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
DDIM code), the free probe (posterior-mean drift beside the divergence numbers), wire the oracle
(the LCM-SDXL adapter and its one-state smoke), calibrate (roughly 240 states against the
teacher, two bars in code), then the grid and the figures. Slices 1 and 2 need no new model;
slice 4 gates slice 5.

## Purpose and goals
Purpose: the parent scope [closing the compositional gap](../../MASTER_PLAN.md), whose mechanism
story needs to say why the correction window ends at step 10.
Goals: the numbered list under Goals below.

## Do this next
One ordering spans the whole tree: see the root MASTER_PLAN.md's `## Do this next`.

## Running order
This scope keeps no table of its own. Its plans join the root MASTER_PLAN.md running-order table
once `populate-plans` creates them.

## Mission
Measure, for every cached pair-and-seed cell, the step at which the final image is decided, with
instruments cheap enough to run across all three trajectory families and honest enough that
every oracle read carries a stability check and a calibration verdict.

## Objectives
1. Establish whether basin structure is measurable on the cached trajectories at all.
2. Measure the speciation step per cell with two independent probes.
3. Adopt or reject LCM-SDXL as an instrument by bars set in code before results are seen.
4. Deliver the three-timestamp figure and the per-family counterfactual compose-rate curves.

## Goals
1. The perturbation check has returned its verdict, with the saved endings and their numbers filed.
2. Speciation numbers sit beside divergence numbers for every cell, and the pre-registered
   ordering (speciation at or before divergence) is judged.
3. The calibration verdict is recorded as adopt, shrink, or fall back.
4. The figures are filed with sidecars, every caption naming the sweep that produced it.

## Expected Outcome
The paper can state when the outcome locks in: either the gap is explained as decide-then-descend
with the speciation numbers to show it, or that story is honestly killed and the review file says
what killed it.

## Definition of Done
1. The slice-1 verdict is recorded in its review file: basins measurable, or the proposal
   re-marked.
2. Two-probe speciation exists per cell and the ordering prediction is judged against its bar in
   code.
3. The adapter runs with the no-double-guiding assert and passes its one-state smoke against a
   teacher ending.
4. Calibration bars live in source and the verdict is recorded. ⚠️ The calibration run's W&B
   curves read by eye and the read noted in the review file [inferred, owner: human].
5. The grid figures are built and filed with sidecars. ⚠️ Filmstrips and destination frames
   eyeballed and judged, the judgement recorded [inferred, owner: human].
6. The scope has a recall gallery: run `/recap-plan-tree` on this MASTER_PLAN.md once every plan
   above is ✅, and record the Artifact URL it publishes.

## Sub-Scopes
(none)

## Plans
- ⚠️ [plans/01-basins-by-hand.md](plans/01-basins-by-hand.md) — proves basins and a ridge exist for the composed flow, one cell, one afternoon
- ⚠️ [plans/02-the-free-probe.md](plans/02-the-free-probe.md) — posterior-mean drift per cell, judged against the pre-registered ordering
- ⚠️ [plans/03-wire-the-oracle.md](plans/03-wire-the-oracle.md) — LCM-SDXL downloaded, adapted, smoke-tested against one teacher ending
- ⚠️ [plans/04-calibrate-the-instrument.md](plans/04-calibrate-the-instrument.md) — 240 states, two bars in code, verdict: adopt, shrink, or fall back
- ⚠️ [plans/05-the-grid-and-the-figures.md](plans/05-the-grid-and-the-figures.md) — both sweeps, the speciation table, the scope's paper figures

## Environment Context
Start at [the environment index](../../../../environment/00-INDEX.md). The facts this scope
leans on: the `co3` absolute python path, a GPU node, large artifacts to `/datasets` only (a
disk guard checks the filesystem it actually writes to), and W&B project
`prime_lab/poe-repair-animals-compose`, where W&B owns the numbers and this tree owns the
verdicts.

## Diagram Prompts
The scope's illustrated map is [the diagram prompts file](diagram-prompts.md): five subject
pieces and a capstone, inheriting the parent map's art direction, palette and glyphs. The
process lane is authored by `populate-plans` once plans exist.
