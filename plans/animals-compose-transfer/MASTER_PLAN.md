# Animals-Compose-Transfer

## Mission
cat×dog through PoE makes a chimera, and that blend-two-animals failure is a
specific, known problem. Train a rank-8 cross-attention LoRA on the cached
guided residual r_t = ε̃_J − ε̃_PoE on a token-disjoint pool of blend-prone
animal×animal pairs, then test on animal pairs and seeds it never saw. If it
composes them, it learned a general "put two animals in one picture" operator,
not a per-pair patch. If an animals-only pool beats a size-matched mixed pool on
animal pairs, the pool composition itself is what buys the fix. Every existing
taxonomy pair mixes an animal with a scene, a style, or an object, or is
object×object, so no existing group isolates "two similar animals that must not
blend". This scope is that isolation.

## Depends on
plans/compose-scorer: this scope's first step halts unless
`scorer_validated.json` exists and reads pass. That file is the machine-checkable
precondition that lets the two scopes run in sequence unattended.

## Reading a result (two-tier)
Every eval is read on two axes, and the plans below refer to this block instead of
restating it:
1. **Did it compose?** The compose-scorer labels each output compose or blend. Over
   the eval seeds this gives a **compose-rate**.
2. **Did the correction point the right way?** The **direction-cosine** is the cosine
   between this run's correction and the pool-mean correction. A separate number,
   **fraction-of-distance-reached**, is how far the correction travelled toward the
   PoE→Mono target. Corrections tend to stall near ~40% of that distance (the "40%
   plateau"), so this axis is watched live.

Keeping these two axes separate is the whole point. A floor compose-rate splits two ways:
- **delivery-null**: direction is right but the correction under-delivered (stalled at
  the plateau). The operator may be fine; the run just didn't push far enough.
- **no-transfer**: direction is wrong. No general operator was learned.

Calling a floor result "no transfer" without checking direction hides this difference.
So a null is diagnosed, not narrated.

## Objectives
1. **Pool**: curate a token-disjoint pool of ~15 blend-prone animal pairs (no
   animal word repeats across pairs, so leave-one-out stays a fair
   concept-disjoint test; ~15 pairs ≈ ~30 distinct animals), fails-by-default
   confirmed by the compose-scorer over 8 seeds as a fail-rate, not by eye. Keep
   a few compose-by-default animal pairs as a do-no-harm control.
2. **Transfer (A)**: leave-one-pair-out: train 15 LoRAs, each holding out one
   pair, eval each held-out pair on its own LoRA. 15 transfer points give a
   compose-rate and a degradation curve (rate vs fraction held out).
3. **Contrast (B)**: an animals-only pool beats a size-matched mixed pool on the
   same animal held-out pairs. Size-matching kills the "more data" confound.
4. **Diagnose**: read every null on both axes (see "Reading a result"), so a floor
   compose-rate is split into delivery-null vs no-transfer before it is called a null.

## Goals
1. **Transfer (A)** (three-way rule):
   - *support* if held-out pairs compose above the do-no-harm baseline across most
     of the 15, direction intact.
   - *null* if compose-rate is at floor (delivery-null or no-transfer, per the
     two-tier read).
   - *inconclusive* between → widen seeds/pairs, do not loosen the threshold.
   [checkpoint: cross-run leaderboard table + the degradation curve]
2. **Contrast (B)** (three-way rule):
   - *support* if animals-only beats size-matched mixed on animal held-out pairs,
     direction intact.
   - *null* if the two pools match.
   - *inconclusive* if only one embedding space shows the gap → report both.
   [checkpoint: paired animals-vs-mixed result on the identical held-out set]

## Expected Outcome
Either a confirmed pair-agnostic animal-compose operator with a transfer rate and
a degradation curve, plus evidence the animals pool specifically buys the fix; or
an honest null saying which: the fix doesn't transfer across animals, or it
transfers but pool composition doesn't matter. Every landing backed by the
two-tier read, so a null is diagnosed, not narrated.

## Definition of Done
1. ✅ `pair_pool.yaml` finalised: ~15 token-disjoint blend-prone animal pairs +
   do-no-harm controls, fails-by-default confirmed over 8 seeds by the
   compose-scorer. (11 training pairs all blend by default; fail_rate.{json,md}.)
2. ⚠️ compose-scorer wired into the eval hook, logging compose-rate +
   direction-cosine + fraction-of-distance-reached as separate live W&B curves.
3. ⚠️ (A) leave-one-pair-out: 15 LoRAs trained, each held-out pair scored,
   cross-run leaderboard table + degradation curve produced.
4. ⚠️ (B) size-matched mixed baseline built and run on the same animal held-out
   pairs; contrast reported.
5. ⚠️ Figure cascade F2–F5 produced (F1 belongs to compose-scorer).

## Sub-Scopes
(none)

## Plans
- ✅ plans/01-pool-and-precondition.md: precondition gate + curate + finalise pair_pool.yaml by fail-rate (DoD 1)
- ⚠️ plans/02-wire-scorer-eval-hook.md: the one build: scorer into eval hook, three live W&B curves, green smoke gate (DoD 2) — compose-rate, direction-cosine, distance-reached all wired in code; the 1-epoch GPU smoke to confirm the three live curves remains
- ⚠️ plans/03a-phase1-pooled.md: one pooled LoRA, held-out transfer read (cheap first pass ahead of the LOPO) — RUN DONE, read near-complete (out_out 0.96 @ 60k, all held-out pairs above floor; steps 70k-100k unscored, go/no-go note + direction axis remain)
- ⚠️ plans/03-run-A-leave-one-pair-out.md: 15 LoRAs, leaderboard, degradation curve (DoD 3)
- ⚠️ plans/04-run-B-contrast.md: size-matched mixed pool, animals-vs-mixed on same held-out set (DoD 4)
- ⚠️ plans/05-figures.md: F2–F5 evidence cascade via /design-figure (DoD 5)

## Running order

This scope keeps no order of its own. The single flat order across every scope
and level is the `## Running order` table in the repo root `MASTER_PLAN.md`.

## Environment Context
See `docs/ENVIRONMENT.md` for this project's environment/architecture facts.
Read before drafting or checking any plan in this scope.
