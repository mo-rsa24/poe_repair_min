# Animals-Compose-Transfer

## Where this scope sits in the order

This scope owns **6 of the 22 steps**, 1 of them done. The steps interleave with the other scopes', so the list below is a filter on the one `## Running order` table in the [repo root MASTER_PLAN.md](../../../../MASTER_PLAN.md), never an order of its own.

**Next in this scope: step 9**, [instrument-02-three-live-curves-while-training](plans/instrument-02-three-live-curves-while-training.md), one epoch on a GPU, and it gates steps 10 to 14.

| Step | Plan | What it does | Status |
|---|---|---|---|
| 3 | ~~[instrument-01-the-clean-pair-pool](plans/instrument-01-the-clean-pair-pool.md)~~ | the pool that blends by default | ✅ |
| 9 | [instrument-02-three-live-curves-while-training](plans/instrument-02-three-live-curves-while-training.md) | the one-epoch smoke | ⚠️ do this next |
| 10 | [hypothesis-01-does-one-pooled-fix-transfer-at-all](plans/hypothesis-01-does-one-pooled-fix-transfer-at-all.md) | finish the pooled read | ◑ read incomplete |
| 11 | [hypothesis-02-transfer-as-a-rate-over-fifteen-pairs](plans/hypothesis-02-transfer-as-a-rate-over-fifteen-pairs.md) | fifteen adapters, one held out each | ⚠️ |
| 12 | [baseline-01-the-size-matched-control-pool](plans/baseline-01-the-size-matched-control-pool.md) | the size-matched mixed pool | ⚠️ |
| 14 | [figure-01-the-transfer-figures](plans/figure-01-the-transfer-figures.md) | the transfer figures | ◑ F8a and F8b built |

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
plans/completed/compose-scorer: this scope's first step halts unless
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
5. ⚠️ Figure cascade A2–A5 produced (F1 belongs to compose-scorer).

## Sub-Scopes
(none)

## Plans

Grouped by the run group each answers to. Statuses live in the review/ files.

**Hypothesis runs: the transfer claim, seen to the end unless the science is wrong**

| Plan | What it does | Status |
|---|---|---|
| instrument-01-the-clean-pair-pool | the clean pool behind the scorer gate (DoD 1) | ✅ |
| instrument-02-three-live-curves-while-training | the three live curves; the sweep's safety gate (DoD 2) | ◑ wired, smoke owed |
| hypothesis-01-does-one-pooled-fix-transfer-at-all | one pooled LoRA: does the fix transfer at all | ◑ run done, read incomplete |
| hypothesis-02-transfer-as-a-rate-over-fifteen-pairs | transfer as a rate: 15 held-out points (DoD 3) | ⚠️ |

**Baseline runs: frozen the moment they land**

| Plan | What it does | Status |
|---|---|---|
| baseline-01-the-size-matched-control-pool | the size-matched mixed pool on the identical held-out set (DoD 4) | ⚠️ |

**Figure runs: settled results only, captions capped by the register**

| Plan | What it does | Status |
|---|---|---|
| figure-01-the-transfer-figures | the A2 to A5 cascade feeding register slot F8 (DoD 5) | ◑ F8a and F8b built; F8 itself waits on the sweep |


## Environment Context
See `docs/ENVIRONMENT.md` for this project's environment/architecture facts.
Read before drafting or checking any plan in this scope.

## Glossary

Terms only this scope uses. Shared vocabulary is in the root `MASTER_PLAN.md`.

- **Compose-rate:** the fraction of outputs showing two separate animals, decided by the
  validated scorer, never by eye.
- **Held-out:** a pair the LoRA never trained on. The only kind that tests transfer.
- **LOPO (leave-one-pair-out):** train fifteen LoRAs, each missing one pair, test each on its
  missing pair. Transfer as a rate instead of an anecdote.
- **Delivery-null vs no-transfer:** the two ways a held-out pair can sit at floor. The fix never
  arrived (distance-reached at floor), or it arrived pointing wrong (direction-cosine low). The
  two-tier read exists to tell them apart.
- **Direction-cosine:** how aligned the run's correction is with the pool-mean correction.
- **Distance-reached:** how far toward the Mono target the fix actually moved the prediction.
- **Do-no-harm control:** pairs that compose fine without any fix; the LoRA must not break them.
