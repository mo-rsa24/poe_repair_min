# Showcase the Trained LoRA

## Where this scope sits in the order

This scope's steps interleave with the other scopes', so the list below is a filter on the one
`## Running order` table in the [repo root MASTER_PLAN.md](../../../../MASTER_PLAN.md), never an
order of its own. Steps 31 to 35 in the root table; `sync-plan-tree` re-verifies on each pass.

**Next in this scope:** [01-read-the-plateau-curves](plans/01-read-the-plateau-curves.md), no GPU,
and it gates any longer training or rank sweep.

| Step | Plan | What it does | Status |
|---|---|---|---|
| 31 | [01-read-the-plateau-curves](plans/01-read-the-plateau-curves.md) | the free read that decides train-longer | ⚠️ do this next |
| 32 | [02-the-dog-x-dog-null-probe](plans/02-the-dog-x-dog-null-probe.md) | the null-input control | ⚠️ |
| 33 | [03-the-lora-dose-sweep](plans/03-the-lora-dose-sweep.md) | the causal dose curve for the shipped artifact | ⚠️ |
| 34 | [04-the-transfer-matrix-figure](plans/04-the-transfer-matrix-figure.md) | the reviewer-credible transfer demo | ⚠️ |
| 35 | [05-assemble-the-showcase-figures](plans/05-assemble-the-showcase-figures.md) | the figures, under the standard | ⚠️ |

## Mission

Turn the trained rank-8 cross-attention LoRA (`phase1_r8_100k`, W&B project
`prime_lab/poe-repair-animals-compose`; the cross-seed run `pueuo7bl` is a replication datum
only) into the paper's results figures, with the causal story carried by the LoRA's own output
rather than the cached oracle correction. The register's imbalance is the reason this scope exists:
roughly twenty figures measure the cached r_t and one measured the LoRA, while the LoRA is the one
artifact the project ships. Every figure obeys the standard in
[decisions-taken-here.md](decisions-taken-here.md), the walk-produced ledger that is this scope's
source of truth for pre-registrations, caption entitlements and gates.

## Objectives

1. Decide the train-longer question from already-logged curves before any GPU is spent on it.
2. Establish what the LoRA learned (a rule, not a stored vector) by the cheapest decisive test.
3. Move the causal dose evidence from the oracle correction to the shipped LoRA.
4. Ship the generalization demo at the tier the repo already named reviewer-credible.
5. Assemble the showcase figures under one standard a reader can trace.
6. Frame the baseline honestly: the joint prompt's own failure counted, so "restores what the
   target loses" is a measured sentence.
7. Read the mechanism beside the training runs (h-space, Jacobian directions), every read ending
   in an intervention or staying out of the main text.
8. Open tier-three captions by re-validating the scorer beyond two-animal scenes.

## Goals

1. A verdict on the `eval/frac_distance_reached` plateau: ceiling or waypoint, with the curve read
   across epochs and checkpoints recorded in plan 01's review file.
2. The dog x dog probe run and judged against its pre-registration (one dog + small ‖r̂‖ supports;
   two dogs falsifies; the preflight identity PoE(A,A) ≈ Mono(A) passed first).
3. The LoRA-dose figure: compose rate against λ on r̂ with the four control rows, AUC beside the
   oracle sweep's 0.387-vs-0.023 for comparison.
4. The transfer matrix at group-pooled, concept-disjoint tier, with the held-out grid as its
   qualitative rung.
5. The showcase figure set in `paper/iclr/figures/`, each with its sidecar, none violating the
   ledger's standard.
6. Experiments A, B and C judged against their pre-set null bars, with the tracking set frozen
   before any launch.
7. The joint-prompt baseline compose rate scored per pair and the three-bar figure built.
8. The scorer re-validation verdict recorded, tier-three captions opened or kept closed.

## Expected Outcome

A results section whose LoRA claims are carried by LoRA-measured figures, a probe verdict that
either sharpens the "learned a rule" caption or honestly kills it, and a settled answer to
whether longer training can crisp the held-out samples.

## Definition of Done

1. Plan 01's verdict recorded; experiments A and B launched only after it, with the plateau read
   as their interpretation key.
2. ⚠️ The probe and dose runs' W&B curves read in the browser and their verdicts written into the
   review files [owner: human].
3. All five plans ✅ with their review files' pre-registered questions answered.
4. No figure in the set violates the ledger's standard (checked at plan 05's close).
5. The scope has a recall gallery: run `/recap-plan-tree @plans/closing-the-compositional-gap/plans/showcase-the-trained-lora/MASTER_PLAN.md`
   once every plan above is ✅, and record the Artifact URL it publishes.

## Sub-Scopes

None.

## Plans

The nine files under [plans/](plans/), in build order; experiment plans 02, 03, 04, 06 and 07
carry review files with their pre-registrations.

## Depends on

- The trained checkpoints under
  `artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k/checkpoints/`
  (resolved via `checkpoints/latest.json`; `scripts/cross_seed_lora_pooling/heldout_pair.sh`
  is the resolution pattern to copy).
- The cached r_t cells (`POE_REPAIR_TRAINING_CACHE`), the validated scorer
  (`scorer_validated.json`), and the spectrum results under
  `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses/`.

## Environment Context

Read [environment/overview.md](../../../../environment/overview.md) and the row in
[environment/00-INDEX.md](../../../../environment/00-INDEX.md) matching what a task touches.
Facts this scope leans on: the `co3` python, `/datasets` for large outputs, the biggpu one-job
rule (long sweeps run nohup outside Slurm), the fp16 upcast rule, W&B project
`prime_lab/poe-repair-animals-compose` plus the cross-seed project the checkpoints came from.

## Diagram Prompts

The scope's illustrated map is [diagram-prompts.md](diagram-prompts.md), inheriting the parent
scope's art direction (Vivid circuit), palette and glyphs; its subject lane draws the walk's six
decided layers as one topology.

## Glossary

- **held-out**: pairs or seeds the LoRA never trained on.
- **compose-rate**: the fraction of generated images showing two separate animals, per the validated instance-count scorer.
- **AUC**: the area under the compose-rate-versus-λ curve on a 0-to-1 scale; the oracle reference is 0.387 real against 0.023 random.
- **r_t**: the cached correction, the joint prompt's guided noise prediction minus PoE's, on the same state.
- **PoE**: product-of-experts composition, the two expert predictions combined with guidance weighting.
- **Mono**: the single joint-prompt run, the ceiling the correction moves toward.
- **oracle**: the cached r_t injected directly, as opposed to the LoRA's own emitted correction.
- **SVD**: the decomposition giving the directions and gains the structure figure plots.
