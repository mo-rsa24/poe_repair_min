# Plan 09 — Single-seed LoRA across the composition taxonomy

> Parent: [LORA_TAXONOMY_PLAN.md](LORA_TAXONOMY_PLAN.md). Builds on
> [04-lora-single-seed.md](04-lora-single-seed.md), which closed the
> PoE gap on the Group-6 beachhead (`a cat × a dog`, seed 42).

## Question

If we *fix* the seed to 42 and the LoRA recipe to the Phase-4 config
(rank-8, `attn2.{to_q,to_k,to_v}`, MSE on guided ε-space residuals),
does the residual-corrector mechanism close the PoE gap on **one
representative pair per group**, across groups G1–G4 and G6 of the
taxonomy (G5 deliberately deferred — see "Representative pairs" below)?

This is single-seed-per-pair training. Each cell is allowed to overfit
its own residual cache. The question is *mechanism breadth*, not seed
generalisation (which is Plan 10).

## Why this phase exists

[04-lora-single-seed.md](04-lora-single-seed.md) only shows the
deployable LoRA works for Group 6's hardest case (`a cat × a dog`,
seed 42). That single cell — even though it is the headline result of
the project — does not by itself rule out the worry that the LoRA
mechanism is *Group-6-specific* (e.g. it learns "split the chimera in
two" rather than the general "follow the joint trajectory" correction).

Plan 09 generates the contrast set we need to retire that worry. Five
single-seed LoRA trainings (one per G1–G4 and G6; G5 deferred), each
producing:

1. A consolidated artefact under `outputs/lora/<pair-slug>/seed_42/`
   matching the layout in Phase 4.
2. A residual-tab inspector view with the existing decoded-image row
   *plus* the new MDS / PCA trajectory panel below it, both wired to
   the same `(epoch, λ)` sliders.
3. A per-pair `inspector_manifest.json` carrying an `mds_cells` block,
   selectable from the inspector's pair dropdown.

If all five cells close the PoE gap by eyeball + visibly bend the
PoE+λ·R arm toward `A ∧ B` in the MDS panel, the LoRA is a mechanism,
not a Group-6 artefact.

## Research-objective alignment

The deployable contribution of this project is the LoRA residual
corrector. Memory-recorded scope discipline
([contribution_scope](../.claude/projects/-home-mscluster-mmolefe-Playground-PhD-poe-repair-min/memory/contribution_scope.md))
treats LoRA as the *deployed* artefact and group-A architectures as the
*failure cases*. For the deployed-artefact claim to land at paper
scale, it must be supported across the taxonomy — otherwise reviewers
read it as a one-pair anecdote.

This plan supports the claim "the LoRA residual corrector closes the
PoE gap across the composition taxonomy, not just on Group-6 concept
collisions" by producing the five per-group data points (G1–G4, G6)
that make the taxonomy-wide statement testable. G5 is deferred — its
prompts under-specify the entangled third concept, so the reference
trajectory is itself ambiguous and the read would be untrustworthy. The MDS panel is the visual evidence
that the corrected sampler walks toward `A ∧ B` rather than just toward
some image-space improvement — i.e. that the *trajectory-level*
mechanism described in
[residual_definition](../.claude/projects/-home-mscluster-mmolefe-Playground-PhD-poe-repair-min/memory/residual_definition.md)
travels.

This plan is also the entry point for Plans 10 and 11. The per-group
representative pair selected here is the same pair that Plan 10 trains
on across seeds, and the union of these pairs (plus their siblings) is
the training pool for Plan 11. So 09's six artefacts gate the rest of
the arc.

## Representative pairs (one per group)

Pulled from `/datasets/mmolefe/neurips2026/pilot_5seeds_interaction/seed_42/`.
The selection is fixed for the duration of plans 09–11 so artefacts are
comparable.

| # | Group | Pair slug | Disk path (under `seed_42/`) |
|---|---|---|---|
| 1 | Co-occurrence | `a_dolphin__x__an_ocean_wave` | `group1_cooccurrence/a_dolphin__x__an_ocean_wave` |
| 2 | Factorization | `a_dog__x__oil_painting_style` | `group2_factorization/a_dog__x__oil_painting_style` |
| 3 | Object + scene | `a_mailbox__x__a_snowfield` | `group3_role_separable_object_scene/a_mailbox__x__a_snowfield` |
| 4 | Dual-object | `a_typewriter__x__a_cactus` | `group4_dual_object_composition/a_typewriter__x__a_cactus` |
| 5 | Entanglement | *(deferred — see note below)* | `group5_concept_prior_entanglement/` |
| 6 | Concept collision | `a_cat__x__a_dog` *(short slug retained)* | already in `outputs/lora/a_cat__x__a_dog/seed_42/` |

Group 6 is the existing Phase-4 artefact and is **not** retrained here.
The MDS pre-render and inspector wiring for `a_cat__x__a_dog` are inherited
from Phase 4; only the pair dropdown surfaces it alongside G1–G4.

**G5 is deferred from this plan.** Candidate G5 pairs
(`a_tuxedo__x__a_flamingo`, `a_wedding_dress__x__a_lobster`,
`a_fur_coat__x__a_goldfish`, etc.) leave the entangled third concept
unspecified by the joint prompt — there is no canonical "what should
the joint trajectory look like" reference. Running a Phase-4-style
LoRA against an ambiguous target wastes compute and produces a read we
can't trust. G5 is listed here so the taxonomy stays complete; it is
not used in 09, 10, or 11. Revisiting it requires first defining a
prompt grammar that pins the entangled concept.

## Code

This plan adds no new architectures or samplers. It reuses Phase 4's
modules and the MDS pre-renderer end-to-end. The only new code is a
thin batch driver.

| Module | Reused from | Role |
|---|---|---|
| `poe_repair.experiments.one_pair_one_seed` (entrypoint) | Phase 4 | Trainer per pair, with `--pair <slug>`. |
| `poe_repair.experiments.one_pair_one_seed.trainer.attach_lora` | Phase 4 | LoRA attachment to `attn2.{to_q,to_k,to_v}`. |
| `poe_repair.methods._sampling.run_lora_residual_inject` | Phase 4 | Mono-free deployment sampler. |
| `poe_repair.experiments.one_pair_one_seed.probe` | Phase 4 | Per-epoch probe across λ grid. |
| `scripts/build_lora_manifest.py` | Phase 4 | Builds `inspector_manifest.json` per pair. |
| `scripts/build_lora_inspector_mds.py` | Phase 4 | Five-stage MDS pre-renderer (per-pair). |
| `scripts/lora_inspector.py` | Phase 4 | Inspector app; route `/` carries the residual tab + MDS panel. |
| `scripts/build_taxonomy_lora.sh` *(new)* | — | One-shot wrapper that loops over the six pair slugs and runs train → manifest → MDS. |

The new wrapper is the only deliverable code in this plan. It walks the
six representative pairs, calls the existing entrypoints, and writes
into per-pair output roots with no cross-pair coupling. It does **not**
touch `outputs/lora/a_cat__x__a_dog/seed_42/` beyond reading and pre-rendering.

## Commands

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
export CUDA_VISIBLE_DEVICES=1
export POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/outputs/training_cache
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
```

### Train one pair (per group), then build its inspector assets

```bash
PAIR=a_typewriter__x__a_cactus   # for example — Group 4

# 1. Train. Same recipe as Phase 4; same checkpoint cadence.
$PY -m poe_repair.experiments.one_pair_one_seed \
    --pair $PAIR --seed 42 --split heldout \
    --total-epochs 600 --probe-every-epochs 50 \
    --lr 1e-4 --lora-rank 8

# 2. Build the decoded-image manifest for the residual tab.
$PY scripts/build_lora_manifest.py \
    --results-root outputs/lora/$PAIR/seed_42/results

# 3. Pre-render the MDS trajectory panels (one PNG per (epoch, λ) cell).
$PY scripts/build_lora_inspector_mds.py \
    --results-root outputs/lora/$PAIR/seed_42/results \
    --pair-slug $PAIR \
    --epochs all --lambdas all \
    --stages collect-static,collect-cells,project,render,update-manifest
```

### Run all five pairs end-to-end

```bash
bash scripts/build_taxonomy_lora.sh
```

`build_taxonomy_lora.sh` iterates over:

```
a_dolphin__x__an_ocean_wave
a_dog__x__oil_painting_style
a_mailbox__x__a_snowfield
a_typewriter__x__a_cactus
a_cat__x__a_dog            # only stages 2–3; no retraining
# G5 intentionally absent — see "Representative pairs"
```

For each pair it calls the three steps above. Re-runs are idempotent —
the MDS pre-renderer skips already-cached trajectories and PNGs unless
`--overwrite` is passed. The a_cat__x__a_dog branch *skips* step 1.

### Smoke set (cheap, before the full sweep)

```bash
# 4 cells per pair × 5 pairs ≈ 15 min on one RTX 8000.
$PY scripts/build_lora_inspector_mds.py \
    --results-root outputs/lora/$PAIR/seed_42/results \
    --pair-slug $PAIR \
    --epochs 0,800 --lambdas 0.00,1.00 \
    --stages collect-static,collect-cells,project,render,update-manifest
```

### Inspector

```bash
$PY scripts/lora_inspector.py --port 5050
# from laptop: ssh -L 5050:127.0.0.1:5050 mscluster106 && open http://127.0.0.1:5050
```

Residual tab → pair dropdown → epoch slider + λ slider. Decoded-image
row on top, MDS panel directly below, both swap together.

## How to read the result

| Bucket | What you see | Means |
|---|---|---|
| **Poor** | One or more pairs fail the Phase-4 sanity guards (λ=0 ≢ vanilla PoE, or NaN probes, or MDS panel renders but static endpoints are not actually static across cells). | Wiring regression — the new pair runs are not byte-comparable to the reference Phase-4 artefact. Fix before reading anything. |
| **Bad** | At λ=1 the decoded image for ≥ 3 of 4 *new* pairs is visually indistinguishable from PoE at every epoch, and the MDS PoE+λ·R arm does not bend toward A∧B. | The mechanism does *not* travel. The Phase-4 result is plausibly Group-6 specific (the LoRA learned "split chimera in two"). Plan 10 still runs, but with the caveat that the per-group claim degenerates to a per-pair claim. |
| **Unknown** | 2 of 4 new pairs close the gap clearly, the others are ambiguous (image moves but not all the way to A∧B; MDS bends partially). | Mixed. Read each group separately; flag the ambiguous groups for longer training (the Phase-4 trajectory was still moving at epoch 600) and inspect their `r_t` magnitude vs Phase 2's curve. |
| **Good** | All 4 new pairs (and Group-6 inherited) show: (a) λ=0 byte-identical to vanilla PoE, (b) λ=1 produces a clear two-concept composition by epoch ~500–600, (c) MDS PoE+λ·R arm visibly bends from "near PoE" at (epoch=0, λ=0) toward A∧B as epoch and λ increase, (d) static A, B, A∧B endpoints sit at the same coordinates across all cells of the same pair. | The LoRA residual corrector is a taxonomy-wide mechanism (across G1–G4, G6). Plans 10 and 11 are justified. The 09 artefacts can be cited as the per-group baselines. |

## What this plan does *not* prove

- **Anything about seeds other than 42.** Per-pair single-seed
  overfitting is allowed; cross-seed generalisation is Plan 10's job.
- **Cross-pair transfer within a group.** A Plan-09 LoRA trained on
  `a_typewriter__x__a_cactus` is *not* expected to work on
  `a_drum_set__x__a_snowman`. That is the held-out-pair flag in
  Plan 10.
- **A single LoRA across the taxonomy.** Six independent LoRAs are
  trained here. Plan 11 trains one.

## Status — 2026-05-23

| Item | Done | To do |
|---|:---:|:---:|
| Representative pair selected per group (G1–G4, G6) | ✅ | |
| G5 explicitly deferred from this arc | ✅ | |
| Group 6 (`a_cat__x__a_dog`) Phase-4 artefact on disk | ✅ | |
| Pilot pair folders under `pilot_5seeds_interaction/seed_42/` exist for the 4 new pairs | ✅ | |
| Training-cache `residuals/` cells at seed 42 for the 4 new pairs (`/datasets/.../training_cache/heldout/`) | | ⬜ (only `a_typewriter__x__a_cactus` and `a_cat__x__a_dog` cached; 3 missing) |
| MDS pre-renderer is per-pair already (Phase 4) | ✅ | |
| `scripts/build_taxonomy_lora.sh` — one-shot driver across five pairs | | ⬜ |
| Train LoRA on each of the 4 new pairs at seed 42 (rank 8, ~600 epochs) | | ⬜ (~3 h GPU each) |
| Build manifests + MDS panels for the 4 new pairs | | ⬜ (~3 h GPU each at full sweep) |
| Pair dropdown in `lora_inspector.py` surfaces all five pairs from the residual tab | | ⬜ (small UI change if not already) |
| Read the five-pair contact sheet; classify each into poor/bad/unknown/good | | ⬜ |
