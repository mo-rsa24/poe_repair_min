# Plan 11 — Cross-pair × multi-seed LoRA (broadest claim)

> Parent: [LORA_TAXONOMY_PLAN.md](LORA_TAXONOMY_PLAN.md). Generalises
> [10-cross-seed-lora-per-group.md](10-cross-seed-lora-per-group.md)
> along the *pair* axis so that one LoRA is trained on the union of all
> six groups' pairs across all their seeds. Two-axis held-outs (pair
> and seed) make this the project's broadest generalisation test.

## Question

Train **one** rank-8 LoRA on a pool of (pair, seed) cells drawn from
the five studied groups of the taxonomy (G1–G4 and G6; G5 deferred —
see Plan 09). Evaluate it on held-out *pairs* (siblings within the
same group as the training pairs) and held-out *seeds* (seeds not seen
at training time for any pair). Does the single LoRA produce
recognisable composition on these unseen (pair, seed) cells, and does
it degrade gracefully or catastrophically when both axes are held out
simultaneously?

## Why this phase exists

Plans 09 and 10 give us the per-cell and per-group views of the LoRA
mechanism. Neither answers the deployment question:

> Can we ship a **single** LoRA that covers the composition taxonomy?

A "yes" here is the strongest possible form of the project's deployable
contribution. A "no" still matters — it tells us pooling-by-group (Plan
10's unit) is the right deployment granularity, and the per-group
LoRAs are the actual catalogue.

This phase is also the only one in the arc that *requires* the
two-axis held-out structure. Held-out seeds alone (Plans 08 and 10)
only stress the seed axis. Held-out pairs alone (the
`--heldout-pair` flag in Plan 10) only stress the pair axis. The
crossbar — pair AND seed both unseen — is what a real deployment would
look like, and it's only meaningful when one LoRA spans both axes at
training.

## Research-objective alignment

The five-thread framing
([project_framing](../.claude/projects/-home-mscluster-mmolefe-Playground-PhD-poe-repair-min/memory/project_framing.md))
identifies LoRA as the deployed thread. The strongest version of that
thread's contribution claim is:

> One LoRA, trained without seeing pair-`P` or seed-`s`, can close the
> PoE gap on cell `(P, s)` at deployment time.

Plan 11 is the test of that claim. The four held-out quadrants are the
testbed:

| Quadrant | Pair | Seed | What it tests |
|---|---|---|---|
| In-pair × in-seed | seen | seen | Training-cell sanity. Floor must be at least as good as Plan 09 on the same cell. |
| In-pair × held-seed | seen | unseen | Pure seed-axis generalisation. Should match Plan 10's pooled result on that pair. |
| Held-pair × in-seed | unseen | seen | Pure pair-axis generalisation. Within-group transfer measured *without* seed confound. |
| **Held-pair × held-seed** | unseen | unseen | The deployment crossbar. The only quadrant Plans 09 and 10 cannot answer. |

The crossbar quadrant is the headline. Everything else is calibration.

The two failure modes are also research-relevant:

- **Group structure persists at scale.** Per-group accuracy on the
  crossbar mirrors Plan 10's per-group buckets. Confirms the taxonomy
  is the right unit of analysis. Strong support for the paper's
  framing.
- **Single-LoRA collapse.** The big LoRA matches per-group LoRAs in
  some quadrants but degrades on others (typically the hardest
  studied group, G6). Confirms there is a deployment cost to
  consolidating into one corrector — a real, paper-worthy tradeoff.

Either landing is a clean result. The bad landing is "the cross-pair
LoRA fails everywhere," in which case the cleanest deployment story is
Plan 10's per-group catalogue.

## Training pool

Let `S_train` and `P_train` be the seed and pair pools. Build training
cells from the cross product `P_train × S_train`, drawing residual
caches from the pilot tree. Pool composition (initial proposal — sized
to fit in one ~24 h training run on one RTX 8000):

- `S_train = {1, 2, 3, 4, 5, 6, 7, 8}` (8 seeds — matches Plan 08).
- `S_heldout = {9, 10, 11, 12}` (4 seeds; disjoint from `S_train`).
- `P_train` = the five representative pairs from Plan 09 (one per
  studied group, G1–G4 and G6; G5 not included). 5 × 8 = 40 cells.
- `P_heldout` = the five sibling pairs from Plan 10's held-out-pair
  table (one per studied group). 5 × 12 = 60 evaluation cells
  (combining held-out seeds and in-pool seeds for the four-quadrant
  analysis).

The pair-pool YAML enforces `P_train ∩ P_heldout = ∅` and the seed-pool
YAML enforces `S_train ∩ S_heldout = ∅`. Optional richer-pool variant
(if cache cells exist):

- `P_train` extended to include 2–3 pairs per group (10–15 pairs
  total, across G1–G4 and G6) with the held-out sibling kept out.
  Stresses the LoRA more; only worth running if the 5-pair baseline
  lands `Good`.

## Code

This plan adds two new modules to the existing
`poe_repair.experiments.cross_seed_lora_pooling` namespace plus a new
two-axis evaluator. Trainer reuses Plan 10's flow but pools across
pairs.

| Module | New / reused | Role |
|---|---|---|
| `poe_repair.experiments.cross_pair_lora_pooling.pair_pool` *(new)* | new | Loads `pair_pool.yaml`; enforces `train ∩ heldout = ∅`. |
| `poe_repair.experiments.cross_pair_lora_pooling.train_pooled` *(new)* | new | Pools cache cells across `(pair, seed)`. Trainer body delegates to the Plan-08 trainer; only the loader differs. |
| `poe_repair.experiments.cross_pair_lora_pooling.sample_crossbar` *(new)* | new | For each `(pair, seed)` in the four-quadrant grid, samples through `run_lora_residual_inject` with the trained adapter. Writes a `quadrant=<...>` tag on each output dir. |
| `poe_repair.experiments.cross_seed_lora_pooling.seed_pool` | reused | Seed-pool YAML loader (verbatim from Plan 08). |
| `poe_repair.experiments.cross_seed_lora_pooling.task_d_bridge` | reused; takes `--pair` and `--seed` | Δ̄_t bridge per (pair, seed). Inputs are now per-cell. |
| `poe_repair.experiments.cross_pair_lora_pooling.contact_sheet` *(new)* | new | Renders the four-quadrant grid: rows × quadrants × pairs. |
| `scripts/cross_pair_lora_pooling/train_all_groups.sh` *(new)* | new | One-shot wrapper: builds the pair pool, calls trainer with `--total-epochs`, evaluates the crossbar, renders the contact sheet. |

The trainer **never** touches `outputs/lora/<pair>/seed_42/` or
`outputs/cross_seed_lora_pooling/<pair>/`. Plan-09 / Plan-10 artefacts
are read-only inputs (for cross-comparison) at most.

## Pool YAMLs

```yaml
# outputs/cross_pair_lora_pooling/pair_pool.yaml
# G5 deliberately omitted — see Plan 09 for the deferral rationale.
train:
  - a_dolphin__x__an_ocean_wave
  - a_dog__x__oil_painting_style
  - a_mailbox__x__a_snowfield
  - a_typewriter__x__a_cactus
  - a_cat__x__a_dog
heldout:
  - a_polar_bear__x__an_iceberg           # G1 sibling
  - a_cat__x__charcoal_drawing_style      # G2 sibling
  - a_fire_hydrant__x__a_snowfield        # G3 sibling
  - a_drum_set__x__a_snowman              # G4 sibling
  - a_wolf__x__a_husky                    # G6 sibling
```

```yaml
# outputs/cross_pair_lora_pooling/seed_pool.yaml
train_pool: [1, 2, 3, 4, 5, 6, 7, 8]
held_out:   [9, 10, 11, 12]
```

## Cache prerequisites

Total cells needed for training: 5 pairs × 8 seeds = 40 residual
caches. Evaluation: 10 pairs × 12 seeds = 120 cells, but only the four
quadrants of interest are sampled (in-pair/in-seed + the three
held-out quadrants; the in-pair/in-seed quadrant samples a small
subset for sanity).

All cells live under
`/datasets/mmolefe/poe_repair_min/outputs/training_cache/heldout/<pair-slug>/seed_<N>/`
and are resolved through `POE_REPAIR_TRAINING_CACHE`. Groups G4 and G6
inherit existing multi-seed caches; G1, G2, G3 (and their sibling
pairs) require cache production via `scripts/build_training_cache.py`
before this plan can train.

## Commands

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
export CUDA_VISIBLE_DEVICES=1
export POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/outputs/training_cache
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
```

### Task A — pool YAMLs + leak guards

```bash
$PY -m poe_repair.experiments.cross_pair_lora_pooling.pair_pool \
    --pair-pool outputs/cross_pair_lora_pooling/pair_pool.yaml \
    --check-only
$PY -m poe_repair.experiments.cross_seed_lora_pooling.seed_pool \
    --seed-pool outputs/cross_pair_lora_pooling/seed_pool.yaml \
    --check-only
```

Deliberately broken YAMLs (a pair in both `train` and `heldout`, or a
seed in both `train_pool` and `held_out`) must abort at load.

### Task B — one-shot training across all 40 cells

```bash
bash scripts/cross_pair_lora_pooling/train_all_groups.sh
```

Wraps:

```bash
$PY -m poe_repair.experiments.cross_pair_lora_pooling.train_pooled \
    --pair-pool outputs/cross_pair_lora_pooling/pair_pool.yaml \
    --seed-pool outputs/cross_pair_lora_pooling/seed_pool.yaml \
    --total-epochs 2400 \
    --lora-rank 8 --lr 1e-4 \
    --output-root outputs/cross_pair_lora_pooling/all_groups \
    --run-id main
```

Epoch budget is the per-cell Phase-4 budget (~600) × 4 (to amortise
across the 40 cells with a smaller per-cell pass). Checkpoint cadence
mirrors Plan 08 (every 200 epochs).

### Task C — four-quadrant evaluation (the headline)

```bash
$PY -m poe_repair.experiments.cross_pair_lora_pooling.sample_crossbar \
    --checkpoint outputs/cross_pair_lora_pooling/all_groups/main/checkpoints/lora_step_<...>.pt \
    --pair-pool outputs/cross_pair_lora_pooling/pair_pool.yaml \
    --seed-pool outputs/cross_pair_lora_pooling/seed_pool.yaml \
    --out-dir   outputs/cross_pair_lora_pooling/all_groups/main/samples
```

For each `(pair, seed)` chosen by the quadrant policy (`in/in`,
`in/out`, `out/in`, `out/out`), runs `run_lora_residual_inject` with
the trained adapter. The PoE-baseline arm is sampled in the same call
for paired panels.

### Task D — Δ̄_t bridge across (pair, seed)

```bash
$PY -m poe_repair.experiments.cross_seed_lora_pooling.task_d_bridge \
    --pooled-run outputs/cross_pair_lora_pooling/all_groups/main \
    --cells outputs/cross_pair_lora_pooling/all_groups/main/samples/cells.jsonl
```

Per cell, cosine alignment of pooled `ε_PoE_lora,t` against (a) that
cell's own `Δ_t`, (b) the pair-averaged `Δ̄_t^(P)`, (c) the
seed-averaged `Δ̄_t^(s)`. Pulls apart whether the LoRA learned a
pair-conditional or seed-conditional correction (or neither).

### Contact sheet

```bash
$PY -m poe_repair.experiments.cross_pair_lora_pooling.contact_sheet \
    --pooled-run outputs/cross_pair_lora_pooling/all_groups/main
```

Renders a grid: rows = pairs (10, both train and heldout), columns =
{PoE, pooled-LoRA, per-group LoRA from Plan 10, per-pair LoRA from
Plan 09 if available, mono}. One sheet per quadrant; the held-pair ×
held-seed sheet is the paper figure.

### Optional richer-pool variant

```bash
$PY -m poe_repair.experiments.cross_pair_lora_pooling.train_pooled \
    --pair-pool outputs/cross_pair_lora_pooling/pair_pool_v2.yaml \
    --seed-pool outputs/cross_pair_lora_pooling/seed_pool.yaml \
    --total-epochs 4800 \
    --run-id v2_dense
```

Only run if the 5-pair version lands `Good` — this version pushes to
10–15 train pairs (G1–G4, G6 with multiple representatives per group)
and tests whether the LoRA scales with pool diversity.

## How to read the result

Per quadrant, count cells (out of `n_cells` in that quadrant) on which
the pooled LoRA produces recognisable composition (eyeball; matches
the Phase-4 acceptance criterion). Report as a 4-row table.

| Quadrant | Expected if "Good" | Expected if "Mixed" | Expected if "Bad" |
|---|---|---|---|
| in / in | ≥ 90% recognisable composition. Sanity floor. | ≥ 70% | < 50%. Training collapsed; stop. |
| in / out | ≥ 70%. Matches Plan-10 pooled per-pair. | ≥ 50% on easier groups; degrades on G6 (hardest studied). | Mostly PoE-equivalent. |
| out / in | ≥ 60%. Within-group pair transfer learned at scale. | Group-dependent — strong on G1–G3, weak on G4/G6. | No transfer; LoRA is per-pair. |
| **out / out** | ≥ 50%, **with per-group structure matching Plans 09–10**. | Per-group structure visible but absolute scores below Plan-10's per-group pooled LoRA. | Indistinguishable from PoE. |

Buckets across the whole experiment:

| Bucket | What you see across the four quadrants | Means |
|---|---|---|
| **Poor** | Sanity quadrant (in/in) fails. | Training didn't actually fit. Likely a loader bug pooling across pairs. Fix and re-run. |
| **Bad** | All non-trivial quadrants ≤ 30% recognisable; pooled LoRA ≈ vanilla PoE. | The single LoRA cannot span the taxonomy. Catalogue at Plan-10's per-group granularity instead. |
| **Unknown** | Pooled LoRA matches Plan-10's per-group LoRAs on the easy groups but visibly degrades on hard ones. | The taxonomy ordering carries deployment-relevant information. Report as "one LoRA covers the easy groups; G5–G6 need their own." This is a defensible, honest paper finding. |
| **Good** | Pooled LoRA is within visual tie of per-group LoRAs across all six groups in the held-pair × held-seed quadrant; Task D shows cosine to per-cell `Δ_t` exceeding cosine to `Δ̄_t^(P)` and `Δ̄_t^(s)`. | One LoRA covers the taxonomy. Strongest version of the deployable contribution. |
| **Surprising-good** | Pooled LoRA *beats* Plan-09 per-pair LoRAs on the in/in cells, and the held-pair × held-seed quadrant matches per-group LoRAs. | The single LoRA learned a *general* corrector — the pair pool is acting as data augmentation for the residual signal, not just as coverage. Motivates a sequel on bigger pools. |

## What this plan does *not* do

- **Architecture sweeps.** Rank 8, `attn2` targets. If the run lands
  `Unknown` or `Bad`, the *first* follow-up (out of scope here but
  flagged) is rank ∈ {16, 32} and `attn1 + attn2` targets, not a new
  trainer.
- **Outcome supervision.** As in every plan in this arc, no DRaFT /
  DDPO / hypernet. Listed in 04's scope discussion as future work.
- **Cross-cell aggregation metrics as the headline.** Eyeball
  contact sheets remain primary; Task D and per-quadrant counts are
  supporting evidence.
- **Generalisation outside the six taxonomy groups.** The pair pool is
  bounded by the pilot tree.

## Risk register (kept minimal)

| Risk | Trigger | Response |
|---|---|---|
| 48-cell training run OOMs or runs slow | First epoch wall time | Pool by streaming the cache (don't load all cells into RAM); already supported by Plan 08's loader. |
| Pair pool too small — LoRA overfits to 5 pairs | Held-pair × held-seed quadrant collapses while in-pair quadrants thrive | Run the richer-pool variant (10–15 pairs). |
| Imbalanced cell counts across groups (a group has fewer good seeds) | Per-group eyeball reads | Weighted sampler in the loader (1/n_cells per group). |
| Single rank-8 LoRA is capacity-limited | Per-group degradation only on G6 (the hardest studied group) | Rank-16 run as a follow-up, not part of this plan. |

## Status — 2026-05-23

| Item | Done | To do |
|---|:---:|:---:|
| Pair pool design (G1–G4, G6; G5 deferred) | ✅ (above) | |
| Seed pool design | ✅ (above) | |
| Cache cells for the 5 representative pairs at seeds {1..8} | Partial (G4 and G6 already cached at seeds {1..12, 42}) | ⬜ for G1, G2, G3 |
| Cache cells for the 5 sibling pairs at seeds {9..12} | | ⬜ |
| `poe_repair.experiments.cross_pair_lora_pooling.pair_pool` | | ⬜ |
| `poe_repair.experiments.cross_pair_lora_pooling.train_pooled` | | ⬜ |
| `poe_repair.experiments.cross_pair_lora_pooling.sample_crossbar` | | ⬜ |
| `poe_repair.experiments.cross_pair_lora_pooling.contact_sheet` | | ⬜ |
| `scripts/cross_pair_lora_pooling/train_all_groups.sh` | | ⬜ |
| Task A — pool YAMLs + leak guards | | ⬜ |
| Task B — train pooled LoRA across 40 cells | | ⬜ (~20 h GPU) |
| Task C — four-quadrant evaluation | | ⬜ |
| Task D — Δ̄_t bridge across (pair, seed) | | ⬜ |
| Contact sheets, headline at held-pair × held-seed quadrant | | ⬜ |
| Per-quadrant classification table | | ⬜ |
| Optional richer-pool variant | | ⬜ (conditional on `Good`) |
