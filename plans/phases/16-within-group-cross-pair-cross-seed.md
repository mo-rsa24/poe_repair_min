# Plan 16 — Within-group cross-pair LoRA (rung 4)

> Parent: [LORA_TAXONOMY_PLAN.md](LORA_TAXONOMY_PLAN.md). Fills the rung
> between [10-cross-seed-lora-per-group.md](10-cross-seed-lora-per-group.md)
> (single pair × N seeds + a single-pair sibling smoke) and
> [15-cross-pair-cross-seed-lora-mscluster.md](15-cross-pair-cross-seed-lora-mscluster.md)
> (all five studied groups × multiple pairs × multiple seeds). For each
> studied group in isolation: train a single LoRA on **7 within-group
> pairs across all 12 seeds**, then evaluate it on **3 held-out pairs
> from the same group across all 12 seeds**. The only axis held out at
> evaluation time is the *pair* axis.

## Question

For each studied group `G ∈ {G1, G2, G3, G4, G6}` (G5 deferred per
Plan 09; see "G5 deferral" below), pool training across `K_train = 7`
within-group pairs × seeds `{1..12}`. Then evaluate on `K_eval = 3`
held-out pairs from the same group, across **all** 12 seeds.

Two reads:

1. **Within-group cross-pair transfer.** On held-out pairs of group
   `G`, does the within-group pool produce recognisable composition?
   If yes, "group" is a meaningful pooling unit and the deployment
   story is "one LoRA per difficulty class."
2. **Within-group pool vs. single-pair baseline.** Does the
   `K_train = 7` pool beat Plan 10's single-pair-trained LoRA on the
   same held-out pair? Plan 10's `--heldout-pair` smoke measured
   single-pair → sibling transfer; this plan asks whether adding more
   within-group pairs helps.

The unit of analysis is the *group*. There are five independent
experiments (one per studied group); they share code, prompts, and
seed pool but no training cells.

### Why no seed-axis held-out

Rung 4 is about the *pair* axis. Seed generalisation was the question
of rungs 2 and 3 (Plans 08 and 10). Mixing the seed axis into the
held-out structure dilutes the question: an `out/out` failure could be
either pair-axis failure or seed-axis failure. By training each
within-group LoRA on all 12 seeds of its 7 training pairs and
evaluating held-out pairs across the same 12 seeds, every held-out
sample is a clean "this LoRA has seen seed `s` (just not on pair `P`)"
test. Any failure is unambiguously pair-axis.

Plan 15 still holds seeds `{9..12}` out — its question is *both* axes
at once, and the crossbar structure makes that read meaningful. Plans
08, 10, and 15 collectively own the seed axis; Plan 16 owns the pair
axis.

## Why this phase exists

The current arc skips a rung. Plan 10 trains on one pair × many seeds
per group and uses `--heldout-pair` as a *single-pair-trained* sibling
smoke. Plan 15 jumps straight to the broadest claim — one LoRA across
all five studied groups × five training pairs × eight seeds, evaluated
at the four-quadrant crossbar.

Plan 15's headline (held-pair × held-seed) cannot distinguish three
failure modes:

- **Group is the right unit, but cross-group pooling hurts.** A
  per-group within-group pool would land `Good`; the all-groups pool
  degrades because group `G'` gradients interfere with group `G`.
- **Group is not a useful unit.** Within-group pooling lands no better
  than Plan 10's single-pair baseline; "group" was just a labelling
  convenience and the deployable artefact is per-pair regardless.
- **The mechanism transfers across all groups.** Within-group pool and
  all-groups pool both land `Good`. Plan 15's claim stands.

Without rung 4 we can read Plan 15 as `Good`/`Bad`/`Unknown` but not
*why*. This phase is the disambiguator and the natural fallback if
Plan 15 lands `Mixed` — within-group pools become the deployable
catalogue rather than the single LoRA.

The cost is bounded because the Plan-15 code (multi-pair trainer,
sampler, contact sheet, Δ̄_t bridge) already runs the experiment shape
this plan needs; only the per-group YAMLs and a cache-build precondition
are new.

## Research-objective alignment

The deployable thread
([project_framing](../.claude/projects/-home-mscluster-mmolefe-Playground-PhD-poe-repair-min/memory/project_framing.md))
is the LoRA residual corrector. The honest deployment story picks one:

- "One LoRA covers the taxonomy" — Plan 15's strongest landing.
- "One LoRA per difficulty class" — Plan 16's strongest landing, and a
  defensible catalogue claim.
- "One LoRA per pair" — Plan 09/10's floor, the weakest deployable
  story but a real one.

Plan 16 picks the middle landing and asks whether it is supported. A
`Good` outcome is paper-grade on its own; combined with a `Mixed` Plan
15 it becomes the headline deployable claim. A `Bad` outcome reduces
the deployment unit to per-pair.

## G5 deferral

G5 (concept-pair entanglement) is deferred from Plan 16 for the same
reason it was deferred from Plans 09, 10, and 15: the entangled third
concept in G5 prompts (e.g. *a tuxedo × a flamingo*) is not specified
by the joint prompt itself. The cache target is therefore ambiguous
and any read on G5 would be untrustworthy. Listed in the pilot tree
for completeness; not trained or evaluated here. See
[plans/09-lora-taxonomy-single-seed.md](09-lora-taxonomy-single-seed.md#L91-L99)
for the original rationale.

## Per-group pair splits (proposed)

Pairs drawn from the new pilot tree at
`/datasets/mmolefe/neurips2026/pilot_5seeds/seed_1/<group>/` (10 pairs
per group). Splits picked to (a) keep already-cached pairs in the
train set wherever possible (minimises cache build), (b) keep Plan 15's
group representative in the train set, (c) put a difficulty-balanced
mix in the held-out set with at least one near-sibling of a train
pair so within-group transfer has something to lean on. Final
selection at plan-execution time.

| Group | Train (7 pairs) | Held-out (3 pairs) |
|---|---|---|
| **G1** co-occurrence | `a_dolphin__x__an_ocean_wave` ✱<br>`a_polar_bear__x__an_iceberg`<br>`a_camel__x__a_desert_landscape`<br>`a_butterfly__x__a_flower_meadow`<br>`a_deer__x__a_forest_clearing`<br>`a_horse__x__a_grassy_field`<br>`a_sailboat__x__a_harbor` | `a_duck__x__a_pond`<br>`a_flamingo__x__a_lagoon`<br>`a_lighthouse__x__an_ocean_with_stormy_waves` |
| **G2** factorization | `a_dog__x__oil_painting_style` ✱<br>`a_cat__x__charcoal_drawing_style`<br>`a_barn__x__pencil_drawing_style`<br>`a_bicycle__x__sketch_style`<br>`a_cactus__x__mosaic_style`<br>`a_camera__x__watercolor_style`<br>`a_castle__x__stained_glass_style` | `a_lighthouse__x__watercolour_style`<br>`a_teapot__x__claymation_style`<br>`a_train__x__pixel_art_style` |
| **G3** object+scene | `a_mailbox__x__a_snowfield` ✱<br>`a_fire_hydrant__x__a_snowfield`<br>`a_bookcase__x__a_glacier`<br>`a_candle__x__a_waterfall`<br>`a_lamppost__x__a_desert_dune`<br>`a_lighthouse__x__a_desert_dune`<br>`a_park_bench__x__a_sand_dune` | `a_phone_booth__x__a_tropical_beach`<br>`a_picnic_table__x__a_snowstorm`<br>`a_rowboat__x__a_cactus_garden` |
| **G4** dual-object | `a_typewriter__x__a_cactus` ✱<br>`a_drum_set__x__a_snowman`<br>`a_bathtub__x__a_streetlamp`<br>`a_birdcage__x__a_watering_can`<br>`a_briefcase__x__a_ceramic_bowl`<br>`a_chessboard__x__a_lantern`<br>`a_feather_pillow__x__a_cast_iron_pan` | `a_lab_microscope__x__a_hay_bale`<br>`a_microwave__x__a_potted_plant`<br>`a_suitcase__x__a_desk_fan` |
| **G6** concept collision | `a_cat__x__a_dog` ✱<br>`a_lion__x__a_dog`<br>`a_cat__x__a_horse`<br>`a_cat__x__a_lion`<br>`a_dog__x__a_horse`<br>`a_lion__x__a_horse`<br>`a_wolf__x__a_horse` | `a_cow__x__a_horse`<br>`a_tiger__x__a_dog`<br>`a_mug__x__a_wine_glass` |

✱ = Plan 15 representative for that group; must remain in `train`.

Joint-prompt convention (mechanical from slug):
- G1, G4, G6: `"<prompt_a> and <prompt_b>"`.
- G2, G3: `"<prompt_a> in <prompt_b>"`.

Both halves of the slug are split on `__x__` and `_`-to-space. Articles
(`a`, `an`, `the`) come from the slug as-is.

Per-group YAMLs live under
`outputs/cross_pair_lora_pooling/within_group/<group>/pair_pool.yaml`.

## Seed pool (shared)

```yaml
# outputs/cross_pair_lora_pooling/within_group/seed_pool.yaml
# All 12 seeds used both at training and evaluation; pair axis is the
# only held-out axis in Plan 16.
train_pool: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
held_out:   []
```

`seed_pool.assert_disjoint` accepts an empty `held_out` (the disjoint
check is `set(train_pool) & set(held_out)`, which is empty when one
side is). Verified against
[poe_repair/experiments/cross_pair_lora_pooling/seed_pool.py:31-36](../poe_repair/experiments/cross_pair_lora_pooling/seed_pool.py#L31-L36).

## Code

Plan 15's code runs this experiment shape verbatim. The only new code
is per-group output routing and a cache-build wrapper. Sampler is run
with `--quadrants in_in,out_in --out-in-train-seeds 12` to express
"sanity floor + held-pair × all-seeds" under the new seed structure.

| Module / file | Status | Role |
|---|---|---|
| `poe_repair.experiments.cross_pair_lora_pooling.pair_pool` | reused | YAML loader + disjoint guard. |
| `poe_repair.experiments.cross_pair_lora_pooling.seed_pool` | reused | Tolerates `held_out: []`. |
| `poe_repair.experiments.cross_pair_lora_pooling.pair_prompts` | reused; **extend YAML** | Add the ~37 new pair slugs across the five groups. |
| `poe_repair.experiments.cross_pair_lora_pooling.multi_pair_trainer` | reused verbatim | Per-step embedding lookup. |
| `poe_repair.experiments.cross_pair_lora_pooling.train_pooled` | reused; `--pair-pool` per group | Same entrypoint as Plan 15. |
| `poe_repair.experiments.cross_pair_lora_pooling.sample_crossbar` | reused with `--quadrants in_in,out_in --out-in-train-seeds 12` | `in_in` = sanity floor; `out_in` = held pair × all 12 seeds (headline). |
| `poe_repair.experiments.cross_pair_lora_pooling.contact_sheet` | reused | Renders `in_in` and `out_in` (the other two quadrants come out empty). |
| `poe_repair.experiments.cross_pair_lora_pooling.task_d_bridge` | reused | Δ̄_t bridge per `(pair, seed)`. |
| `scripts/cross_pair_lora_pooling/build_plan16_caches.sh` *(new)* | new | Iterates the per-group `pair_pool.yaml`s + `pair_prompts.yaml`, calls `scripts/build_training_cache.py` for every (pair, seed) cell not already on disk. Idempotent. |
| `scripts/cross_pair_lora_pooling/run_within_group.sh` *(updated)* | updated | Per group: leak-guard → train → sample (`in_in,out_in`) → contact-sheet → Task D. |
| `outputs/cross_pair_lora_pooling/within_group/{g1,g2,g3,g4,g6}/pair_pool.yaml` | new | 5 YAMLs (Pair-splits table above). |
| `outputs/cross_pair_lora_pooling/within_group/seed_pool.yaml` | new | One YAML; `train_pool=[1..12], held_out=[]`. |

Output namespace stays separate from Plan 15: per-group runs land under
`outputs/cross_pair_lora_pooling/within_group/<group>/main/`. Plan 15's
`outputs/cross_pair_lora_pooling/all_groups/main/` is read-only here —
it provides the all-groups comparison column in the per-group contact
sheets, where present.

## Cache prerequisites

Each group needs 10 pairs × 12 seeds = **120 cells**. Current coverage
on `/datasets/mmolefe/poe_repair_min/outputs/training_cache/heldout/`
(audited 2026-05-27):

| Group | Pairs already cached at `{1..12}` (12 cells) | Pairs cached at `{9..12}` only (4 cells) | Pairs at 0 cells | New cells to build |
|---:|---|---|---:|---:|
| G1 | dolphin/ocean_wave | polar_bear/iceberg | 8 of 10 | 104 |
| G2 | dog/oil_painting | cat/charcoal | 8 of 10 | 104 |
| G3 | mailbox/snowfield | fire_hydrant/snowfield | 8 of 10 | 104 |
| G4 | typewriter/cactus | drum_set/snowman | 8 of 10 | 104 |
| G6 | cat/dog | lion/dog | 8 of 10 | 104 |
| **Total** | 5 | 5 | 40 | **~520 new cells** |

`scripts/cross_pair_lora_pooling/build_plan16_caches.sh` walks the
required (group, pair, seed) tuples and calls
`scripts/build_training_cache.py` only for missing cells. Estimated
build wall time at ~75 s/cell on RTX 8000: **~11 h GPU**.

The Plan 15 `wolf/husky` (G6 sibling) cache is *not* used by Plan 16 —
that slug is not in the new pilot tree's G6 list. Its cache stays on
disk for Plan 10/15 use; Plan 16 ignores it.

## Pair prompts (canonical strings)

Generated mechanically from the slug, with per-group connector. The
five Plan-15 train pairs and five Plan-15 siblings are already in
`outputs/cross_pair_lora_pooling/pair_prompts.yaml`; this plan extends
that file with the remaining 37 entries.

Example for one G6 entry:

```yaml
a_lion__x__a_dog:
  prompt_a: "a lion"
  prompt_b: "a dog"
  joint_prompt: "a lion and a dog"
```

Full additions live in the extended `pair_prompts.yaml` after Step 1
runs.

## Commands

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
export CUDA_VISIBLE_DEVICES=0
export POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/outputs/training_cache
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
```

### Step 1 — extend `pair_prompts.yaml` with the 37 new slugs

Edit
[outputs/cross_pair_lora_pooling/pair_prompts.yaml](../outputs/cross_pair_lora_pooling/pair_prompts.yaml).
This is mechanical; the slug-parsing rule above gives one entry per
pair. Performed at plan-execution time; no GPU.

### Step 2 — build the ~520 missing cache cells (~11 h GPU)

```bash
bash scripts/cross_pair_lora_pooling/build_plan16_caches.sh
```

Driver walks `outputs/cross_pair_lora_pooling/within_group/<g>/pair_pool.yaml`
× seeds `{1..12}`, skips cells already on disk, and calls
`scripts/build_training_cache.py` only for the missing ones.
Idempotent — safe to re-run after partial completion.

Verification:

```bash
for g in g1 g2 g3 g4 g6; do
  PAIRS=$( $PY -c "import yaml,sys; d=yaml.safe_load(open('outputs/cross_pair_lora_pooling/within_group/$g/pair_pool.yaml')); print(' '.join(d['train']+d['heldout']))" )
  for p in $PAIRS; do
    n=$(ls /datasets/mmolefe/poe_repair_min/outputs/training_cache/heldout/$p/ 2>/dev/null | grep -c '^seed_\(0\?[1-9]\|1[012]\)$')
    printf '%s  %-50s  %s\n' "$g" "$p" "$n"
  done
done
```

Expect every row to show **12**.

### Step 3 — leak-guard smoke on all six YAMLs (no GPU)

```bash
$PY -m poe_repair.experiments.cross_pair_lora_pooling.seed_pool \
    --seed-pool-path outputs/cross_pair_lora_pooling/within_group/seed_pool.yaml \
    --check-only

for g in g1 g2 g3 g4 g6; do
  $PY -m poe_repair.experiments.cross_pair_lora_pooling.pair_pool \
      --pair-pool outputs/cross_pair_lora_pooling/within_group/$g/pair_pool.yaml \
      --check-only
done
```

### Step 4 — run one group end-to-end (smoke; start with G6)

```bash
GROUPS=g6 bash scripts/cross_pair_lora_pooling/run_within_group.sh
```

`run_within_group.sh` per group: leak-guards → trains pooled LoRA →
samples `--quadrants in_in,out_in --out-in-train-seeds 12` against the
final checkpoint → renders the contact sheet → invokes Task D.

Loss-curve sanity gate during the first hour:

```bash
tail -F outputs/cross_pair_lora_pooling/within_group/g6/main.log
```

The bucket loss must decrease monotonically through the first ~600
epochs. Flat / oscillating loss = per-step embedding lookup regression
in `multi_pair_trainer`; kill and inspect before letting it run for
~30 h.

### Step 5 — run all five groups (~150 h GPU total)

```bash
bash scripts/cross_pair_lora_pooling/run_within_group.sh
```

Default `GROUPS="g1 g2 g3 g4 g6"`, `EPOCHS=2400`, `RUN_ID=main`. Use
`tmux`/`nohup`; checkpoints every 200 epochs allow `--resume-from` if
a session drops.

If GPU time is tight: run G6 + G1 first (most opposite groups —
hardest concept-collision vs. easiest co-occurrence). If both land
`Good`, the rung is read; G2/G3/G4 become extension runs.

### Step 6 — Δ̄_t bridge per group

Folded into Step 5's driver. Direct invocation if needed:

```bash
for g in g1 g2 g3 g4 g6; do
  $PY -m poe_repair.experiments.cross_pair_lora_pooling.task_d_bridge \
      --pooled-run outputs/cross_pair_lora_pooling/within_group/$g/main \
      --pair-pool outputs/cross_pair_lora_pooling/within_group/$g/pair_pool.yaml \
      --seed-pool-path outputs/cross_pair_lora_pooling/within_group/seed_pool.yaml \
      --cells outputs/cross_pair_lora_pooling/within_group/$g/main/samples/cells.jsonl
done
```

Per cell, three cosines: against own `Δ_t`, against the within-group
pair-mean `Δ̄_t^(P)` (averaged over that group's 7 train pairs), and
against the seed-mean `Δ̄_t^(s)` (averaged over all 12 seeds × that
pair). The within-group pair-mean is a richer average than Plan 15's
(7 vs. 5 pairs) so the cosine is somewhat less noisy than Plan 15's
Task D.

## How to read the result

Plan 16 has *one* meaningful quadrant: `out_in` (held-pair × any-seed).
`in_in` is a sanity floor. The sample plan produces empty `in_out`
and `out_out` outputs by construction.

Per group, classify against the same eyeball criterion as Plans
09/10/15: the subject of the joint prompt is recognisably composed
(not the chimera, not just a partial component).

| Region (per group) | `Good` | `Mixed` | `Bad` |
|---|---|---|---|
| `in_in` (sanity, 7 cells × 1 seed) | ≥ 90% recognisable | ≥ 70% | < 50% — training collapsed; stop. |
| **`out_in` (headline, 3 pairs × 12 seeds = 36 cells)** | ≥ 50% recognisable, **and matches or exceeds Plan 10's `heldout_pair` smoke for that group on seeds `{9..12}`** | held-out pair composes on a subset of seeds; degrades on one of the three held pairs | indistinguishable from Plan 10's smoke (no benefit from the within-group pool). |

The "matches or exceeds Plan 10's `heldout_pair` smoke" criterion is
the key one for this rung. Plan 10's smoke is a one-pair-trained LoRA
applied to a sibling at seeds `{9..12}`. Plan 16 is a *seven-pair*
LoRA applied to held-out pairs at the same seeds (a subset of its
12-seed eval). If Plan 16's seed-`{9..12}` slice is no better than
Plan 10's smoke, the within-group pool isn't adding anything and
"group" is not a pooling unit.

Whole-rung bucket (aggregated across G1–G4, G6):

| Bucket | Landing pattern | Means |
|---|---|---|
| **Poor** | Any group's `in_in` fails. | Loader bug; fix and re-run. |
| **Bad** | All groups: `out_in` ≤ Plan 10's `heldout_pair` smoke. | "Group" is a diagnostic label, not a deployment unit. Catalogue is per-pair if Plan 15 also `Bad`; else Plan 15's `out/out` is the deployable. |
| **Mixed** | Some groups (typically G1–G3) `Good`; G4/G6 degrade or no better than Plan 10's smoke. | Taxonomy ordering carries deployment-relevant information. Honest finding: "within-group pooling helps for easier groups; hardest groups need per-pair LoRAs." |
| **Good** | All five groups: `out_in` ≥ 50% on at least 2 of 3 held pairs, exceeding Plan 10's smoke; Task D shows alignment to the within-group pair-mean. | "Group" is a deployable unit. Catalogue is five per-group LoRAs. If Plan 15 *also* `Good`, choose between the two on size/maintenance grounds; otherwise Plan 16 is the headline deployable. |
| **Surprising-good** | All groups `Good` AND the seed-`{9..12}` slice of `out_in` matches Plan 15's `out/out` per-group breakdown. | Per-group LoRAs are as good as one big LoRA. The bigger Plan-15 pool is acting as data augmentation, not coverage. Strong support for "one LoRA per difficulty class." |

## What this plan does *not* do

- **Cross-group transfer.** Each within-group LoRA is evaluated only
  on its own group. Cross-group transfer is Plan 15's territory.
- **Seed-axis held-out.** By design — see "Why no seed-axis held-out"
  above. Plans 08, 10, and 15 own the seed axis.
- **Architecture sweeps.** Rank 8, `attn2` targets, same as the rest
  of the arc.
- **Outcome supervision.** None.
- **G5.** Deferred per Plan 09.

## Risk register

| Risk | Trigger | Response |
|---|---|---|
| ~520-cell cache build OOMs or is interrupted mid-flight | `build_plan16_caches.sh` aborts | Driver is idempotent (skips finished cells). Re-launch and it picks up where it stopped. |
| New training pairs' residual norms differ visibly from cached pairs | First 100 epochs of training: per-pair loss buckets diverge | Inspect cached `Δ_t` magnitudes per pair; if norms differ by > 2×, add a per-cell residual normaliser to the loader (out of scope here; flag follow-up). |
| 7-pair training pool overfits to 7 pairs and fails on the held-out 3 | `out_in` collapses while `in_in` thrives | Either accept (the answer is "group is not a pooling unit at this scale") or extend training set to ~8-9 pairs and reduce held-out to 1-2 — losing some eval power but stressing transfer. |
| Plan 10's `heldout_pair` smoke missing for some groups | Contact sheet's reference column has placeholders | Re-run Plan 10's `sample_heldout --heldout-pair <slug>` for the missing groups before Plan 16's read. |
| ~30 h-per-group × 5 groups exceeds available GPU window | Schedule pressure | Run G6 + G1 only first (rung 4's most-opposite read); G2/G3/G4 become extension runs, not blockers. |
| G6 held-out pair `a_mug__x__a_wine_glass` is *too far* from training distribution (out-of-class for canid/ungulate group) | `out_in` clearly degrades only on `mug/wine_glass` | Read directionally — mug/wine_glass is a stress test, not the headline. Tiger/dog and cow/horse are the load-bearing G6 held-outs. |
| `multi_pair_trainer` regressed since Plan 15 last ran | First-epoch loss has wrong shape | Re-run Plan 15's leak-guard smoke and a one-epoch dry-run before Step 4. |

## Estimated wall time

| Phase | Code | GPU | Human |
|---|---:|---:|---:|
| Step 1 (extend prompts YAML) | 30 min | 0 | — |
| Step 2 (build ~520 cache cells) | 0 | ~11 h | — |
| Step 3 (leak guards) | 5 min | 0 | — |
| Step 4 (G6 smoke train + sample + sheet + Task D) | 0 | ~30 h | — |
| Step 5 (remaining 4 groups) | 0 | ~120 h | — |
| Read result | 0 | 0 | ~45 min |
| **Total (all 5 groups)** | **~35 min** | **~161 h** | **~45 min** |
| **Total (G6 + G1 smoke only)** | **~35 min** | **~71 h** | **~30 min** |

## Status — 2026-05-27 (v2; new pilot tree, 7+3 split, all-seeds eval)

| Item | Done | To do |
|---|:---:|:---:|
| Rung-4 design — within-group cross-pair, pair-axis-only held-out | ✅ (this plan) | |
| G5 deferred (Plan 09 rationale carried forward) | ✅ | |
| Plan 15 code (`pair_pool`, `multi_pair_trainer`, `sample_crossbar`, `contact_sheet`, `task_d_bridge`) | ✅ (on disk) | |
| Per-group 7 + 3 pair splits proposed | ✅ (this plan) | |
| Per-group `pair_pool.yaml` (× 5) | | ⬜ (Step 1) |
| Shared `seed_pool.yaml` (`train_pool=[1..12], held_out=[]`) | | ⬜ (Step 1) |
| `pair_prompts.yaml` extended with 37 new slugs | | ⬜ (Step 1) |
| `scripts/cross_pair_lora_pooling/build_plan16_caches.sh` | | ⬜ (Step 2 driver) |
| `scripts/cross_pair_lora_pooling/run_within_group.sh` (updated to `--quadrants in_in,out_in`) | | ⬜ (driver update) |
| Cache: 5 pairs at `{1..12}` (legacy Plan-15 train pairs) | ✅ (on `/datasets/`) | |
| Cache: 5 pairs at `{9..12}` (legacy Plan-15 siblings — 4 of 5 reusable in Plan 16) | ✅ (on `/datasets/`) | |
| **Cache: ~520 missing cells for the 40 new pair slugs** | | ⬜ (Step 2, ~11 h GPU) |
| Leak-guard smoke for all 5 groups | | ⬜ (Step 3) |
| G6 within-group train + sample + sheet + Task D (smoke) | | ⬜ (Step 4; ~30 h GPU) |
| G1 within-group train + sample + sheet + Task D (smoke) | | ⬜ (~30 h GPU) |
| G2/G3/G4 within-group runs | | ⬜ (~90 h GPU; conditional on G6 + G1 read) |
| Per-group classification + rung-4 bucket landing | | ⬜ |
