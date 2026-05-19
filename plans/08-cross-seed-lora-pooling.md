# Phase 8 — Cross-seed pooled LoRA

## Question

Phase 4 trained one LoRA on one seed. Phase 7 measured whether `Δ_t`
is the same kind of object across seeds and landed close to "no — at
the cross-seed mean it's noise, with detectable but small structure."

Now: train a single LoRA on a *pool* of seeds for the same pair, then
evaluate it on held-out seeds. Two answers we need:

1. **Does pooling generalise at all?** Held-out seeds get sampled with
   the pooled LoRA. Do those samples look like cat + dog, or like the
   PoE chimera?
2. **Is the pooled LoRA recovering the seed-average direction Δ̄_t, or
   doing something genuinely seed-conditional?** Compare the per-step
   `ε_LoRA,t` on held-out trajectories to both Δ̄_t and to the
   held-out seed's own Δ_t.

If Phase 7 landed "shared signal" we'd expect cosine alignment with
Δ̄_t to be high. Phase 7 actually landed close to "seed noise," so the
honest prior is that the pooled LoRA either recovers a weak seed-mean
or finds a different but useful prompt-conditional correction.

## Why this phase exists

This phase decides whether the single-seed LoRA generalises along the
seed axis with no architectural changes (the cheapest possible
extension), or whether cross-seed generalisation needs new machinery
(hypernet, outcome supervision, per-cell library — out of scope here).

## Code

All cross-seed pooling code lives under
`poe_repair/experiments/cross_seed_lora_pooling/` with thin shell
runners under `scripts/cross_seed_lora_pooling/`. The trainer never
touches the single-seed Phase-4 reference artefact under
`outputs/lora/cat_dog/seed_42/results/`.

| Module | What it does |
|---|---|
| `seed_pool.py` | Loads `seed_pool.yaml`; enforces `train_pool ∩ held_out = ∅` at startup. |
| `train_pooled.py` | Pooled trainer. `--k <int>` picks the first *k* seeds from `train_pool`. `--single-seed-pick <s>` overrides for the k=1 picks. |
| `sample_heldout.py` | Loads a checkpoint, attaches LoRA, samples held-out seeds via `run_lora_residual_inject`. `--record-eps` dumps per-step `ε_PoE_lora,t` for Task D. |
| `step0_prescreen.py` | Inference-time mono-average pre-screen. Per held-out seed samples four conditions: PoE / r̄_t-inject / oracle r_t^(s*)-inject / mono. No training. |
| `task_d_bridge.py` | Reads dumped ε records + cached Δ_t; emits cos / norm-ratio / residual-of-residual curves vs t and a spatial residual-of-residual heatmap. |
| `contact_sheet.py` | Renders Task B / Task C grids from finished runs. |
| `trajectory_diagram.py` | Decodes per-step Tweedie x̂_0 for a chosen run to visualise the pooled-LoRA trajectory. |

Shared dependencies:

- Loader / leak guard: `poe_repair/training_cache.py::CellPath`,
  `load_cached_steps`, `delta_t_from_raw`. `DEFAULT_CACHE_ROOT` reads
  `POE_REPAIR_TRAINING_CACHE` env var (defaults to
  `/datasets/mmolefe/poe_repair_min/outputs/training_cache`).
- Sampler: `poe_repair/methods/_sampling.py::run_lora_residual_inject`.
  Already supports a `record_eps_path` style hook via
  `sample_heldout --record-eps`.

Shell runners (one per task — read them before running, they show the
canonical flag combinations and parametrise through env vars):

- `scripts/cross_seed_lora_pooling/task_b_learning_curve.sh`
- `scripts/cross_seed_lora_pooling/task_c_per_seed_ceiling.sh`
- `scripts/cross_seed_lora_pooling/sweep_s1_rank.sh`

## Cache prerequisites

Inventory on the cluster:
`/datasets/mmolefe/poe_repair_min/outputs/training_cache/heldout/a_cat__x__a_dog/seed_{1..12,42}/`
— 13 cells, ~461 MB total. **The cache directory `outputs/` is
git-ignored** (including `seed_pool.yaml`); a fresh clone has *no*
cache cells and no seed pool YAML.

For k=16 we'd need 16 train + 4 held-out = 20 cells. Either generate
seeds 13–19 or stay at k ∈ {1, 4, 8} with the cells already in hand.
**Default: k ∈ {1, 4, 8}.** No cache-generation script ships in this
repo — the cells were produced on the cluster and should be rsync'd
to any new machine, not regenerated.

### Bootstrapping a fresh checkout (e.g. `hippo`)

```bash
# From a machine that has the cache (e.g. mscluster106), push to the new host:
ssh <host> 'mkdir -p <repo>/outputs/training_cache/heldout/a_cat__x__a_dog \
                     <repo>/outputs/cross_seed_lora_pooling'

rsync -avzP \
  /datasets/mmolefe/poe_repair_min/outputs/training_cache/heldout/a_cat__x__a_dog/ \
  <host>:<repo>/outputs/training_cache/heldout/a_cat__x__a_dog/

rsync -avzP \
  outputs/cross_seed_lora_pooling/seed_pool.yaml \
  <host>:<repo>/outputs/cross_seed_lora_pooling/seed_pool.yaml
```

Then on the new host, before any cross-seed run:

```bash
export POE_REPAIR_TRAINING_CACHE=<repo>/outputs/training_cache
```

(Or pass `--cache-root <repo>/outputs/training_cache` to each module
— the shell runners don't forward it, so the env var is simpler.)

## Commands

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
export CUDA_VISIBLE_DEVICES=1
export POE_REPAIR_TRAINING_CACHE=$PWD/outputs/training_cache    # if not the cluster default
```

### Step 0 — inference-time mono-average pre-screen (no training)

```bash
$PY -m poe_repair.experiments.cross_seed_lora_pooling.step0_prescreen
```

Per held-out seed reads the cached Δ_t for the train pool, builds
`r̄_t = (1/K) Σ_s r_t^(s)`, then samples four conditions (PoE /
r̄_t-inject / oracle r_t^(s*)-inject / mono) and writes a contact-sheet
PNG plus metadata under
`outputs/cross_seed_lora_pooling/step0_prescreen/`. The pool is read
from the YAML — no `--train-pool` flag.

### Task A — seed pool config

The pool lives at `outputs/cross_seed_lora_pooling/seed_pool.yaml`
(committed already in the cluster checkout but git-ignored, so it must
be present on every host before any pooled run). Verify the leak guard:

```bash
# Make a deliberately-broken pool YAML, point the trainer at it, expect abort.
$PY -m poe_repair.experiments.cross_seed_lora_pooling.train_pooled \
    --k 1 --total-epochs 1 \
    --seed-pool-path /tmp/leak.yaml --dry-run
```

Where `/tmp/leak.yaml` puts seed 9 in both `train_pool` and `held_out`.
The run aborts at `seed_pool.load_seed_pool(...)` with a leak error.

### Task B — pooled learning curve in k

Use the shell runner (handles run IDs, sampling, and W&B wiring):

```bash
bash scripts/cross_seed_lora_pooling/task_b_learning_curve.sh
```

Default `KSET="1a 1b 4 8"` (k=1 with `--single-seed-pick 1`, k=1 with
`--single-seed-pick 5`, k=4, k=8). Override via env vars:

```bash
EPOCHS=1600 KSET="4 8" CUDA_VISIBLE_DEVICES=1 \
    bash scripts/cross_seed_lora_pooling/task_b_learning_curve.sh
```

Each run writes a checkpoint under
`outputs/cross_seed_lora_pooling/task_b_learning_curve/k<label>__ep<EPOCHS>/`
and immediately calls `sample_heldout --record-eps` on the four
held-outs (records ε_PoE_lora,t for Task D).

Direct invocation if you don't want the runner:

```bash
$PY -m poe_repair.experiments.cross_seed_lora_pooling.train_pooled \
    --k 4 --total-epochs 1600 \
    --output-root outputs/cross_seed_lora_pooling/task_b_learning_curve \
    --run-id k04__ep1600

$PY -m poe_repair.experiments.cross_seed_lora_pooling.sample_heldout \
    --checkpoint outputs/cross_seed_lora_pooling/task_b_learning_curve/k04__ep1600/checkpoints/lora_step_<...>.pt \
    --out-dir   outputs/cross_seed_lora_pooling/task_b_learning_curve/k04__ep1600/samples/heldout \
    --record-eps
```

### Task C — per-seed ceiling (epoch parity)

The runner generates a temporary one-seed pool YAML per held-out seed
(since the standing `seed_pool.yaml` only has seeds 1–8 in
`train_pool`):

```bash
bash scripts/cross_seed_lora_pooling/task_c_per_seed_ceiling.sh
```

Defaults: `CEILING_SEEDS="9 10 11 12"`, `EPOCHS=1600` (8 × 200 = epoch
parity for the k=8 pooled run). Each per-seed LoRA samples only on its
own seed — that's what "ceiling" means.

### Task D — Δ̄_t bridge

Requires Task B's sample step to have been run with `--record-eps`
(the shell runner does this by default).

```bash
$PY -m poe_repair.experiments.cross_seed_lora_pooling.task_d_bridge \
    --pooled-run outputs/cross_seed_lora_pooling/task_b_learning_curve/k08__ep1600
```

Writes cosine / norm-ratio / residual-of-residual curves per held-out
seed and a spatial heatmap for one seed in the commit window.

### Contact sheets (final read)

```bash
$PY -m poe_repair.experiments.cross_seed_lora_pooling.contact_sheet --task B
$PY -m poe_repair.experiments.cross_seed_lora_pooling.contact_sheet --task C
```

### Optional sweep S1 — rank at k=8

```bash
bash scripts/cross_seed_lora_pooling/sweep_s1_rank.sh    # ranks 16, 32 by default
```

## How to read the result

| Bucket | What you see | Means |
|---|---|---|
| **Poor** | Pooled trainer's load-time assertion does not fire on a deliberate leak (put seed 9 in both pools). | Leak guard is broken; downstream comparisons are not trustworthy. Fix first. |
| **Bad** | Held-out samples at every k look identical to PoE. Per-seed LoRA on the same held-out seed *also* fails. | The held-out seeds are themselves hard for the LoRA approach — pooling can't be blamed and the per-seed ceiling itself is a no. Re-pick held-out seeds or report this as a seed-difficulty study, not a pooling result. |
| **Unknown** | Pooled-k8 produces visibly more separated animals than PoE on 2 of 4 held-out seeds; the other 2 are ambiguous. Per-seed ceilings are similar — neither cleanly wins. Task D shows moderate cosine alignment to Δ̄_t (0.4–0.6) that exceeds the random-seed-pair baseline but not by much. | Real but small pooling effect. Note in writeup; do not over-claim. The cheap inference-time average pre-screen (Step 0) probably looked similar — pooled training isn't adding much over averaging the residuals offline. |
| **Good** | Pooled-k8 produces clear cat + dog on ≥ 3 of 4 held-out seeds. Per-seed ceiling on the same held-outs is comparable (within visual tie). Task D shows pooled `ε_PoE_lora,t` is cosine-aligned with Δ̄_t at a level meaningfully above the random-seed-pair baseline. Step-0 mono-average pre-screen *also* gives recognisable cat+dog (consistent with Phase 7's "the structured part is weak but pooled training extracts it"). | Pooled LoRA is a real deployable extension of Phase 4. Cross-seed generalisation works for this pair at this rank. Honest scope: this is "this pair" not "all pairs." |
| **Surprising-good** | Pooled-k8 *beats* per-seed ceilings on most held-outs, AND Task D's cosine alignment to Δ̄_t is small while alignment to held-out Δ_t^(s) is high. | The pooled LoRA learned something prompt-conditional that is not the seed-average. This would be the most interesting outcome and motivates a follow-on hypernet / prompt-conditioning study (out of scope here). |

The Phase-7 result on disk pushes the expected outcome toward
**Unknown** unless the pooled training extracts more structure than
the offline mean. Step 0 is designed to surface that distinction
before training any pooled LoRA.

## Risk register (kept minimal)

| Risk | Trigger | Response |
|---|---|---|
| Step 0 mono-average already ≈ mono on held-outs | Eyeball Step-0 grid | Task D becomes confirmatory; still run Tasks B & C for the deployment-side question. |
| k=16 cache cells not ready in time | Day 1 | Run k ∈ {1, 4, 8} only — three points still readable. |
| Per-seed LoRA fails on a specific held-out | Task C sheet | Carve that seed out of the pooled-vs-ceiling read; document as a difficulty case. |
| Task D cosine ambiguous (~0.5) at most t | Curves | Compare to the random-seed-pair baseline cosine. If LoRA cosine exceeds raw-seed-pair cosine, the LoRA is at least as seed-invariant as the data permits. |

## What this phase does *not* do

- Cross-pair generalisation. Cat × dog only.
- Outcome supervision (DRaFT / DDPO / hypernet LoRAs). All listed as
  follow-on work in [04-lora-single-seed.md](04-lora-single-seed.md)'s
  scope discussion; not in this plan.
- Bootstrap CIs or preregistered thresholds. Eyeball-first; expand to
  larger N held-outs only if any of the four reads is ambiguous.
- Touching `outputs/lora/cat_dog/seed_42/results/` — that is the
  reference single-seed artefact consumed by the inspector and must
  not be overwritten by pooled training.

## Status — 2026-05-19

| Item | Done | To do |
|---|:---:|:---:|
| Plan written and harmonised with Phase 7's `landing_6` result | ✅ | |
| Cache cells `seed_{1..12,42}` produced under `/datasets/.../training_cache/` (cluster) | ✅ | |
| `outputs/cross_seed_lora_pooling/seed_pool.yaml` written (cluster) | ✅ | |
| Pooled multi-seed loader (`training_cache.load_cached_steps` + seed-pool slicing) | ✅ | |
| Trainer entrypoint `cross_seed_lora_pooling.train_pooled` (`--k`, `--single-seed-pick`) | ✅ | |
| Held-out evaluator `cross_seed_lora_pooling.sample_heldout` (`--record-eps` for Task D) | ✅ | |
| Step-0 pre-screen module `cross_seed_lora_pooling.step0_prescreen` | ✅ | |
| `seed_pool.py` leak guard (`train_pool ∩ held_out = ∅`) | ✅ | |
| Task-D bridge module `cross_seed_lora_pooling.task_d_bridge` | ✅ | |
| Contact-sheet builder `cross_seed_lora_pooling.contact_sheet` | ✅ | |
| Shell runners `scripts/cross_seed_lora_pooling/{task_b,task_c,sweep_s1}*.sh` | ✅ | |
| Seeds 13–19 cache cells (only for k=16 curve point) | | ⬜ (optional) |
| Bootstrap fresh host: rsync cache + `seed_pool.yaml`, set `POE_REPAIR_TRAINING_CACHE` | | ⬜ (per new machine) |
| **Step 0** — actually run the pre-screen and inspect the contact sheet | | ⬜ |
| **Task A** — verify the leak guard aborts on a deliberately-broken pool | | ⬜ |
| **Task B** — pooled learning curve at k ∈ {1a, 1b, 4, 8} (`task_b_learning_curve.sh`) | | ⬜ (running on hippo as of 2026-05-19) |
| **Task B** — k=16 extension (only if cache cells 13–19 land) | | ⬜ (optional) |
| **Task C** — per-seed ceiling × 4 held-outs at epoch parity (`task_c_per_seed_ceiling.sh`) | | ⬜ |
| **Task D** — Δ̄_t bridge against Task B's recorded ε | | ⬜ |
| **Contact sheets** — Task B & Task C grids | | ⬜ |
| **Sweep S1** — rank ∈ {16, 32} at k=8 (`sweep_s1_rank.sh`) | | ⬜ (conditional on Task C verdict) |
| Final writeup answering the four "done criteria" questions | | ⬜ |
