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

## Code (what exists vs. what needs to be built)

Existing.

- Single-seed trainer: `poe_repair/experiments/lora/` — copy as
  scaffold for pooled trainer.
- Cache loader: `poe_repair/training_cache.py::CellPath`,
  `load_cached_steps`, `delta_t_from_raw`.
- Sampler: `run_lora_residual_inject` (Phase 4) — accepts pinned init
  latents per seed.
- Inspector + manifest builders for the single-seed run.

To build.

| Component | Where it lives | Why |
|---|---|---|
| `load_cached_steps_pooled(cells, ...)` | `training_cache.py` | Concatenate cached steps across multiple cells; tag each entry with `source_seed`. |
| Pooled trainer entrypoint | `poe_repair/experiments/lora_pooled/` (sibling of `lora/`) | Don't touch the reference single-seed run dir; pooled trainer takes `--seed-pool` and `--held-out-seeds`. |
| Held-out evaluator | `lora_pooled/probe.py` | Loop the existing probe across held-out seeds with per-seed pinned init latents. |
| `record_eps_path` kwarg on `run_lora_residual_inject` | `_sampling.py` | Pickle per-step `ε_PoE_lora,t` and matching `x_t` — used for Task D's cosine analysis. |
| Inference-time mono-average sampler | `_sampling.py::run_constant_residual_inject` (new, ~50 lines) | Inject a `{step_index: r̄_t}` dict at every step. Used by the free Step-0 pre-screen. |
| `seed_pool.yaml` + load-time leak assertion | `outputs/cross_seed_lora_pooling/` | One file, fail-fast: any overlap between train pool and held-out aborts the run. |

## Cache prerequisites

Inventory:
`/datasets/.../training_cache/heldout/a_cat__x__a_dog/seed_{1..12,42}`
— 13 cells exist.

For k=16 we'd need 16 train + 4 held-out = 20. Either generate seeds
13–19 in parallel, or drop k=16 from the curve and run k ∈ {1, 4, 8}
with the cells already in hand. **Default: start with k ∈ {1, 4, 8}.**

## Commands

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
export CUDA_VISIBLE_DEVICES=1
```

### Step 0 — inference-time mono-average pre-screen (no training)

```bash
$PY -m poe_repair.experiments.lora_pooled.prescreen \
    --train-pool 1,2,3,4,5,6,7,8 \
    --held-out 9,10,11,12
```

Builds `r̄_t = (1/K) Σ_s r_t^(s)` from cached Δ_t, injects it at every
step on each held-out seed's PoE trajectory, decodes. Outputs two
grids: `[PoE | r̄_t-injected | r_t^(s*)-injected | mono]` per held-out
seed; per-step `‖r̄_t‖` vs `‖r_t^(s*)‖`.

### Task A — write seed-pool config + assert

```bash
$PY -m poe_repair.experiments.lora_pooled.write_seed_pool \
    --train-pool 1,2,3,4,5,6,7,8 --held-out 9,10,11,12 \
    --out outputs/cross_seed_lora_pooling/seed_pool.yaml
```

### Task B — pooled learning curve in k

```bash
# k=1 picks (two single-seed runs, just so the curve has a left edge)
$PY -m poe_repair.experiments.lora_pooled --seed-pool 1   --held-out 9,10,11,12 --total-epochs 200
$PY -m poe_repair.experiments.lora_pooled --seed-pool 5   --held-out 9,10,11,12 --total-epochs 200

# k=4
$PY -m poe_repair.experiments.lora_pooled --seed-pool 1,2,3,4         --held-out 9,10,11,12 --total-epochs 200

# k=8
$PY -m poe_repair.experiments.lora_pooled --seed-pool 1,2,3,4,5,6,7,8 --held-out 9,10,11,12 --total-epochs 200
```

Each run evaluates the trained LoRA on all four held-out seeds at the
final epoch and writes the decoded PNGs. The contact sheet is the
read.

### Task C — per-seed ceiling (with epoch parity)

For each held-out seed `s ∈ {9, 10, 11, 12}`, train one per-seed LoRA
using the existing single-seed CLI. To match the pooled-k8 gradient
budget, run for 1600 effective epochs:

```bash
for S in 9 10 11 12; do
  $PY -m poe_repair.experiments.lora \
      --pair a_cat__x__a_dog --seed $S --split heldout \
      --total-epochs 1600 --probe-every-epochs 200 \
      --lr 1e-4 --lora-rank 8
done
```

Comparison sheet: 4 × 4 grid, rows = held-out seeds, cols = [PoE |
pooled-k8 | per-seed | mono].

### Task D — Δ̄_t bridge

Re-run the held-out evaluator with `--record-eps-path` enabled to
dump per-step `ε_PoE_lora,t` for each held-out seed. Then:

```bash
$PY -m poe_repair.experiments.lora_pooled.bridge \
    --pooled-run outputs/cross_seed_lora_pooling/k_08/<run_id> \
    --held-out 9,10,11,12
```

Computes per-step cosine to Δ̄_t and to each Δ_t^(s), the
norm-ratio, and the residual-of-residual. Writes 3-panel curves per
held-out seed plus the cross-seed-pair baseline cosines.

### Optional sweep S1 — rank at k=8

Only if Task C reads "pooled < per-seed" and you suspect capacity:

```bash
for R in 16 32; do
  $PY -m poe_repair.experiments.lora_pooled --seed-pool 1,2,3,4,5,6,7,8 \
      --held-out 9,10,11,12 --total-epochs 200 --lora-rank $R
done
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
| Cache cells `seed_{1..12,42}` on disk under `/datasets/.../training_cache/` | ✅ | |
| **Infra I1** — `load_cached_steps_pooled(cells, ...)` multi-seed loader | | ⬜ |
| **Infra I2** — `poe_repair/experiments/lora_pooled/` trainer entrypoint | | ⬜ |
| **Infra I3** — held-out evaluator across a list of seeds | | ⬜ |
| **Infra I4** — `record_eps_path` kwarg on `run_lora_residual_inject` | | ⬜ |
| **Infra I5** — `seed_pool.yaml` + load-time leak assertion | | ⬜ |
| **Infra I6** — generate seeds 13–19 cache cells (only for k=16 curve point) | | ⬜ (optional) |
| New sampler `run_constant_residual_inject` for Step 0 pre-screen | | ⬜ |
| **Step 0** — inference-time mono-average pre-screen on held-outs | | ⬜ |
| **Task A** — write `seed_pool.yaml`, verify leak abort | | ⬜ |
| **Task B** — pooled learning curve at k ∈ {1, 4, 8} | | ⬜ |
| **Task B** — k=16 extension (only if I6 cells land in time) | | ⬜ (optional) |
| **Task C** — per-seed ceiling at epoch parity (4 held-outs × 1600 epochs) | | ⬜ |
| **Task D** — Δ̄_t bridge: cosine, norm-ratio, residual-of-residual curves | | ⬜ |
| **Sweep S1** — rank ∈ {16, 32} at k=8 (only if Task C reads "pooled < per-seed") | | ⬜ (conditional) |
| Final writeup answering the four "done criteria" questions | | ⬜ |
