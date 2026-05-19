# Phase 5 — Group-A external-corrector failures

## Question

Phase 4 puts the correction *inside* the UNet (LoRA on cross-attn).
Group A asks the opposite: what if the corrector is a small,
standalone network that reads `(z_t, t, e_J)` and outputs an additive
ε-correction `r̂_t`, leaving SDXL completely frozen? Three flavours:

- **A1 — Latent CNN.** Flat 5-block CNN at latent resolution with FiLM
  conditioning on `t` and `e_J_pool`. ~1.2 M params.
- **A2 — Latent UNet.** Small 2-scale UNet with skip connections. Same
  FiLM conditioning. ~6 M params.
- **A3 — Frozen-feature MLP.** Runs SDXL frozen with the joint prompt,
  hooks the mid-block output (`(1, 1280, 32, 32)`), and feeds it to a
  small CNN head. ~3 M params.

Do any of these match Phase 4's LoRA on the same cell?

## Why this phase exists

The Phase-4 LoRA's success could plausibly be a property of the LoRA
hypothesis space (rank-8 perturbation of cross-attn projections), the
training data (cached PoE↔Mono residuals), or both. Group A swaps the
hypothesis space while keeping the data identical. If any A-variant
matches LoRA, the LoRA's choice of attachment isn't load-bearing. If
all three fail, the inside-the-UNet placement is the load-bearing
ingredient.

**As of 2026-05-19, all three are reported negatively** — they don't
reach the Phase-4 visual quality on the headline cell. This phase
documents that result and explains why we ran each one.

## Code

- Shared scaffolding: `poe_repair/students/common.py` (FiLM block,
  sinusoidal time embedding, pool projection).
- Architectures:
  - `poe_repair/students/latent_cnn.py`,
  - `poe_repair/students/latent_unet.py`,
  - `poe_repair/students/frozen_feature_mlp.py`.
- Experiment package: `poe_repair/experiments/group_a_failure/`
  (one package; `--technique` selects which architecture).
- Sampler: `poe_repair/methods/_sampling.py::run_external_corrector_inject`.
  Takes a `corrector` callback. At `λ=0` reproduces vanilla PoE
  bit-identically (canary).
- Replay-baseline sampler: `run_teacher_residual` (already used in
  Phase 1) doubles as the "inject the cached `r_t` directly with no
  learner" sanity check that has to pass before training.

Run-directory layout:
`outputs/group_a_failure/<technique>/cat_dog/seed_42/<run_id>/` with
the same probe / figure / checkpoint conventions as the LoRA experiment.

## Commands

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
export CUDA_VISIBLE_DEVICES=0   # one technique per GPU
```

### Run all three (parallel on a 2-node × 2-GPU layout)

```bash
# A1 — latent CNN
CUDA_VISIBLE_DEVICES=0 $PY -m poe_repair.experiments.group_a_failure \
    --technique latent_cnn \
    --pair a_cat__x__a_dog --seed 42 --split heldout \
    --total-epochs 600 --probe-every-epochs 50 --lr 1e-4

# A2 — latent UNet
CUDA_VISIBLE_DEVICES=1 $PY -m poe_repair.experiments.group_a_failure \
    --technique latent_unet \
    --pair a_cat__x__a_dog --seed 42 --split heldout \
    --total-epochs 600 --probe-every-epochs 50 --lr 1e-4

# A3 — frozen-feature MLP
CUDA_VISIBLE_DEVICES=0 $PY -m poe_repair.experiments.group_a_failure \
    --technique frozen_feature_mlp \
    --pair a_cat__x__a_dog --seed 42 --split heldout \
    --total-epochs 600 --probe-every-epochs 50 --lr 1e-4 --train-batch-size 2
```

### Dry-run wiring check (any technique, no training)

```bash
$PY -m poe_repair.experiments.group_a_failure \
    --technique latent_cnn --dry-run
```

Runs one probe at epoch 0 against an untrained corrector, writes the
λ=0 / λ=1 strip. λ=0 must be byte-identical to vanilla PoE.

## How to read the result

| Bucket | What you see | Means |
|---|---|---|
| **Poor** | Probe at λ=0 differs from vanilla PoE. Or `run_teacher_residual` replay at λ=1 (no learner) doesn't reproduce Mono on this cell. | Sampler abstraction is wrong, or the cache is corrupt. Fix before reading any technique. The Phase-1 cache was supposed to validate this. |
| **Bad** | Loss collapses to ~0 within a few epochs and the probe at λ=1 is identical to PoE. | Trivial-solution failure — corrector outputs zero. Likely σ-window sampler underweights everything, or the FiLM conditioning is broken. Diagnose, don't move on. |
| **Unknown** | Loss decreases steadily, probes show *some* qualitative movement toward two animals at the highest λ value, but the morph is incomplete by epoch 600. Strict ordering of A1 vs A2 vs A3 is unclear. | Architecture-level partial morph — record as "less complete than Phase 4" without escalating to ablation. The headline failure case is established; the comparison between A1/A2/A3 is a footnote. |
| **Good (this phase's *good* is documented failure)** | All three architectures train cleanly (loss decreases monotonically, `‖r̂_t‖` reaches the same order of magnitude as `‖r_t‖` from the cache), but the final-epoch probe still shows a chimera or single-concept image at λ=1. Phase 4's LoRA on the same cell is visibly better. | We've cleanly contrasted external correctors against the inside-the-UNet LoRA on identical data. Reports for Phase 5 say "this is *why* the LoRA placement matters" not "this is broken." |

## Why each architecture was worth running anyway

- **A1 (Latent CNN).** Cheapest plausible learner that respects
  spatial structure. If A1 had worked, we wouldn't have needed A2/A3.
- **A2 (Latent UNet).** Tests whether multi-scale residuals (coarse
  split + fine texture corrections) are what the latent-CNN can't
  capture.
- **A3 (Frozen-feature MLP).** Tests whether the right inductive bias
  is "use what SDXL already encoded about `(z_t, t, e_J)`." A failure
  here is informative: the residual cannot be read off SDXL's frozen
  mid-block features at all.

## What this phase does *not* do

- Cross-seed or cross-pair runs (single-seed beachhead).
- Outcome-supervised fine-tuning (DRaFT / DDPO).
- Quantitative VQA gating — pass/fail is by the same eyeball criterion
  as Phase 4.
- Hyperparameter sweeps. One default config per technique. The
  failure is reported under those defaults; tuning is out of scope.

## Status — 2026-05-19

| Item | Done | To do |
|---|:---:|:---:|
| `run_external_corrector_inject` sampler landed (canary at λ=0 passes) | ✅ | |
| Shared scaffolding (`students/common.py`, FiLM block, time embedding) | ✅ | |
| A1 latent CNN — 600-epoch training run + probes | ✅ | |
| A2 latent UNet — 600-epoch training run + probes | ✅ | |
| A3 frozen-feature MLP — 600-epoch training run + probes | ✅ | |
| Cumulative grids (epoch × λ) per technique | ✅ | |
| Combined 3-row comparison figure (A1 / A2 / A3 × λ at final epoch) | ✅ | |
| Negative-result writeup in `group-a-failure.md` (root) | ✅ | |
| Final verdict notes per technique in master writeup | | ⬜ (one paragraph each) |
