# Mono-during-inference residual diagnostic

The goal of this thread is **not** a deployable method. It is a *diagnostic*
that measures the gap between the failing PoE composition and the working Mono
(literal joint prompt) composition, step by step. Mono is treated as an
oracle / ceiling: at λ=1 the sampler is Mono; at λ=0 the sampler is PoE; in
between we get the λ-interpolated residual.

The central object is the guided residual
``r_t = ε̃_Mono(x_t) − ε̃_PoE(x_t)``
evaluated *in guided epsilon space, not raw UNet output*. The
`teacher_residual` composer caches `Δ_t = ε̃_Mono − ε̃_PoE` per step alongside
the decoded image for each λ on the grid.

## What's in this thread

  - `poe_repair/composers/teacher_residual.py` — λ-interpolated PoE↔Mono
    composer.
  - `poe_repair/methods/_sampling.py::run_teacher_residual` — the underlying
    sampler. Records residual norms / direction stats per step.
  - `poe_repair/composers/poe_internal.py`,
    `poe_repair/methods/_poe_internal.py` — basin-template based corrector
    used by **idea1**.
  - `poe_repair/experiments/idea1/` — basin-template barrier ablation.
  - `poe_repair/experiments/idea5a/` — λ-trajectory snapshots: latent
    differences, attention drift, decoded thumbnails along the PoE→Mono path.
  - `poe_repair/experiments/veracity/` — the λ-grid sweep that produces the
    canonical `teacher_residual_const_lam{000,010,...,100}/` output tree
    consumed by idea1 and idea5a.

## Reproducing from a clean checkout

Outputs and checkpoints are not preserved for this thread (it is code-only).
Use the `co3` conda env's python:

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
export CUDA_VISIBLE_DEVICES=1
```

### Step 1 — veracity λ-sweep (produces the shared output cells)

```bash
$PY -m poe_repair.experiments.veracity \
    --pair "a cat|a dog" --seed 42
```

Writes `outputs/veracity/pairs/a_cat__x__a_dog/seed_42/teacher_residual_const_lam{NNN}/`
for `NNN ∈ {000, 010, 020, ..., 100}` (PoE → Mono in λ steps of 0.1).
Each cell directory contains the decoded image, the cached per-step
residuals, and per-step attention maps if `--record-cross-attention`.

### Step 2 — idea1 (basin-template barrier ablation)

Reads the veracity λ=0 (PoE) and λ=1 (Mono) cells as anchors.

```bash
$PY -m poe_repair.experiments.idea1 --pair "a cat|a dog" --seed 42
```

### Step 3 — idea5a (λ-trajectory snapshots)

Reads the full λ grid from step 1.

```bash
$PY -m poe_repair.experiments.idea5a --pair "a cat|a dog" --seed 42
```

## Why Mono is the right ceiling, not the right method

Mono uses the *literal* joint prompt e_J — "a cat and a dog" encoded as one
SDXL caption. That works because SDXL's text encoder has co-occurrence
knowledge baked in: the joint encoding routes attention correctly. PoE has
to *infer* that co-occurrence from two independent prompts and gets it
wrong on collision pairs. The residual measures exactly how wrong.

Mono is not deployable as a real composition method — it requires the joint
prompt at inference, which defeats the purpose of compositional generation
("given two trained concepts I never co-trained, generate them together").
But as a *diagnostic ceiling* it pins down what a perfect repair would
recover.
