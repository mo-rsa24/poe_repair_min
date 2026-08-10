# Residual diagnostics

Two sub-experiments under `poe_repair/experiments/residual_diagnostics/`
characterise the guided Mono–PoE residual

```
r_t = ε̃_Mono(x_t) − ε̃_PoE(x_t)
```

evaluated in *guided* epsilon space, not raw UNet output. The residual is
not a deployable method — it requires the literal joint prompt e_J at
inference. Its job is to define the diagnostic ceiling the LoRA repair is
chasing.

## Sub-experiments

**`existence/`** — verifies r_t exists and is structured. Runs the
11-point λ-sweep on cat × dog seed 42 (PoE at λ=0 → Mono at λ=1),
caches per-step residuals + decoded images, and runs the *PMI identity
check* (confirms Δ_t = ε̃_J − ε̃_PoE equals the algebraic rearrangement
`w·(ε_J + ε_∅ − ε_A − ε_B)` to numerical precision). Renders Fig 1
(existence + PMI identity), Fig 4 (sufficiency), App A (trajectory
independence: PoE-anchor vs Mono-anchor), App B' (per-seed detection
failure modes).

Outputs land under `outputs/residual_diagnostics/existence/`.

**`clip_window/`** — CLIP-as-diagnostic on the commitment window.
Pure post-hoc analysis: reads existence's cached residuals,
reconstructs Tweedie x̂_0 at chosen step indices for chosen λ values,
decodes through SDXL's VAE, and scores against CLIP text targets.
Answers:

- At which denoising step does CLIP first separate PoE from Mono?
- Does CLIP cleanly distinguish "cat and dog" from "cat" / "dog" alone
  on the Mono trajectory?
- Is there a step window where CLIP is informative *and* the trajectory
  is still correctable (before commitment)?

Outputs land under `outputs/residual_diagnostics/clip_window/`.

## Reproducing from a clean checkout

Both sub-experiments generate their own data; no external training cache
is needed. Use the `co3` conda env:

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
export CUDA_VISIBLE_DEVICES=1
```

### Step 1 — run both back-to-back

```bash
$PY -m poe_repair.experiments.residual_diagnostics \
    --pair "a cat|a dog" --seed 42
```

This runs `existence` first (produces the λ-grid + figures), then
`clip_window` (reads from existence's outputs and produces its own
figures).

### Or run them individually

```bash
$PY -m poe_repair.experiments.residual_diagnostics.existence \
    --pair "a cat|a dog" --seed 42

$PY -m poe_repair.experiments.residual_diagnostics.clip_window \
    --pair "a cat|a dog" --seed 42
```

The `existence` stage is idempotent: existing λ-cell outputs are reused
unless `--overwrite` is passed.

## Why Mono is the right ceiling, not the right method

Mono uses the *literal* joint prompt e_J — "a cat and a dog" encoded as
one SDXL caption. That works because SDXL's text encoder has
co-occurrence knowledge baked in: the joint encoding routes attention
correctly. PoE has to *infer* that co-occurrence from two independent
prompts and gets it wrong on collision pairs. The residual measures
exactly how wrong.

Mono is not deployable as a real composition method — it requires the
joint prompt at inference, defeating compositional generation's whole
point ("given two trained concepts I never co-trained, generate them
together"). But as a *diagnostic ceiling* it pins down what a perfect
repair would recover.
