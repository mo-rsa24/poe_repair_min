# poe_repair_min

Ask SDXL for "a cat" and for "a dog" separately, then combine the two predictions with the
Product-of-Experts rule, and you rarely get a cat next to a dog. You get one fused animal, or only
one of the two, or noise. This repo asks what is missing from that combination, and whether a small
trained adapter can add it back while the image is being generated.

The work splits into five threads of code, outputs, and checkpoints:

1. **The LoRA that works.** A rank-8 LoRA on SDXL's UNet cross-attention layers, trained one pair
   and one seed at a time. The training timeline and the generation-time tests for cat × dog at
   seed 42 are kept at `outputs/lora/a_cat__x__a_dog/seed_42/results/`. See
   [the LoRA writeup](artifacts/_quarantine/results-archive/lora-success.md).
2. **What the missing piece looks like (the Mono ceiling).** Two smaller studies of the residual
   `r_t`, the difference between what the joint prompt predicts and what the Product-of-Experts
   combination predicts. `residual_diagnostics/existence/` asks whether that residual is
   well-defined and structured, and `residual_diagnostics/clip_window/` asks when during
   generation it matters. Code only, since the outputs can be regenerated. See
   [the residual writeup](artifacts/_quarantine/results-archive/residual-diagnostics.md).
3. **Correctors that live outside the UNet, and fail.** A latent CNN, a latent UNet, and a
   frozen-feature MLP, all of which demonstrably do *not* fix the combination. Their outputs and
   checkpoints are kept under `outputs/group_a_failure/`. See
   [the external-corrector writeup](artifacts/_quarantine/results-archive/group-a-failure.md).
4. **A corrector inside the UNet, which also fails.** Corrective forces computed from the model's
   own attention and scores (attention overlap plus score alignment), needing no joint prompt at
   all. Another repair attempt that fails alongside thread 3. See
   [the internal-force writeup](artifacts/_quarantine/results-archive/internal-force-failure.md).

   > "Needing no joint prompt" is what Mono-free means throughout this repo: the method never gets
   > to encode the combined phrase "a cat and a dog", only the two separate phrases.
5. **How much the prompt alone can do, with no LoRA.** Clean SDXL, generating the same image many
   times over while switching classifier-free guidance on and off over different ranges of steps.
   It finds the smallest range of guided steps that still produces a recognisable cat and dog, and
   that is the no-correction baseline thread 1 measures the LoRA against. See
   [the conditioning-window writeup](artifacts/_quarantine/results-archive/conditioning-window.md).

Reference code from published papers (AAE, CO3, FOCUS, P2P, reduce-reuse-recycle) sits untouched
in `composition/`.

## Setup

```bash
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
```

**Always use the `co3` conda env's python.** Each experiment checks its imports at startup, and
exits with code 2 if your interpreter is missing a dependency.

All commands assume `CUDA_VISIBLE_DEVICES=1` (override as needed).

The training cache defaults to `/datasets/mmolefe/poe_repair_min/outputs/training_cache`, which is
the canonical path on the cluster. Override it with the `POE_REPAIR_TRAINING_CACHE` env var, or
per run with `--cache-root <path>` on the LoRA and external-corrector trainers.

## Repo layout

```
poe_repair/
  config.py runtime.py run.py training_cache.py
  _sdxl/                       SDXL model loading
  embeddings/cache_dataset.py  Training-cache dataset for students
  diagnostics/residual.py      attention_overlap (one function)
  methods/
    _sampling.py               run_cfg, run_cfg_poe, run_teacher_residual,
                               run_lora_residual_inject,
                               run_external_corrector_inject,
                               run_direct_eps_inject, _CrossAttnRecorder
    _poe_internal.py           Basin-template corrector (used by
                               internal_force_failure)
  composers/
    mono.py poe.py             literal Mono / vanilla PoE
    teacher_residual.py        λ-interpolated PoE↔Mono (diagnostic)
    poe_internal.py            basin-template corrector wrapper
    direct_eps.py              direct-ε student wrapper (group_a_failure)
  experiments/
    lora/                                  Thread 1
    residual_diagnostics/
      existence/                           Thread 2a
      clip_window/                         Thread 2b
    group_a_failure/                       Thread 3
    internal_force_failure/                Thread 4
    conditioning_window/                   Thread 5
  students/                    latent_cnn, latent_unet, frozen_feature_mlp,
                               direct_eps (group-A architectures)
  figures/                     plotting helpers
outputs/
  lora/a_cat__x__a_dog/seed_42/results/                  consolidated LoRA artifact
  group_a_failure/                               failure-mode outputs + ckpts
  residual_diagnostics/existence/                (regenerable; not in git)
  residual_diagnostics/clip_window/              (regenerable; not in git)
  internal_force_failure/                        (regenerable; not in git)
  conditioning_window/a_cat__x__a_dog/seed_42/           (regenerable; not in git)
composition/                   vendored published-paper reference repos
scripts/
  build_lora_manifest.py       scan results/ -> JSON manifest for inspector
  lora_inspector.py            Flask app for the LoRA training timeline
  run_lora_inspector.sh        launcher with SSH-tunnel-friendly defaults
  watch_and_visualize.py       live student-checkpoint visualiser
```

> "Basin" in the corrector names refers to the region of starting noise that ends at one particular
> image. The corrector pushes a run out of the fused-animal region and toward the two-animal one.

## Run order

### Thread 1: LoRA generation and inspection

The finished run is at `outputs/lora/a_cat__x__a_dog/seed_42/results/`. Rebuild the inspector's
manifest and serve it locally:

```bash
$PY scripts/build_lora_manifest.py
$PY scripts/lora_inspector.py --port 5050
# from your laptop: ssh -L 5050:localhost:5050 mscluster106
#                   open http://localhost:5050
```

To generate images from an existing checkpoint without training (it loads, runs the startup check,
and exits):

```bash
$PY -m poe_repair.experiments.one_pair_one_seed \
    --resume-from outputs/lora/a_cat__x__a_dog/seed_42/results/checkpoints/lora_step_062500.pt \
    --total-epochs 0
```

To train the LoRA from scratch:

```bash
$PY -m poe_repair.experiments.one_pair_one_seed \
    --pair a_cat__x__a_dog --seed 42 --split heldout \
    --total-epochs 200 --probe-every-epochs 50 \
    --lr 1e-4 --lora-rank 8
```

### Thread 2: what the missing piece looks like

See [the residual writeup](artifacts/_quarantine/results-archive/residual-diagnostics.md). Both
studies run together with:

```bash
$PY -m poe_repair.experiments.residual_between_mono_and_poe \
    --pair "a cat|a dog" --seed 42        # runs both existence + clip_window
```

Or one at a time:

```bash
$PY -m poe_repair.experiments.residual_between_mono_and_poe.existence    --pair "a cat|a dog" --seed 42
$PY -m poe_repair.experiments.residual_between_mono_and_poe.clip_window  --pair "a cat|a dog" --seed 42
```

### Thread 3: correctors outside the UNet, which fail

See [the external-corrector writeup](artifacts/_quarantine/results-archive/group-a-failure.md):

```bash
$PY -m poe_repair.experiments.correction_outside_the_unet --technique latent_unet
$PY -m poe_repair.experiments.correction_outside_the_unet --technique latent_cnn
$PY -m poe_repair.experiments.correction_outside_the_unet --technique frozen_feature_mlp
```

### Thread 4: the corrector inside the UNet, which also fails

See [the internal-force writeup](artifacts/_quarantine/results-archive/internal-force-failure.md).
It needs the residual-existence study first, because that is what calibrates how hard the
corrector has to push:

```bash
$PY -m poe_repair.experiments.residual_between_mono_and_poe.existence --pair "a cat|a dog" --seed 42
$PY -m poe_repair.experiments.internal_force_failure         --pair "a cat|a dog" --seed 42
```

### Thread 5: how much the prompt alone can do

See [the conditioning-window writeup](artifacts/_quarantine/results-archive/conditioning-window.md).
This generates a series of images on clean SDXL, one per on/off pattern of classifier-free guidance
across the steps. No LoRA and no joint prompt anywhere. It starts from the same initial noise
`x_T` as the LoRA experiment, which is what makes the comparison between them fair.

```bash
$PY -m poe_repair.experiments.cfg_window_without_lora --sanity-only   # equivalence checks
$PY -m poe_repair.experiments.cfg_window_without_lora --smoke         # 2-schedule wiring test
$PY -m poe_repair.experiments.cfg_window_without_lora                  # full STANDARD_SUITE (~10–15 min)
```

The results are easiest to read in the inspector at
`http://127.0.0.1:5050/conditioning_window` (linked from the main LoRA page).

## Cached baselines (PoE / Mono / solo_a / solo_b)

`run.py` dispatches these and caches them. The first call generates the image, every later call
reuses the saved PNG.

```bash
$PY -c "
from poe_repair.run import make_ctx, run_method
from poe_repair.experiments._eval_common import cell_for
ctx = make_ctx()
cell = cell_for('a cat', 'a dog', 42)
for m in ['solo_a', 'solo_b', 'poe', 'mono']:
    run_method(m, cell, ctx)
"
```

## What is not in this codebase

This is a snapshot of the state in which the LoRA works, and the exploratory threads around it
have been pruned. To recover anything from before 2026-05-18 (sched-M2, the ê_J synthesiser, ULA,
residual-prompt, the e_* experiments on pairs the model never trained on, thread_c_structure,
idea2/idea5b, and so on), check out the `v0.1-pre-cleanup` tag or the
`archive/pre-cleanup-2026-05-18` branch.
