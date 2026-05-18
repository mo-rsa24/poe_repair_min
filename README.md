# poe_repair_min

PoE composition repair on SDXL. The repo is organized around three coherent
threads of code, outputs, and checkpoints:

1. **LoRA (success).** Per-arm rank-8 LoRA on SDXL UNet cross-attention.
   Training timeline and inference probes for cat × dog, seed 42 are
   preserved at `outputs/lora/cat_dog/seed_42/results/`. See
   [`lora-success.md`](lora-success.md).
2. **Mono-during-inference residual (diagnostic ceiling).** The
   λ-interpolated PoE↔Mono sampler and the three downstream diagnostic
   experiments (idea1, idea5a, veracity). Code-only — outputs are
   regenerable. See [`mono-residual-diagnostic.md`](mono-residual-diagnostic.md).
3. **Group A (failure cases).** Latent-CNN, latent-UNet, frozen-feature-MLP
   external correctors that demonstrably *don't* fix PoE. Outputs and
   checkpoints kept under `outputs/group_a_failure/`. See
   [`group-a-failure.md`](group-a-failure.md).

Published-paper reference codebases (AAE, CO3, FOCUS, P2P,
reduce-reuse-recycle) live untouched in `composition/`.

## Setup

```bash
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
```

**Always use the `co3` conda env's python.** Each experiment runs an
import-sanity check at startup; if your interpreter is missing a dep it
exits with code 2.

All commands assume `CUDA_VISIBLE_DEVICES=1` (override as needed).

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
    _poe_internal.py           Basin-template corrector (used by idea1)
  composers/
    mono.py poe.py             literal Mono / vanilla PoE
    teacher_residual.py        λ-interpolated PoE↔Mono (diagnostic)
    poe_internal.py            basin-template corrector wrapper
    direct_eps.py              direct-ε student wrapper (group_a_failure)
  experiments/
    lora/                      Thread 1: LoRA success
    group_a_failure/           Thread 3: failure cases (corrector ablation)
    idea1/ idea5a/ veracity/   Thread 2: mono-residual diagnostic
  students/                    latent_cnn, latent_unet, frozen_feature_mlp,
                               direct_eps (group-A architectures)
  figures/                     plotting helpers
outputs/
  lora/cat_dog/seed_42/results/   consolidated LoRA training artifact
  group_a_failure/                failure-mode outputs + checkpoints
composition/                   vendored published-paper reference repos
scripts/
  build_lora_manifest.py       scan results/ -> JSON manifest for inspector
  lora_inspector.py            Flask app for the LoRA training timeline
  run_lora_inspector.sh        launcher with SSH-tunnel-friendly defaults
  lora_resume*.sh              (none; one-off resume scripts were dropped)
  watch_and_visualize.py       live student-checkpoint visualizer
```

## Run order

### Thread 1 — LoRA inference / inspection

The consolidated training run is at `outputs/lora/cat_dog/seed_42/results/`.
Rebuild the inspector manifest and serve it locally:

```bash
$PY scripts/build_lora_manifest.py
$PY scripts/lora_inspector.py --port 5050
# from your laptop: ssh -L 5050:localhost:5050 mscluster106
#                   open http://localhost:5050
```

To retrain from scratch (requires the training cache, which is not part of
this checkpoint):

```bash
$PY -m poe_repair.experiments.lora \
    --pair a_cat__x__a_dog --seed 42 --split heldout \
    --total-epochs 200 --probe-every-epochs 50 \
    --lr 1e-4 --lora-rank 8
```

### Thread 2 — Mono-residual diagnostic

See [`mono-residual-diagnostic.md`](mono-residual-diagnostic.md) for the
step-by-step reproduction. Summary:

```bash
$PY -m poe_repair.experiments.veracity --pair "a cat|a dog" --seed 42
$PY -m poe_repair.experiments.idea1    --pair "a cat|a dog" --seed 42
$PY -m poe_repair.experiments.idea5a   --pair "a cat|a dog" --seed 42
```

### Thread 3 — Group A failure cases

See [`group-a-failure.md`](group-a-failure.md) for design and protocol.
Training (per architecture):

```bash
$PY -m poe_repair.experiments.group_a_failure \
    --technique latent_unet --pair a_cat__x__a_dog --seed 42
```

(``latent_unet`` ↔ ``latent_cnn`` ↔ ``frozen_feature_mlp``.)

## Cached baselines (PoE / Mono / solo_a / solo_b)

`run.py` provides a cached dispatcher; first call inferences, later calls
reuse the PNG.

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

## What's not in this codebase

This is a checkpoint of the LoRA-working state; the surrounding exploratory
threads have been pruned. If you need to recover any of:

- sched-M2 + ê_J synthesiser pipeline
- ULA / MCMC correctors (Du et al. AnnealedULA port)
- residual-prompt / teacher-residual stand-alone trainers
- CLIP-guided / adaptive-schedule / PoE-internal-repair compositions
- the e_* held-out / cfg-isolation / residual-decomposition / synth-audit
  experiments
- thread_c_structure (D4A, VLM-grid)
- idea2 / idea5b experiments

… check out the `v0.1-pre-cleanup` tag or the `archive/pre-cleanup-2026-05-18`
branch — those preserve the full pre-cleanup state of `main`.
