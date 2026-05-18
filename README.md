# poe_repair_min

PoE composition repair on SDXL. The repo is organised around four threads
of code, outputs, and checkpoints:

1. **LoRA (success).** Per-arm rank-8 LoRA on SDXL UNet cross-attention.
   Training timeline and inference probes for cat × dog, seed 42 are
   preserved at `outputs/lora/cat_dog/seed_42/results/`. See
   [`lora-success.md`](lora-success.md).
2. **Residual diagnostics (Mono ceiling).** Two sub-experiments characterising
   the guided PoE→Mono residual r_t:
   `residual_diagnostics/existence/` (residual is well-defined + structured)
   and `residual_diagnostics/clip_window/` (commitment window). Code-only —
   outputs are regenerable. See [`residual-diagnostics.md`](residual-diagnostics.md).
3. **Group-A (failure cases).** Latent-CNN, latent-UNet, frozen-feature-MLP
   external correctors that demonstrably *don't* fix PoE. Outputs and
   checkpoints kept under `outputs/group_a_failure/`. See
   [`group-a-failure.md`](group-a-failure.md).
4. **Internal-force failure case.** Mono-free PoE-internal corrective
   forces (attention-overlap + score-alignment). Another repair attempt
   that fails alongside group-A. See [`internal-force-failure.md`](internal-force-failure.md).

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

The training-cache root defaults to
`/datasets/mmolefe/poe_repair_min/outputs/training_cache` (canonical
cluster path). Override with the `POE_REPAIR_TRAINING_CACHE` env var
or per-run `--cache-root <path>` on the LoRA and group-A trainers.

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
  students/                    latent_cnn, latent_unet, frozen_feature_mlp,
                               direct_eps (group-A architectures)
  figures/                     plotting helpers
outputs/
  lora/cat_dog/seed_42/results/                  consolidated LoRA artifact
  group_a_failure/                               failure-mode outputs + ckpts
  residual_diagnostics/existence/                (regenerable; not in git)
  residual_diagnostics/clip_window/              (regenerable; not in git)
  internal_force_failure/                        (regenerable; not in git)
composition/                   vendored published-paper reference repos
scripts/
  build_lora_manifest.py       scan results/ -> JSON manifest for inspector
  lora_inspector.py            Flask app for the LoRA training timeline
  run_lora_inspector.sh        launcher with SSH-tunnel-friendly defaults
  watch_and_visualize.py       live student-checkpoint visualiser
```

## Run order

### Thread 1 — LoRA inference / inspection

Consolidated run at `outputs/lora/cat_dog/seed_42/results/`. Rebuild the
inspector manifest and serve locally:

```bash
$PY scripts/build_lora_manifest.py
$PY scripts/lora_inspector.py --port 5050
# from your laptop: ssh -L 5050:localhost:5050 mscluster106
#                   open http://localhost:5050
```

LoRA inference-only (load an existing checkpoint, run the startup probe,
exit — no training):

```bash
$PY -m poe_repair.experiments.lora \
    --resume-from outputs/lora/cat_dog/seed_42/results/checkpoints/lora_step_062500.pt \
    --total-epochs 0
```

LoRA re-training from scratch:

```bash
$PY -m poe_repair.experiments.lora \
    --pair a_cat__x__a_dog --seed 42 --split heldout \
    --total-epochs 200 --probe-every-epochs 50 \
    --lr 1e-4 --lora-rank 8
```

### Thread 2 — Residual diagnostics

See [`residual-diagnostics.md`](residual-diagnostics.md). Summary:

```bash
$PY -m poe_repair.experiments.residual_diagnostics \
    --pair "a cat|a dog" --seed 42        # runs both existence + clip_window
```

Or run them individually:

```bash
$PY -m poe_repair.experiments.residual_diagnostics.existence    --pair "a cat|a dog" --seed 42
$PY -m poe_repair.experiments.residual_diagnostics.clip_window  --pair "a cat|a dog" --seed 42
```

### Thread 3 — Group-A failure cases

See [`group-a-failure.md`](group-a-failure.md):

```bash
$PY -m poe_repair.experiments.group_a_failure --technique latent_unet
$PY -m poe_repair.experiments.group_a_failure --technique latent_cnn
$PY -m poe_repair.experiments.group_a_failure --technique frozen_feature_mlp
```

### Thread 4 — Internal-force failure case

See [`internal-force-failure.md`](internal-force-failure.md). Depends on
the residual-existence diagnostic for basin-barrier calibration:

```bash
$PY -m poe_repair.experiments.residual_diagnostics.existence --pair "a cat|a dog" --seed 42
$PY -m poe_repair.experiments.internal_force_failure         --pair "a cat|a dog" --seed 42
```

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

This is a checkpoint of the LoRA-working state; surrounding exploratory
threads have been pruned. To recover anything from the pre-2026-05-18 state
(sched-M2, ê_J synthesiser, ULA, residual-prompt, e_* held-out experiments,
thread_c_structure, idea2/idea5b, etc.), check out the `v0.1-pre-cleanup`
tag or the `archive/pre-cleanup-2026-05-18` branch.
