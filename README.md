# poe_repair_min

PoE composition repair on SDXL via a learned interaction-residual synthesiser.

> **Deployed method: sched-M2 + ê_J.** Compose two concepts without a literal
> joint prompt by synthesising ê_J(e_A, e_B) and mixing it with PoE under a
> Phase-11 β-schedule.

See `OBJECTIVE.md` for the science. See `.claude/plans/serialized-seeking-pebble.md`
for the current work plan.

## Setup

```bash
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
```

**Always use the `co3` conda env's python.** Each experiment runs an
import-sanity check at startup; if your interpreter is missing a dep it
exits with code 2.

All commands assume `CUDA_VISIBLE_DEVICES=1` (override as needed).

## Run order

### Run 1 — synthesiser audit (cheap; precondition for everything else)

Quantifies how well ê_J approximates e_J on held-out pairs.

```bash
CUDA_VISIBLE_DEVICES=1 $PY -m poe_repair.experiments.e2_synth_audit
```

Outputs:
- `outputs/e2_synth_audit/summary.json` — per-pair `seq_cosine`, `pool_cosine`,
  `seq_mse`, `pool_mse`, `unet_rmse`.
- `outputs/e2_synth_audit/figures/embed_cosine_by_pair.png` — bar chart.

Cost: ~10–15 min. If `seq_cosine` < 0.95 or `unet_rmse` is large on most pairs,
the synthesiser is the bottleneck — iterate on training before running E1.

### Run 2 — diagnostic context (cheap; supports the story)

Generates the residual-decomposition figures showing r_t is structured and
concentrated in the early window.

```bash
CUDA_VISIBLE_DEVICES=1 $PY -m poe_repair.experiments.e_residual_decomposition \
    --seeds 42 4 --steps 0 3 5 10
```

Outputs in `outputs/e_residual_decomposition/`.

### Run 3 — E1 pilot (verify the refactor before the full run)

```bash
CUDA_VISIBLE_DEVICES=1 $PY -m poe_repair.experiments.e1_held_out \
    --seeds 42 4 --pairs "a cat|a dog"
```

2 seeds × 1 pair × 5 columns. ~10 min on GPU. Confirms the 5-column layout
(PoE / Mono(e_J) / sched-M2(e_J) / sched-M2(ê_J) / CO3) and JSON schema before
committing to the full headline run.

### Run 4 — E1 headline (the load-bearing result)

```bash
CUDA_VISIBLE_DEVICES=1 $PY -m poe_repair.experiments.e1_held_out \
    --seeds 42 1 2 3 4 5 6 7
```

8 seeds × 19 held-out pairs × 5 columns ≈ 760 generations. ~5–6 GPU-hours.

Outputs:
- `outputs/e1_held_out/pairs/<slug>/seed_<n>/<method>/<method>.png`
- `outputs/e1_held_out/figures/aggregate__<slug>.png`
- `outputs/e1_held_out/summary.json`

### Optional — diagnostic appendix experiments

```bash
# Per-step decoded x̂_0 + token attention maps (Mono / Mono+CO3-step0).
CUDA_VISIBLE_DEVICES=1 $PY -m poe_repair.experiments.e_diag_trajectory \
    --seeds 4 42 --include-co3

# CO3 confound control: real CO3 across CFG regimes.
CUDA_VISIBLE_DEVICES=1 $PY -m poe_repair.experiments.e_cfg_isolation \
    --seeds 42 4 --include-mono
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

## Synthesiser

Training pool / held-out lists / config:

- `poe_repair/embeddings/dataset.py` — pair pool
- `poe_repair/embeddings/holdout_pairs.py` — `COLLISION_PAIRS` (10) +
  `COOPERATIVE_PAIRS` (9) = `ALL_HOLDOUT_PAIRS` (19, the headline set).
- `poe_repair/embeddings/synthesizer.py` — three architectures
  (linear / residual_mlp / gated_attn). Default is `residual_mlp`.
- `poe_repair/embeddings/train.py` — embedding-MSE + cosine loss training.
- `poe_repair/embeddings/train_distill_unet.py` — UNet-level distillation:
  `‖ε(x_t, t, ê_J) − ε(x_t, t, e_J)‖²` sampled over training trajectories.

Trained checkpoints:
- `checkpoints/synthesizer/residual_mlp/best.pt`
- `checkpoints/synthesizer_distilled/residual_mlp_distilled/best.pt`

## What's NOT in this codebase

This repo has been pruned to the surviving scope. Removed exploratory branches
(FOCUS / AAE / P2P composers, the composite stack, VLM rewards, the perp-gated
dispatcher, λ-sweeps, scoring routers) are gone. Their outputs live under
`outputs/_archive/` if you need to inspect them.

## Repo layout

```
poe_repair/
  config.py runtime.py run.py
  _sdxl/                       SDXL model loading
  embeddings/                  Synthesiser training + inference
  diagnostics/residual.py      attention_overlap (one function)
  methods/_sampling.py         Reference samplers (PoE / Mono / sched-M2)
                               + diagnostic trajectory + LS decomposition
  composers/                   PoE / Mono / sched-M2 / CO3 (5 files)
  experiments/                 e1_held_out + e2_synth_audit + 3 diagnostics
  figures/_common.py           plotting helpers
checkpoints/                   synthesiser + distilled-synthesiser
composition/                   Vendored reference (CO3 source still imported)
outputs/                       Generated artifacts (+ _archive/ for legacy)
```

## Honest limitations (declared up front)

- `sched-M2 + ê_J` approaches Mono's behaviour as λ_t → 1. It inherits Mono's
  failure modes: subject duplication, neglect, and same-class hybridisation.
- Synthesiser generalisation is bounded by training distribution; far-OOD
  pairs may have low cosine to the literal e_J.
- Seed variance is real. We report per-seed outcomes transparently.
- We do not propose a new sampler. sched-M2 is from Phase 11; CO3 is from
  Dutta et al. The contribution is the synthesiser plus the deployment story.
