# Internal-force failure case

`poe_repair/experiments/internal_force_failure/` is a *repair attempt*
that does not invoke Mono at inference. At every denoising step it
constructs a corrective force from PoE's own UNet outputs (per-concept
ε + cross-attention maps), scales it to clear the basin barrier
measured by the residual-existence diagnostic, and adds it to ε̃_PoE.

There is no 4th UNet branch, no joint e_J encoding, no synthesiser at
inference. Two force variants are reported alongside the group-A
architectural correctors:

- **`overlap`** — *attention-overlap repulsion* (Force A). Penalises
  spatial overlap between concept A's and concept B's cross-attention
  maps; the gradient is added as a corrective force on x_t.
- **`alignment`** — *score-alignment damping* (Force B). Damps the
  component of ε_PoE along the bisector of (ε̃_A − ε_∅) and
  (ε̃_B − ε_∅); the bisector is the direction Mono's score consistently
  diverges from when PoE collapses to a chimera.

## Status: failure case

This thread is reported negatively alongside `group_a_failure`: the
Mono-free PoE-internal forces don't recover Mono-quality compositions
on the headline cell. The interest is *diagnostic* — they show which
PoE-internal signals carry partial repair information and which don't.

## Reproducing

Depends on `residual_diagnostics/existence/` outputs to read the basin
barrier height the force is calibrated against.

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
export CUDA_VISIBLE_DEVICES=1

# Step 1 — produce the residual-existence cells the calibration reads
$PY -m poe_repair.experiments.residual_diagnostics.existence \
    --pair "a cat|a dog" --seed 42

# Step 2 — run the internal-force sweep
$PY -m poe_repair.experiments.internal_force_failure \
    --pair "a cat|a dog" --seed 42
```

Outputs land under `outputs/internal_force_failure/`.
