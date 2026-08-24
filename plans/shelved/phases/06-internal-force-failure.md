# Phase 6 — Internal-force failures

## Question

Group A used an *external* learner. This phase tries the other extreme:
no learner at all, no joint prompt at inference, no new parameters
anywhere. Instead, construct a corrective force at each denoising step
from PoE's own UNet outputs (the per-concept ε predictions and the
cross-attention maps), scale that force by the basin-barrier height
measured in Phase 2, and add it to `ε̃_PoE`. Two variants:

- **`overlap`** — *attention-overlap repulsion* (Force A). Penalises
  spatial overlap between concept A's and concept B's cross-attention
  maps; the gradient of that penalty becomes a corrective force on `z_t`.
- **`alignment`** — *score-alignment damping* (Force B). Damps the
  component of `ε̃_PoE` along the bisector of `(ε̃_A − ε_∅)` and
  `(ε̃_B − ε_∅)`. That bisector is the direction Mono's score
  consistently diverges from when PoE collapses to a chimera.

Do either of these forces recover Mono-like compositions on the
beachhead cell?

## Why this phase exists

If the PoE-internal signal alone is enough — without any joint prompt
encoding and without any training — then the LoRA's value would be
"caching that signal in weights" rather than "extracting a new signal
from the data." Distinguishing these matters for how we explain the
LoRA result.

**As of 2026-05-19, both forces are reported negatively.** The PoE
chimera is not undone by either force at the calibrated scale. This
phase documents that result.

## Code

- Force implementations: `poe_repair/methods/_poe_internal.py`
  (basin-template corrector).
- Composer wrapper: `poe_repair/composers/poe_internal.py`.
- Experiment package: `poe_repair/experiments/internal_force_failure/`
  (one package; `--force overlap|alignment`).
- Cross-attention recorder: `poe_repair/methods/_sampling.py::_CrossAttnRecorder`.
- Basin barrier calibration consumes residual-existence outputs
  (Phase 2) — the force magnitude is calibrated against `‖r_t‖`.

Outputs land under `outputs/internal_force_failure/`.

## Commands

``bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
export CUDA_VISIBLE_DEVICES=1
``

### Prerequisite — residual-existence cache (from Phase 2)

``bash
$PY -m poe_repair.experiments.residual_between_mono_and_poe.existence \
    --pair "a cat|a dog" --seed 42
``

This is what the force magnitude is calibrated against. If the cache
is missing, the sweep aborts.

### Run the force sweep

``bash
$PY -m poe_repair.experiments.internal_force_failure \
    --pair "a cat|a dog" --seed 42
``

Sweeps both forces at calibrated magnitudes and decodes per-step
results to `outputs/internal_force_failure/`.

## How to read the result

| Bucket | What you see | Means |
|---|---|---|
| **Poor** | Sweep aborts with a calibration error (Phase 2 cache missing or `‖r_t‖` curve undefined). | Run Phase 2 first. |
| **Bad** | Both forces blow up the trajectory — output is structureless noise — even at `force_scale × 0.1`. | Scale calibration is wrong, or the cross-attention recorder is hooking the wrong layer. Investigate before reporting. |
| **Unknown** | One force visibly nudges the image partially toward separated animals on a couple of steps, but doesn't close the gap by the final step. The other does effectively nothing. | Partial signal; report which side the partial signal came from and stop. Do not tune. |
| **Good (this phase's *good* is documented failure)** | Both forces run cleanly without instability. At the calibrated scale, the decoded image at the final step is essentially indistinguishable from vanilla PoE — the chimera remains. The per-step `‖r̂_t‖` from the force is at least 10× smaller than the cached `‖r_t‖` from Phase 2. | The PoE-internal signal carries some failure information (visible in attention maps and score bisectors) but not enough to actuate the correction. The LoRA's value isn't "caching this signal" — it's reading something the PoE-internal forces don't expose. |

## Why each force was worth running anyway

- **`overlap`.** Tests whether the attention maps themselves are
  informative enough to drive separation when the two concept tokens
  are competing for the same spatial region.
- **`alignment`.** Tests whether the score-bisector direction (which
  Phase 1's anti-corroboration analysis identifies as the direction
  Mono and PoE consistently disagree on) is enough to *act on*
  without the joint prompt.

If either had worked, Phase 4's framing would have to change to "the
LoRA is a parametric cache of [this internal signal]." Since neither
worked, Phase 4's framing stands: the LoRA learns something not
available from the unaltered PoE forward.

## What this phase does *not* do

- Train anything.
- Encode the joint prompt at inference (the whole point is to avoid
  that).
- Cross-seed or cross-pair runs.
- Force-magnitude sweeps beyond the single calibrated scale. If a
  reader wants to know "what if you scaled the force 10×," the answer
  is "we tried, it blew up the trajectory; see the bad bucket above."

## Status — 2026-05-19

| Item | Done | To do |
|---|:---:|:---:|
| Basin-template corrector (`_poe_internal.py`) landed | ✅ | |
| `_CrossAttnRecorder` hook installed in the sampler | ✅ | |
| Phase-2 residual-existence cache (calibration prerequisite) on disk | ✅ | |
| `overlap` force run on cat × dog seed 42 at calibrated scale | ✅ | |
| `alignment` force run on cat × dog seed 42 at calibrated scale | ✅ | |
| Per-step `‖r̂_t‖` vs cached `‖r_t‖` comparison | ✅ | |
| Negative-result writeup in `docs/results-archive/internal-force-failure.md` (root) | ✅ | |
| One-paragraph verdict added to master writeup | | ⬜ |
