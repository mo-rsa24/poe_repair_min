# Phase 4 — Single-seed LoRA (the deployed result)

## Question

Can a rank-8 LoRA on SDXL's cross-attention projections — trained
purely on the cached PoE↔Mono residual — drive the per-arm PoE
prediction toward Mono at inference, *without ever encoding the joint
prompt at inference*?

## Why this phase exists

Phases 1–3 only used the oracle. This is the first phase where a
*learner* has to predict the residual from `(z_t, t, e_A, e_B, e_∅)`
alone. If it works, we have the headline deployable artefact for the
project. If it fails, every downstream phase is a study of how it
fails, not how to scale it.

What "works" means here is specific:

- Inference is **Mono-free** — the deployment sampler never encodes
  the joint prompt `e_J`.
- At λ=0 (LoRA bypass) the sampler reproduces vanilla PoE
  bit-identically. At λ=1 the sampler walks PoE → toward Mono in
  image space.
- A timeline of probes across training epochs shows a smooth morph
  from "blob of fur" chimera through to two visibly distinct animals.

## Code

- LoRA attachment: `poe_repair/experiments/lora/trainer.py::attach_lora`.
  Targets `attn2.{to_q, to_k, to_v}` on the UNet.
- Training loop: `poe_repair/experiments/lora/trainer.py::train_epoch`.
  Target per step is `r_t = guided(ε_J) − guided(ε_PoE)` from cache;
  loss is MSE between LoRA-corrected `ε̃_PoE_lora` and `ε̃_J_cached`.
- Inference sampler:
  `poe_repair/methods/_sampling.py::run_lora_residual_inject`. Runs
  the 3-branch (A, B, ∅) forward twice (adapter OFF for the frozen
  PoE prediction, adapter ON for the LoRA-modified prediction),
  `Δ̂_t = ε̃_PoE_lora − ε̃_PoE_frozen`, then composes
  `ε_final = ε̃_PoE_frozen + λ · Δ̂_t`. **No joint forward.**
- Masked variant (for the CFG-window × LoRA inspector):
  `run_lora_residual_inject_masked` in the same file, with a
  `cfg_mask` and a `composition_mode` of `with_prompt` or `always`.
- Probe + figures: `poe_repair/experiments/lora/probe.py`,
  `poe_repair/experiments/lora/figures.py`.
- Inspector: `scripts/lora_inspector.py` (routes `/`,
  `/conditioning_window`, `/conditioning_window_lora`). The residual
  tab `/` carries the image row (PoE / PoE+λ·r / Mono) and, directly
  below it, a per-cell **MDS / PCA trajectory panel** swapped by the
  same `(epoch, λ)` sliders. Static endpoints A, B, A∧B are at fixed
  coordinates across panels; only the PoE+λ·R path moves.
- Manifest builders: `scripts/build_lora_manifest.py`,
  `scripts/build_cwl_manifest.py`.
- MDS panel pre-renderer: `scripts/build_lora_inspector_mds.py`.
  Five stages — `collect-static` (solo A, solo B, mono trajectories
  via `run_cfg`), `collect-cells` (per `(epoch, λ)`, load checkpoint
  with `lora_trainer.load_lora_state`, run
  `run_lora_residual_inject(lambda_value=λ)`, cache the 51-step flat
  latent trajectory), `project` (fit one global PCA — or MDS via
  `--projection mds` — on the union so static endpoints really are
  static), `render` (one PNG per cell at
  `mds_probes/<epoch>/<λ>/mds.png`, style copied from
  `neurips2026/scripts/render_taxonomy_paper_figure.py`),
  `update-manifest` (writes the `mds_cells` map into
  `inspector_manifest.json`). Cache lives at
  `outputs/lora/<pair>/seed_42/results/mds_cache/` (~7 MB per
  trajectory; ~1.6 GB at full 48 × 5 sweep).

Consolidated artefact on disk:
`outputs/lora/a_cat__x__a_dog/seed_42/results/`. Headline checkpoint is
`checkpoints/lora_step_062500.pt`.

The companion CFG-window-with-LoRA experiment lives at
`poe_repair/experiments/conditioning_window_lora/` and writes to
`outputs/conditioning_window_lora/a_cat__x__a_dog/seed_42/`. See
[conditioning-window-lora.md](../../../.claude/plans/conditioning-window-lora.md) for the
detailed sampler grammar and inspector wiring — this plan only enumerates
the run commands.

## Commands

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
export CUDA_VISIBLE_DEVICES=1
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
```

### Inference / inspection only (existing checkpoint)

```bash
$PY scripts/build_lora_manifest.py
$PY scripts/lora_inspector.py --port 5050
# from laptop: ssh -L 5050:localhost:5050 mscluster106 && open http://127.0.0.1:5050
```

### Pre-render the MDS trajectory panels (residual tab, below image row)

```bash
# Smoke set on cat × dog (4 cells, ~3 min on one RTX 8000):
$PY scripts/build_lora_inspector_mds.py \
    --epochs 0,800 --lambdas 0.00,1.00 \
    --stages collect-static,collect-cells,project,render,update-manifest

# Full sweep on cat × dog (48 epochs × 5 λ, ~3 h):
$PY scripts/build_lora_inspector_mds.py --epochs all --lambdas all \
    --stages collect-static,collect-cells,project,render,update-manifest

# Another pair (each pair has its own results-root, slug, cache, PNGs,
# and mds_cells block in inspector_manifest.json):
$PY scripts/build_lora_inspector_mds.py \
    --results-root outputs/lora/a_camel__x__a_desert_landscape/seed_42/results \
    --pair-slug a_camel__x__a_desert_landscape \
    --epochs all --lambdas all \
    --stages collect-static,collect-cells,project,render,update-manifest
```

`--projection pca` (default) is cleaner on diffusion latents than the
metric MDS the paper figure uses; pass `--projection mds` for parity
with `trajectory_g1g4.png`. Re-running stages skips already-cached
trajectories and panels unless `--overwrite` is passed.

Or a programmatic startup probe with no training:

```bash
$PY -m poe_repair.experiments.lora \
    --resume-from outputs/lora/a_cat__x__a_dog/seed_42/results/checkpoints/lora_step_062500.pt \
    --total-epochs 0
```

### Re-train from scratch

```bash
$PY -m poe_repair.experiments.lora \
    --pair a_cat__x__a_dog --seed 42 --split heldout \
    --total-epochs 200 --probe-every-epochs 50 \
    --lr 1e-4 --lora-rank 8
```

Probe artefacts land at `outputs/lora/<run>/probes/epoch_NNNN/lambda_*/`
with decoded PNGs and `delta_overlays/step_NN.pt` for the inspector.

### CFG-mask × LoRA companion sweep

```bash
$PY -m poe_repair.experiments.conditioning_window_lora \
    --lora-ckpts outputs/lora/a_cat__x__a_dog/seed_42/results/checkpoints/lora_step_062500.pt \
    --lambda-values 0.0,0.5,1.0 \
    --modes with_prompt,always
```

~2.5 h on one GPU for 3 λ × 2 modes × 59 schedules. The inspector route
`/conditioning_window_lora` consumes the resulting manifest.

## How to read the result

| Bucket | What you see | Means |
|---|---|---|
| **Poor** | At λ=1, output is identical to PoE at every epoch. Or output is pure noise after a few hundred steps. Sanity `masked(all_on, λ=1) ≡ run_lora_residual_inject(λ=1)` fails. | Sampler wiring is broken (adapter not actually applied; PoE composition not matching the cached one; mask off-step path drifting). Stop and fix; nothing else is interpretable. |
| **Bad** | Training loss decreases but probes never visually move off PoE. `‖Δ̂_t‖` is small (< 5% of `‖ε̃_PoE‖`) at every step. | LoRA is collapsing to zero — likely sign convention, residual normalisation, or LR. Phase 2's `‖r_t‖` curve is the ground truth target magnitude; if Δ̂ is orders of magnitude smaller, the trainer never started fitting. |
| **Unknown** | Probes morph but plateau short of two-distinct-animals (e.g. cat tucked into dog's chest). Latent-L1 reaches ~30% of cache-to-Mono distance and stops improving past epoch 400. | This is on-trend but unfinished. Train longer (the published trajectory was still moving at epoch 600). Don't fold into "failure case" — flag as a budget issue. |
| **Good (the result on disk)** | At λ=0 the output is byte-identical to vanilla PoE (canary). At λ=1, by epoch ~500–600 the image shows two distinct white animals touching, GroundingDINO finds two boxes with confidence > 0.55, VQAScore on `"is the cat clearly separate from the dog?"` ≥ 0.5. Latent-L1 reaches roughly 40% of the way from PoE to Mono and is still slowly moving. The masked sampler at `all_on` is bit-identical to the standard LoRA sampler at λ=1. | The deployable repair works on the beachhead cell. Phases 5 + 6 can now be reported negatively against this result; Phases 7 + 8 are justified. |

## What this phase does *not* prove

- Anything about pairs other than `cat × dog`, or seeds other than 42.
- That the LoRA is recovering the *true* `r_t` rather than some
  cheap-and-different correction — that's Phase 7's job.
- That the LoRA generalises to drifted trajectories — at deployment
  with non-zero λ the trajectory leaves the training distribution; the
  ~40% plateau is plausibly explained by this drift.

## Status — 2026-05-22

| Item | Done | To do |
|---|:---:|:---:|
| `run_lora_residual_inject` sampler landed | ✅ | |
| `run_lora_residual_inject_masked` (CFG-window companion) landed | ✅ | |
| Bit-exact sanity (`masked(all_on) ≡ run_lora_residual_inject` at λ=1, Δ = 0.0) | ✅ | |
| LoRA training on cat × dog seed 42, rank 8, ~600 epochs | ✅ | |
| Consolidated artefact at `outputs/lora/a_cat__x__a_dog/seed_42/results/` | ✅ | |
| Per-epoch probes across λ-grid with `where_applied` overlays | ✅ | |
| Inspector route `/` with epoch × λ sliders | ✅ | |
| `conditioning_window_lora` companion at one cell (epoch 062500, λ=1.0, both modes) | ✅ | |
| Inspector route `/conditioning_window_lora` with epoch + λ sliders | ✅ | |
| MDS / PCA trajectory pre-renderer `scripts/build_lora_inspector_mds.py` (5 stages, per-pair) | ✅ | |
| Residual tab wired with MDS panel below image row (same `(epoch, λ)` sliders) | ✅ | |
| MDS smoke set on cat × dog seed 42 (epochs {0, 800} × λ ∈ {0, 1}) rendered | ✅ | |
| Full MDS sweep on cat × dog seed 42 (48 epochs × 5 λ) | | ⬜ (~3 h wall-clock) |
| MDS sweep on other pairs under `outputs/lora/*/seed_42/results/` | | ⬜ |
| One-shot wrapper that loops every pair under `outputs/lora/*/seed_42/` | | ⬜ |
| Multi-λ companion sweep at the final checkpoint (`--lambda-values 0.0,0.5,1.0`) | | ⬜ (~2.5 h wall-clock) |
| Multi-epoch companion sweep (5 sampled epochs × 5 λ) | | ⬜ (~21 h; only if λ sweep is informative) |
| Longer training run past 600 epochs to test the plateau | | ⬜ (the trajectory was still moving) |
| Quantitative VQA / GroundingDINO gating | | ⬜ (deferred — eyeball is the headline) |
