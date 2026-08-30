# Phase 2 — Residual diagnostics

## Question

Phase 1 proved injecting `r_t` works. This phase asks: *what is `r_t`
actually like* on the cell where we know it works? Two narrow
sub-questions:

1. **Existence sub-experiment.** Does `r_t` have repeatable structure
   across the 11-point λ-grid? Does the PMI identity
   `Δ_t = w · (ε_J + ε_∅ − ε_A − ε_B)` hold to numerical precision?
   How does it behave when we anchor it to the *Mono* trajectory
   instead of the PoE trajectory?
2. **CLIP-window sub-experiment.** When in the trajectory can a CLIP
   text encoder already separate "a cat and a dog" from "a cat" / "a
   dog" alone? Is there a window where CLIP is informative *and* the
   trajectory can still be steered?

These two together tell us what time-locus and what spatial
characteristics any future corrector has to match.

## Why this phase exists

The Phase-1 result by itself is a yes/no. The Phase-2 sub-experiments
turn that yes into a *target spec*: a corrector should fire mostly in
the basin-commit window (roughly steps 10–25 of 50), should match the
shape of `r_t` when the trajectory hasn't drifted too far, and should
be expected to lose accuracy off the PoE-anchor path.

Without this characterisation, Phase 4's LoRA result is interpretable
only as "it works"; with it, we can say *why* it works.

## Code

- Sub-experiment 2a: `poe_repair/experiments/residual_between_mono_and_poe/existence/`.
- Sub-experiment 2b: `poe_repair/experiments/residual_between_mono_and_poe/clip_window/`.
- Top-level driver: `poe_repair/experiments/residual_between_mono_and_poe/__main__.py`.
- The figures (`Fig 1`, `Fig 4`, `App A`, `App B'`) live in the
  veracity experiment package — diagnostic figures are rendered from
  caches both experiments produce.

`clip_window` is post-hoc: it reads the cached residuals from
`existence`, reconstructs Tweedie `x̂_0` at chosen step indices for
chosen λ values, decodes through SDXL's VAE, and scores against CLIP
text targets.

## Commands

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
export CUDA_VISIBLE_DEVICES=1
```

### Both back to back

```bash
$PY -m poe_repair.experiments.residual_between_mono_and_poe \
    --pair "a cat|a dog" --seed 42
```

`existence` runs first (caches per-step residuals + decoded grids),
then `clip_window` reads those caches and produces its own figures.

### Or individually

```bash
$PY -m poe_repair.experiments.residual_between_mono_and_poe.existence \
    --pair "a cat|a dog" --seed 42

$PY -m poe_repair.experiments.residual_between_mono_and_poe.clip_window \
    --pair "a cat|a dog" --seed 42
```

`existence` is idempotent — existing λ-cell outputs are reused unless
`--overwrite` is passed. Outputs land under
`outputs/residual_diagnostics/{existence,clip_window}/`.

## How to read the result

| Bucket | What you see | Means |
|---|---|---|
| **Poor** | PMI identity fails. `‖r_t‖` is essentially zero, or noise-like across t. | Either the encoding of `e_J` is wrong, the CFG scale used in `ε̃_PoE` and `ε̃_J` is mismatched, or the cached PoE trajectory was sampled with a different scheduler. Stop and fix; do not trust Phase 1's pass. |
| **Bad** | `r_t` is structured but the trajectory-independence appendix (Mono-anchor vs PoE-anchor) shows them *agreeing* on cat × dog. CLIP fails to separate the joint prompt from the singletons at *any* step. | The residual is not a meaningful failure signature — both trajectories look the same in ε-space. Either the cell isn't a real collision case, or CLIP isn't the right grader. Switch grader before Phase 4. |
| **Unknown** | `r_t` has the right shape but the commit-window placement is ambiguous (no obvious peak between steps 5 and 30). CLIP separates the prompts only at the very last step. | Useable but flag in writeup: a learner should be allowed to fire across the whole trajectory rather than be windowed. Proceed to Phase 4 with the σ-weighted (not commit-window) loss. |
| **Good (the result on disk)** | PMI identity passes to fp16 tolerance everywhere. `‖r_t‖` along the *PoE anchor* grows quickly after step ~7 and stays high; along the *Mono anchor* it's small and front-loaded. CLIP separates `"a cat and a dog"` from `"a cat"` / `"a dog"` cleanly inside steps 10–25. | The basin-commit window is real, the residual is the failure signature in ε-space, and a corrector that fires in the commit window has a target it can match. Phase 4 inherits a known time-locus and a known loss-weighting prior. |

## What this phase does *not* prove

- That `r_t` is the same across seeds (Phase 7).
- That any *learner* can match `r_t` (Phase 4).
- Anything outside `cat × dog, seed 42`.
- That CLIP is the right grader for the deployed LoRA — only that it's
  informative on the diagnostic ceiling.

## Status — 2026-05-19

| Item | Done | To do |
|---|:---:|:---:|
| `existence/` λ-grid run on cat × dog seed 42 | ✅ | |
| PMI-identity check (Δ_t = w·(ε_J + ε_∅ − ε_A − ε_B)) | ✅ | |
| Trajectory-independence appendix (PoE-anchor vs Mono-anchor) | ✅ | |
| Per-seed detection failure-mode figure (App B′) | ✅ | |
| `clip_window/` Tweedie-x̂_0 reconstruction + CLIP scoring | ✅ | |
| Commitment-window identification (steps 10–25 band) | ✅ | |
| Cross-link from this plan back into Phase 4 writeup | | ⬜ |
