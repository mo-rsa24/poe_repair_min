# 📊 The four disk-only analyses

## Description
Four analyses that read the cached trajectories and residuals directly, no
image generation: the correction's size against noise level, where the PoE and
Mono paths fork, the plausibility climb along each path, and the factorization
that asks whether the correction is small and shared.

## Purpose
The theory core: the term is universal in noise level (Goal 4), the trajectory
fork corroborates the timing read (Goal 2), the climb is the cheap tier of the
density instrument, and the factorization answers why rank-8 suffices
(Goal 3). Serves DoD 5.

## Goal
Four figures, each with one number attached: collapse spread across pairs,
fork elbow step, climb gap between PoE and Mono paths, energy-at-k with the
held-out projection.

## Environment Facts This Plan Depends On
- Cached tensors are fp16: upcast to fp32 before any accumulation (the SVD
  especially).
- Runs in-session on mscluster85 (CPU heavy, 123GB RAM covers the stacked
  matrix; restrict rows to the measured window).
- The cache stores per-step states along one recorded path per cell; the fork
  analysis needs BOTH the PoE and Mono paths.

## Tasks
- [x] ✅ ‖r_t‖ against log-SNR, every pair overlaid, normalized per plan 01;
      report the collapse spread
      ✓ verified: spread 19.7% ("loose") over 17 pairs / 34 curves, using the
      plan-01 committed relative_norm. Under that measure the median curve has
      NO interior peak: it is still rising at the right edge. The raw-‖r_t‖
      measure peaks at log-SNR -0.90 instead, because ‖ε_PoE‖ falls ~15% along
      the trajectory. Do not read either as plan 04's timing answer.
- [ ] ⚠️ check both trajectories exist per cell for the fork analysis; if only
      one path was recorded, add a cheap regeneration task for the other
      [inferred]
- [ ] ⚠️ fork curve d(t) = ‖x_t_PoE − x_t_Mono‖ from shared inits; report the
      elbow step and compare against plan 04's peak band
- [ ] ⚠️ plausibility climb: sum of r_t · Δx_t along each path, PoE vs Mono
      distributions
- [ ] ⚠️ factorization: stack windowed target residuals in fp32, SVD, curves
      for pooled vs same-shape Gaussian floor vs per-pair blocks
- [ ] ⚠️ held-out projection: fit top-k on train pairs, report energy explained
      on held-out pairs vs k
- [ ] ⚠️ /pair-figure decision: per-timestep rows vs time-averaged rows as the
      spectrum's statistical entity

## Success/Failure Outcomes
- **factorization**
  - Success: pooled curve beats the Gaussian floor clearly; per-pair
    comparison readable either way.
  - Failure: fp16 accumulation artifacts (NaN/inf in singular values). Upcast
    was skipped; redo in fp32.

## Recommended skill
▶ `/pair-figure` ✅ before the spectrum plot; `/demonstrate` ✅ per analysis
   (one plot, one number).

## Engagement Instructions
```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
$PY scripts/snr_collapse.py        # figure + "collapse spread: X%" printed
$PY scripts/fork_curve.py          # figure + "elbow at step N" printed
$PY scripts/climb.py               # PoE vs Mono climb distributions
$PY scripts/spectrum.py            # energy-at-k table + held-out projection
```
