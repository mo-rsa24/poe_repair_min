# 📊 What the cached trajectories already tell us

Design only. Findings and run state live in
[../review/hypothesis-04-what-the-cached-runs-already-show.md](../review/hypothesis-04-what-the-cached-runs-already-show.md).

## What this asks, in one line
Four questions the already-cached predictions can answer, with no image generation and no
queue at all.

## Description
Four analyses, each reading the cached predictions directly:

- **Does the correction's size track the noise level?** If one curve fits every pair, the
  correction is a property of the noise level rather than of the particular animals.
- **Where do the two paths separate?** Walk the broken path and the working path from the
  same starting noise, and measure the distance between them at each step. Where that
  distance takes off is where the outcome gets decided.
- **Does the correction push along the direction sampling is already moving?** With two
  controls: a random vector, and the right correction taken from the wrong step.
- **Is the correction low-rank enough for a small adapter to learn?** Stack the
  corrections and ask how few directions carry most of their energy, against a
  same-shape random floor.

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
Plain checkboxes: each analysis either ran or it did not. What each one FOUND is in the review
file, question by question.

- [x] The correction's size against noise level, every pair overlaid, using the normalization
      committed in `instrument-02-fix-the-size-measure-before-any-result`. Feeds slot F3.
- [x] Confirm both trajectories exist per cell for the fork read. They did not: the cache walks
      only the PoE path, so the Mono paths had to be generated (next task).
- [x] Generate the missing Mono paths from the same pinned inits.
      `scripts/mechanism_study/generate_fork_paths.sh`, resumable.
- [x] The fork curve: distance between the two paths per step, elbow reported. Feeds the F4 band.
- [x] The climb: does the correction align with the sampling motion, with a random control and a
      wrong-step control.
- [x] The factorization: stack the windowed residuals in fp32, SVD, energy-at-k against a
      same-shape random floor, and the held-out projection. Feeds slot F6.
- [ ] Decide the spectrum's statistical entity (per-timestep rows or time-averaged rows) with
      `/pair-figure`, then freeze F6's caption.

## Next

1. `/pair-figure` on the spectrum: choose the statistical entity, write the choice into the open
   review question.
2. `/design-figure` rides plan 10 for F3 and F6's final forms; do not design them here.

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
