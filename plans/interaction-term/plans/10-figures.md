# 🖼️ The seven-figure cascade

## Description
One /design-figure pass per figure, then the build via /evidence-ladder. The
figures, in story order: the two meanings of "and"; the gap seen (‖r_t‖ vs
log-SNR); the cure dosed; the cure timed; three spaces one dial; why it is
learnable; the learned version.

## Purpose
The paper's evidence set, each figure carrying one claim with one control.
Serves DoD 10.

## Goal
Seven design specs committed and seven built figures, each pairing a
qualitative view with its number.

## Environment Facts This Plan Depends On
- Consumes outputs of plans 02-08; builds run in-session.
- The two purely illustrative figures (the three-regime diagram; the method
  schematic) have ready-to-paste ChatGPT image prompts from the
  hypothesis-to-scope pass; all other figures must come from real grids.

## Tasks
- [ ] ⚠️ /design-figure: the two meanings of "and" (three-panel density
      diagram over real λ=0 exemplars; ChatGPT prompt exists)
- [ ] ⚠️ /design-figure: the gap seen (y normalized ‖r_t‖, x log-SNR, thin
      line per pair, one bold mean)
- [ ] ⚠️ /design-figure: the cure dosed (image strip above the curve, shared
      λ axis, controls gray below)
- [ ] ⚠️ /design-figure: the cure timed (W1+W2 curves, fork elbow as a
      vertical band)
- [ ] ⚠️ /design-figure: three spaces one dial (manifold walk, caption
      crossover, density climb; shared λ colorbar)
- [ ] ⚠️ /design-figure: why it is learnable (spectrum with Gaussian floor
      shaded, held-out projection inset)
- [ ] ⚠️ /design-figure: the learned version (mechanism panel beside the
      transfer table and replication strip; method-schematic ChatGPT prompt
      exists)
- [ ] ⚠️ /evidence-ladder build of the approved specs
- [ ] ⚠️ run the two paper-illustration prompts (three-regime diagram, method
      schematic) through ChatGPT and drop the images in
      plans/interaction-term/assets/  [owner: human]

## Recommended skill
▶ `/design-figure` ✅ per figure task; `/evidence-ladder` ✅ for the build;
   `/plan-figures` ✅ as the set-level check if the order shifts.

## Engagement Instructions
```bash
ls paper/figures/                 # expect 7 built figures + specs
# each figure: qualitative element present, control visible, one-line caption
```
