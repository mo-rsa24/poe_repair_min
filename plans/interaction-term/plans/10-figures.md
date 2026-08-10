# 🖼️ The seven figures the paper needs

## What this asks, in one line
Seven figures, in the order a reader meets them, each making one claim with one
control beside it. Decide the shape of each before drawing it, then draw it from
real grids rather than from an impression of what the result looked like.

## Description
One `/design-figure` pass per figure, then the build via `/evidence-ladder`. The
figures, in story order: the two meanings of "and"; the correction's size against
noise level; the correction dosed; the correction timed; three spaces one dial;
why it is learnable; the learned version.

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

## Next

The figure chain, in order. Each line says what it writes, so you can see where the
previous step's output goes.

1. `/plan-figures` on this plan. Writes the set-level order: which seven, in what
   sequence, and whether any two collapse into one. Run once, not per figure.
2. `/design-figure` on one figure at a time. Writes 2 to 3 layout options into that
   figure's task line above: what it plots, what a reader is meant to see, and what
   the plot cannot tell them. Pick one and strike the others.
3. `/pair-figure` on the chosen layout. Names the qualitative partner that goes beside
   the number, which for the dose figure is the five-picture strip.
4. `/evidence-ladder` on the approved specs. Builds the files as
   `figures/<name>_case/step<N>_quant.png` and `step<N>_qual.png`.
5. Add a row per figure to `paper/iclr/figures.md`: what it claims, this plan as its
   source, the file path, and the review question its claim rests on.

**The short version, when time is short.** Skip 1 and 2. Use the figure that already
exists (`dose_curves.png`), run step 3 to get its qualitative partner, and do step 5.
A registered figure with an honest caption beats a beautifully designed one that is not
in the paper.

**Do not skip step 5.** A figure not in the register is invisible, and there are 1611
figure files in this repo against 3 in the manuscript.

## Engagement Instructions
```bash
ls paper/figures/                 # expect 7 built figures + specs
# each figure: qualitative element present, control visible, one-line caption
```
