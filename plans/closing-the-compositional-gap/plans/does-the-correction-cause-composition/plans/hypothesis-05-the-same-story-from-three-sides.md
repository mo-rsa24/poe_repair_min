# 🔭 The same story from three independent sides

Design only. Verdicts live in [../review/hypothesis-05-the-same-story-from-three-sides.md](../review/hypothesis-05-the-same-story-from-three-sides.md).

## What this asks, in one line
Three reads that do not share the dose experiment's machinery: does the image slide out of the
blend region as the dose rises (and not under a random push), does language-space arithmetic
predict which pairs need a big correction, and are chimeras wrong content rather than low
quality.

## Description
Three independent reads of the same story: the image-map walk (dose-sweep
outputs sliding out of the blend region), the language probes (is the joint
prompt more than the sum of its parts; does the chimera match a blend caption),
and the quality check (chimeras are wrong content, not bad images).

## Purpose
The causal claim shown from three sides (Goal 1's secondary reads). The
language probes give the composition-type regimes predictive teeth; the
quality check removes the "the correction just improves image quality"
objection. Serves DoD 6.

## Goal
The manifold slide with its random-path control, the caption-match crossover
curve, the additivity-gap and binding-direction results, and the quality-gap
table.

## Environment Facts This Plan Depends On
- L1/L3 read cached embeddings.pt per cell; use SDXL's OWN two text encoders,
  probe pooled AND 77-token sequence forms.
- Manifold and L2 consume plan 03's dose images; runnable only after that
  sweep lands. L1, L3, and the quality check run now from cache.
- CLIP embedding runs in-session on the 3090 (light).

## Tasks
- [ ] L1 additivity gap per pair (both encoders, pooled and sequence);
      scatter against normalized ‖r_t‖
- [ ] L3 binding direction: b = e_J − normalized(e_A+e_B) per pair; cosine
      matrix and SVD across pairs
- [ ] chimera quality control on cached poe.png/mono.png: quality proxies,
      expect no gap
- [ ] manifold slide: embed plan 03's λ-sweep outputs on the existing CLIP
      axes; random-direction path as the control
- [ ] L2 caption readback on plan 03's images: caption bank including the
      blend caption; crossover curve vs λ

## Success/Failure Outcomes
- **L1 additivity gap**
  - Success: gap computed for all pairs; correlation with ‖r_t‖ reported
    either way (a null is a finding: binding info lives in joint processing,
    not the embedding).
  - Failure: pooled-only probing (the sequence form is the one cross-attention
    consumes; both must be reported).

## Next

1. `/demonstrate` the two cache-only probes now (additivity gap, binding direction) and the
   quality check; no GPU queue, they answer three review questions today.
2. After plan 03's re-score: the manifold slide and the caption readback on the dose images.
3. Answer the review questions; final figure forms ride plan 10.

**The short version:** the quality check alone. It removes the one standing objection ("the
correction just improves image quality") and runs from cache in minutes.

## Engagement Instructions
```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
$PY scripts/language_probes.py --probe l1 --probe l3   # per-pair table printed
$PY scripts/quality_control.py                          # gap table, expect ~0
$PY scripts/manifold_slide.py                           # needs plan 03 outputs
```
