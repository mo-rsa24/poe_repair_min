# 🔭 The same story from three independent sides

**Step 7 of 22.** Waits on step 4. The one order is the `## Running order` table in the [repo root MASTER_PLAN.md](../../../../../MASTER_PLAN.md).

| Step | Plan | Status |
|---|---|---|
| 6 | [hypothesis-03-when-in-the-run-it-matters](hypothesis-03-when-in-the-run-it-matters.md) | ◑ timing tab owed |
| **7** | **this plan** | **✅** |
| 8 | ~~[hypothesis-01-what-the-fix-changes-inside-the-model](hypothesis-01-what-the-fix-changes-inside-the-model.md)~~ | ✅ |

Design only. Verdicts live in [../review/hypothesis-05-the-same-story-from-three-sides.md](../review/hypothesis-05-the-same-story-from-three-sides.md).

## What this asks, in one line
Three checks that share none of the strength-sweep's machinery, so they can agree with it
independently rather than repeat it.

## Description
Three independent reads:

- **The picture moves out of the blend region.** Place the strength-sweep's images on the
  existing image-similarity axes and watch them slide as the strength rises. The control is
  the same-sized push in a random direction, which should not slide.
- **The prompt's own arithmetic predicts which pairs are hard.** Ask whether the joined
  prompt is more than the sum of its two parts in text space, and whether a blended picture
  reads back as a blend caption.
- **A blended animal is wrong content, not a bad picture.** Measure image quality on the
  cached broken and working outputs and expect no gap. This removes the objection that the
  correction merely improves quality.

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
- L1/L3 read cached `embeddings.pt` per cell. SDXL's own two text encoders give
  four views: pooled (1280, from text_encoder_2), the CLIP-L 77-token sequence
  (first 768 channels of the cached 2048), the bigG sequence (last 1280), and
  the concatenation cross-attention actually consumes. Concatenation order is
  fixed by `poe_repair/_sdxl/runtime.py`.
- The manifold slide and L2 consume plan 03's dose images, which are on disk:
  440 renders under `outputs/interaction_term/dose/pairs`, λ ∈ {0, .25, .5,
  .75, 1}, with 32 cells carrying the `_random` and `_wrong_pair` control rows.
- The quality check reads `poe.png` and `mono.png` from the training cache:
  749 paired cells.
- CLIP embedding and GroundingDINO run in-session on the 3090 (light).
- λ=1 reproduces ε_J exactly, so the full-dose picture is the joint render
  (measured at 1.9 grey levels of 255). Every dose bar is therefore read at the
  largest interior dose, λ=0.75.

## Tasks
- [x] L1 additivity gap per pair (both encoders, pooled and sequence);
      scatter against normalized ‖r_t‖
- [x] L3 binding direction: b = e_J − normalized(e_A+e_B) per pair; cosine
      matrix and SVD across pairs, against a mismatched-solos control
- [x] chimera quality control on cached poe.png/mono.png: quality proxies,
      expect no gap
- [x] manifold slide: embed plan 03's λ-sweep outputs on the existing CLIP
      axes; random-direction path as the control
- [x] L2 caption readback on plan 03's images: caption bank including the
      blend caption; crossover curve vs λ

## Success/Failure Outcomes
- **L1 additivity gap**
  - Success: gap computed for all pairs; correlation with ‖r_t‖ reported
    either way (a null is a finding: binding info lives in joint processing,
    not the embedding).
  - Failure: pooled-only probing (the sequence form is the one cross-attention
    consumes; both must be reported).

## Next

1. Final figure forms ride plan 10. The quality check and the caption readback are the two
   that carry their own argument; F5 takes the manifold slide.
2. The `wrong_pair` control reaches 44% of the oracle's interior travel on the manifold
   slide, against a 50% bar. Plan 10 should either widen that gap or say plainly that a
   mis-aimed correction gets you nearly half the slide.
3. L1 and L3 both come back null. If a later plan wants a language-space predictor, the
   place to look is joint processing (cross-attention maps), not the prompt embedding.

**The short version:** the quality check alone. It removes the one standing objection ("the
correction just improves image quality") and runs from cache in minutes.

## Engagement Instructions
```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python

# L1 + L3, 75 pairs from cache, ~1 min. Prints a per-pair table, then the
# correlation against the preregistered correction-size measure and the
# shared-direction read against its mismatched-solos control.
$PY scripts/language_probes.py --probe l1 --probe l3

# 749 paired poe.png/mono.png, ~20 min (GroundingDINO on every image).
# Four content-blind quality proxies decide; the compose rate is the positive
# control. Expect flat quality and a large content gap.
$PY scripts/quality_control.py

# 32 cells x 5 doses x 3 rows in CLIP image space, ~3 min. Checks the lambda=1
# endpoint really is the mono render before reading anything off the curves.
$PY scripts/manifold_slide.py

# Same 32 cells against a four-way caption bank, ~3 min.
$PY scripts/caption_readback.py
```

Outputs land in `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses/`
as `language_probes.json`, `quality_control_cache.json`, `manifold_slide_clip.json`,
`caption_readback.json`, each beside its figure.
