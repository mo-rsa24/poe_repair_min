# 🔬 What gets painted: probe how the LoRA fixes composition without re-aiming attention

## Description
Plan 01 found that the LoRA fixes cat×dog composition (the image goes from a fused chimera to
two animals around checkpoint step 20000 on seed 9) but does NOT separate the cat- and dog-token
cross-attention peaks: the two maps stay ~98% correlated in both plain PoE and LoRA. So the fix
does not act by pointing the two words at two places. This plan probes the alternative: the LoRA
changes WHAT gets written at a location, not WHERE the words look. Four views, cheapest first.

## Purpose
Turns the plan-01 preliminary finding into a mechanism. If cross-attention weights don't move
but the image separates, the change must live somewhere else (attention values, self-attention
grouping, or the score/correction field). Also settles two reader questions raised on the seed-9
artifact: why the cat/dog maps look weak (they aren't: ~11x uniform share, just diffuse), and
what other attention views exist.

## Goal
Four figures on seed 9, each pairing a picture with a number, that together say where the LoRA's
fix lives if not in cross-attention weights.

## Tasks
- [x] **Reweighted cross-attention (renorm half done; self-attn still TODO).** Added
  `--renorm-tokens` (AAE softmax renorm over the real words). Finding: renorm makes the cat/dog
  maps punchier (cat max/mean 1.39→2.29) but they stay 96.5% correlated — both word tokens attend
  to BOTH animals. The tokens are non-selective, not weak: cat gets ~11× the uniform share. So
  the "weak" look was diffuse raw probability. STILL TODO: the `track_self_attn=True` pixel-to-pixel
  grouping view (does the model treat the scene as one object or two, λ=0 vs λ=1).
- [x] **Value / content maps.** Added `track_values` to `_CrossAttnRecorder` (stores `to_v`) and
  `aggregate_painted_content` (per-location |Σ attn·value|). `value_probe.py` captures cat/dog
  weight + content maps for the same x_t, adapter OFF vs ON.
  FINDING (seed 9): the LoRA changes CONTENT ~2.9× more than WEIGHTS (30.6% vs 10.6%), and it is
  almost entirely the DOG token (dog content 53-61%, ratio 3.7-5.4×; cat content <7%, ratio <1).
  The fix rewrites what the dog token writes, leaving the cat mostly intact and both tokens'
  aim unchanged. This is a different mechanism from Attend-and-Excite weight-steering, and is the
  direct positive evidence for the "changes what gets painted" hypothesis. Caveat: content-NORM is
  a coarse proxy; value-DIRECTION would be the fuller story. Figure: scratchpad/value_compare.png.
- [x] **Δ-correction vector field.** `value_probe.py` now stores `delta = ε_LoRA − ε_PoE` per
  captured step; `scratchpad/plot_delta_field.py` renders it as a magnitude heatmap + arrow field.
  FINDING (seed 9): the correction is ~95% concentrated on the DOG (right) half at every step
  (step 40: 4807 right vs 117 left) and sharpens from the central chimera onto the dog region as
  denoising proceeds. Third independent confirmation. Figure: scratchpad/delta_field.png.
- [x] **Manifold density plot.** `gen_reference_sets.py` makes pure-cat/pure-dog (A=B same prompt,
  12 seeds each); `manifold_plot.py` CLIP-embeds pure-cat, pure-dog, cat×dog λ=0/λ=1, and the
  seed-9 sweep, laying them on a cat↔dog axis (reference centroids) + orthogonal PCA.
  FINDING: pure cats cluster left, pure dogs right; fixed two-animal images occupy a distinct
  upper-middle region the singles and broken images don't; seed 9 detours to the 17.5k chimera
  (pulled toward the dog cloud) then settles into the fixed cluster. Caveat: the "two animals vs
  one" separation lives more on the orthogonal PCA axis than the cat↔dog axis; the cat↔dog axis
  mostly reads balance. Figure: scratchpad/manifold.png.

## Summary of the mechanism (all views agree)
The LoRA fixes seed 9's cat×dog composition by rewriting what the DOG token means and writes, not
by re-aiming attention: (1) cross-attention weights barely move and stay 96.5% cat-dog correlated;
(2) painted content changes 2.9× more than weights, almost all on the dog (53-61% vs cat <7%);
(3) value DIRECTION — the refinement — rotates 39.2% for the dog vs 9.9% for the cat (4×), so the
LoRA points "dog" at genuinely different content, not just louder; (4) the Δ-correction leans dog
~55% per step (honest figure; an earlier ~95% came from a different slicing and is retracted);
(5) on the CLIP manifold the fixed image lands in a distinct two-animal region seed 9 moves into
over training. DIFFERENT mechanism from Attend-and-Excite weight-steering — evidence toward the
scope's "different channel" outcome. Per-step data (weight/content/value-cos/delta at all 50 steps)
drives the interactive mechanism tabs in the seed-9 sweep artifact
(claude.ai f2f2938e-99d6-4407-a7dc-ef6641545dbe). Remaining: generality beyond seed 9 / cat×dog;
the self-attention grouping view (task 1) is still unbuilt (OOM-prone, deferred).

## Engagement Instructions
```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
# self-attn capture writes an extra self_attn/ dir per seed
ls /datasets/mmolefe/poe_repair_min/outputs/attn_mechanism/*/a_cat__x__a_dog/seed_9/  # expect attn_maps + (new) self_attn / values
# value maps: a .pt with a 'value_map' key alongside the weight map
$PY -c "import torch; sd=torch.load('<value .pt>', weights_only=False); print(sd.keys())"
# manifold: a 2D embedding array + per-set labels
$PY -c "import numpy as np; d=np.load('<manifold .npz>'); print(list(d.keys()), d['emb'].shape)"
```
