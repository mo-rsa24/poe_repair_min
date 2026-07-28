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
- [ ] **Δ-correction vector field.** The sampler already computes `delta_hat = ε_LoRA − ε_frozen`
  per denoising step. Capture it for seed 9 and plot it as a 2D arrow field over the latent (a
  slice, or PCA-projected), at the denoising steps around the composition-commit window. Checkpoint:
  an arrow field, like the Langevin/score pictures, showing the push that splits the chimera; pair
  with its norm-per-step curve. The diffusion-dynamics view of the fix.
- [ ] **Manifold density plot.** Generate sample sets: pure "a cat", pure "a dog", and the cat×dog
  composition across the training sweep. Embed (CLIP or latent), project to 2D, and draw each as a
  density. A good composition covers both clouds; a chimera sits in the collapsed valley between
  them. Checkpoint: seed 9's composition point plotted across checkpoints 12500→100000, showing it
  move from the collapsed valley toward the two-mode region as training proceeds. Reuses the
  `veracity`/basin-projection idea and the repo's `manifold/` scaffolding. Biggest payoff.

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
