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
- [ ] **Self-attention + reweighted cross-attention.** Re-capture seed 9 with
  `track_self_attn=True` and render (a) pixel-to-pixel self-attention grouping (does the model
  treat the scene as one object or two?), and (b) cross-attention softmax-renormalized over the
  real words only (`drop_bos`/`text_token_count`, already in `aggregate_token_map`) so the maps
  show each word's SHARE, not raw prob. Checkpoint: the renormalized cat/dog maps are punchier
  (peak/mean up) and the self-attn grouping is shown λ=0 vs λ=1. Answers "why weak" + "other
  attention forms". Cheap: recorder flags already exist.
- [ ] **Value / content maps.** Extend `_CrossAttnRecorder` to also store `to_v(encoder)` so we
  can map what the cat/dog tokens WRITE at each location, not just the weight. Compare λ=0 vs λ=1
  value-norm and value-direction maps on seed 9. Checkpoint: if the fix lives in the values, the
  value maps differ between regimes more than the weight maps do (they were only ~6-8% apart).
  This is the direct test of "changing what gets painted".
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
