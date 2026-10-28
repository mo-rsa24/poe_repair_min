/drip-idea A small SDXL-shaped diffusion model on 32 × 32 procedurally drawn shapes can reproduce the cat-dog hybrid under conditions I control, and probing it can show where the hybrid comes from. I want it to cover everything the 2D toy covered, plus the parts 2D cannot hold (a text encoder, cross-attention, word similarity), and to see how deep the probing can go. Correct me where the idea is wrong.

## Where this comes from

Repo: /home-mscluster/mmolefe/PhD/reduce-reuse-recycle. It grows out of claim 12 of the idea walk at artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md. Read that map in full first; the parent walk continues separately, so write nothing into it. This walk keeps its own map at artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md.

What the 2D walk settled, which this testbed must reproduce at image scale before it claims anything new:
- **The support rule.** A product of experts reaches the region that is both A and B only as far as each expert's density does. A missing combination has to be borrowed (support kept from pretraining; strongest as one conditional model with "contains" labels, composed as p(x|A)·p(x|B)/p(x), which holds `0.989` in A∩B down to a data share of `0.001`, while separately fine-tuned experts cap at `0.5`) or built (experts that each model only their own part of the image, `0.9999` with no A∩B data). Answer key: artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/.
- **Two constructions.** score(x|A)+score(x|B)−score(x) and score_A+score_B compose different things, and the second divided by a background expert becomes the first. The first is unchanged by the space it is computed in; the plain product is not (run: runs/compose-before-or-after-a-warp/).
- **Sampler against data.** With exact experts, annealed ULA, MALA (fixed) and UHA reach the composed density, and reverse diffusion with summed scores misses it by `0.049` total variation above the exact-sample floor (runs/samplers-on-exact-experts/). With learned scores the two errors may interact rather than add (Soiffer et al. 2026, quoted in artifacts/ideas/poe-overlap-dial/composition-terms.md).
- **Kinds of composition.** Co-occurrence, intersection, interaction (the wet dog), contention (the chimera or hybrid) and negation, defined in artifacts/ideas/poe-overlap-dial/composition-terms.md. The earlier walk found chimera mass `0.66` when both experts saw their object in the same place.

## The testbed, as I picture it

- **Data.** 32 × 32 RGB images of procedurally drawn shapes (a red clean circle, blue scattered dots, and a few more shapes and colours), so every image's ground truth is known exactly and a parser of the drawing, not a learned classifier, scores outputs as both present, one missing, or fused into a hybrid.
- **An SDXL-shaped model at small scale.** A tiny text encoder over a small vocabulary, a UNet with cross-attention at two resolutions, and a pooled text embedding added to the time embedding, so each structural part of SDXL has a counterpart. My learning journey covers these parts: /home-mscluster/mmolefe/goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/plans/ (04 cross-attention, 07 SDXL's two text encoders, 08 classifier-free guidance, 15 which layer each research question tests). Read 07, 08 and 15.
- **Dials.** Overlap of the two concepts in the training data (the 2D dial); how close the two words' embeddings sit (cat and dog sit close in CLIP; Bradley et al. 2025a, Figure 8); whether each expert sees only its own region (masks or boxes) or the whole image; the joint caption against construction (1) against the plain product against the background-divided product; the sampler; and fine-tuning on a small number of joint examples against a joint model trained on the same few.
- **Probes.** Each word's cross-attention map (does it land on its own region?), the token embeddings' similarity, the interaction term where it can be computed, and the parser's hybrid rate per setting.
- Prior synthetic-shape studies to borrow from: Okawa et al. 2023, Compositional Abilities Emerge Multiplicatively (confident); Park et al. 2024, Emergence of Hidden Capabilities (likely, verify).

## What I want from the walk

The claims this rests on, each with its mark, starting from the one it dies without: that a hybrid in this testbed has the same cause as a cat-dog hybrid in SDXL. Then how deep the probing can go: which cause each probe can isolate, and which it cannot. Python, NumPy and PyTorch only. Constraints worth knowing: the cluster's GPU nodes and the `co3` and `co3_bw` conda environments; experiments stay small enough to run in minutes to an hour.

## Return

When the walk compiles, write a fenced return package to artifacts/ideas/when-poe-reaches-a-and-b/routes/08-sdxl-shaped-hybrid-testbed.return.md:
  origin: artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md · claim 12 · route 8
  ask: an SDXL-shaped testbed that reproduces and probes hybrids, covering the 2D walk's findings
  deliverable: the testbed walk's map path, its compiled verdict, and its route out
  changes for the walk: one line on how claim 12 should be marked
Then I paste it into the parent walk and say `integrate`.
