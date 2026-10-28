# ⭐ Layer 10: latent space and VAEs

⬅️ [Layer 9: the correction](09-the-correction-learned-back.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 11: the testbed](11-the-text-conditioned-testbed.md) ➡️

**Why it is critical.** The latent is the one variable this rung adds ([the axes](../PROMPTS.md#the-axes-one-new-variable-per-rung), rung f). Construction (1) and negation are unchanged by an invertible change of space, so the guided compositions Stable Diffusion runs would carry over to its latent if its [VAE](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Variational%20Autoencoders/Variational%20Autoencoders.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/understanding-deep-learning.pdf#page=349 "Understanding deep learning, sections 17.5 to 17.7, The variational approximation to the posterior, then the VAE itself: the encoder predicts μ and Σ, a samp..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/probabilistic-machine-learning-advanced-topics.pdf#page=814 "Probabilistic machine learning: advanced topics, section 21.2, The VAE as a deep latent variable model with its modelling assumptions, and CelebA faces recon..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/the-little-book-of-generative-ai-foundations.pdf#page=54 "The little book of generative AI foundations, chapter 3, From the deterministic autoencoder to the VAE, the ELBO in three equivalent forms, and the reparamet..."), [▶1](https://www.youtube.com/watch?v=WUUq6TVAu8U&t=1372s "CME296 lecture 4, 22:52 to 31:07, The encoder outputs a mean and variance, z is sampled from that Gaussian and pushed toward a N(0, I) prior, and the decoder..."), [▶2](https://www.youtube.com/watch?v=WUUq6TVAu8U&t=1867s "CME296 lecture 4, 31:07 to 46:46, Derives the VAE ELBO on the board (marginalise over z, bring in the encoder, apply Jensen, split into reconstruction and KL..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/plans/05-a-vae-on-mnist-from-the-elbo.md "A VAE on MNIST, from the ELBO, deep-learning/diffusion-models/00-from-scratch-to-sdxl, plan 05. Times how convolution cost grows with image size, derives the...")</sub> were invertible. It is not: the [decoder](../../../../../../Obsidian/Artificial%20Intelligence/Deep%20Learning/Encoder-Decoder.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/understanding-deep-learning.pdf#page=233 "Understanding deep learning, sections 12.6 to 12.8, The three transformer layouts: an encoder model (BERT), a decoder model (GPT3), and an encoder-decoder fo..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/deep-learning-textbook.pdf#page=412 "Deep learning textbook, section 10.4, The recurrent sequence-to-sequence architecture: an encoder reads the input sequence into a context C, and a decoder ge..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/understanding-deep-learning.pdf#page=193 "Understanding deep learning, section 10.5.3, A segmentation network whose downsampling side is called the encoder and upsampling side the decoder, an hourgla..."), [▶1](https://www.youtube.com/watch?v=L8HKweZIOmg&t=223s "StatQuest, Sequence-to-Sequence (seq2seq) Encoder-Decoder Neural Networks, Building the Encoder, 3:43 to 8:27, Builds the encoder half of a small seq2seq net..."), [▶2](https://www.youtube.com/watch?v=RQowiOF_FvQ&t=157s "CS231n lecture 8, 2:37 to 9:11, A recurrent network translating 'we see the sky' to Italian: the encoder squeezes the input into one context vector and the d..."), [◆](../../../../../../goal-setting/learning/deep-learning/transformers/plans/46-run-cross-attention-with-a-2-word.md "Run cross-attention with a 2-word decoder reading a 3-word encoder, deep-learning/transformers, plan 46. Works the numbers: decoder words D = [[0.9, 0.1], [0...")</sub> loses detail, so whether the invariance survives has to be measured on a VAE, not assumed from the theorem. The plain product in a latent is a second reason, since it is not invariant even under an invertible warp.

**Depth.** Own: why a composition whose [exponents](../../../../../../goal-setting/learning/mathematics/textbooks/algebra/algebra-handbook.pdf#page=109) sum to 1 survives an invertible warp. Build: composing before and after a warp. Quiz R.1 already covered [change of variables](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=220), so this layer has one quiz. You 2.2 to 4.7 h in all.

**Study first.** [Theorem 6.1 in the feature-space read-ahead](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/sources/mechanisms-of-projective-composition/read-ahead/06-feature-space.md) and [its proof in appendix I](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/sources/mechanisms-of-projective-composition/read-ahead/18-appendix-i-reparameterization.md). Textbooks: Deisenroth et al. 6.7 (p220) and Chen 7.1.1 (p140).

**Ideas to tie in.** From [the ideas that tie the ladder together](../../../ideas/ideas-that-tie-the-ladder-together.md): [fixed decoders before training a VAE](../../../ideas/ideas-that-tie-the-ladder-together.md#2-use-fixed-decoders-before-training-a-vae) (could help, 3 to 5 h without training); [decoder geometry as a known measure correction](../../../ideas/ideas-that-tie-the-ladder-together.md#3-treat-decoder-geometry-as-a-known-measure-correction-not-as-the-interaction) (could help as a diagnostic, 3 to 5 h). Each comes with an exact toy check.

## The flow

```text
 AI, in the background                         │  You, in order
───────────────────────────────────────────────┼───────────────────────────────────────────────
 1. quiz 10.1, prepared ─────────────────────▶ │  2. /drip --math: change of variables
   (paste when you start this layer)           │       ▼
                                               │  3. build: walk10.py, the warp
 the warp check's run.log ───────────────────▶ │  4. compare: the same four rows from both
                                               │       ▼
                                               │  5. quiz 10.1: warp invariance
                                               │       ▼
                                               │  6. /drip-idea, claim 11: the VAE check
                                               │       ▼
 xref-cadence links the layer ◀─────────────── │  7. /drip-retro, then layer 11
```

## The steps

### 1. AI, in the background: quiz 10.1

When you start the layer, paste the prompt so quiz 10.1 is waiting after the build. It checks why a weighted product of densities picks up |det J| to the power minus the sum of its exponents under an invertible map, so construction (1) and negation survive a warp and the plain product does not.

**Prompt:** [build quiz 10.1](prompts/10-1-build-quiz-10-1.md)

⏱ you 2 min to paste · AI about 40 min (estimate: the setup's ~2 h for three quizzes; neither quiz agent has a measured run), beside the maths drip and the build

### 2. You: the maths drip

You derive, one rung per message, how a density changes under an invertible map and why a composition whose exponents sum to 1 is unchanged by it. The worked instance is the build's warp, where det J = 49 at z = 0 for a = 3. Your quiz R.1 misses on change of variables become the first rungs.

**Prompt:** [the maths drip](prompts/10-2-the-maths-drip.md)

⏱ you 10 to 20 min · AI ~10 min

### 3. You: the build

You type the warp check into `walk10.py`, cut into 4 to 6 pieces: the four-blob toy in a latent, a warp at four strengths, and each composition built before and after the warp. Construction (1) should agree across the warp to rounding (about 4e-17 total variation), and the plain product should differ by 0.1261 at the strongest warp, where the map stretches area up to 49 times.

**Prompt:** [the build](prompts/10-3-the-warp-build.md)

⏱ you 1 to 2 h · AI ~30 min

**Outcome:** the invariance shown in code.

### 4. You: compare your numbers with the reference

```bash
cat /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/artifacts/ideas/when-poe-reaches-a-and-b/runs/compose-before-or-after-a-warp/run.log
cd ~/Playground/PhD/simple-distributions-walk
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python walk10.py
```

⏱ you ~10 min · stop: the four rows agree. `detJ_max` reads 1, 2.56, 9 and 49; `tv_plain_product` reads 0, 0.03539, 0.07957 and 0.1261; construction (1) and negation sit near 1e-17 in both. The rounding-level values will not match digit for digit, and they do not need to. A plain-product number that differs is either a lesson or a bug in one of the two files: find which before the quiz.

### 5. You: quiz 10.1

Quiz 10.1 checks warp invariance: the factor a map adds to a density, the exponent sums of the three compositions, and what a decoder that cannot be inverted changes.

Quiz 10.1: (URL lands here)

Built by step 1. ⏱ you 20 to 40 min · stop: every *needed now* question right and at least 80% of the rest; below that, read the items your misses point at and retake.

### 6. You: the idea walk

You resume claim 11 of the when-poe idea map: whether a VAE trained on the 2D toy shows what happens inside Stable Diffusion's latent, and what the smallest check is. The walk ends with that check written down beside what each outcome means, or with the claim marked dead.

To see the space SDXL composes in, open the [latent part of the SDXL notebook scene](https://claude.ai/artifact/1GayZthQkRL4FmukdCV9WW#nb-latent). The noisy latent there is 4 × 128 × 128, the compressed picture the denoiser works on, and the VAE decoder that turns it back into pixels is the map this walk asks about.

**Prompt:** [the idea walk](prompts/10-6-the-vae-idea-walk.md)

⏱ you ~20 min (the median of 13 measured `drip-idea` runs; the middle half took 5 to 61 minutes), AI ~10 min · stop: the walk ends when the smallest VAE check is chosen or claim 11 is marked dead

**Outcome:** a decision on the smallest VAE experiment worth running.

### 7. You: close the layer

Run the `/drip-retro` command the build printed at its close (about 25 minutes). Then paste [the linking prompt](prompts/10-7-link-the-layer.md), which links the maths drip, the retrospective and claim 11 to the feature-space read-ahead, the latent-space brick row, and claims 10 and 11 with the warp run's card.

⏱ you 2 min to paste · AI ~10 min (estimate)

---

⬅️ [Layer 9: the correction](09-the-correction-learned-back.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 11: the testbed](11-the-text-conditioned-testbed.md) ➡️
