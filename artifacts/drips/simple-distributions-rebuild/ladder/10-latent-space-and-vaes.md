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

Paste this when you start the layer, so the quiz is waiting after the build.

```text
In the background, prepare and build one quiz for the composition ladder in
artifacts/drips/simple-distributions-rebuild/: quiz 10.1, on warp invariance.

Task: first dispatch prep-cadence (quiz-prep-pulse --build) with the goal, minutes and topics below,
then dispatch quiz-cadence (mode build, repo /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain). The
goal is "say why a composition whose exponents sum to 1 survives an invertible warp".
- Quiz 10.1, --minutes 40: a density pushed through an invertible map x = g(z) picks up the factor
  |det J|⁻¹; a weighted product ∏ p_i^(w_i) then picks up |det J|^(−Σw_i); construction (1),
  p(x|A)·p(x|B)/p(x), has exponents 1 + 1 − 1 = 1 and negation p_A^(1+w)/p_B^w has (1 + w) − w =
  1, so both are unchanged; the plain product p_A·p_B has exponents summing to 2 and is reweighted
  by how much the map stretches each region; what a non-invertible VAE decoder changes.
Starts from: nothing before it in this layer. The material is
artifacts/drips/simple-distributions-rebuild/ladder/10-latent-space-and-vaes.md and its Study first
line, claim 10 of artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, and the run card
artifacts/ideas/when-poe-reaches-a-and-b/runs/compose-before-or-after-a-warp/README.md.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- quiz-cadence stops when a quiz has no coverage file in .quiz-prep/, so prep-cadence runs first
  (quiz-cadence, "The preparation gate").
- Quiz R.1 at the reading stop already covers change of variables with a Jacobian, so this quiz
  starts from |det J| as known and asks only what layer 10 adds (this file's Depth line).
- pdftotext is not installed on mscluster, so
  ~/.claude/skills/index-source/scripts/find-in-textbooks.py fails; open the PDFs with PyMuPDF from
  the co3 environment (checked on mscluster109: import pymupdf works in co3, version 1.28.2, and
  import pypdf fails everywhere).
- The shelf index is ~/Obsidian/Mathematics/references/textbooks/Textbooks.md and the
  worked-examples index is ~/Obsidian/Mathematics/references/exercises/Worked Examples.md
  (quiz-prep-pulse, "The bank").
- The textbook passages are Deisenroth et al. 6.7 (p220) and Chen 7.1.1 (p140), and the paper is
  Bradley et al. 2025a (arXiv 2502.04549), Theorem 6.1 with its proof in appendix I (this file's
  Study first line).
- The warp is x1 = z1 + a·tanh(2 z1), x2 = z2 + a·tanh(2 z2) + 0.4 z1. Its Jacobian is lower
  triangular, so det J = (1 + 2a·sech²(2 z1))(1 + 2a·sech²(2 z2)), largest at z = 0 where it is
  (1 + 2a)²: 2.56, 9 and 49 at a = 0.3, 1 and 3 (warp_check.py; warp_check.json).
- Across that warp, construction (1) agrees to 4e-17 total variation and negation to 7e-17, which
  is rounding; the plain product differs by 0.035, 0.080 and 0.126 at largest area stretches 2.6,
  9 and 49 (the run card; claim 10).
- A VAE decoder is not invertible; its analogue of |det J| is the volume factor √det(JᵀJ) of the
  pullback metric, marked "likely, verify" (claim 10). Questions on it are marked as such.
Outputs:
- one coverage file under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, with a verdict
  per question: ready, blocked, or unprepared by choice;
- rows in the worked-examples index for every banked problem;
- one published quiz page, opening on a read-first list with minutes per item, a short path and a
  full path.
Target: no question carries the verdict blocked; about 4 to 6 banked problems for the concept at
the own depth; every number in an answer traces to the run card, warp_check.json or a cited
textbook page. I judge, from the page.
Progress: prep-cadence reports its question count, node count and verdicts; quiz-cadence reports
the URL when it publishes.
Proof: the coverage file exists under .quiz-prep/ and lists no question as blocked; the URL opens.
Limits: change nothing in the idea map, the run folder or rebuild/; in this ladder file, write only
the URL.
Hand-off: write the URL into step 5 of
artifacts/drips/simple-distributions-rebuild/ladder/10-latent-space-and-vaes.md, replacing "(URL
lands here)".
```

added: the whole quiz-building prompt, in the setup's shape, with the preparation gate, the missing `pdftotext` and the PyMuPDF route, the two indexes (from the setup's quiz prompt) · the warp, its [Jacobian](../../../../../../Obsidian/Mathematics/Calculus/Jacobian.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=155 "Mathematics for machine learning, section 5.3, Gradients of vector-valued functions as the Jacobian, then the Jacobian determinant as an area scale worked to..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/calculus/calculus-textbook.pdf#page=1082 "Calculus textbook, section 14.7, The Jacobian of a change of variables in two variables (Definition 14.7.1, page 1086) and three (Definition 14.7.3, page 109..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/deep-learning-textbook.pdf#page=102 "Deep learning textbook, section 4.3.1, The Jacobian matrix J with entries ∂fᵢ/∂xⱼ, and the Hessian as the Jacobian of the gradient."), [▶1](https://www.youtube.com/watch?v=ne2ngVAoMG8&t=2569s "Stanford CS229, lecture 8, 42:49 to 48:29, Derives the backward rule for u = Wz + b: the Jacobian entries are ∂uⱼ/∂zᵢ = Wⱼᵢ, so ∂J/∂z = Wᵀ ∂J/∂u. Caption-read."), [▶2](https://www.youtube.com/watch?v=25zD5qJHYsk&t=3873s "Stanford CS231n, lecture 4, 1:04:33 to 1:11:16, Scalar derivative, gradient vector and Jacobian matrix side by side, and ReLU's diagonal Jacobian that is nev..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/textbook/l0-simple-maths/01-log-sum-exp-and-the-softmax-gradient.md "1. Log-sum-exp and the softmax gradient, deep-learning/diffusion-models/00-from-scratch-to-sdxl, volume L0, section 1. The softmax Jacobian diag(p) − ppᵀ, de...")</sub> and the run's numbers (from `warp_check.py`, `warp_check.json` and the run card) · the [VAE](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Variational%20Autoencoders/Variational%20Autoencoders.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/understanding-deep-learning.pdf#page=349 "Understanding deep learning, sections 17.5 to 17.7, The variational approximation to the posterior, then the VAE itself: the encoder predicts μ and Σ, a samp..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/probabilistic-machine-learning-advanced-topics.pdf#page=814 "Probabilistic machine learning: advanced topics, section 21.2, The VAE as a deep latent variable model with its modelling assumptions, and CelebA faces recon..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/the-little-book-of-generative-ai-foundations.pdf#page=54 "The little book of generative AI foundations, chapter 3, From the deterministic autoencoder to the VAE, the ELBO in three equivalent forms, and the reparamet..."), [▶1](https://www.youtube.com/watch?v=WUUq6TVAu8U&t=1372s "CME296 lecture 4, 22:52 to 31:07, The encoder outputs a mean and variance, z is sampled from that Gaussian and pushed toward a N(0, I) prior, and the decoder..."), [▶2](https://www.youtube.com/watch?v=WUUq6TVAu8U&t=1867s "CME296 lecture 4, 31:07 to 46:46, Derives the VAE ELBO on the board (marginalise over z, bring in the encoder, apply Jensen, split into reconstruction and KL..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/plans/05-a-vae-on-mnist-from-the-elbo.md "A VAE on MNIST, from the ELBO, deep-learning/diffusion-models/00-from-scratch-to-sdxl, plan 05. Times how convolution cost grows with image size, derives the...")</sub> caveat (from claim 10) · the URL write-back
dropped: "ask for it with the layer 8 prompt's shape" (why: a fresh session cannot read a shape from another file; the prompt now stands alone)

⏱ you 2 min to paste · AI about 40 min (estimate: the setup's ~2 h for three quizzes; neither quiz agent has a measured run), beside the maths drip and the build

### 2. You: the maths drip

```text
/drip --math change of variables and the Jacobian, and why a composition whose exponents sum to 1 is
unchanged by an invertible warp

Starts from: quiz R.1 passed at the reading stop; any question it marked missed on change of
variables becomes a first rung.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- A density moved through an invertible map x = g(z) picks up |det J|⁻¹. Each conditional and the
  unconditional pick up the same factor, so p(x|A)·p(x|B)/p(x) picks it up once, exactly as the
  latent composition pushed through g does (claim 10 of
  artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md).
- Construction (1) has exponents 1 + 1 − 1 = 1 and negation (1 + w) − w = 1; the plain product
  has 2, so it picks up |det J|⁻¹ twice in pixel space and once from the latent (claim 10).
- The worked instance is the run's warp, x1 = z1 + a·tanh(2 z1), x2 = z2 + a·tanh(2 z2) + 0.4 z1:
  lower triangular, so det J is the product of the two diagonal terms, (1 + 2a)² at z = 0, which
  gives the largest area stretches 2.56, 9 and 49 at a = 0.3, 1 and 3
  (runs/compose-before-or-after-a-warp/warp_check.py and warp_check.json).
- The general statement is Theorem 6.1 of Bradley et al. 2025a (arXiv 2502.04549), proved in
  appendix I (this file's Study first line).
- A VAE decoder is not invertible; √det(JᵀJ) as its volume factor is "likely, verify" (claim 10),
  so the drip names it and does not derive it.
Outputs: the drip's thread folder under artifacts/drips/, with dispatch.md recording my notation,
the worked numbers I used and what clicked.
Target: I show without notes that ∏ p_i^(w_i) pushed through g picks up |det J|^(−Σw_i), and
compute det J = 49 at z = 0 for a = 3 by hand; I judge.
Hand-off: the thread folder's path, which step 3's build and step 6's idea walk read.
```

added: the factor argument and both exponent sums (from claim 10) · the run's warp and its three largest stretches as the worked instance (from `warp_check.py` and `warp_check.json`) · the theorem and the VAE caveat (from this file's Study first line and claim 10) · quiz R.1's misses as first rungs
dropped: nothing

⏱ you 10 to 20 min · AI ~10 min

### 3. You: the build

```text
/drip-execute-plan --request compose before and after a warp on the quadrant toy, reproducing
artifacts/ideas/when-poe-reaches-a-and-b/runs/compose-before-or-after-a-warp/, I type every block
myself into ~/Playground/PhD/simple-distributions-walk/walk10.py; bar: difference 4e-17 for
construction (1), 0.126 for the plain product at area stretch 49

Task: cut the warp check into 4 to 6 pieces and build, teach and test one per round. I type every
block myself into ~/Playground/PhD/simple-distributions-walk/walk10.py and run it from that folder
with /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python.
Starts from: the setup's folder and environment; the layer 10 --math drip folder under
artifacts/drips/, whose dispatch.md holds my notation (step 2).
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The reference is warp_check.py in the run folder: the quadrant toy in a latent z, one Gaussian
  blob per quadrant at (±0.5, ±0.5) with std 0.15, A∩B share r = 0.05 on the top-right blob and
  (1 − r)/3 on each other blob (warp_check.py).
- Expert A is the right half's data (top-right and bottom-right weights) and expert B the top
  half's (top-right and top-left); the unconditional p is all four blobs (warp_check.py).
- Everything is evaluated on one 801 × 801 latent grid over [−1.6, 1.6]². A pixel-space expert is
  the pushforward p_Z(z)/|det J|, and a cell's mass is p_X(x)·|det J|·dz, so both routes are
  compared as probability over the same cells by total variation (warp_check.py; the run card).
- The warp is x1 = z1 + a·tanh(2 z1), x2 = z2 + a·tanh(2 z2) + 0.4 z1 at a = 0, 0.3, 1 and 3;
  det J's smallest and largest values are 1 and 1, 1.008 and 2.56, 1.027 and 9, and 1.081 and 49
  (warp_check.json).
- Negation uses w = 0.3, so p_A^1.3 / p_B^0.3 (warp_check.py).
- The reference numbers: construction (1) 3.72e-17, 3.71e-17 and 3.82e-17 at a = 0.3, 1 and 3,
  which the card rounds to 4e-17; negation 5.0e-17 to 6.6e-17; the plain product 0.03539, 0.07957
  and 0.1261; all three are 0 at a = 0 (warp_check.json; run.log).
- The plain product's top-right share barely moves with the warp: 0.9997 in the latent, 0.9998 in
  pixels (run.log), so total variation is the number that shows the change, not the share.
- No rebuild/latent.py exists; the REBUILD_WALK.md brick "Composing in a latent space" names it as
  not built, so the run folder is the only reference.
Outputs:
- walk10.py: one script that prints, per warp strength, det J's range and the three total
  variations;
- the step file and walk map the --request open writes, with each piece ticked;
- a figure of the plain product composed before and after the warp at a = 3, side by side.
Target: walk10.py prints 0.1261 (to four digits) for the plain product at a = 3 and a construction
(1) difference under 1e-15 at every a. I judge, from the printed lines.
Progress: the first piece prints the latent mixture's mass summing to 1; the warp piece prints det
J's range per a; each construction's piece prints its four total variations.
Proof: python walk10.py prints four rows matching run.log in the run folder: tv_plain_product 0,
0.03539, 0.07957, 0.1261, and tv_construction_1 under 1e-15.
Limits: never edit the run folder or rebuild/; I type, I do not paste; no commit.
Hand-off: the retrospective the walk closes on, and the /drip-retro command it prints.
```

added: the toy's blobs, weights and experts, the grid, the mass-per-cell comparison, the warp's four strengths and det J ranges, w = 0.3 for negation, every reference number to four digits, the barely moving top-right share (all from `warp_check.py`, `warp_check.json`, `run.log` and the run card) · that no `rebuild/latent.py` exists (from `REBUILD_WALK.md`) · the tolerance on the rounding-level difference
dropped: nothing

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

Quiz 10.1: (URL lands here)

Built by step 1. ⏱ you 20 to 40 min · stop: every *needed now* question right and at least 80% of the rest; below that, read the items your misses point at and retake.

### 6. You: the idea walk

```text
/drip-idea resume artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md at claim 11: whether a
beta-VAE, beta-TCVAE or FactorVAE on the 2D toy shows what happens inside Stable Diffusion's latent,
read against Theorem 6.1 of arXiv 2502.04549, and what the smallest check is

Task: walk claim 11 from its place under Held claims until the smallest VAE check is chosen or the
claim is marked dead, in repo /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain.
Starts from: walk10.py and the step 4 comparison, which reproduce run 6 (claim 10's warp check); the
layer 10 --math drip folder under artifacts/drips/ (step 2).
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- Claim 11 is held. Unsupervised disentanglement is not identifiable without supervision or
  inductive bias (Locatello et al. 2019, confident), so a β-VAE on 2D data may not put A and B on
  separate axes (the map's Held claims row).
- A 2D latent from 2D data speaks to coordinates, whether A and B fall on separate axes (the
  orthogonal-mean-difference condition of Bradley et al. 2025a, Lemma 8.1), not to a
  high-dimensional latent (same row).
- The map's own route to settle it: claim 10's warp check first, then a 2D VAE whose latent is
  deliberately rotated 45° against A and B, composed in the latent, against one aligned with them
  (same row).
- Claim 10's warp check is done (run 6): construction (1) and negation agree to 4e-17 and 7e-17
  total variation across the warp; the plain product differs by 0.035, 0.080 and 0.126 at largest
  area stretches 2.6, 9 and 49 (Runs table, run 6).
- A VAE decoder is not invertible; its analogue of |det J| is √det(JᵀJ) of the pullback metric,
  marked "likely, verify" (claim 10).
- Route 7, the pressure-test on whether exponent-sum invariance is known, is emitted with no return
  file (Routes table, route 7; no routes/07-exponent-sum-invariance.return.md exists).
- Theorem 6.1 and Lemma 6.2 of arXiv 2502.04549 say projective composition in pixel space is
  projective composition in any feature space an invertible map reaches (route 7's prompt).
- The quadrant toy: A is x1 > 0, B is x2 > 0, one blob per quadrant at (±0.5, ±0.5), std 0.15
  (Checks outstanding, claim 2a).
Outputs: claim 11's row in the Position table and its Held claims row, updated with the mark; if a
check is chosen, a Checks outstanding row naming the check and what each outcome means.
Target: the smallest VAE check is written with its outcomes, or claim 11 is marked dead with the
reason; I judge.
Progress: each round closes one part of the claim with a mark: holds, needs a check, wrong as
stated with its repair, or dead.
Proof: grep -n "^| 11\." artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md prints the claim's
new mark.
Limits: write only in that idea map; launch no run; no commit.
Hand-off: claim 11's row, which step 7's cross-links read.
```

added: the held claim's two reasons and its settle path, run 6's numbers, the VAE caveat, route 7 having no return, the theorem and lemma, the toy's definition (all from `artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md` and its route 7 prompt)
dropped: nothing

⏱ you ~20 min (the median of 13 measured `drip-idea` runs; the middle half took 5 to 61 minutes), AI ~10 min · stop: the walk ends when the smallest VAE check is chosen or claim 11 is marked dead

**Outcome:** a decision on the smallest VAE experiment worth running.

### 7. You: close the layer

Run the `/drip-retro` command the build printed at its close (⏱ about 25 minutes). Then paste this:

```text
In the background, dispatch xref-cadence over the files layer 10 of the composition ladder wrote:
the layer 10 --math drip folder under artifacts/drips/, the retrospective the layer 10 walk closed
on, and claim 11 of artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md.

Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The reference sets to link against are the two read-ahead chunks on the Study first line of
  artifacts/drips/simple-distributions-rebuild/ladder/10-latent-space-and-vaes.md (the feature-space
  chunk and appendix I), the "Composing in a latent space" row of the Bricks table in
  artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, and claims 10 and 11 of
  artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md with the run card
  artifacts/ideas/when-poe-reaches-a-and-b/runs/compose-before-or-after-a-warp/README.md.
- ~/Playground/PhD/simple-distributions-walk/ is scratch space; link nothing into it (CLAUDE.md, the
  playground is disposable).
Outputs: links written both ways between those files and the reference sets.
Hand-off: the agent's report, listing every link it wrote, read before layer 11 starts.
```

added: the reference sets, by path (from this layer's Study first line, `REBUILD_WALK.md` and the idea map)
dropped: nothing

⏱ you 2 min to paste · AI ~10 min (estimate)

---

⬅️ [Layer 9: the correction](09-the-correction-learned-back.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 11: the testbed](11-the-text-conditioned-testbed.md) ➡️
