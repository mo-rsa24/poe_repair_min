# /immerse: Product-of-Experts Composition — Mathematical Foundations

## Target
Master the complete mathematical derivation of product-of-experts (PoE) composition from first principles, including:
- Conditional independence and factorization of joint probabilities
- Bayes' rule and its role in score-based composition
- The transition from probability theory to diffusion models (score functions)
- How noise prediction relates to score functions via the diffusion connection
- Why the independence assumption fails for real concepts and what signal is lost

## Purpose
This immersion directly supports **the introduction of the PoE repair paper** ([paper/iclr/iclr2027_conference.tex](../paper/iclr/iclr2027_conference.tex)), specifically the derivation now in lines 94–110 (the four-step align block and its surrounding prose). You are reading and working through this to own every step so that when a reviewer asks "why does that formula work?" you can explain it cold, and to build the intuition for why $r_t$ (the interaction residual) is the exact quantity that fixes composition.

## Anchors
- **Live derivation:** The align block you just integrated ([iclr2027_conference.tex:96–101](../paper/iclr/iclr2027_conference.tex#L96-L101))
- **Paper claim:** Lines 94–110 assert that conditional independence + Bayes' rule + score functions = PoE formula
- **The gap:** Line 110 names $r_t$ as the interaction residual; you need to know exactly what it measures and why
- **Your data:** N/A (this is pure math)

## What Will Be Solid When Done
1. You can state the conditional independence assumption in plain English and know why it's an assumption (not a law)
2. You can write out the Bayes' rule derivation $p(x|c_1, c_2) \propto \frac{p(x|c_1)p(x|c_2)}{p(x)}$ and explain each step
3. You understand the score function $\nabla_x \log p(x|c)$ as a gradient and why taking gradients moves us into the space where diffusion models operate
4. You can derive the noise-prediction formula $\epsilon_\theta(x_t, t | c) = -\sigma_t \nabla_x \log p(x|c)$ from first principles (or trace it back to the original paper)
5. You can state why real concepts violate conditional independence and what $r_t$ measures: the per-step difference in noise predictions when independence fails
6. You can draw and explain a diagram showing: Mono (joint prompt) vs. PoE (two experts) vs. PoE+correction, with $r_t$ labeled as the bridge

## Journey Structure (Rungs)
This journey is structured to move from concrete to abstract, then back down to your paper.

### Rung 1: Conditional Probability and Independence
**What to understand:**
- Definition of conditional probability: $P(A|B) = \frac{P(A,B)}{P(B)}$
- What "independent" means: $P(A, B) = P(A) P(B)$, or equivalently $P(A|B) = P(A)$
- **Conditional independence:** Two events are conditionally independent given a third if $P(A, B | C) = P(A|C) P(B|C)$

**Why this matters for PoE:**
When you say "concepts $c_1$ and $c_2$ are conditionally independent given the image $x$," you're asserting that once you know what image you're generating, knowing whether it has property $c_1$ tells you nothing additional about whether it has property $c_2$. This is a strong claim and it will turn out to be false for real concepts.

**Checkpoint:**
- Write out: $P(c_1, c_2 | x) = P(c_1|x) P(c_2|x)$ in your own words
- Give an example where this is true (e.g., coin flips)
- Give an example where this fails (e.g., "has a cat" and "has a dog" in the same image)

### Rung 2: Bayes' Rule and Factorization
**What to understand:**
- Bayes' rule: $P(A|B) = \frac{P(B|A) P(A)}{P(B)}$
- Rearranging to get: $P(A, B) = \frac{P(A|B) P(B|A) P(B)}{P(B)} = P(A|B) P(B|A) P(B) / P(B)$ (this is messy; focus on the clean form below)
- **The key move:** If $P(A, B | C) = P(A|C) P(B|C)$, then using Bayes on $P(x | c_1, c_2)$:

$$P(x | c_1, c_2) \propto \frac{P(x | c_1) P(x | c_2)}{P(x)}$$

This says: the joint is proportional to the product of the conditionals divided by the unconditional.

**Why this is the PoE formula in disguise:**
The "$\propto$" (proportional to) sign hides a normalization constant, but the *shape* of the distribution is determined by this product. This is exactly what PoE tries to implement.

**Checkpoint:**
- Derive $P(x|c_1, c_2) \propto \frac{P(x|c_1) P(x|c_2)}{P(x)}$ starting from conditional independence
- Explain why we divide by $P(x)$ (Hint: it's the unconditional baseline)
- Check: does the formula make intuitive sense? If both $P(x|c_1)$ and $P(x|c_2)$ are high, the product is high; if $P(x)$ is high, we "discount" (divide down) the product. Why?

### Rung 3: Moving to Log Space
**What to understand:**
- Taking the logarithm of a probability product turns it into a sum: $\log(AB) = \log A + \log B$
- Taking the logarithm of a ratio turns it into a difference: $\log(A/B) = \log A - \log B$
- So: $\log P(x|c_1, c_2) = \log P(x|c_1) + \log P(x|c_2) - \log P(x)$ (up to a constant)

**Why log space:**
Log probabilities are easier to work with computationally (numerically stable) and algebraically (products become sums). More importantly, the *gradient* of log probability (the score function) is what diffusion models operate in.

**Checkpoint:**
- Write the log version of the Bayes factorization
- Verify: take the exponential of both sides; do you recover the original formula? (You should, up to a constant)

### Rung 4: Score Functions and Gradients
**What to understand:**
- The **score function** is the gradient of the log probability: $\mathbf{s}(x) = \nabla_x \log p(x)$
- Gradient means: the direction and magnitude of steepest increase
- For a conditional, the score is: $\nabla_x \log p(x|c)$
- **Key property:** If $\log p(x) = \log p(x|c_1) + \log p(x|c_2) - \log p(x)$, then taking $\nabla_x$ of both sides preserves the sum:

$$\nabla_x \log p(x|c_1, c_2) = \nabla_x \log p(x|c_1) + \nabla_x \log p(x|c_2) - \nabla_x \log p(x)$$

**Why scores:**
Diffusion models don't sample directly from a distribution; they iteratively denoise by following the score (gradient of log-probability). The score tells you which direction in image space to move to increase the probability of landing in regions that satisfy your condition.

**Checkpoint:**
- Draw a contour plot of a 2D Gaussian and sketch the score vectors (gradients) at various points. They should point toward the center.
- State the score-function version of the Bayes factorization from memory
- Explain: why is the sum of gradients the same as the gradient of the sum? (Linearity of differentiation)

### Rung 5: Diffusion Models and Noise Prediction
**What to understand:**
- Diffusion models reverse a noising process. At step $t$, the image $x_t$ is a noisy version of the true image $x_0$
- A diffusion model learns to predict the noise: $\epsilon_\theta(x_t, t | c)$ is a neural network that predicts what Gaussian noise was added
- The relationship between noise and score: $\epsilon_\theta(x_t, t | c) = -\sigma_t \nabla_{x_t} \log p(x_t | c)$
  - This says: predicting the noise is equivalent to predicting (and negating, scaled by noise level) the score
  - The factor $\sigma_t$ is the noise standard deviation at step $t$; it scales the score into the noise prediction space

**Why this connection:**
The derivation comes from the forward diffusion process. If you add Gaussian noise to an image, the score of the resulting noisy image tells you the direction to move to reduce the noise (i.e., the negative of the direction the noise pushed you). So predicting the noise is the same as predicting the anti-score.

**Checkpoint:**
- State the noise-prediction formula and the role of $\sigma_t$
- Explain: if you know the score $\nabla_x \log p(x|c)$, how do you compute the noise prediction? (Multiply by $-\sigma_t$)
- Read the original diffusion model paper (Ho et al., 2020, "Denoising Diffusion Probabilistic Models") section on the connection between score and noise; note the exact equation

### Rung 6: Assembling PoE
**What to understand:**
- Substitute the noise-prediction relationship into the score-function Bayes formula:

$$\epsilon_\theta(x_t, t | c_1, c_2) = \epsilon_\theta(x_t, t | c_1) + \epsilon_\theta(x_t, t | c_2) - \epsilon_\theta(x_t, t)$$

This is the PoE rule. It says: to get the noise prediction for the joint concept, add the two conditional noise predictions and subtract the unconditional prediction.

- **In practice:** You don't have a model trained on the joint $p(x | c_1, c_2)$. You have three models (or one model with three forward passes):
  - $\epsilon_\theta(\cdot | c_1)$: model conditioned on $c_1$
  - $\epsilon_\theta(\cdot | c_2)$: model conditioned on $c_2$
  - $\epsilon_\theta(\cdot)$: model with no condition (unconditional)
  
  You combine them at every denoising step using the PoE formula.

**Why it works (and why it fails):**
- It works if the conditional independence assumption holds: then you're exactly sampling from $p(x | c_1, c_2)$
- It fails when $c_1$ and $c_2$ interact: cats and dogs look similar, occupy space, and compete for visual "real estate." The true $p(x | \text{cat and dog})$ is not just the product of the marginals; it's shaped by how the concepts negotiate shared space.

**Checkpoint:**
- Write out the PoE formula from memory
- Explain each term: why do you add the two conditionals? Why subtract the unconditional?
- State the assumption that makes PoE exact, and why it fails for real concepts

### Rung 7: The Interaction Residual $r_t$
**What to understand:**
- Define: $r_t := \epsilon_\text{true}(x_t, t | c_1, c_2) - \hat{\epsilon}_\mathrm{PoE}(x_t, t)$
- $\epsilon_\text{true}$ is what a model actually trained on the joint prompt $(c_1, c_2)$ would predict
- $\hat{\epsilon}_\mathrm{PoE}$ is what the PoE formula gives you
- **$r_t$ is the missing signal:** everything that PoE composition drops when it assumes independence

**Why it matters:**
- If you add back $r_t$ to the PoE prediction, you recover what the true joint model would have predicted
- This is the key insight of your paper: you learn to predict $r_t$ without ever seeing the true joint prompt, then add it back during denoising
- Understanding $r_t$ transforms the problem from "why does PoE fail?" (hard) to "what is the signal we can learn to recover?" (tractable)

**Checkpoint:**
- State the definition of $r_t$ in plain English
- Explain: what does $r_t$ measure? (The per-step difference due to violated independence)
- Sketch: how would you visualize $r_t$ on a curve showing noise predictions over denoising steps?

## Learning Arc (Self-Paced Drips)

**Session 1: Probability Foundations (45 min)**
- Read: [3Blue1Brown, "Bayes' Theorem"](https://www.youtube.com/watch?v=HZGCoVwiada) (13 min)
- Work: Rung 1 checkpoint (10 min)
- Work: Rung 2 checkpoint, deriving the factorization by hand (20 min)
- Pause and recite: state Bayes' rule and the conditional-independence form without notes

**Session 2: Score Functions and Calculus (60 min)**
- Read: [Jeremy Kun, "A Primer on Calculus"](https://jeremykun.com/2013/02/08/methods-of-differentiation/) sections on gradients (15 min, skim for intuition)
- Work: Rung 3 checkpoint, verify log-space identities (10 min)
- Work: Rung 4 checkpoint, draw gradients and state the score formula (20 min)
- Pause and recite: state the log-Bayes formula and the score version without notes

**Session 3: Diffusion Models (90 min)**
- Read: [Ho et al. 2020, "Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239) sections 2–3 (30 min, focus on forward/reverse process and the noise prediction connection)
- Read: [Yang Song, "Generative Modeling by Estimating Gradients of the Data Distribution"](https://yang-song.github.io/blog/2021/score/) (15 min, the score-matching perspective)
- Work: Rung 5 checkpoint, state the noise-prediction formula and derive it (20 min)
- Pause and recite: explain why predicting noise is equivalent to predicting the (negative) score

**Session 4: PoE Derivation and Integration (60 min)**
- Review: Rungs 1–5 in sequence, filling in any gaps from Sessions 1–3 (15 min)
- Work: Rung 6 checkpoint, assemble the full PoE formula and explain each term (20 min)
- Hands-on: Take the derivation from your paper (lines 96–101 of iclr2027_conference.tex) and explain it line-by-line to yourself (15 min)
- Pause and recite: derive PoE from the conditional-independence assumption, cold

**Session 5: The Residual and the Fix (60 min)**
- Work: Rung 7 checkpoint, define $r_t$ and state what it captures (10 min)
- Hands-on: Read your paper's contributions (lines 115–128) and note how each one hinges on understanding $r_t$ (10 min)
- Create artifact: Draw a three-panel diagram (or ASCII sketch first):
  - Panel 1: True joint model prediction over denoising steps
  - Panel 2: PoE prediction (independence assumed)
  - Panel 3: $r_t$ as the difference, labeled with why it arises (cat/dog interaction, spatial negotiation, etc.)
- Pause and recite: explain $r_t$ to someone who has not read your paper; what does it measure, why is it small at the start and large mid-run?

## Assessment Checkpoints

After completing all rungs, you should be able to:

1. **State the independence assumption** in plain English and explain why it's an assumption (not provable from data alone)
2. **Derive Bayes' rule version** starting from conditional-independence definition:
   - Blackboard/paper: write $P(x | c_1, c_2) = P(x|c_1) P(x|c_2)$ → $P(x|c_1, c_2) \propto \frac{P(x|c_1) P(x|c_2)}{P(x)}$ (3 lines max)
3. **Explain the score-function derivation:**
   - State why we move to log space (algebraic convenience, numerical stability, gradients)
   - Write $\log P(x|c_1, c_2) = \log P(x|c_1) + \log P(x|c_2) - \log P(x)$
   - Take $\nabla_x$ of both sides and name the result
4. **Connect to diffusion models:**
   - State the noise-prediction formula: $\epsilon_\theta = -\sigma_t \nabla_x \log p(x|c)$
   - Explain $\sigma_t$ (noise level at step $t$)
   - Derive PoE by substitution
5. **Explain why PoE fails:**
   - Name two reasons real concepts violate independence (visual similarity, spatial layout)
   - Define $r_t$ as the per-step residual
   - Explain why $r_t$ is learnable and why adding it back recovers composition
6. **Answer a skeptical question:**
   - "Why should I believe the PoE formula is correct if you're using it to fix a problem?"
   - Your answer: "PoE is exact when concepts are independent. Real concepts aren't. The formula is sound; we're learning what it misses."

## Deliverables When Done

1. **Handwritten or typed derivation** (one page, no notes): Start from conditional independence, land on the PoE formula, state assumptions and where they break
2. **Diagram** (hand-drawn or digital): Show Mono vs. PoE vs. PoE+correction, with $r_t$ labeled as the bridge
3. **Explanation (2 min, video or voice memo):** Record yourself explaining the full chain cold: why PoE is sound, why it fails, what $r_t$ measures, why it's learnable
4. **Recitation checkpoint:** State each of the 6 assessment points above without notes, in front of someone or recorded
5. **Integration:** Reread lines 94–110 of your paper and annotate in the margin which rung each step comes from

## Blockages and What to Do

- **If Bayes' rule confuses you:** Rung 2 is the bottleneck. Do not move forward. Work through concrete examples (coin flips, drawing from urns) until $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$ feels intuitive. The 3Blue1Brown video is the fastest fix.
- **If gradients feel slippery:** Rung 4 is the bottleneck. Draw contour plots. Sketch vectors. Watch [Grant Sanderson's calc series](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr28EL8nyhWWh6hy_c9sJ) ("Essence of Calculus," episodes 1–3) for 30 min.
- **If the noise-prediction connection is unclear:** Rung 5 is the bottleneck. Work through the forward-diffusion math from Ho et al. (2020) section 2 in detail. This is the crux; do not skip.
- **If the PoE formula still feels magical:** Rung 6 is incomplete. Go back to Rungs 1–5 and check your understanding of each. The formula *should* feel like a straightforward substitution once the setup is solid.

## References (Fully Qualified)

- **Conditional Probability & Bayes:** 
  - 3Blue1Brown (2019). "Bayes' theorem." https://www.youtube.com/watch?v=HZGCoVwiada
  - Pearl (2009). *Causality*. Chapter 1 (Primer on Probability).

- **Score Functions:**
  - Song, Y. (2021). "Generative Modeling by Estimating Gradients of the Data Distribution." https://yang-song.github.io/blog/2021/score/
  - Hyvarinen, A. (2005). "Estimation of non-normalized statistical models." JMLR.

- **Diffusion Models & Noise Prediction:**
  - Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS 2020*. https://arxiv.org/abs/2006.11239 (Sections 2–3)
  - Dhariwal, P., & Nichol, A. (2021). "Diffusion Models Beat GANs on Image Synthesis." *NeurIPS 2021*. https://arxiv.org/abs/2105.05233 (Section 2, classifier-free guidance, which uses score composition)

- **PoE Composition (in diffusion context):**
  - Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." *ICML 2021*. (CLIP, the conditioning paradigm)
  - Nichol, A., et al. (2021). "GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models." *ICML 2022*. https://arxiv.org/abs/2112.10741 (Section 3, classifier-free guidance as weighted PoE)
  - Prompt: search arXiv for "product of experts diffusion" or "concept composition"; your paper is part of an emerging literature

- **Your Paper:**
  - [paper/iclr/iclr2027_conference.tex](../paper/iclr/iclr2027_conference.tex), lines 94–110 (the introduction PoE derivation)
  - [paper/iclr/MASTER_PLAN.md](../paper/iclr/MASTER_PLAN.md) (context on what the paper is claiming)

## Why This Matters

Reviewers will read lines 94–110 and think one of two things:
1. "This is standard; I'll trust it and move on." (Means: the writing is clear and the formula looks familiar.)
2. "Why should I believe this? Show me it's sound." (Means: you need to own it, be ready to explain, and defend the assumption.)

By the end of this immersion, you'll own it cold. You'll be able to explain it to a diffusion expert, a probabilist, and an engineer, each in their own language. And when someone asks "why does the independence assumption fail for concepts?", you'll have a ready answer: "Visual similarity, spatial layout, and shared features create dependencies that PoE's linear combination cannot capture."

That confidence is what separates a paper you wrote from a paper you understand.
