# ⭐ Layer 8: the interaction term, defined and shown

⬅️ [Layer 7: A without B](07-a-without-b.md) · 🏠 [the guide](../PROMPTS.md) · [Correction prior art](08a-correction-prior-art.md) ➡️

**Why it is critical.** This is the first half of the research goal. It is defined exactly here before any learned version of it is trusted. On [the axes](../PROMPTS.md#the-axes-one-new-variable-per-rung), this is rung e: the interaction term, defined here and learned back as a correction in layer 9.

**Depth.** Build: r(x) in log-density form, its ε-space form, and the label term. You 3.3 to 6.7 h in all (rough).

**Study first.** Bayes and [conditional independence](../../../../../../Obsidian/Mathematics/Statistics/Probability/Conditional%20Independence.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/computer-vision/computer-vision-textbook.pdf#page=219 "Computer vision textbook, section 10.1, Works it on a joint table of three discrete variables taking 4, 3 and 2 values, 24 probabilities in all, where x₁ and..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=201 "Mathematics for machine learning, section 6.4.5, Definition 6.11, p(x, y ∣ z) = p(x ∣ z) p(y ∣ z) for every z, read as 'given knowledge about z, the distribu..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=389 "Pattern recognition and machine learning, section 8.2, The definition p(a ∣ b, c) = p(a ∣ c), then three-node graphs where observing c makes a and b independ..."), [▶1](https://www.youtube.com/watch?v=zRdE8A4UZes&t=4587s "CS229 lecture 5, 1:16:27 to 1:20:34, Makes the naive Bayes assumption that words are conditionally independent given the class, cutting the parameters from 2..."), [◆](../../../../../../goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/04-conditional-independence.md "Check independence inside each slice of a third variable, deep-learning/generative-models/03-sampling-from-a-graph, plan 04. Conditional independence checked...")</sub>: [the assumption the product rests on](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/02-the-assumption-poe-rests-on.md), worked in [joint tables and independence](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/01-joint-tables-and-independence.md), and [the route through p(c1, c2 | x), section 8.2](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/08-assembling-the-poe-rule.md). The interaction term: [Kinds Of Composition](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/Kinds%20Of%20Composition.md), [where independence fails](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/15-where-independence-fails.md) and its ε-space form in [a corrector on the cached trajectories, equation 27.1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/27-a-corrector-on-the-cached-trajectories.md). Textbook: Deisenroth et al. 6.3 ([Bayes](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=191), p189). [Log-odds](../../../../../../Obsidian/Artificial%20Intelligence/Deep%20Learning/Logit.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/ai-engineering/ai-engineering.pdf#page=190 "AI engineering, chapter 2, Sampling fundamentals, A language model outputs one logit per vocabulary token (figure 2-15); logits can be negative and do not su..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/probabilistic-machine-learning-an-introduction.pdf#page=76 "Probabilistic machine learning: an introduction, section 2.4.2, Shows the sigmoid's input a equals the log odds log(p / (1 − p)), and names the inverse of th..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/probabilistic-machine-learning-an-introduction.pdf#page=79 "Probabilistic machine learning: an introduction, section 2.5.2, Calls the inputs to the softmax logits, 'a generalization of the log odds', and writes a = Wx..."), [▶1](https://www.youtube.com/watch?v=pdqofxJeBN8&t=3450s "CS231n lecture 2, 57:30 to 1:00:36, Turns a cat image's raw scores 3.2, 5.1 and −1.7 into probabilities with a softmax, giving the cat class 0.13, and names ..."), [▶2](https://www.youtube.com/watch?v=pwQ0l4hFCVI&t=1746s "CS229 lecture 14, 29:06 to 36:12, Divides the logits (2, 1, −1) by a tiny temperature to get (200, 100, −100) and a near one-hot softmax, then top-k sampling..."), [◆](../../../../../../goal-setting/learning/deep-learning/transformers/textbook/10-find-the-gradient-of-the-cross-entropy.md "Find the gradient of the cross-entropy loss through a softmax, deep-learning/transformers, chapter 10. Defines a logit as a score before the softmax, takes t...")</sub> has no passage that teaches it yet beyond Murphy 10.2 ([binary logistic regression](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/probabilistic-machine-learning-an-introduction.pdf#page=377), p377); quiz 8.1's preparation writes the missing passage. The interaction term on a four-image toy, where a hidden cause breaks the fork and the product's error is measured: [break the fork with a hidden cause](../../../../../../goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/18-poe-fork-built.md).

**Ideas to tie in.** From [the ideas that tie the ladder together](../../../ideas/ideas-that-tie-the-ladder-together.md): [conditional independence made visible as an interaction residual](../../../ideas/ideas-that-tie-the-ladder-together.md#1-make-conditional-independence-visible-as-an-interaction-residual) (needed, 2 to 3 h, no training); [the interaction score as the object a correction can learn](../../../ideas/ideas-that-tie-the-ladder-together.md#5-learn-the-identifiable-object-which-is-the-interaction-score) (needed, 2 to 3 h, no training); [the interaction's own diffusion-time trajectory](../../../ideas/ideas-that-tie-the-ladder-together.md#6-give-the-interaction-its-own-diffusion-time-trajectory) (needed, 2 to 4 h, no training). Each comes with an exact toy check.

**Achieve.** `walk08.py` plotting r with the wet rule off and on at every noise level, and `walk08b.py` printing the label term · on the whiteboard: the two-by-two table of labels with r in each cell · problems: the ones quiz 8.2 banks · quizzes 8.1 and 8.2 passed.

## The flow

```text
 AI, in the background                         │  You, in order
───────────────────────────────────────────────┼───────────────────────────────────────────────
 quizzes 8.1 and 8.2, prepared ──────────────▶ │  1. quiz 8.1: Bayes for two labels
   (paste during layer 7's build)              │       ▼
                                               │  2. /drip --math: r(x) and its ε form
                                               │       ▼
                                               │  3. quiz 8.2: needed-now right, 80% of the rest
                                               │       ▼
                                               │  4. build: walk08.py, walk08b
 builds rebuild/phase8_interaction.py ───────▶ │  5. compare: p8. numbers equal yours
                                               │       ▼
 math-scene, beside prior art ◀─────────────── │  6. scene
                                               │       ▼
 xref-cadence links the layer ◀─────────────── │  7. /drip-retro, then the next layer
```

## The steps

### 0. AI, in the background: the layer 8 reference and quizzes 8.1 and 8.2, one layer ahead

**AI, one layer ahead.** Paste both when you start layer 7's build.

The first prompt has an AI session write the reference your step 5 compares against: `rebuild/interaction.py` and `rebuild/phase8_interaction.py`. They compute the interaction term exactly in the parts world with a wet rule (the cat half shifts when a dog is present), then r_t and its gradient at each noise level, and add the numbers to `walk_numbers` as `p8.` keys. The second prompt prepares and builds quizzes 8.1 and 8.2, so they are waiting at steps 1 and 3.

**Prompt:** [the layer 8 reference](prompts/08-0-the-interaction-reference.md)

**Prompt:** [build quizzes 8.1 and 8.2](prompts/08-0-quizzes-8-1-and-8-2.md)

⏱ AI ~2 h for both (estimate), beside layer 7's build

### 1. You: quiz 8.1

Quiz 8.1 checks the college maths under this layer: Bayes' rule for two labels, conditional independence, and log-odds.

Quiz 8.1: (URL lands here)

⏱ you 20 to 40 min · stop: the bar, every *needed now* question right and at least 80% of the rest; below that, read the items your misses point at and retake.

### 2. You: the maths drip

You derive, one rung per message, Bayes' rule for two labels and conditional independence, ending at the interaction term r(x) = log p(x|A,B) − [log p(x|A) + log p(x|B) − log p(x)] and its noise-prediction form ε_joint − ε_A − ε_B + ε_∅. Your quiz 8.1 misses become the first rungs.

**Prompt:** [the maths drip](prompts/08-2-the-maths-drip.md)

⏱ rough: you 30 to 60 min · AI ~10 min · stop: you derive r and its ε-space form without notes

**Outcome:** you can derive r, say why it is constant when the labels are fixed functions of the image, and write its noise-prediction form, the residual `poe_repair_min` already caches for 25 pairs.

### 3. You: quiz 8.2

Quiz 8.2 checks the layer itself: r(x), why it is constant when the labels are fixed functions of the image, its ε-space form, and the label term for report-style labels, with banked problems.

Quiz 8.2: (URL lands here)

⏱ you 30 to 60 min, the banked problems included · stop: the bar, then the build. Below the bar, the misses become the first rungs of a second `/drip --math`, and you retake before building.

### 4a. You: the build, walk08.py

You type the interaction term into `walk08.py` from an empty file, one piece per round: the world's weights and four densities, r at t = 0 with the wet rule off and on, r_t per noise level, ∇r_t with its identity check, and the plot. With clean labels r stays constant at t = 0 whether the rule is on or off. The rule's effect shows at t > 0, off the joint's support, or for experts whose training sets lack the wet cat-and-dog images.

**Prompt:** [the build](prompts/08-4a-the-interaction-build.md)

⏱ rough: you 1 to 2 h (about 5 pieces) · AI ~35 min · stop: the prompt's bar

**Outcome:** r(x) in the parts world, plotted with the wet rule off and on, and r_t with its gradient at every noise level.

### 4b. You: the label term, walk08b.py

You type the label term into `walk08b.py`. On the quadrant toy, A is read with probability 0.9 when true, and inside A∩B, B is read only when A was read. A four-way classifier over the label pairs, set against two separate classifiers, gives the term: +0.105 nats for (A,B) inside A∩B, zero elsewhere, and strongly negative for (not A, B).

**Prompt:** [the label term](prompts/08-4b-the-label-term-build.md)

⏱ rough: you 1 to 2 h (one build concept) · AI ~35 min · stop: the prompt's bar

**Outcome:** the interaction term in both places it lives. In the pixels it is the wet dog. In the labels it is zero for clean labels and real for report-mined ones like NIH's, and it hits the negation cells hardest.

**What it looks like at scale (look by hand).** Open `artifacts/results/joint-against-poe-by-kind-of-composition/README.md` and its `interaction-grids/`. On Stable Diffusion v1-4, the product fuses one cat-dog in 4 of 4 seeds, and it paints rain streaks over a dry dog in 4 of 4. Those are the two things the term you just computed leaves out, on a real model.

<a href="../../../results/joint-against-poe-by-kind-of-composition/pairs/interaction.png"><img src="../../../results/joint-against-poe-by-kind-of-composition/pairs/interaction.png" width="420" alt="Stable Diffusion v1-4, seed 0: on the left the joint prompt 'a wet dog in the rain' gives a dog with soaked fur standing in a puddle; on the right the product of 'a dog' and 'rain' gives a dry, clean dog with rain streaks drawn over the picture"></a>

Look at the fur: the joint's dog is soaked, and the product's dog is dry with the rain drawn over the picture. The product does this in [all 4 of its seeds](../../../results/joint-against-poe-by-kind-of-composition/tiles/interaction-poe-seeds0to3.png).

### 5. You: compare your numbers with the reference

**Compare.** `python -m rebuild.walk_numbers` prints the `p8.` keys; your `walk08.py` prints the same numbers.

```bash
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m rebuild.walk_numbers | grep 'p8\.'
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m rebuild.phase8_interaction --out ~/Playground/PhD/simple-distributions-walk/phase8-reference | grep 'identity'
cd ~/Playground/PhD/simple-distributions-walk
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python walk08.py
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python walk08b.py
```

⏱ you ~10 min (`--out` keeps the reference run from rewriting `rebuild/out/phase8/`) · stop: every `p8.` line and the number your scripts print agree to the digits shown; both identity lines read under `1e-10`; the label term for (A,B) inside A∩B reads `0.1054` from the reference and within 0.05 of it from `walk08b.py`. A number that differs is either a lesson or a bug in one of the two files; find which before the prior-art read.

### 6. AI, in the background: the scene

An AI session builds a scene in your notation of the parts world's interaction term, with the wet rule as a toggle and the noise level as a slider. Every number on it comes from the `p8.` keys.

**Prompt:** [the scene](prompts/08-6-the-interaction-scene.md)

Scene: (URL lands here)

⏱ you ~20 min · AI ~45 min, beside the correction prior-art read · stop: the scene is published and its numbers match your walk's

**Outcome:** a scene of the parts world's interaction term, with the wet rule as a toggle and the noise level as a slider.

### 7. You: close the layer

Run the `/drip-retro` commands the two builds printed at their close (⏱ about 25 minutes each). Then paste [the linking prompt](prompts/08-7-link-the-layer.md), which links what the maths drip and the two builds wrote to the journey plans, textbook chapters, vault notes, brick row and idea-map claims the layer rests on.

⏱ you 2 min to paste · AI ~10 min (estimate)

---

⬅️ [Layer 7: A without B](07-a-without-b.md) · 🏠 [the guide](../PROMPTS.md) · [Correction prior art](08a-correction-prior-art.md) ➡️
