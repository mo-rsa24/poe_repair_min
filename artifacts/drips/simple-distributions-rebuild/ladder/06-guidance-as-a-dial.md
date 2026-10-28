# ⭐ Layer 6: guidance as a dial

⬅️ [Reading stop: Bradley](05a-reading-stop-projective-composition.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 7: A without B](07-a-without-b.md) ➡️

**Why it is critical.** Stable Diffusion's cat-dog hybrid happens at guidance weight 7.5, and everything above ran at 1. The correction in layer 9 has to work at the weight the real model uses. On [the axes](../PROMPTS.md#the-axes-one-new-variable-per-rung), this is rung d: guidance, added with the data held at a fixed share.

**Depth.** Build: a [density](../../../../../../Obsidian/Mathematics/Statistics/Distributions/Probability%20Density%20Function.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=188 "Introduction to probability for data science, section 4.1, Builds the intuition of probability per unit length, defines the PDF and its properties (Definitio..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=186 "Mathematics for machine learning, section 6.2.2, The density and the CDF (Definitions 6.1 and 6.2), a uniform example (Example 6.3), and discrete against con..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=36 "Pattern recognition and machine learning, section 1.2.1, The density's two conditions (equations 1.25 and 1.26), its link to the CDF, and why a density picks..."), [▶1](https://www.youtube.com/watch?v=QKA4HNEw3aY&t=52s "The Organic Chemistry Tutor, Probability Density Functions, 0:52 to 3:39, Verifies that a given function is a density by checking that its integral equals 1...."), [▶2](https://www.youtube.com/watch?v=oI3hZJqXJuc&t=135s "StatQuest, The Main Ideas behind Probability Distributions, 2:15 to 5:14, A curve as a distribution, with probability read as area under it, not its height. ..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/02-densities-and-the-gaussian.md#21-a-density-can-exceed-1 "Chapter 2: Read a density as area and draw a Gaussian from its covariance, deep-learning/diffusion-models/01-poe-derivation-foundations, chapter 02, section ..."), [◈](https://claude.ai/artifact/DBmJtn6xy75sPtQwPEvQZU#nb-sec1 "Kinds of composition notebook, Setup: the ideal expert, The ideal expert's log-density on a grid, in four steps from the 2D Gaussian to its logsumexp form, t...")</sub> [raised to a power](../../../../../../goal-setting/learning/mathematics/textbooks/algebra/algebra-handbook.pdf#page=109), and the guided target on both toys. You 2.3 to 4.7 h in all (rough).

**Study first.** A density raised to a power: [what is annealed](../../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/24-what-is-annealed.md) and [The Tempered Density And Its Score](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/The%20Tempered%20Density%20And%20Its%20Score.md). The guided target: [guidance as a tilted density, section 8.4](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/08-assembling-the-poe-rule.md), [classifier-free guidance trained on MNIST](../../../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/plans/08-cfg-trained-on-mnist.md) and [Powers Of Gaussians](../../../../../../Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Powers%20Of%20Gaussians.md). Textbook: Goodfellow et al., *Deep Learning* 17.5.1 (tempering, from p615). Video: Umar Jamil on [classifier-free guidance](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Classifier-Free%20Guidance.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/probabilistic-machine-learning-advanced-topics.pdf#page=909 "Probabilistic machine learning: advanced topics, section 25.6.3, Classifier-free guidance, Derives the implicit classifier from Bayes' rule, log p(c∣x) = log..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/understanding-deep-learning.pdf#page=379 "Understanding deep learning, section 18.6.3, Conditional generation, Feeds the class into the U-Net the way the time step is added and trains on conditional ..."), [3](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/sampling-composed-models/compositional-visual-generation-with-composable-diffusion-models.pdf#page=7 "Liu and others 2022, Compositional visual generation with composable diffusion models, section 4.2, Composes concepts as p(x) ∏ p(c_i∣x) and notes that with ..."), [▶1](https://www.youtube.com/watch?v=Edr4uZFh4EE&t=2730s "CS231n lecture 14, 45:30 to 50:05, Drops the label to a null value half the time in training and combines v_cfg = (1 + w) v_y − w v_null at sampling, where w..."), [▶2](https://www.youtube.com/watch?v=WUUq6TVAu8U&t=5665s "CME296 lecture 4, 1:34:25 to 1:40:38, Derives the implicit classifier from the conditional and unconditional predictions with a weight w of about 3, the cond..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/plans/08-cfg-trained-on-mnist.md "CFG trained on MNIST, a w sweep, deep-learning/diffusion-models/00-from-scratch-to-sdxl, plan 08. Trains a guided model on MNIST and sweeps the guidance weig..."), [◈](https://claude.ai/artifact/1GayZthQkRL4FmukdCV9WW#nb-poe "The SDXL notebook in ten parts, PoE composition and CFG inside it, Two prompts composed as a product of experts, a run that shows the composition failing, an...")</sub>, [22:20 to 31:00](https://www.youtube.com/watch?v=ZBKpAp_6TGI&t=1340s).

**Ideas to tie in.** From [the ideas that tie the ladder together](../../../ideas/ideas-that-tie-the-ladder-together.md): [guidance as tempering, and negation tested for normalisability](../../../ideas/ideas-that-tie-the-ladder-together.md#4-treat-guidance-as-tempering-and-test-negation-for-normalisability) (needed, 2 to 4 h, no training). Each comes with an exact toy check.

**Achieve.** `walk06.py` printing A∩B share, [total variation](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/probabilistic-machine-learning-advanced-topics.pdf#page=94) and Z for each w from 1 to 7.5 · on the whiteboard: a two-mode 1D density raised to w = 1, 2 and 7.5 · problems: the ones quiz 6.2 banks · quizzes 6.1 and 6.2 passed.

## The flow

```text
 AI, in the background                         │  You, in order
───────────────────────────────────────────────┼───────────────────────────────────────────────
 2. quizzes 6.1 and 6.2, prepared ───────────▶ │  3. quiz 6.1: powers of densities, Bayes
   (paste during the reading stop)             │       ▼
                                               │  4. /drip --math: the guided target
                                               │       ▼
                                               │  5. quiz 6.2: needed-now right, 80% of the rest
                                               │       ▼
                                               │  6. build: walk06.py (--request)
 1. builds rebuild/phase6_guidance.py ───────▶ │  7. compare: p6. numbers equal yours
                                               │       ▼
 math-scene, beside layer 7 ◀───────────────── │  8. scene
                                               │       ▼
 xref-cadence links the layer ◀─────────────── │  9. /drip-retro, then the next layer
```

## The steps

**AI, one layer ahead.** Paste steps 1 and 2 when you start the reading stop. The first builds the reference your compare step needs; the second builds the quizzes.

### 1. AI, in the background: the layer 6 reference

An AI session writes the reference your step 7 compares against: `rebuild/guidance.py` and `rebuild/phase6_guidance.py`. They compute the guided target p(x)·p(A|x)^w·p(B|x)^w exactly on the quadrant toy and the parts world for w from 1 to 7.5, print the A∩B share, total variation and Z per w, and add them to `walk_numbers` as `p6.` keys.

**Prompt:** [the layer 6 reference](prompts/06-1-the-guidance-reference.md)

⏱ you 2 min to paste · the AI's time is in step 2's line

### 2. AI, in the background: quizzes 6.1 and 6.2

An AI session prepares and builds both quizzes, so they are waiting at steps 3 and 5.

**Prompt:** [build quizzes 6.1 and 6.2](prompts/06-2-quizzes-6-1-and-6-2.md)

⏱ AI ~2 h for both (estimate), beside the reading stop

### 3. You: quiz 6.1

Quiz 6.1 checks the college maths under this layer: raising a density to a power and renormalising, and Bayes' rule p(A|x) ∝ p(x|A)p(A).

Quiz 6.1: (URL lands here)

Built by step 2. ⏱ you 20 to 40 min · stop: every *needed now* question right and at least 80% of the rest; below that, read the items your misses point at and retake.

### 4. You: the maths drip

You derive, one rung per message, what raising a density to a power does to its modes, ending at classifier-free guidance written as p(x)·p(A|x)^w·p(B|x)^w. Your quiz 6.1 misses become the first rungs, and on the four-quadrant toy you show by hand that Z = r at w = 1.

**Prompt:** [the maths drip](prompts/06-4-the-maths-drip.md)

⏱ rough: you 30 to 60 min · AI ~10 min · stop: you write the guided target without notes

**Outcome:** you can say what w > 1 does to a density, and write the guided target exactly. Factored [classifier-free guidance](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Classifier-Free%20Guidance.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/probabilistic-machine-learning-advanced-topics.pdf#page=909 "Probabilistic machine learning: advanced topics, section 25.6.3, Classifier-free guidance, Derives the implicit classifier from Bayes' rule, log p(c∣x) = log..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/understanding-deep-learning.pdf#page=379 "Understanding deep learning, section 18.6.3, Conditional generation, Feeds the class into the U-Net the way the time step is added and trains on conditional ..."), [3](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/sampling-composed-models/compositional-visual-generation-with-composable-diffusion-models.pdf#page=7 "Liu and others 2022, Compositional visual generation with composable diffusion models, section 4.2, Composes concepts as p(x) ∏ p(c_i∣x) and notes that with ..."), [▶1](https://www.youtube.com/watch?v=Edr4uZFh4EE&t=2730s "CS231n lecture 14, 45:30 to 50:05, Drops the label to a null value half the time in training and combines v_cfg = (1 + w) v_y − w v_null at sampling, where w..."), [▶2](https://www.youtube.com/watch?v=WUUq6TVAu8U&t=5665s "CME296 lecture 4, 1:34:25 to 1:40:38, Derives the implicit classifier from the conditional and unconditional predictions with a weight w of about 3, the cond..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/plans/08-cfg-trained-on-mnist.md "CFG trained on MNIST, a w sweep, deep-learning/diffusion-models/00-from-scratch-to-sdxl, plan 08. Trains a guided model on MNIST and sweeps the guidance weig..."), [◈](https://claude.ai/artifact/1GayZthQkRL4FmukdCV9WW#nb-poe "The SDXL notebook in ten parts, PoE composition and CFG inside it, Two prompts composed as a product of experts, a run that shows the composition failing, an...")</sub> (Xia and Glocker 2025) is construction (1) with one weight per condition, already used on MIMIC-CXR.

### 5. You: quiz 6.2

Quiz 6.2 checks the layer itself: guidance as p(x)·p(A|x)^w·p(B|x)^w, what w > 1 does to the modes, and w = 1 as construction (1), with banked problems.

Quiz 6.2: (URL lands here)

⏱ you 30 to 60 min, the banked problems included · stop: every *needed now* question right and at least 80% of the rest, then the build. Below that, the misses become the first rungs of a second `/drip --math`, and you retake before building.

### 6. You: the build

You type the guidance dial into `walk06.py` from an empty file, one piece per round, after the walk cuts the layer into 4 to 6 pieces and writes them as the layer's step file. You end with the A∩B share and Z per w on both toys, each sampler's total variation beside the floor, and either the w where the parts-world blend appears or the finding that none up to 7.5 does.

**Prompt:** [the build](prompts/06-6-the-guidance-build.md)

⏱ rough: you 1 to 2 h (about 5 pieces) · AI ~35 min · stop: the prompt's bar

**Outcome:** the A∩B share, total variation and Z for each w from 1 to 7.5 on both toys, and the w where the parts-world blend appears.

### 7. You: compare your numbers with the reference

**Compare.** `python -m rebuild.walk_numbers` prints the `p6.` keys the AI lane added; your `walk06.py` prints the same numbers.

```bash
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m rebuild.walk_numbers | grep 'p6\.'
cd ~/Playground/PhD/simple-distributions-walk
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python walk06.py
```

⏱ you ~10 min (estimate) · stop: every exact number your script prints equals its `p6.` line to the digits shown, and every sampler number sits within 0.03 of `rebuild/out/phase6/phase6.json`. A number that differs is either a lesson or a bug in one of the two files; find which before layer 7.

### 8. AI, in the background: the scene

An AI session builds a scene in your notation with w as a slider from 1 to 7.5, showing the guided target on both toys. Every number on it comes from the `p6.` keys.

**Prompt:** [the scene](prompts/06-8-the-guidance-scene.md)

Scene: (URL lands here)

⏱ you ~20 min · AI ~45 min, beside layer 7's maths · stop: the scene is published and its numbers match your walk's

**Outcome:** a scene with w as a slider, showing the guided target on both toys.

### 9. You: close the layer

Run the `/drip-retro` command the build printed at its close (⏱ about 25 minutes). Then paste [the linking prompt](prompts/06-9-link-the-layer.md), which links what this layer wrote to the journey plans, vault notes, brick row and idea-map claims it rests on.

⏱ you 2 min to paste · AI ~10 min (estimate)

---

⬅️ [Reading stop: Bradley](05a-reading-stop-projective-composition.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 7: A without B](07-a-without-b.md) ➡️
