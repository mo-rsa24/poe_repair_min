# ⭐◇ Layer 3: ideal experts, and the seven kinds of composition

⬅️ [Layer 2: composing](02-composing-two-experts.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 4: the parts world](04-the-parts-world.md) ➡️

**Why it is half critical.** The ideal expert is critical: the training points smoothed by N(0, σ²I). Under a denoising objective taken over the whole distribution, with unlimited model capacity and a global optimum, its score is the best score field at that smoothing level. The rebuild then treats this smoothed set as the clean expert, which makes every later target exact, so a failure can be blamed on the sampler or the data rather than on a network's approximation. In the cited 2D run, trained networks matched it within `0.04` nats. Of the seven kinds, only co-occurrence, intersection, the chimera and the wet dog matter for the goal. On [the axes](../PROMPTS.md#the-axes-one-new-variable-per-rung), this is rung a: the ideal expert that layer 1's learned expert is measured against.

**Depth.** Build: the ideal expert, the exact target and its perfect-sample baseline, the four kinds that matter, and the weak-sampler rebuild. Link only: disjunction and dominance. You 4.3 to 8.7 h in all.

**Study first.** The kinds and their words: [Kinds Of Composition](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/Kinds%20Of%20Composition.md), [The Product As An Intersection](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/The%20Product%20As%20An%20Intersection.md), [What An Expert Learns From Its Training Set](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/What%20An%20Expert%20Learns%20From%20Its%20Training%20Set.md), and [the composition terms](../../../ideas/poe-overlap-dial/composition-terms.md). The exact noised mixture: [the exact score of a Gaussian mixture](../../../../../../goal-setting/learning/deep-learning/diffusion-models/06-diffusion-speciation-dynamics/plans/02-speciation/plans/02-exact-mixture-score.md). Why a product of noised experts differs from the noised product: [score additivity and the product density](../../../../../../goal-setting/learning/deep-learning/diffusion-models/06-diffusion-speciation-dynamics/plans/05-composition-and-the-experiment/plans/01-score-additivity-product-density.md). Smoothing by a [Gaussian](../../../../../../Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Multivariate%20Normal%20Distribution.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=227 "Introduction to probability for data science, section 4.6, The one-dimensional Gaussian random variable: its density with plots of μ and σ moving and widenin..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=306 "Introduction to probability for data science, section 5.6, The d-dimensional Gaussian density with Σ⁻¹, its mean vector and covariance, and the diagonal cova..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=203 "Mathematics for machine learning, section 6.5, The univariate and multivariate densities (equations 6.62 and 6.63, page 204) drawn with 100 samples, then mar..."), [▶1](https://www.youtube.com/watch?v=rzFX5NWojp0&t=0s "StatQuest, The Normal Distribution, Clearly Explained, 0:00 to 3:27, The bell-shaped curve and what its mean and standard deviation set, in the video's openi..."), [▶2](https://www.youtube.com/watch?v=oI3hZJqXJuc&t=135s "StatQuest, The Main Ideas behind Probability Distributions, 2:15 to 5:14, A curve as a distribution, where probability is the area under it over an interval ..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/02-densities-and-the-gaussian.md "Chapter 2: Read a density as area and draw a Gaussian from its covariance, deep-learning/diffusion-models/01-poe-derivation-foundations, chapter 02. The Gaus...")</sub> as a convolution: [summing out a joint, equation 2.3](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/textbook/02-joint-and-marginal-summing-out.md). Textbooks: Murphy, *Probabilistic Machine Learning* 16.3 (kernel density estimation, p605) and Chan, *Introduction to Probability for Data Science* 5.5 (the sum of two [random variables](../../../../../../Obsidian/Mathematics/Statistics/Probability/Random%20Variable.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=121 "Introduction to probability for data science, section 3.1, Works it on four outcomes ♣, ♢, ♡, ♠ mapped to the numbers 1 to 4, explains why a function is call..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-statistics.pdf#page=80 "Introduction to statistics (Wits lecture notes), section 4.2, A discrete random variable and its probability distribution, with a worked example (Example 4.1..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=180 "Mathematics for machine learning, section 6.1.2, A random variable as a function from outcomes to a target space of states, discrete or continuous."), [▶1](https://www.youtube.com/watch?v=OSPr6G6Ka-U&t=90s "StatQuest, Expected Values for Continuous Variables, 1:30 to 3:42, Discrete and continuous variables defined, before the exponential distribution. Chapter-ti..."), [▶2](https://www.youtube.com/watch?v=shrTfDUB9Hk&t=0s "Professor Leonard, Statistics lecture 6.2, Introduction to the Normal Distribution and Continuous Random Variables, 0:00 to 41:13, Continuous distributions, ...")</sub>, p296).

**Achieve.** `walk03.py` with your ideal experts and four kinds, each shown succeeding and failing · on the whiteboard: three training points smoothed by N(0, σ²) at two values of σ · worked: what `python -m rebuild.phase3_kinds` prints · problems: the ones quiz 3.2 banks · quizzes 3.1 and 3.2 passed.

## The flow

```text
 AI, in the background                         │  You, in order
───────────────────────────────────────────────┼───────────────────────────────────────────────
 quizzes 3.1 and 3.2, prepared ──────────────▶ │  1. quiz 3.1: convolution, kernel densities
   (paste during layer 2's build)              │       ▼
                                               │  2. /drip --math: the ideal expert
                                               │       ▼
                                               │  3. quiz 3.2: needed-now right, 80% of the rest
                                               │       ▼
                                               │  4. build: walk03.py, walk03b
 rebuild/ reference, built ──────────────────▶ │  5. compare: p3. and phase3_kinds
                                               │       ▼
 xref-cadence links the layer ◀─────────────── │  7. /drip-retro, then the next layer
```

## The steps

### 0. AI, in the background: quizzes 3.1 and 3.2, one layer ahead

Claude prepares and publishes quizzes 3.1 and 3.2 while you build layer 2. Paste the prompt when you start layer 2's build.

**Prompt:** [build layer 3's two quizzes](prompts/03-0-build-layer-3-quizzes.md)

⏱ AI ~1.5 h (estimate), beside layer 2's build

### 1. You: quiz 3.1

Quiz 3.1 checks the college maths under this layer: convolving a density with a Gaussian, kernel density estimates, and softmax weights over training points. Its link lands here when step 0 publishes it: (URL lands here)

⏱ you 20 to 40 min · stop: the bar, every *needed now* question right and at least 80% of the rest; below that, read the items your misses point at and retake.

### 2. You: the maths drip

You derive why smoothing the training points by N(0, σ²I) gives the best score field a diffusion model can learn at that smoothing, under which assumptions, and what the smoothed set becomes at every noise level. Your quiz 3.1 misses become the first rungs.

<a href="../figures/walk03-step2-noised-variance.png"><img src="../figures/walk03-step2-noised-variance.png" width="420" alt="The exact component variance v_t of the noised ideal expert against diffusion step t at σ = 0.05: 0.00313 at t = 0, 0.1665 at 25, 0.5229 at 50, 0.8668 at 75 and 1 at 99"></a>

Each training point's kernel widens from the narrow clean 0.00313 to the full 1 by the last step, following v_t = 1 − ᾱ_t + ᾱ_t σ². The drip has you reproduce the 0.5229 at t = 50 by hand.

**Prompt:** [the maths drip](prompts/03-2-the-maths-drip.md)

⏱ you 45 to 90 min · AI ~10 min · stop: you derive the Outcome below without notes

**Outcome:** you can derive the ideal expert's score and say why its composition is exact at every noise level.

### 3. You: quiz 3.2

Quiz 3.2 checks the layer itself, with banked problems: the ideal expert's score, why its composition is exact at every noise level, the perfect-sample baseline, and co-occurrence, intersection, contention and interaction. Its link lands here when step 0 publishes it: (URL lands here)

⏱ you 45 to 90 min, the banked problems included · stop: the bar, then the build. Below the bar, the misses become the first rungs of a second `/drip --math`, and you retake before building.

### 4a. You: the build, walk03.py

You type the ideal experts, the exact grid, the perfect-sample baseline and the kinds of composition into `walk03.py`, following [the step file](../03-ideal-experts-and-the-seven-kinds.md) one step per message, steps 0 to 13. Disjunction and dominance are typed so the data stay in the notebook's order, but not taught.

Before the first step, walk [the kinds-of-composition scene](https://claude.ai/artifact/DBmJtn6xy75sPtQwPEvQZU#system) one cell at a time. It lays out the 15-cell [notebook](../../../ideas/poe-overlap-dial/what-an-expert-learns-and-kinds-of-composition.ipynb) as a walk, and its closing cells compute co-occurrence, contention and interaction as products of two experts' densities on one grid, the same kinds this build types.

<a href="../../../../toy/rebuild/out/phase3/figure1-every-sampler-on-every-kind.png"><img src="../../../../toy/rebuild/out/phase3/figure1-every-sampler-on-every-kind.png" width="420" alt="Seven kinds of composition, one per row (intersection, disjunction, co-occurrence, contention, dominance, interaction, negation): the two experts' training data, the exact composed density, then 2,000 samples each from perfect draws, reverse diffusion, ULA, MALA and UHA, each titled with its total variation"></a>

This is the kinds notebook's Figure 1, which `phase3_kinds` rebuilds. On the intersection row reverse diffusion leaves a faint ring between the two modes (TV 0.31) where perfect draws read 0.11 and UHA 0.13. The dominance row stays high for every sampler (0.29 to 0.58), which is why this layer skips it.

**Prompt:** [the build](prompts/03-4a-the-build.md)

⏱ you 1.5 to 3 h (12 of the 14 steps; disjunction and dominance are skipped) · AI ~30 min · stop: the prompt's bar

**Outcome:** your own ideal experts, plus four kinds of composition, each shown succeeding and failing.

### 4b. You: the idea-walk rebuild, walk03b.py

You rebuild run 2 of [the kinds notebook on the paper's toys](../../../ideas/poe-overlap-dial/kinds-of-composition-on-the-papers-toys.ipynb) in `walk03b.py`: three kinds at the steep settings where a weak sampler draws the picture of a data failure, such as reverse diffusion's chimera column at contention with 5% of each set rotated.

**Prompt:** [rebuild the steep settings](prompts/03-4b-rebuild-the-steep-settings.md)

⏱ you 1 to 2 h (one build concept) · AI ~35 min · stop: the prompt's bar

**Outcome:** you can show that near the edge of what the data supports, a weak sampler draws the picture of a data failure. That is why every later result is graded against an exact target.

### 5. You: compare your numbers with the reference

```bash
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m rebuild.walk_numbers | grep 'p3\.'
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m rebuild.phase3_kinds --out ~/Playground/PhD/simple-distributions-walk/phase3-reference
cd ~/Playground/PhD/simple-distributions-walk
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python walk03.py
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python walk03b.py
```

**Compare.** `python -m rebuild.walk_numbers` prints the `p3.` keys and `python -m rebuild.phase3_kinds` prints a row per kind; your `walk03.py` prints the same numbers.

⏱ you ~15 min (`phase3_kinds` takes about 2 minutes on one card, and `--out` keeps it from rewriting `rebuild/out/phase3/`) · stop: the 7 `p3.` lines your `walk03.py` prints equal `walk_numbers`' to the digits shown (`walk_numbers` prints 25 `p3.` keys; the other 18 back the step file's derivations); the `exact` column of `phase3_kinds`' Figure 1 table reads `0.71`, `0.81`, `0.67` and `-0.41` for the four kinds, as your `walk03.py` does; each row of its `Run 2` table is within 0.05 of `walk03b.py`'s. A number that differs is either a lesson or a bug in one of the two files; find which before layer 4.

### 7. You: close the layer

This layer has no scene, so the flow skips step 6. Run the `/drip-retro` command step 4a's build printed at its close (about 25 minutes). Then paste [the linking prompt](prompts/03-7-link-the-layer.md), which links what this layer wrote to the vault notes, journey plans and idea maps that teach it.

⏱ you 2 min to paste · AI ~10 min (estimate)

**Skip:** disjunction and dominance. Dominance is unequal mode weights that no sampler here fixes (total variation `0.29` to `0.58`, against a perfect-draw total variation of `0.11` in that run). It is a sampler problem, not a composition one, and it stays open in `artifacts/ideas/poe-overlap-dial/IDEA_MAP.md`, claims 4 and 8.

---

⬅️ [Layer 2: composing](02-composing-two-experts.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 4: the parts world](04-the-parts-world.md) ➡️
