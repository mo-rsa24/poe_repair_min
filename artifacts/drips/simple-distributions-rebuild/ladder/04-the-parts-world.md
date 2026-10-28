# ⭐ Layer 4: the parts world

⬅️ [Layer 3: ideal experts](03-ideal-experts-and-the-seven-kinds.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 5: the overlap dial](05-the-overlap-dial.md) ➡️

**Why it is critical.** It is the only place the cat-dog product is exact in more than two dimensions, and layer 8 computes the interaction term inside it. On [the axes](../PROMPTS.md#the-axes-one-new-variable-per-rung), this is rung c: co-occurrence in the data, with the rule and the samplers held as on rung b.

**Depth.** Build: the product of diagonal [Gaussians](../../../../../../Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Multivariate%20Normal%20Distribution.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=227 "Introduction to probability for data science, section 4.6, The one-dimensional Gaussian random variable: its density with plots of μ and σ moving and widenin..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=306 "Introduction to probability for data science, section 5.6, The d-dimensional Gaussian density with Σ⁻¹, its mean vector and covariance, and the diagonal cova..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=203 "Mathematics for machine learning, section 6.5, The univariate and multivariate densities (equations 6.62 and 6.63, page 204) drawn with 100 samples, then mar..."), [▶1](https://www.youtube.com/watch?v=rzFX5NWojp0&t=0s "StatQuest, The Normal Distribution, Clearly Explained, 0:00 to 3:27, The bell-shaped curve and what its mean and standard deviation set, in the video's openi..."), [▶2](https://www.youtube.com/watch?v=oI3hZJqXJuc&t=135s "StatQuest, The Main Ideas behind Probability Distributions, 2:15 to 5:14, A curve as a distribution, where probability is the area under it over an interval ..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/02-densities-and-the-gaussian.md "Chapter 2: Read a density as area and draw a Gaussian from its covariance, deep-learning/diffusion-models/01-poe-derivation-foundations, chapter 02. The Gaus...")</sub> (the mosaic and the blend), the product of Gaussian mixtures (two animals), and the summed-score error. You 3.3 to 6.7 h in all.

**Study first.** The product of diagonal Gaussians: [Products And Quotients Of Diagonal Gaussians](../../../../../../Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Products%20And%20Quotients%20Of%20Diagonal%20Gaussians.md) and [Multivariate Normal Distribution](../../../../../../Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Multivariate%20Normal%20Distribution.md). The normaliser of a product: [the PoE factorisation and normaliser](../../../../../../goal-setting/learning/deep-learning/diffusion-models/05-poe-composition-diffusion/plans/07-poe-factorisation-and-normaliser.md) and [Why The Normaliser Has No Slope In X](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/Why%20The%20Normaliser%20Has%20No%20Slope%20In%20X.md). The mosaic, blend and two animals: [Kinds Of Composition, mosaic, blend or two animals](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/Kinds%20Of%20Composition.md#mosaic-blend-or-two-animals-from-one-product), the [parts-world write-up](../../../ideas/poe-overlap-dial/example-9-parts-world/example-9-parts-world.tex) and [the parts-world scene](https://claude.ai/artifact/6XzsUMYdEb2eopRyGkpYkh).

**Achieve.** `walk04.py` producing the mosaic, the blend and two animals from one product · on the whiteboard: two 1D Gaussians multiplied, their precisions adding and the mean landing precision-weighted · worked: `numbers.json` in example 9 · problems: the ones quiz 4.2 banks · quizzes 4.1 and 4.2 passed.

## The flow

```text
 AI, in the background                         │  You, in order
───────────────────────────────────────────────┼───────────────────────────────────────────────
 quizzes 4.1 and 4.2, prepared ──────────────▶ │  1. quiz 4.1: products of 1D Gaussians
   (paste during layer 3's build)              │       ▼
                                               │  2. /drip --math: products and mixtures
                                               │       ▼
                                               │  3. quiz 4.2: needed-now right, 80% of the rest
                                               │       ▼
                                               │  4. build: walk04.py, walk04b
 rebuild/ reference, built ──────────────────▶ │  5. compare: numbers.json, phase4.json
                                               │       ▼
 parts-world scene ✅ built ──────────────────▶ │  6. scene: operate it
                                               │       ▼
 xref-cadence links the layer ◀─────────────── │  7. /drip-retro, then the next layer
```

## The steps

### 0. AI, in the background: quizzes 4.1 and 4.2

Claude prepares and publishes quizzes 4.1 and 4.2 while you build layer 3. Paste the prompt when you start layer 3's build.

**Prompt:** [build layer 4's two quizzes](prompts/04-0-build-layer-4-quizzes.md)

⏱ AI ~1.5 h (estimate), beside layer 3's build

### 1. You: quiz 4.1

Quiz 4.1 checks the college maths under this layer: the product of two 1D Gaussians (precisions add, the mean is precision-weighted), diagonal covariance, and Gaussian mixtures with their weights. Its link lands here when step 0 publishes it: (URL lands here)

⏱ you 20 to 40 min · stop: the bar, every *needed now* question right and at least 80% of the rest; below that, read the items your misses point at and retake.

### 2. You: the maths drip

You derive, one rung per message, the product of two diagonal Gaussians and the product of two Gaussian mixtures, every pair with its weight, over 16 pixels. Your quiz 4.1 misses become the first rungs.

**Prompt:** [the maths drip](prompts/04-2-the-maths-drip.md)

⏱ you 30 to 60 min · AI ~10 min · stop: you derive the Outcome below without notes

**Outcome:** you can derive that precisions add and that the mean is precision-weighted, plus the mixture-product weight w_ij ∝ π_i ρ_j N(μ_i; ν_j, Σ_i + Λ_j).

### 3. You: quiz 4.2

Quiz 4.2 checks the layer itself, with banked problems: the mosaic and the blend, the mixture-product weight, and the summed-score error at ᾱ = 0.5. Its link lands here when step 0 publishes it: (URL lands here)

⏱ you 30 to 60 min, the banked problems included · stop: the bar, then the build. Below the bar, the misses become the first rungs of a second `/drip --math`, and you retake before building.

### 4. You: the build: walk04.py

You type the 16-pixel cat and dog experts and their products into `walk04.py`, following [the step file](../04-the-parts-world.md) one step per message, steps 0 to 8: the mosaic, the blend, two animals, and every sampler scored against exact draws.

<a href="../../../../toy/rebuild/out/phase4/parts-world-samplers-against-exact.png"><img src="../../../../toy/rebuild/out/phase4/parts-world-samplers-against-exact.png" width="420" alt="The parts world's three settings, one per row (mosaic, blend, two animals), each sample drawn at its two half-means: exact draws in purple, then perfect draws, reverse diffusion, ULA, MALA, tuned MALA and UHA, each titled with projected total variation and weight error"></a>

On the mosaic row reverse diffusion's cloud sits off the exact point (0.98, −0.98), projected TV 0.73, while tuned MALA and UHA read 0.04 against perfect draws' 0.01. On the two-animals row every sampler finds both side-by-side pairs.

**Prompt:** [the build](prompts/04-4-the-build.md)

⏱ you 1 to 2 h (9 steps) · AI ~25 min · stop: the prompt's bar

**Outcome:** the mosaic, the blend and two animals side by side, all from one product.

### 4b. You: the second build: the summed-score error in walk04b.py

You rebuild setting (d) of [the parts-world script](../../../ideas/poe-overlap-dial/example-9-parts-world/parts_world.py) in `walk04b.py`, without `walk04.py` open: summed scores at ᾱ = 0.5 put the head at 0.233 where the noised product puts it at 0.693.

**Prompt:** [the summed-score error](prompts/04-4b-the-summed-score-error.md)

⏱ you 1 to 2 h (one build concept) · AI ~35 min · stop: the prompt's bar

**Outcome:** Du et al.'s summed-score error, computed exactly in the parts world.

### 5. You: compare your numbers with the reference

**Compare.** The reference numbers are `artifacts/ideas/poe-overlap-dial/example-9-parts-world/numbers.json` and `rebuild/out/phase4/phase4.json`; your `walk04.py` prints the same numbers.

```bash
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m json.tool ../artifacts/ideas/poe-overlap-dial/example-9-parts-world/numbers.json
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -c "import json; d = json.load(open('rebuild/out/phase4/phase4.json'))['settings']; [print(s, '|', c, '| TV', round(x['tv'], 3), '| weight error', round(x['weight_error'], 3), '| half-means', [round(m, 3) for m in x['half_means']]) for s in d for c, x in d[s]['samplers'].items()]"
cd ~/Playground/PhD/simple-distributions-walk
export PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python walk04.py
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python walk04b.py
```

⏱ you ~10 min · stop: every closed-form number your two scripts print equals `numbers.json` to 9 significant figures: `a.head_mean` 0.98019801980, `a.logZ` −46.465787495, `b.logZ` −1583.4068325, `b.product_sd` 0.070710678119, the side-by-side weight 0.5 (its float64 value: exactly it is 1/(2(1 + e^−779.12)), and the same-half weights underflow to 0.0), the same-half `logw_unnorm` −803.21380759, and `d` 0.23257000444 and 0.69310466671. The sampler lines are random draws, so read them for order rather than exact digits. In `phase4.json`, tuned MALA and UHA both have projected TV near 0.036, against the exact-draw baseline of 0.013. In the 1,000-sample run, tuned MALA is 0.059 and UHA is 0.048. Reverse diffusion is far above both at 0.732 in `phase4.json`. A closed-form number that differs is either a lesson or a bug in one of the files; find which before layer 5.

### 6. You: the scene, already built

✅ **Scene, already built:** [the parts-world scene](https://claude.ai/artifact/6XzsUMYdEb2eopRyGkpYkh). **Quiz, already built, for extra practice:** [the diagonal Gaussian quiz](https://claude.ai/artifact/Vmd1FBg7oWwZq1onu9Jofa), section "Products of [Gaussian](../../../../../../Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Multivariate%20Normal%20Distribution.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=227 "Introduction to probability for data science, section 4.6, The one-dimensional Gaussian random variable: its density with plots of μ and σ moving and widenin..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=306 "Introduction to probability for data science, section 5.6, The d-dimensional Gaussian density with Σ⁻¹, its mean vector and covariance, and the diagonal cova..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=203 "Mathematics for machine learning, section 6.5, The univariate and multivariate densities (equations 6.62 and 6.63, page 204) drawn with 100 samples, then mar..."), [▶1](https://www.youtube.com/watch?v=rzFX5NWojp0&t=0s "StatQuest, The Normal Distribution, Clearly Explained, 0:00 to 3:27, The bell-shaped curve and what its mean and standard deviation set, in the video's openi..."), [▶2](https://www.youtube.com/watch?v=oI3hZJqXJuc&t=135s "StatQuest, The Main Ideas behind Probability Distributions, 2:15 to 5:14, A curve as a distribution, where probability is the area under it over an interval ..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/02-densities-and-the-gaussian.md "Chapter 2: Read a density as area and draw a Gaussian from its covariance, deep-learning/diffusion-models/01-poe-derivation-foundations, chapter 02. The Gaus...")</sub> mixtures".

⏱ you ~20 min to operate it (estimate, as for layer 1's scene) · stop: you can set a slider on each expert's [variance](../../../../../../Obsidian/Mathematics/Statistics/Variance.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=150 "Introduction to probability for data science, section 3.4.4, Defines variance as E[(X − µ)²] and the standard deviation as its root (Definition 3.7), proves ..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=196 "Mathematics for machine learning, section 6.4.1, Variance as the covariance of a variable with itself (Definition 6.7), then the empirical mean and variance ..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/deep-learning-textbook.pdf#page=76 "Deep learning textbook, section 3.8, A short definition of variance and standard deviation beside expectation and covariance."), [▶1](https://www.youtube.com/watch?v=4-FysvtxQEU&t=1551s "Professor Leonard, Statistics 3.3, A standard deviation calculated by hand, 25:51 to 36:43, Works a standard deviation on a small data set by hand. Chapter-t..."), [▶2](https://www.youtube.com/watch?v=SzZ6GpcfoQY&t=236s "StatQuest, Calculating the Mean, Variance and Standard Deviation, Calculating population variance, 3:56 to 5:54, The population variance as the average squar..."), [◆](../../../../../../goal-setting/learning/deep-learning/transformers/textbook/06-show-that-variances-of-independent-terms-add.md "Show that variances of independent terms add, deep-learning/transformers, chapter 06. Predicts the spread of a dot product of random vectors by adding the va...")</sub> and say before you look whether the head follows the cat, lands at the midpoint, or splits into two animals.

### 7. You: close the layer

Run the `/drip-retro` command each build printed at its close, walk04 first, then walk04b (about 25 minutes each). Then paste [the linking prompt](prompts/04-7-link-the-layer.md), which links what this layer wrote to the vault notes, journey plan and idea map that teach it.

⏱ you 2 min to paste · AI ~10 min (estimate)

---

⬅️ [Layer 3: ideal experts](03-ideal-experts-and-the-seven-kinds.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 5: the overlap dial](05-the-overlap-dial.md) ➡️
