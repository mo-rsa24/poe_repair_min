# ⭐ Layer 2: composing two experts, and the samplers that reach the composition

⬅️ [Layer 1: one expert](01-one-diffusion-expert.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 3: ideal experts](03-ideal-experts-and-the-seven-kinds.md) ➡️

**Why it is critical.** The two constructions and the sampler problem both start here. Every later result reports which construction and which sampler it used. On [the axes](../PROMPTS.md#the-axes-one-new-variable-per-rung), this is rung b: the combination rule and the sampler, with the experts held as on rung a.

**Depth.** Build: the two constructions, summed scores missing the product, the MALA accept step, and the two idea-walk rebuilds. Own: total variation distance. Link only: the Skip list. You 5.8 to 11.7 h in all.

**Study first.** The product rule: [the PoE rule, assembled](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/12-the-poe-rule-assembled.md), [PoE on two Gaussians](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/13-poe-on-two-gaussians.md) (with its scenes), and the vault note [Product, Mixture And Negation](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/Product%2C%20Mixture%20And%20Negation.md). Why summed scores miss the product: [The Order Of Multiplying And Noising](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/The%20Order%20Of%20Multiplying%20And%20Noising.md). The samplers: [Langevin dynamics](../../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/21-langevin-dynamics.md), [Metropolis-Hastings](../../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/12-metropolis-hastings.md), [detailed balance](../../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/11-stationary-and-detailed-balance.md), [the Hamiltonian corrector](../../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/23-the-hamiltonian-corrector.md), [what is annealed](../../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/24-what-is-annealed.md) and [the accept step needs the height](../../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/36-the-accept-step-needs-the-height.md), whose scene is [the accept-step scene](https://claude.ai/artifact/LtVupoHgCdoyZgXtxB9817). Total variation distance has no passage that teaches it yet, only [what r_t costs the samples](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/23-what-rt-costs-the-samples.md), which writes it from [histograms](../../../../../../Obsidian/Mathematics/Statistics/Histogram.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=129 "Introduction to probability for data science, sections 3.2.4 to 3.2.5, Works a histogram of English letter frequencies and of N die throws approaching the PM..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=139 "Pattern recognition and machine learning, section 2.5, The histogram as a density estimate, p_i = n_i / (N Δ_i), drawn for three bin widths on 50 points from..."), [▶1](https://www.youtube.com/watch?v=qBigTkBLU6g&t=34s "StatQuest, Histograms, Clearly Explained, 0:34 to 3:42, Builds a histogram by stacking measurements into bins. Chapter-timed."), [▶2](https://www.youtube.com/watch?v=AbHn39y8eUo&t=3170s "Professor Leonard, Statistics Lecture 2.2, Data visualization, 52:50 to 1:02:41, Draws the frequency distribution built earlier in the lecture as graphs. Cha..."), [◆](../../../../../../goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/06-inverse-cdf-in-code.md "Write the inverse-CDF draw as a function and check its histogram, deep-learning/generative-models/03-sampling-from-a-graph, plan 06. Checks draws from the ex...")</sub>; quiz 2.1's preparation writes the missing passage. Textbook: Bishop, *Pattern Recognition and Machine Learning* 11.2.2 ([Metropolis-Hastings](../../../../../../Obsidian/Mathematics/Statistics/Markov%20Chain%20Monte%20Carlo/Markov%20Chain%20Monte%20Carlo.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=553 "Pattern recognition and machine learning, section 11.2, The Metropolis algorithm with a symmetric proposal, drawn as accepted and rejected steps on a 2D Gaus..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/probabilistic-machine-learning-advanced-topics.pdf#page=526 "Probabilistic machine learning: advanced topics, sections 12.1 to 12.2, The Metropolis-Hastings algorithm (Algorithm 12.1), why it works, and the choice of p..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/deep-learning-textbook.pdf#page=611 "Deep learning textbook, section 17.3, MCMC for energy-based models: running many chains in parallel, the distribution q(t) over their states converging to th..."), [▶1](https://www.youtube.com/watch?v=KmqTrm-bn8k&t=141s "ritvikmath, Coding MCMC, 2:21 to 5:50, One chain written in Python, at its accept or reject step. Chapter-timed, from the video lessons index."), [▶2](https://www.youtube.com/watch?v=yApmR-c_hKU&t=247s "ritvikmath, Markov Chain Monte Carlo, 4:07 to 8:24, Drawing samples by running a Markov chain whose long-run distribution is the target. Chapter-timed, from ..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/textbook/l3-stochastic-processes/03-metropolis-hastings-and-why-z-cancels.md "3. Metropolis-Hastings, and why Z cancels, deep-learning/diffusion-models/00-from-scratch-to-sdxl, volume L3, section 3. Why a product of densities can be sa...")</sub>, p557).

**Achieve.** `walk02.py` with your product, annealed ULA and per-sample MALA reporting acceptance per [sample](../../../../../../Obsidian/Mathematics/Statistics/Sampling%20From%20A%20Distribution.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=539 "Pattern recognition and machine learning, chapter 11, Why expectations need samples, ancestral sampling of a directed graph (page 541), standard distribution..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=245 "Introduction to probability for data science, section 4.8, Generates random numbers for any PDF by running a uniform through the inverse CDF, in Python, with..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/deep-learning-textbook.pdf#page=606 "Deep learning textbook, section 17.1, Why machine learning samples at all (Monte Carlo estimates of sums and integrals, and sampling as the goal of a generat..."), [▶1](https://www.youtube.com/watch?v=9ixzzPQWuAY&t=474s "ritvikmath, Inverse Transform Sampling, 7:54 to 10:04, Inverts the exponential CDF and turns a uniform draw into an exponential one. Chapter-timed, from the ..."), [▶2](https://www.youtube.com/watch?v=OXDqjdVVePY&t=232s "ritvikmath, Accept-Reject Sampling, 3:52 to 7:45, An easy proposal g scaled by M into an envelope above f, then each draw accepted with probability f/(M g). ..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/textbook/l3-stochastic-processes/13-drawing-samples-by-hand.md "Drawing samples by hand: Monte Carlo, inverse transform and accept-reject, deep-learning/diffusion-models/00-from-scratch-to-sdxl, volume L3, section 13. Thr...")</sub> · on the whiteboard: ring × bar as two densities and their product, and one MALA step with its accept test · worked: the sampler table below · problems: the ones quiz 2.2 banks · quizzes 2.1 and 2.2 passed.

## The flow

```text
 AI, in the background                         │  You, in order
───────────────────────────────────────────────┼───────────────────────────────────────────────
 quizzes 2.1 and 2.2, prepared ──────────────▶ │  1. quiz 2.1: accept ratio, TV, normalisers
   (paste during layer 1's build)              │       ▼
                                               │  2. /drip --math: two constructions, MALA
                                               │       ▼
                                               │  3. quiz 2.2: needed-now right, 80% of the rest
                                               │       ▼
                                               │  4. build: walk02.py, 02b, 02c
 rebuild/ reference, built ──────────────────▶ │  5. compare: p2. numbers equal yours
                                               │       ▼
 math-scene, beside layer 3 ◀───────────────── │  6. scene
                                               │       ▼
 xref-cadence links the layer ◀─────────────── │  7. /drip-retro, then the next layer
```

## The steps

### 0. AI, in the background: quizzes 2.1 and 2.2, one layer ahead

Claude prepares and publishes quizzes 2.1 and 2.2 while you build layer 1, so both are waiting when you reach steps 1 and 3. Paste the prompt when you start layer 1's build.

**Prompt:** [build layer 2's two quizzes](prompts/02-0-build-layer-2-quizzes.md)

⏱ AI ~1.5 h (estimate), beside layer 1's build

### 1. You: quiz 2.1

Quiz 2.1 checks the college maths under this layer: normalising constants, Markov chains and the Metropolis-Hastings accept ratio, total variation distance, and a Langevin step as gradient ascent plus noise. Its link lands here when step 0 publishes it: (URL lands here)

⏱ you 20 to 40 min · stop: the bar, every *needed now* question right and at least 80% of the rest; below that, read the items your misses point at and retake.

### 2. You: the maths drip

You derive, one rung per message, what the two constructions of a composition target, why summed noised scores miss the product, and why MALA's proposal divides by 4·ss. Your quiz 2.1 misses become the first rungs.

<a href="../figures/walk02-step2-three-means.png"><img src="../figures/walk02-step2-three-means.png" width="420" alt="Three bars of the head pixel's mean position: 0.9802 for the clean product, 0.6931 for the product noised at ᾱ = 0.5, and 0.2326 for the product of the two noised experts, which is where summed noised scores point"></a>

This is the summed-score error as three numbers. Noise the product and its mean sits at 0.6931; multiply the noised experts and it sits at 0.2326, a third of the way. The drip derives the gap for A = N(1, 0.1²) and B = N(−1, 1) at ᾱ = 0.5.

**Prompt:** [the maths drip](prompts/02-2-the-maths-drip.md)

⏱ you 55 to 110 min · AI ~15 min · stop: you state the four things below without notes

**Outcome:** four things you can state:

- construction (1) is construction (2) divided by a background expert;
- summed scores miss the product;
- for a fixed MALA step, the accept step makes the target a stationary law of the chain, which removes the proposal's step-size bias; a finite run can still miss through poor mixing or too few moves, and the tuned MALA here adapts while it [samples](../../../../../../Obsidian/Mathematics/Statistics/Sampling%20From%20A%20Distribution.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=539 "Pattern recognition and machine learning, chapter 11, Why expectations need samples, ancestral sampling of a directed graph (page 541), standard distribution..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=245 "Introduction to probability for data science, section 4.8, Generates random numbers for any PDF by running a uniform through the inverse CDF, in Python, with..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/deep-learning-textbook.pdf#page=606 "Deep learning textbook, section 17.1, Why machine learning samples at all (Monte Carlo estimates of sums and integrals, and sampling as the goal of a generat..."), [▶1](https://www.youtube.com/watch?v=9ixzzPQWuAY&t=474s "ritvikmath, Inverse Transform Sampling, 7:54 to 10:04, Inverts the exponential CDF and turns a uniform draw into an exponential one. Chapter-timed, from the ..."), [▶2](https://www.youtube.com/watch?v=OXDqjdVVePY&t=232s "ritvikmath, Accept-Reject Sampling, 3:52 to 7:45, An easy proposal g scaled by M into an envelope above f, then each draw accepted with probability f/(M g). ..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/textbook/l3-stochastic-processes/13-drawing-samples-by-hand.md "Drawing samples by hand: Monte Carlo, inverse transform and accept-reject, deep-learning/diffusion-models/00-from-scratch-to-sdxl, volume L3, section 13. Thr...")</sub>, so it is judged against the exact-sample floor;
- every composition reports three numbers together: the A∩B share, the total variation to the exact target, and the normaliser Z.

### 3. You: quiz 2.2

Quiz 2.2 checks the layer itself, with banked problems: the two constructions, why summed scores miss the product, MALA's proposal divisor 4·ss, and the three numbers every composition reports. Its link lands here when step 0 publishes it: (URL lands here)

⏱ you 65 to 130 min, the banked problems included · stop: the bar, then the build. Below the bar, the misses become the first rungs of a second `/drip --math`, and you retake before building.

### 4a. You: the build, walk02.py

You type the product, annealed ULA and per-sample MALA into `walk02.py` from an empty file, following [the step file](../02-composing-and-sampling.md) one step per message, steps 0 to 10. Claude's reference prints the same numbers.

<a href="../../../../toy/rebuild/out/phase2/product-mixture-negation-every-sampler.png"><img src="../../../../toy/rebuild/out/phase2/product-mixture-negation-every-sampler.png" width="420" alt="Three compositions of the trained phase 1 experts, one per row (ring times bar, left-or-right mixture, box 0.7 without box 0.3): the two experts' data, the exact density, then 1,000 samples from each sampler titled with its share and total variation; on the product row perfect draws read 0.93, tuned MALA 0.85 and reverse diffusion 0.58"></a>

This is step 10's run on the trained experts. Read the top row: reverse diffusion scatters samples round the whole ring (share 0.58) while tuned MALA lands on the two in-bar modes (0.85, against 0.93 for perfect draws). The exact panel's two modes are uneven because the trained energy at t = 0 splits the in-bar mass 0.10 and 0.90, so step 10 takes 2D ground truth from ideal experts.

**Prompt:** [the build](prompts/02-4a-the-build.md)

⏱ you 1.5 to 3 h (11 steps) · AI ~30 min · stop: the prompt's bar

**Outcome:** your own product, annealed ULA and per-sample MALA.

**Two findings from the idea walks, rebuilt here.**

### 4b. You: the first idea-walk rebuild, walk02b.py

You rebuild sections 1 to 3 of [the labels notebook](../../../ideas/poe-overlap-dial/what-an-expert-learns-and-kinds-of-composition.ipynb) by hand in `walk02b.py`: an expert's gradient never reads its labels, the meaning of a label decides construction (1), and a classifier trained on set membership guides sideways in A∩B. The notebook's printed outputs are the answer key.

**Prompt:** [rebuild the labels notebook](prompts/02-4b-rebuild-the-labels-notebook.md)

⏱ you 1 to 2 h (one build concept) · AI ~35 min (the median over 7 measured `--request` walks is 33) · stop: the prompt's bar

**Outcome:** you can show three things:

- a [density](../../../../../../Obsidian/Mathematics/Statistics/Distributions/Probability%20Density%20Function.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=188 "Introduction to probability for data science, section 4.1, Builds the intuition of probability per unit length, defines the PDF and its properties (Definitio..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=186 "Mathematics for machine learning, section 6.2.2, The density and the CDF (Definitions 6.1 and 6.2), a uniform example (Example 6.3), and discrete against con..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=36 "Pattern recognition and machine learning, section 1.2.1, The density's two conditions (equations 1.25 and 1.26), its link to the CDF, and why a density picks..."), [▶1](https://www.youtube.com/watch?v=QKA4HNEw3aY&t=52s "The Organic Chemistry Tutor, Probability Density Functions, 0:52 to 3:39, Verifies that a given function is a density by checking that its integral equals 1...."), [▶2](https://www.youtube.com/watch?v=oI3hZJqXJuc&t=135s "StatQuest, The Main Ideas behind Probability Distributions, 2:15 to 5:14, A curve as a distribution, with probability read as area under it, not its height. ..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/02-densities-and-the-gaussian.md#21-a-density-can-exceed-1 "Chapter 2: Read a density as area and draw a Gaussian from its covariance, deep-learning/diffusion-models/01-poe-derivation-foundations, chapter 02, section ..."), [◈](https://claude.ai/artifact/DBmJtn6xy75sPtQwPEvQZU#nb-sec1 "Kinds of composition notebook, Setup: the ideal expert, The ideal expert's log-density on a grid, in four steps from the 2D Gaussian to its logsumexp form, t...")</sub> expert never reads its labels;
- the meaning of a label decides construction (1);
- a [classifier](../../../../../../Obsidian/Artificial%20Intelligence/Machine%20Learning/Classifier.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/natural-language-processing/hands-on-large-language-models.pdf#page=160 "Hands-on large language models, chapter 4, Works a classifier end to end on real data: 5,331 positive and 5,331 negative Rotten Tomatoes reviews labelled by ..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/probabilistic-machine-learning-an-introduction.pdf#page=30 "Probabilistic machine learning: an introduction, section 1.2.1, Classification on the iris data: 150 flowers with four measurements each and one of three spe..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/understanding-deep-learning.pdf#page=78 "Understanding deep learning, sections 5.4 to 5.5, A network becomes a binary classifier by predicting a Bernoulli probability through the logistic sigmoid, a..."), [▶1](https://www.youtube.com/watch?v=pdqofxJeBN8&t=995s "CS231n lecture 2, 16:35 to 20:12, Builds the nearest neighbour classifier: training memorises the data, prediction returns the label of the closest training ..."), [▶2](https://www.youtube.com/watch?v=DATnpGoGhM8&t=759s "CS229 lecture 1, 12:39 to 14:39, Separates regression (continuous y) from classification (discrete y) with a house against townhouse boundary, and frames nex..."), [◆](../../../../../../goal-setting/learning/machine-learning/classical-ml-from-scratch/MASTER_PLAN.md "Classical ML from Scratch, machine-learning/classical-ml-from-scratch. Goal 1 classifies a toy dataset with logistic regression written by hand, and goal 7 b..."), [◈](https://claude.ai/artifact/DBmJtn6xy75sPtQwPEvQZU#nb-sec4 "Kinds of composition notebook, A classifier learns 'not B', A classifier trained on the labels the data carries learns the hidden-B points as 'not B'.")</sub> trained on the labels the data carries reproduces a shortcut.

### 4c. You: the second idea-walk rebuild, walk02c.py

You type tuned MALA and the ULA comparison into `walk02c.py`, reproducing run 1 of [the kinds notebook on the paper's toys](../../../ideas/poe-overlap-dial/kinds-of-composition-on-the-papers-toys.ipynb). MALA adapted to 0.6 acceptance takes steps 18 to 82 times larger and reaches the exact-sample floor on four kinds; ULA at those steps diverges.

**Prompt:** [rebuild tuned MALA](prompts/02-4c-rebuild-tuned-mala.md)

⏱ you 1 to 2 h (one build concept) · AI ~35 min · stop: the prompt's bar

**Outcome:** you can show that the accept step is what makes large steps safe, and that large steps are what let a sampler mix.

### 5. You: compare your numbers with the reference

```bash
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m rebuild.walk_numbers | grep 'p2\.'
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m pytest tests/test_anneal_samplers.py -s
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m rebuild.phase3_kinds --out ~/Playground/PhD/simple-distributions-walk/phase3-reference | sed -n '/Run 1/,/Run 2/p'
grep -o 'largest difference[^"]*' ../artifacts/ideas/poe-overlap-dial/what-an-expert-learns-and-kinds-of-composition.ipynb
cd ~/Playground/PhD/simple-distributions-walk
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python walk02.py
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python walk02b.py
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python walk02c.py
```

⏱ you ~15 min (`phase3_kinds` takes about 2 minutes on one card, and `--out` keeps it from rewriting `rebuild/out/phase3/`) · stop: the 28 `p2.` lines and the numbers `walk02.py` prints agree to the digits shown; the 6 sampler tests pass; `walk02b.py` prints the notebook's `0.0`; each tuned TV in `walk02c.py` is within 0.03 of the `Run 1` lines. A number that differs is either a lesson or a bug in one of the two files; find which before layer 3.

**Answer key for the samplers on exact experts** (`artifacts/ideas/when-poe-reaches-a-and-b/runs/samplers-on-exact-experts/`). The quadrant toy at overlap 0.05 gives these total variations against an exact-sample floor of `0.133`:

| Sampler | Total variation |
|---|---|
| ULA | `0.128` |
| fixed MALA (acceptance `0.600`) | `0.124` |
| UHA | `0.132` |
| reverse diffusion | `0.182` |

These total variations use 60 × 60 bins on $[-1.5,1.5]^2$, each bin's sample count divided by the full sample count, with samples outside the box counted as error (`total_variation` in that run's `exact_toy.py`). The floor is the same estimator applied to exact samples.

Reverse diffusion's A∩B share still looks right (`0.983` against `0.989`). That is why the share alone is never enough.

### 6. AI, in the background: the scene

Claude builds a scene, in your notation, where reverse diffusion with summed scores leaves a faint ring between ring × bar's two modes and annealed MALA lands on them. Paste it during layer 3's maths drip.

**Prompt:** [the scene](prompts/02-6-the-scene.md)

Scene: (URL lands here)

⏱ you ~20 min · AI ~45 min, beside layer 3's maths · stop: the scene is published and its numbers match your drip's

**Outcome:** a scene where reverse diffusion with summed scores leaves a faint ring between ring × bar's two modes, and annealed MALA lands on them.

### 7. You: close the layer

Run the `/drip-retro` command step 4a's build printed at its close (about 25 minutes). Then paste [the linking prompt](prompts/02-7-link-the-layer.md), which links what this layer wrote to the journey plans, notes and idea maps that teach it.

⏱ you 2 min to paste · AI ~10 min (estimate)

Layer 3 comes next, and its build opens on [the kinds-of-composition scene](https://claude.ai/artifact/DBmJtn6xy75sPtQwPEvQZU#system): the 15-cell notebook whose sections 1 to 3 you rebuilt in step 4b, walked one cell at a time.

**Skip as build topics:** UHA and Hamiltonian moves, the CHA bug, and the mixture wrapper. UHA stays link-only here. Tuned MALA reaches the exact-sample floor on intersection, disjunction, co-occurrence and interaction in run 1, and stays above it on contention (`0.16` against `0.10`), dominance (`0.30` against `0.11`) and negation (`0.21` against `0.15`).

---

⬅️ [Layer 1: one expert](01-one-diffusion-expert.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 3: ideal experts](03-ideal-experts-and-the-seven-kinds.md) ➡️
