# ⭐ Reading stop: the two Bradley papers (the read is critical; the rest is optional)

⬅️ [Layer 5: the overlap dial](05-the-overlap-dial.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 6: guidance](06-guidance-as-a-dial.md) ➡️

**Why the read is critical.** Bradley et al. 2025a (arXiv 2502.04549) prove when summed scores are exact: under factorized conditionals, where each expert matches the background outside its own region (Theorem 5.3). They also prove when a composition is exact in a feature space (Theorem 6.1). That is the mathematical reason for layer 5's finding that an expert loose outside its own region needs no overlap. Bradley 2025b (arXiv 2509.16447) ties it to a local score. On [the axes](../PROMPTS.md#the-axes-one-new-variable-per-rung), this is a side rung: a reading stop, which adds no variable.

**Depth.** Own: the factorized-conditionals condition, and exactness in a feature space. Later: the CLEVR reproduction and the Bradley rebuild, both optional below. You 1.7 to 3.3 h in all.

**Study first.** The 2025a paper walked chunk by chunk: [projective composition, Definition 4.1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/sources/mechanisms-of-projective-composition/read-ahead/04-projective-composition.md) and [the simple construction and factorized conditionals, Definition 5.2 and Theorem 5.3](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/sources/mechanisms-of-projective-composition/read-ahead/05-simple-construction.md), with the vault note [Composing Models With A Background](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/Composing%20Models%20With%20A%20Background.md). An earlier full read of the same paper stopped at chunk 6 of [its walk map](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/drips/read-mechanisms-of-projective-composition/READ_MAP.md). For quiz R.1: Deisenroth et al., *Mathematics for Machine Learning* 6.7 (change of variables, p220) and 6.4 to 6.5 on [conditional independence](../../../../../../Obsidian/Mathematics/Statistics/Probability/Conditional%20Independence.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/computer-vision/computer-vision-textbook.pdf#page=219 "Computer vision textbook, section 10.1, Works it on a joint table of three discrete variables taking 4, 3 and 2 values, 24 probabilities in all, where x₁ and..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=201 "Mathematics for machine learning, section 6.4.5, Definition 6.11, p(x, y ∣ z) = p(x ∣ z) p(y ∣ z) for every z, read as 'given knowledge about z, the distribu..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=389 "Pattern recognition and machine learning, section 8.2, The definition p(a ∣ b, c) = p(a ∣ c), then three-node graphs where observing c makes a and b independ..."), [▶1](https://www.youtube.com/watch?v=zRdE8A4UZes&t=4587s "CS229 lecture 5, 1:16:27 to 1:20:34, Makes the naive Bayes assumption that words are conditionally independent given the class, cutting the parameters from 2..."), [◆](../../../../../../goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/04-conditional-independence.md "Check independence inside each slice of a third variable, deep-learning/generative-models/03-sampling-from-a-graph, plan 04. Conditional independence checked...")</sub> (Definition 6.11, p201). The same condition drawn as a graph, where c₁ and c₂ are independent given x: [conditional independence inside each slice of a third variable](../../../../../../goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/04-conditional-independence.md), [reading it off with d-separation](../../../../../../goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/16-d-separation.md) and [the product-of-experts rule derived from the fork](../../../../../../goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/17-poe-rule-from-the-fork.md).

**Ideas to tie in.** From [the ideas that tie the ladder together](../../../ideas/ideas-that-tie-the-ladder-together.md): [conditional independence made visible as an interaction residual](../../../ideas/ideas-that-tie-the-ladder-together.md#1-make-conditional-independence-visible-as-an-interaction-residual) (needed, 2 to 3 h, no training). Each comes with an exact toy check.

**Achieve.** On the whiteboard: the quadrant toy with its background, marking where each expert matches the background and where it does not · quizzes R.1 and R.2 passed.

## The flow

```text
 AI, in the background                         │  You, in order
───────────────────────────────────────────────┼───────────────────────────────────────────────
 quizzes R.1 and R.2, prepared ──────────────▶ │  1. quiz R.1: independence, Jacobians
   (paste during layer 5's build)              │       ▼
                                               │  2. /drip-read --combine --lean
                                               │       ▼
                                               │  3. quiz R.2: needed-now right, 80% of the rest
                                               │       ▼
 concept-scene, beside layer 6 ◀────────────── │  4. scene
                                               │       ▼
 xref-cadence links the stop ◀──────────────── │  5. the next layer
```

## The steps

### 0. AI, in the background: quizzes R.1 and R.2, one layer ahead

You paste this when layer 5's build starts. An AI session prepares and builds both quizzes, so they are waiting here when you reach step 1; each quiz's link lands in its own step below.

**Prompt:** [build quizzes R.1 and R.2](prompts/05a-0-quizzes-r1-and-r2.md)

⏱ AI ~1.5 h (estimate), beside layer 5's build

### 1. You: quiz R.1

Quiz R.1 checks the college maths under the stop: conditional independence, and a change of variables with a Jacobian.

Quiz R.1: (URL lands here)

⏱ you 20 to 40 min · stop: the bar, every *needed now* question right and at least 80% of the rest; below that, read the read-first items your misses point at and retake.

### 2. You: the combined read

You read the two Bradley papers side by side, one idea per round, each passage as printed with its plain reading. You come out able to say when summed scores are exact, and where the quadrant toy and the parts world stand against that condition.

**Prompt:** [the combined read](prompts/05a-2-the-combined-read.md)

⏱ you 40 to 80 min (the one measured `--lean` read, of a single paper, took 38 minutes over 6 sittings) · AI ~1 h · stop: you state the condition and place both toys against it

**Outcome:** you can state the factorized-conditionals condition and show where the quadrant toy and the parts world satisfy it or break it.

### 3. You: quiz R.2

Quiz R.2 checks the stop itself: the factorized-conditionals condition (Theorem 5.3), exactness in a feature space (Theorem 6.1), and where the quadrant toy and the parts world satisfy or break the condition.

Quiz R.2: (URL lands here)

⏱ you 40 to 80 min · stop: the bar, every *needed now* question right and at least 80% of the rest.

### 4. AI, in the background: the scene

An AI session builds a scene on the four-quadrant toy where you switch the background (empty, the four-quadrant marginal, the unconditional) and watch each of Definition 5.2's three conditions hold or break. Its link lands below once it is published.

**Prompt:** [the scene](prompts/05a-4-the-background-scene.md)

Scene: (URL lands here)

⏱ you ~20 min · AI ~45 min, beside layer 6's maths · stop: the scene is published

**Outcome:** a scene where you change the background on the quadrant toy and watch the factorized-conditionals condition hold or break.

### 5. You: close the stop

Run the `/drip-retro` command the read printed at its close (⏱ about 25 minutes). Then paste [the linking prompt](prompts/05a-5-link-the-stop.md), which links what the read wrote (its map, the combined reading and the retrospective) to the read-aheads, the vault note and the idea-map claims the stop rests on.

⏱ you 2 min to paste · AI ~10 min (estimate)

### ◇ Optional, expensive: the paper's CLEVR figures

You reproduce the paper's CLEVR figures on mscluster, one figure per round, choosing only the ones the 2D toys and the parts world cannot stand in for. Two EDM2 checkpoints trained on CLEVR already sit on `/datasets`, so Figure 3 C is reachable without new training.

<a href="../../../results/joint-against-poe-by-kind-of-composition/crops/structural-bradley2025a-fig2-projective-composition-clevr.png"><img src="../../../results/joint-against-poe-by-kind-of-composition/crops/structural-bradley2025a-fig2-projective-composition-clevr.png" width="420" alt="Bradley et al. 2025a, Figure 2: four CLEVR models each draw one object, the background model draws an empty scene, and composing the background with 1, 2, 3 and 4 of the models gives scenes with 1, 2, 3 and 4 objects"></a>

This is the paper's own Figure 2, one of the figures the walk can reproduce. Count the objects on the right: composing the background with 1, 2, 3 and 4 models gives 1, 2, 3 and 4 objects, and that count is what a reproduction has to match.

**Prompt:** [the CLEVR figures](prompts/05a-optional-the-clevr-figures.md)

⏱ you ~1 h across its opening and checks (estimate) · cluster hours: listed per figure by its opening · box: set per figure you pick

**Outcome:** the paper's CLEVR figures on the cluster. Its opening lists each figure and its cost, so pick only what the toys cannot stand in for. The CLEVR grids already generated in `artifacts/results/joint-against-poe-by-kind-of-composition/` may cover some of them.

### ◇ Optional, recommended if the read raised questions: the Bradley rebuild

You rebuild both papers' constructions by hand in a new folder, one piece per round: on the quadrant toy and the parts world first, then in a learned invertible feature space. Each piece prints its number beside the read-ahead's answer key.

**Prompt:** [the Bradley rebuild](prompts/05a-optional-the-bradley-rebuild.md)

⏱ you 2 to 4 h (two build concepts) · AI ~1 h · stop: the local score computed on both toys

**Outcome:** projective composition and the local score rebuilt by hand on the quadrant toy and the parts world.

---

⬅️ [Layer 5: the overlap dial](05-the-overlap-dial.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 6: guidance](06-guidance-as-a-dial.md) ➡️
