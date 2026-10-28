# ⭐ Correction prior art (read before layer 9)

⬅️ [Layer 8: the interaction term](08-the-interaction-term.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 9: the correction](09-the-correction-learned-back.md) ➡️

**Why it is critical.** Two published methods already correct a composition, and one of them gets worse with learned scores. Layer 9's correction has to be placed against both. On [the axes](../PROMPTS.md#the-axes-one-new-variable-per-rung), this is a side rung: it adds no variable, and places rung e's correction against the two published ones.

**Depth.** Own: CO3's division of a joint model by its parts, and Feynman-Kac reweighting. Gloss: sliced Wasserstein-2. You 2 to 4 h in all (rough).

**Study first.** [Importance weights](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/probabilistic-machine-learning-advanced-topics.pdf#page=518) as ratios of densities: [annealed importance sampling and its proof](../../../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/textbook/l3-stochastic-processes/10-annealing-on-three-axes-and-annealed-importance-sampling.md). [The Feynman-Kac weights, derived and run on a toy](../../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/26-feynman-kac-correctors.md). Textbook: Goodfellow et al. 17.2 (importance [sampling](../../../../../../Obsidian/Mathematics/Statistics/Sampling%20From%20A%20Distribution.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=539 "Pattern recognition and machine learning, chapter 11, Why expectations need samples, ancestral sampling of a directed graph (page 541), standard distribution..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=245 "Introduction to probability for data science, section 4.8, Generates random numbers for any PDF by running a uniform through the inverse CDF, in Python, with..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/deep-learning-textbook.pdf#page=606 "Deep learning textbook, section 17.1, Why machine learning samples at all (Monte Carlo estimates of sums and integrals, and sampling as the goal of a generat..."), [▶1](https://www.youtube.com/watch?v=9ixzzPQWuAY&t=474s "ritvikmath, Inverse Transform Sampling, 7:54 to 10:04, Inverts the exponential CDF and turns a uniform draw into an exponential one. Chapter-timed, from the ..."), [▶2](https://www.youtube.com/watch?v=OXDqjdVVePY&t=232s "ritvikmath, Accept-Reject Sampling, 3:52 to 7:45, An easy proposal g scaled by M into an envelope above f, then each draw accepted with probability f/(M g). ..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/textbook/l3-stochastic-processes/13-drawing-samples-by-hand.md "Drawing samples by hand: Monte Carlo, inverse transform and accept-reject, deep-learning/diffusion-models/00-from-scratch-to-sdxl, volume L3, section 13. Thr...")</sub>, p608). Sliced Wasserstein-2 has no passage that teaches it; Chen, *The Little Book of Generative AI Foundations* 8.2.2 (p168) covers plain Wasserstein, enough for its gloss.

**Achieve.** On the whiteboard: one toy with CO3's division and Feynman-Kac's weights drawn side by side · quizzes P.1 and P.2 passed.

## The flow

```text
 AI, in the background                         │  You, in order
───────────────────────────────────────────────┼───────────────────────────────────────────────
 quizzes P.1 and P.2, prepared ──────────────▶ │  1. quiz P.1: importance weights
   (paste during layer 8's build)              │       ▼
                                               │  2. /drip-read --combine --lean, 3 papers
                                               │       ▼
                                               │  3. quiz P.2: needed-now right, 80% of the rest
                                               │       ▼
 xref-cadence links the stop ◀──────────────── │  4. layer 9
```

## The steps

### 0. AI, in the background: quizzes P.1 and P.2, one layer ahead

When you start layer 8's build, paste the prompt so both quizzes are waiting at this stop. Quiz P.1 covers the maths under it: importance weights as a ratio of two densities, and sliced Wasserstein-2 in two plain lines. Quiz P.2 covers the three papers by their role, and asks which Table 1 panel each quoted number comes from.

**Prompt:** [build quizzes P.1 and P.2](prompts/08a-0-quizzes-p1-and-p2.md)

⏱ AI ~1.5 h (estimate), beside layer 8's build

### 1. You: quiz P.1

Quiz P.1 checks the maths under this stop: importance weights as a ratio of two densities, and sliced Wasserstein-2 at gloss depth.

Quiz P.1: (URL lands here)

⏱ you 20 to 40 min · stop: the bar, every *needed now* question right and at least 80% of the rest; below that, read the read-first items your misses point at and retake.

### 2. You: the combined read

You read one idea across three papers, one idea per round: CO3 (arXiv 2509.25940), Soiffer et al. 2026 (arXiv 2606.23920) and Feynman-Kac correctors (arXiv 2503.02819), with plan 26 of the sampler-correctors journey as the fourth source. You come out able to say, for each method, what it divides or reweights, what it needs, and which Table 1 panel shows it helping or hurting, and then how layer 9's learned correction differs from both.

**Prompt:** [the combined read](prompts/08a-2-the-combined-read.md)

⏱ rough: you 60 to 120 min (three papers) · AI ~1.5 h · stop: you place your correction against both methods below

**Outcome:** you can place your correction against two existing ones:

- CO3 divides a joint model by its parts, p(x|C)/∏p(x|c_i).
- Soiffer et al.'s Table 1 shows Feynman-Kac correction improving the fit with exact scores (sliced Wasserstein-2 from `0.64` to `0.08`) and worsening it with learned scores (`0.39` to `2.82`).

The two pairs come from different panels, so they differ in two things at once: `0.64` to `0.08` is analytic scores on a target that does not factorise (panel d), and `0.39` to `2.82` is scores learned from 10,000 [samples](../../../../../../Obsidian/Mathematics/Statistics/Sampling%20From%20A%20Distribution.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=539 "Pattern recognition and machine learning, chapter 11, Why expectations need samples, ancestral sampling of a directed graph (page 541), standard distribution..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=245 "Introduction to probability for data science, section 4.8, Generates random numbers for any PDF by running a uniform through the inverse CDF, in Python, with..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/deep-learning-textbook.pdf#page=606 "Deep learning textbook, section 17.1, Why machine learning samples at all (Monte Carlo estimates of sums and integrals, and sampling as the goal of a generat..."), [▶1](https://www.youtube.com/watch?v=9ixzzPQWuAY&t=474s "ritvikmath, Inverse Transform Sampling, 7:54 to 10:04, Inverts the exponential CDF and turns a uniform draw into an exponential one. Chapter-timed, from the ..."), [▶2](https://www.youtube.com/watch?v=OXDqjdVVePY&t=232s "ritvikmath, Accept-Reject Sampling, 3:52 to 7:45, An easy proposal g scaled by M into an envelope above f, then each draw accepted with probability f/(M g). ..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/textbook/l3-stochastic-processes/13-drawing-samples-by-hand.md "Drawing samples by hand: Monte Carlo, inverse transform and accept-reject, deep-learning/diffusion-models/00-from-scratch-to-sdxl, volume L3, section 13. Thr...")</sub> on a target that does (panel c). Read each on its own, never the gap between them as the effect of learned scores alone. That is the same lesson as layer 9's trained experts.

### 3. You: quiz P.2

Quiz P.2 checks the stop itself: CO3's division of a joint model by its parts, Feynman-Kac reweighting, and why the correction helps with exact scores and hurts with learned ones.

Quiz P.2: (URL lands here)

⏱ you 40 to 80 min · stop: the bar, every *needed now* question right and at least 80% of the rest.

### 4. You: close the stop

Run the `/drip-retro` command the read printed at its close (about 25 minutes). Then paste [the linking prompt](prompts/08a-4-link-the-stop.md), which links the combined read's map and notes to the textbook chapter, plan 26, the layer 8 and layer 9 rows they rest on, and the claims in both idea maps.

⏱ you 2 min to paste · AI ~10 min (estimate)

---

⬅️ [Layer 8: the interaction term](08-the-interaction-term.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 9: the correction](09-the-correction-learned-back.md) ➡️
