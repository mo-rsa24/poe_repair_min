# ⭐ Layer 9: the correction, learned back

⬅️ [Correction prior art](08a-correction-prior-art.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 10: latent space](10-latent-space-and-vaes.md) ➡️

**Why it is critical.** This is the second half of the goal: a fix you can train. The fair test says whether composition plus a correction needs fewer joint examples than training on the joint directly. On [the axes](../PROMPTS.md#the-axes-one-new-variable-per-rung), this is rung e, second half: the correction learned back from the interaction term layer 8 defined.

**Depth.** Build: [score matching](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Score%20Matching.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/the-little-book-of-generative-ai-foundations.pdf#page=119 "The little book of generative AI foundations, sections 6.2 to 6.3, Works the one-dimensional case first, removing the unknown data score from the Fisher dive..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/probabilistic-machine-learning-advanced-topics.pdf#page=879 "Probabilistic machine learning: advanced topics, section 24.3, Score matching as minimising the Fisher divergence for an energy model, then denoising score m..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/deep-learning-textbook.pdf#page=633 "Deep learning textbook, sections 18.4 to 18.5, Score matching as a way to train a model without its partition function, its per-example loss (equation 18.25)..."), [▶1](https://www.youtube.com/watch?v=_WaR2fjZpEQ&t=1113s "CME296 lecture 2, 18:33 to 21:08, Writes the score matching loss, notes it needs the unknown true score, and names implicit and sliced score matching as the ..."), [▶2](https://www.youtube.com/watch?v=_WaR2fjZpEQ&t=1771s "CME296 lecture 2, 29:31 to 38:28, States denoising score matching (score matching on the noised distribution equals regression on the tractable conditional s..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/13-the-training-loss-two-ways.md "Chapter 13: Derive the training loss two ways: score matching and the ELBO, deep-learning/diffusion-models/01-poe-derivation-foundations, chapter 13. Hyvärin...")</sub> as [regression](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=65) with ∇r_t as its target, and the fair test. You 2.3 to 4.7 h in all (rough), plus about 1 h of training beside you.

**Study first.** [Denoising score matching and its optimum](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/13-the-training-loss-two-ways.md), with [the objective derived three ways](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/20-score-matching-the-training-objective.md). [The conditional mean is the best guess](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/textbook/05-the-conditional-mean-is-the-best-guess.md) for quiz 9.1. Textbooks: Chen 6.3.3 (p126) and Goodfellow et al. 18.5 (p635). Video: Stanford CS236 lecture 13, [29:16 to 44:39](https://www.youtube.com/watch?v=8G-OsDs1RLI&t=1756s).

**Ideas to tie in.** From [the ideas that tie the ladder together](../../../ideas/ideas-that-tie-the-ladder-together.md): [the interaction score as the object a correction can learn](../../../ideas/ideas-that-tie-the-ladder-together.md#5-learn-the-identifiable-object-which-is-the-interaction-score) (needed, 2 to 3 h, no training); [an oracle correction cube for the fair test](../../../ideas/ideas-that-tie-the-ladder-together.md#7-turn-the-evaluation-into-an-oracle-correction-cube) (needed, 4 to 8 h). Each comes with an exact toy check.

**Achieve.** `walk09.py` printing the three arms' [total variation](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/probabilistic-machine-learning-advanced-topics.pdf#page=94) at overlap 0.05 and 0.001, with exact and trained experts · on the whiteboard: the regression target ∇r_t at one point of the parts world · problems: the ones quiz 9.2 banks · quizzes 9.1 and 9.2 passed.

## The flow

```text
 AI, in the background                         │  You, in order
───────────────────────────────────────────────┼───────────────────────────────────────────────
 quizzes 9.1 and 9.2, prepared ──────────────▶ │  1. quiz 9.1: least squares, gradients
   (paste during the prior-art read)           │       ▼
                                               │  2. /drip --math: score matching as regression
                                               │       ▼
                                               │  3. quiz 9.2: needed-now right, 80% of the rest
                                               │       ▼
                                               │  4. build: walk09.py (--request)
 builds rebuild/phase9_correction.py ────────▶ │  5. compare: p9. numbers equal yours
                                               │       ▼
 results-scene, beside layer 12 ◀───────────── │  6. scene
                                               │       ▼
 xref-cadence links the layer ◀─────────────── │  7. /drip-retro, then the next layer
```

## The steps

### 0. AI, in the background: the layer 9 reference and quizzes 9.1 and 9.2, one layer ahead

When you start the prior-art read, paste both prompts. The first builds Claude's reference for this layer: it reproduces run 7, then trains a small correction network on the exact gradient of r_t and runs the fair test (joint condition, composition, composition plus correction) at overlap 0.05 and 0.001, with exact and with trained experts. The second builds quizzes 9.1 and 9.2.

**Prompt:** [the layer 9 reference](prompts/09-0-the-reference-build.md)

**Prompt:** [quizzes 9.1 and 9.2](prompts/09-0-quizzes-9-1-and-9-2.md)

⏱ AI ~2.5 h for both, the training included (estimate), beside the prior-art read

### 1. You: quiz 9.1

Quiz 9.1 checks the college maths under this layer: least-squares regression recovers a conditional expectation, and the gradient of a scalar function.

Quiz 9.1: (URL lands here)

⏱ you 20 to 40 min · stop: the bar, every *needed now* question right and at least 80% of the rest; below that, read the items your misses point at and retake.

### 2. You: the maths drip

You derive, one rung per message, why training with a squared loss is a regression onto a conditional mean, and why a correction's target is ∇r_t, the part of the joint's score that composition misses. Your quiz 9.1 misses become the first rungs.

**Prompt:** [the maths drip](prompts/09-2-the-maths-drip.md)

⏱ rough: you 30 to 60 min · AI ~10 min · stop: you write the regression and its target without notes

**Outcome:** you can write the regression a correction network solves, and say why its target is ∇r_t.

### 3. You: quiz 9.2

Quiz 9.2 checks the layer itself: score matching as regression, why the score residual is ∇r_t, the fair test's three arms, and why trained experts cost more under a better sampler, with banked problems.

Quiz 9.2: (URL lands here)

⏱ you 30 to 60 min, the banked problems included · stop: the bar, then the build. Below the bar, the misses become the first rungs of a second `/drip --math`, and you retake before building.

### 4. You: the build, walk09.py

You type the correction and the fair test into `walk09.py`, cut into about six pieces. First you reproduce run 7's eight cells, so you know your experts and sampler match the reference; then you train the correction two ways (regression on the exact ∇r_t, and denoising score matching with the experts frozen) and print the fair-test table.

<a href="../../../ideas/when-poe-reaches-a-and-b/runs/sampler-by-expert-interaction/samples-2x2-r0.001.png"><img src="../../../ideas/when-poe-reaches-a-and-b/runs/sampler-by-expert-interaction/samples-2x2-r0.001.png" width="420" alt="Composed samples on the four-blob toy at overlap r = 0.001, exact experts on the top row and trained experts below, reverse diffusion on the left and annealed MALA on the right: exact MALA fills the top-right ring at TV 0.125 against a floor of 0.124, trained MALA sits in the same ring but off-centre at TV 0.391, and both reverse-diffusion panels trail toward the origin at TV 0.680 and 0.715"></a>

This is run 7, the picture your reproduction has to match before the correction goes in. Look at the bottom right: trained MALA puts a 0.989 share in the top-right ring, the answer key's value, but its centre is (0.441, 0.523) and its total variation is 0.391 against a floor of 0.124. The share alone hides that miss, which is why the fair test reports the share, the total variation and the centre together.

**Prompt:** [the build](prompts/09-4-the-correction-build.md)

⏱ rough: you 1 to 2 h (about 6 pieces) · AI ~35 min, plus about 1 h of training beside you (estimate) · stop: the prompt's bar

**Outcome:** the fair test's answer. It also shows why the correction has to be accurate at low noise: a better sampler makes expert error cost more, because the trained scores are worst at low noise (relative error `1.30`), exactly where MALA spends its last steps.

### 5. You: compare your numbers with the reference

**Compare.** `python -m rebuild.walk_numbers` prints the `p9.` keys; your `walk09.py` prints the same total variations within the sampling noise the reference reports.

```bash
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m rebuild.walk_numbers | grep 'p9\.'
grep -A 9 '^\*\*r = ' ../artifacts/ideas/when-poe-reaches-a-and-b/runs/sampler-by-expert-interaction/README.md
cd ~/Playground/PhD/simple-distributions-walk
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python walk09.py
```

⏱ you ~15 min · stop: your eight reproduction cells sit within 2 sd of the run card's table (exact MALA at r = 0.001 inside `+0.001 ± 0.012`, trained MALA inside `+0.267 ± 0.112`), and each fair-test cell sits within 2 sd of its `p9.` value. A cell outside is either a lesson or a bug in one of the two files; find which before layer 12.

### 6. AI, in the background: the scene

Claude builds a scene from your fair-test table: total variation above the floor for each of the three arms at overlap 0.05 and 0.001, exact and trained experts, with the samples behind each cell. It keeps its own copy of every number, since your walk folder is scratch space.

**Prompt:** [the scene](prompts/09-6-the-fair-test-scene.md)

Scene: (URL lands here)

⏱ you ~15 min · AI ~30 min, beside layer 12's gate · stop: the scene is published and its numbers match your walk's

### 7. You: close the layer

Run the `/drip-retro` command the build printed at its close (about 25 minutes). Then paste [the linking prompt](prompts/09-7-link-the-layer.md), which links the maths drip, the build's step file and the retrospective to the textbook chapters, the brick row, claims 4 and 9 of the when-poe idea map, and run 7's card.

⏱ you 2 min to paste · AI ~10 min (estimate)

---

⬅️ [Correction prior art](08a-correction-prior-art.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 10: latent space](10-latent-space-and-vaes.md) ➡️
