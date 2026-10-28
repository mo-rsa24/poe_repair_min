# ⭐ Layer 12: the correction at scale, and the chest X-ray question

⬅️ [Layer 11: the testbed](11-the-text-conditioned-testbed.md) · 🏠 [the guide](../PROMPTS.md)

**Before this: layer 11b, curated cats and dogs.** It is the rung between the testbed and SDXL: real photos, with the share of captions naming both animals set by hand. It has no page yet; [what the evidence means for rung 11b](../../../notes/what-poe-loses-when-it-composes/evidence-poe-fails-at-scale.md#what-this-means-for-rung-11b) holds its design.

**Why it is critical.** It is the goal. On [the axes](../PROMPTS.md#the-axes-one-new-variable-per-rung), this is rung k: SDXL itself, taking the correction from rung j. Three facts from the medical sessions set its order:

- **Drainless pneumothorax is not rare in NIH.** 1,906 of the 3,543 NEATX pneumothorax films have no drain, 54%, so any overlap dial on real films is deliberate subsampling.
- **The drain shortcut is not yet shown in this project.** TorchXRayVision's DenseNet moves a median of `+0.001` between drain and drain-free films of the same patient over 237 pairs, and it scores the drain film higher in 134 of them.
- **RadEdit removes the pneumothorax and keeps the drain.** Only FastDiME removes and adds drains.

So the drain-gap test is a gate: a gap under 5 points, or reversed, ends the drain case. Composition's own medical job may then be a shortcut that cannot be masked, such as AP against PA view or the scanner.

**Depth.** Build: the drain-gap measurement and the correction on the real model. Own: [LoRA](../../../../../../Obsidian/Artificial%20Intelligence/Large%20Language%20Models/LoRA.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/ai-engineering/ai-engineering.pdf#page=649 "AI engineering, chapter 7, LoRA, Works the parameter saving with numbers (a 9 × 9 matrix of 81 parameters against a 9 × 1 and a 1 × 9 factor of 18), gives th..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/natural-language-processing/hands-on-large-language-models.pdf#page=497 "Hands-on large language models, chapter 12, LoRA, Rebuilds a 10 × 10 matrix from two smaller ones, 20 weights in place of 100, and why language models' low i..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/natural-language-processing/hands-on-large-language-models.pdf#page=503 "Hands-on large language models, chapter 12, Instruction tuning with QLoRA, Fine-tunes TinyLlama in code with a 4-bit base and LoRA adapters, explains the ran..."), [▶1](https://www.youtube.com/watch?v=_kREM2UAiJ8&t=3738s "CS229 lecture 12, 1:02:18 to 1:08:24, Works the count with numbers: W₀ + AB with inner rank r cuts the free parameters from d² to 2dr, at d = 1000 and r = 10..."), [▶2](https://www.youtube.com/watch?v=VlA_jt_3Qc4&t=5868s "CME295 lecture 4, 1:37:48 to 1:44:56, Freeze W₀ and learn BA with rank about 4 to 10, one pair per task; it helps most in the feed-forward layers, wants abou..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/latent-diffusion-architecture-rebuilt/plans/08-lora-update.md "The low-rank update, B times A, deep-learning/diffusion-models/latent-diffusion-architecture-rebuilt, plan 08. Derives the parameter count r · (in + out) for..."), [◈](https://claude.ai/artifact/1GayZthQkRL4FmukdCV9WW#nb-lora "The SDXL notebook in ten parts, LoRA repair, Low-rank adapters on the cross-attention projections, trained against the single-prompt teacher.")</sub> as a low-rank update. The plans below set their own depth beyond that. You 8.3 to 16.2 h for the parts that run whatever the gate decides.

**Study first** for quizzes 12.1 and 12.2. [The LoRA update worked on paper](../../../../../../goal-setting/learning/deep-learning/diffusion-models/latent-diffusion-architecture-rebuilt/plans/08-lora-update.md) and [the SVD of the real matrix](../../../../../../goal-setting/learning/deep-learning/diffusion-models/08-spectral-structure-of-the-correction/plans/06-svd-of-the-real-matrix.md). Textbooks: Anton and Rorres, *Elementary Linear Algebra* 4.8 ([rank](../../../../../../goal-setting/learning/mathematics/textbooks/linear-algebra/linear-algebra-textbook.pdf#page=263), p262) and 9.4 to 9.5 (the [SVD](../../../../../../Obsidian/Mathematics/Linear%20Algebra/Singular%20Value%20Decomposition.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=125 "Mathematics for machine learning, section 4.5, Works an SVD by hand (Example 4.13, page 131) after its geometric picture (Example 4.12) and its construction ..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/linear-algebra/linear-algebra-textbook.pdf#page=528 "Linear algebra textbook, section 9.4, Singular values, then the SVD in brief and expanded form, with Example 2 for a matrix that is not square."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/linear-algebra/linear-algebra-textbook.pdf#page=535 "Linear algebra textbook, section 9.5, The reduced SVD, and compressing an image by keeping only the largest singular values."), [▶1](https://www.youtube.com/watch?v=HZl32cDhWEM&t=1007s "tilestats, PCA and SVD: the math explained simply, The singular value decomposition, 16:47 to 22:59, The SVD as the second way to compute PCA, after the eige..."), [▶2](https://www.youtube.com/watch?v=FgakZw6K1QQ&t=728s "StatQuest, Principal Component Analysis (PCA), Step-by-Step, Singular vector/value, Eigenvector/value and loading scores defined, 12:08 to 12:56, Names the s..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/08-spectral-structure-of-the-correction/plans/06-svd-of-the-real-matrix.md "SVD of your own 440 x 65,536 matrix, deep-learning/diffusion-models/08-spectral-structure-of-the-correction, plan 06. Stacks cached correction rows and runs ...")</sub> and compression, p528 to p535); Deisenroth et al. 4.5 to 4.6 (the SVD and [matrix](../../../../../../Obsidian/Mathematics/Linear%20Algebra/Matrix.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/linear-algebra/linear-algebra-textbook.pdf#page=39 "Linear algebra textbook, section 1.3, Matrix notation, then the row-by-column product worked on a 2 by 3 times a 3 by 4 matrix (Example 5, page 43)."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=28 "Mathematics for machine learning, section 2.2, Matrix addition and multiplication, inverse and transpose, and a system of linear equations written compactly ..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/deep-learning-textbook.pdf#page=50 "Deep learning textbook, section 2.2, The matrix product, the element-wise product and its properties, as deep learning uses them."), [▶1](https://www.youtube.com/watch?v=QtAZsWseIKk&t=511s "tilestats, Matrices and matrix operations for beginners 1/2, Matrix multiplication, 8:31 to 12:27, Matrix multiplication for beginners, followed in the next ..."), [▶2](https://www.youtube.com/watch?v=kYB8IZa5AuE&t=212s "3Blue1Brown, Linear transformations and matrices (Essence of linear algebra, chapter 3), Basis vectors and matrices, 3:32 to 6:18, A matrix read as where the..."), [◆](../../../../../../goal-setting/learning/deep-learning/transformers/textbook/03-multiply-two-matrices-row-by-column.md "Multiply two matrices row by column and say why the order matters, deep-learning/transformers, chapter 03. The same two matrices multiplied in both orders, t...")</sub> approximation, p125 to p135). Video: [Strang's MIT 18.065 lecture 6](https://www.youtube.com/watch?v=rYz83XPxiZo&t=468s).

**Ideas to tie in.** From [the ideas that tie the ladder together](../../../ideas/ideas-that-tie-the-ladder-together.md): [the interaction score as the object a correction can learn](../../../ideas/ideas-that-tie-the-ladder-together.md#5-learn-the-identifiable-object-which-is-the-interaction-score) (needed, 2 to 3 h, no training); [the interaction's own diffusion-time trajectory](../../../ideas/ideas-that-tie-the-ladder-together.md#6-give-the-interaction-its-own-diffusion-time-trajectory) (needed, 2 to 4 h, no training); [whether the chest target is observational or counterfactual](../../../ideas/ideas-that-tie-the-ladder-together.md#8-specify-whether-the-chest-target-is-observational-or-counterfactual) (needed before the chest experiment, 2 to 4 h). Each comes with an exact toy check.

**Achieve.** The drain gap with its interval · a transfer plan whose terms are defined · the LoRA's effect on compose rate on held-out pairs beside the interaction term per pair · on the whiteboard: a rank-1 update to a 2 × 2 weight matrix · quizzes 12.1 and 12.2 passed. This is the guide's last Achieve line.

## The flow

```text
 AI, in the background                         │  You, in order
───────────────────────────────────────────────┼───────────────────────────────────────────────
 1. quizzes 12.1 and 12.2, prepared ─────────▶ │  2. the drain-gap measurement
   (paste during layer 9's build)              │       ▼
                                               │  3. /drip-idea, claim 8: which shortcut
                                               │       ▼
                                               │  4. only if the gap passes: /frame-hypothesis
                                               │       ▼
                                               │  5. /drip-refine-plan: the transfer plan
                                               │       ▼
                                               │  6. quiz 12.1, 7. LoRA maths, 8. quiz 12.2
                                               │       ▼
                                               │  9. build: the LoRA on held-out pairs
 layer 9's correction on the toys ───────────▶ │ 10. compare: the direction it predicts
                                               │       ▼
 xref-cadence links the layer ◀─────────────── │ 11. /drip-retro, and the ladder is done
```

## The steps

### 1. AI, in the background: layer 12's quizzes

When you start layer 9's build, paste the prompt so both quizzes are waiting here. Quiz 12.1 covers matrix rank, a low-rank factorisation and its parameter count, and the SVD at gloss depth. Quiz 12.2 covers LoRA as a low-rank weight update, why its residual is layer 8's ε-space interaction term, and compose rate as the measure.

**Prompt:** [build quizzes 12.1 and 12.2](prompts/12-1-build-the-two-quizzes.md)

⏱ AI ~1.5 h (estimate), beside layer 9's build

### 2. You: the gate first, the drain-gap measurement

You resume the paused drain-gap walk in the medical-imaging project (B1's plan `01-measure-the-drain-gap.md`) up to task 10.1. You come out with the difference in classifier sensitivity between pneumothorax films with a chest drain and films without one, with its 95% interval. A gap of at least 5 points with the interval excluding 0 lets the drain case continue; anything else ends it or leaves it inconclusive.

**Prompt:** [the drain-gap measurement](prompts/12-2-measure-the-drain-gap.md)

⏱ you ~4 h (task groups 0 to 10, about 8 at 30 minutes each) · cluster: five fold models, hours unknown until task 8.2 times one [epoch](../../../../../../Obsidian/Artificial%20Intelligence/Machine%20Learning/Epoch.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/ai-engineering/ai-engineering.pdf#page=686 "AI engineering, chapter 7, Number of epochs, Puts numbers on the choice: 1 to 2 epochs may be enough for millions of examples, 4 to 10 can still help for tho..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/understanding-deep-learning.pdf#page=99 "Understanding deep learning, section 6.2.1, Batches are drawn without replacement until the data is used up, and a single pass through the entire training se..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/deep-learning-textbook.pdf#page=262 "Deep learning textbook, section 7.8, Learning curves over 250 epochs of a network on MNIST, where training loss keeps falling and validation loss turns back ..."), [▶1](https://www.youtube.com/watch?v=Q0TtSM9aRew&t=1080s "TileStats, Stochastic gradient descent (SGD) vs mini-batch GD, iterations vs epochs, Epochs vs iterations, 18:00 to 20:00, The closing segment that separates..."), [▶2](https://www.youtube.com/watch?v=dyNGd06MWn4&t=2189s "CS231n lecture 3, 36:29 to 43:01, Defines stochastic gradient descent on random minibatches of 256 examples and the epoch, then its failure modes. Caption-read.")</sub>, as the plan itself says · box 8 h of your time · stop: task 10.1 prints the gap with its interval, and instruction 11.1 records the verdict

**Outcome:** the drain gap with its interval, which decides whether the drain case continues. Its answer keys are that plan and the README of `pneumothorax-score-per-film.csv` in the same project.

### 3. You: then the open questions, in one walk

You walk claim 8 of the when-poe idea map with the drain-gap result in hand, one question per round: which shortcut leads (the drain, or one that cannot be masked, such as view or scanner); whether NIH's drainless pneumothoraces are mostly large; whether FastDiME trains classifiers on its edits; and how many of the 4,312 follow-up films with no pneumothorax label still carry a drain.

**Prompt:** [the claim 8 walk](prompts/12-3-walk-claim-eight.md)

⏱ you ~1.5 h (four questions, two of them counted from data) · AI ~30 min · box 3 h · stop: the shortcut to target is named, and the drain case is marked as standing or ended. This walk runs whatever the gate's verdict is.

**Outcome:** the shortcut composition should target, and whether the drain case stands.

### 4. You: then the experiment, only if the gate passes

Only if step 2 recorded a gap, you sharpen claim 8's training arms into a falsifiable plan: the toy's overlap-dial curve should predict the share of real drainless films below which composition recovers the cell and joint conditioning does not. The compile is saved to `artifacts/ideas/when-poe-reaches-a-and-b/HYPOTHESIS.md`.

**Prompt:** [frame the hypothesis](prompts/12-4-frame-the-hypothesis.md)

⏱ untimed: it runs only if the gate passes, and its size depends on which shortcut the walk above picks.

**Outcome:** a falsifiable plan for the thesis chapter. The baselines to beat:

- FastDiME: AUROC `0.98` on films that keep the drain association and `0.65` without it.
- Kina and Petersen's prevalence recalibration: `0.226` to `0.726`.

The interval at 100 films against 100 is about ±0.07.

### 5. You: then the transfer plan, repaired before it runs

You repair poe_repair_min's transfer plan before it runs, one question per piece: whether compose rate is binary per pair or a fraction over seeds, which 15 pairs it means (its pool file lists 17), what its x axis becomes when every run holds out exactly one pair, and what count out of 15 "most" means.

**Prompt:** [repair the transfer plan](prompts/12-5-repair-the-transfer-plan.md)

⏱ you ~1 h (the one measured `drip-refine-plan` took 66 minutes, too few runs to trust) · AI ~30 min · box 1.5 h · stop: each of the four questions has a written answer in the plan

**Outcome:** a transfer plan whose compose rate, pair set and curve are defined before any run, so the last bar of this layer measures something.

### 6. You: quiz 12.1

The rest of the layer puts the correction on the real model, and quiz 12.1 opens it: matrix rank, a low-rank factorisation and its parameter count, and the SVD at gloss depth.

Quiz 12.1: (URL lands here)

Built by step 1. ⏱ you 20 to 40 min · stop: every *needed now* question right and at least 80% of the rest; below that, read the items your misses point at and retake.

### 7. You: the maths drip

You derive, one rung per message, a LoRA as a frozen weight plus a trained rank-r update, and why the residual poe_repair_min's LoRA trains on is 7.5 times layer 8's ε-space interaction term. Your quiz 12.1 misses become the first rungs.

The SDXL notebook scene draws one denoising step with this correction in it, in three parts. Its [PoE part](https://claude.ai/artifact/1GayZthQkRL4FmukdCV9WW#nb-poe) composes the two guided branches as eps_guided_a + eps_guided_b − eps_empty, the prediction whose gap to the joint prompt's you derive here. Its [LoRA part](https://claude.ai/artifact/1GayZthQkRL4FmukdCV9WW#nb-lora) puts low-rank adapters on the cross-attention projections and trains them against the joint prompt's guided prediction, the update this drip counts the parameters of. Its [inference part](https://claude.ai/artifact/1GayZthQkRL4FmukdCV9WW#nb-infer) adds the adapter's correction to the composed prediction and hands the sum to DDIM, the sampling path step 9 scores. The scene shows no output pictures, because `notebooks/sdxl_crossattn_poe_lora.ipynb` in `~/PhD/SDXL` ran under Matplotlib's Agg backend; rerun it in Jupyter with `%matplotlib inline` to see them.

**Prompt:** [the maths drip](prompts/12-7-the-maths-drip.md)

⏱ you 10 to 20 min, AI ~10 min

**Outcome:** the LoRA understood as a learned interaction term.

### 8. You: quiz 12.2

Quiz 12.2 checks the layer itself: LoRA as a low-rank weight update, why its residual is the ε-space interaction term, and compose rate as the measure.

Quiz 12.2: (URL lands here)

Built by step 1. ⏱ you 20 to 40 min · stop: every *needed now* question right and at least 80% of the rest. Below that, the misses become the first rungs of a second `/drip --math`, and you retake before building.

### 9. You: the build

You type `walk12.py`, cut into 4 to 6 pieces. It reads poe_repair_min's cached ε_joint, ε_A, ε_B and ε_∅ for the 25 animal pairs, never writing to that repo, and prints one row per held-out pair: compose rate without and with the correction, beside the pair's interaction term.

**Prompt:** [the build](prompts/12-9-the-build.md)

⏱ build (rough): you 1 to 2 h, AI ~35 min plus GPU [sampling](../../../../../../Obsidian/Mathematics/Statistics/Sampling%20From%20A%20Distribution.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=539 "Pattern recognition and machine learning, chapter 11, Why expectations need samples, ancestral sampling of a directed graph (page 541), standard distribution..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=245 "Introduction to probability for data science, section 4.8, Generates random numbers for any PDF by running a uniform through the inverse CDF, in Python, with..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/deep-learning-textbook.pdf#page=606 "Deep learning textbook, section 17.1, Why machine learning samples at all (Monte Carlo estimates of sums and integrals, and sampling as the goal of a generat..."), [▶1](https://www.youtube.com/watch?v=9ixzzPQWuAY&t=474s "ritvikmath, Inverse Transform Sampling, 7:54 to 10:04, Inverts the exponential CDF and turns a uniform draw into an exponential one. Chapter-timed, from the ..."), [▶2](https://www.youtube.com/watch?v=OXDqjdVVePY&t=232s "ritvikmath, Accept-Reject Sampling, 3:52 to 7:45, An easy proposal g scaled by M into an envelope above f, then each draw accepted with probability f/(M g). ..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/textbook/l3-stochastic-processes/13-drawing-samples-by-hand.md "Drawing samples by hand: Monte Carlo, inverse transform and accept-reject, deep-learning/diffusion-models/00-from-scratch-to-sdxl, volume L3, section 13. Thr...")</sub> time that the walk's first timed pair sets · stop: the build's bar

**Outcome:** the LoRA understood as a learned interaction term, measured on held-out animal pairs. `inf_sample.py` with `anneal_samplers.py` is the repo's image-scale [MCMC](../../../../../../Obsidian/Mathematics/Statistics/Markov%20Chain%20Monte%20Carlo/Markov%20Chain%20Monte%20Carlo.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=553 "Pattern recognition and machine learning, section 11.2, The Metropolis algorithm with a symmetric proposal, drawn as accepted and rejected steps on a 2D Gaus..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/probabilistic-machine-learning-advanced-topics.pdf#page=526 "Probabilistic machine learning: advanced topics, sections 12.1 to 12.2, The Metropolis-Hastings algorithm (Algorithm 12.1), why it works, and the choice of p..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/deep-learning-textbook.pdf#page=611 "Deep learning textbook, section 17.3, MCMC for energy-based models: running many chains in parallel, the distribution q(t) over their states converging to th..."), [▶1](https://www.youtube.com/watch?v=KmqTrm-bn8k&t=141s "ritvikmath, Coding MCMC, 2:21 to 5:50, One chain written in Python, at its accept or reject step. Chapter-timed, from the video lessons index."), [▶2](https://www.youtube.com/watch?v=yApmR-c_hKU&t=247s "ritvikmath, Markov Chain Monte Carlo, 4:07 to 8:24, Drawing samples by running a Markov chain whose long-run distribution is the target. Chapter-timed, from ..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/textbook/l3-stochastic-processes/03-metropolis-hastings-and-why-z-cancels.md "3. Metropolis-Hastings, and why Z cancels, deep-learning/diffusion-models/00-from-scratch-to-sdxl, volume L3, section 3. Why a product of densities can be sa...")</sub> composition code for this step.

### 10. You: compare with layer 9's correction

**Compare.** The layer 9 correction on the toys is the reference behaviour: the per-pair effect should move in the direction the toy's fair test predicts for the pair's interaction term.

```bash
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m rebuild.walk_numbers | grep 'p9\.'
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -c "import json, numpy as np; p = json.load(open('/datasets/mmolefe/poe_repair_min/outputs/showcase/interaction_bound/interaction_bound.json'))['pairs']; [print(k, v['fail_rate'], round(float(np.mean([v['median_bound_by_step'][str(s)] for s in range(10, 41)])))) for k, v in sorted(p.items())]"
cd ~/Playground/PhD/simple-distributions-walk
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python walk12.py
```

⏱ you ~15 min · stop: the `p9.` lines (they exist once layer 9's background prompt has built the reference) say how composition plus a correction did against composition alone at each overlap; the 25 bound lines fall in the two bands, roughly 100 to 400 and 1,300 to 3,200; your table's interaction column matches them pair for pair. Then read the direction: write down, before looking at your table, which way the `p9.` lines say a pair's compose rate should move as its interaction term grows, and check each held-out pair against it. A pair that moves the other way is the finding to write down.

**Caveats carried from earlier layers:**

- Samplers recover only a third to four fifths of the exact mass at 2% to 10% overlap (layer 5).
- Trained experts plus a better sampler cost more at low overlap (layer 9).
- The interaction term's lower bound did not track which of the 25 pairs blend in `poe_repair_min`.

### 11. You: close the layer

Run the `/drip-retro` command the build printed at its close (about 25 minutes). Then paste [the linking prompt](prompts/12-11-link-the-layer.md), which links the maths drip, claim 8, the hypothesis if step 4 ran, and the retrospective to the LoRA and SVD journey plans, the brick row, and the drain-gap and transfer plans. This is the ladder's last layer.

⏱ you 2 min to paste · AI ~10 min (estimate)

---

⬅️ [Layer 11: the testbed](11-the-text-conditioned-testbed.md) · 🏠 [the guide](../PROMPTS.md)
