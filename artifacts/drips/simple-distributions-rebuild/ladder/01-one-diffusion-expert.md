# ⭐ Layer 1: one diffusion expert

⬅️ [Setup](00-setup.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 2: composing](02-composing-two-experts.md) ➡️

**Why it is critical.** Everything later composes experts, so you need one expert you built and understand: what it learns (the score) and how it is trained (the denoising loss). On [the axes](../PROMPTS.md#the-axes-one-new-variable-per-rung), this is rung a: a learned expert, the first variable.

**Depth.** Build: the forward process, the ε-score identity, and training and sampling the expert. Link only: the Skip list at the foot. You 3.3 to 6.7 h in all.

**Study first.** The forward process and the schedule: [the schedule and one reverse step](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/10-schedule-and-reverse-step.md) and the textbook's [forward process by hand](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/09-the-forward-process-by-hand.md). Noise prediction as the score: [noise prediction is the score](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/11-noise-prediction-is-the-score.md). The energy: [a scalar energy whose gradient is the noise prediction](../../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/34-a-scalar-energy-whose-gradient-is-the-score.md) and the vault note [The Energy Parameterisation](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/The%20Energy%20Parameterisation.md). The big picture: [Diffusion Models](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Diffusion%20Models.md) and [Multivariate Normal Distribution](../../../../../../Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Multivariate%20Normal%20Distribution.md). The [step file](../01-one-expert-from-scratch.md) links a textbook page and a timed video segment beside each fundamental it uses.

**Achieve.** `walk01.py`, run end to end: its noise prediction equals 2x at initialisation, and after the full 15,001 steps the bar dataset's total variation sits within 0.10 of its floor · on the whiteboard: one point x_0 noised at three values of ᾱ_t, with the direction ε points · worked: the 41 phase 1 numbers `walk_numbers` prints · problems: the ones quiz 1.2 banks · quizzes 1.1 and 1.2 passed.

## The flow

```text
 AI, in the background                         │  You, in order
───────────────────────────────────────────────┼───────────────────────────────────────────────
 quizzes 1.1 and 1.2 ✅ built ────────────────▶ │  1. quiz 1.1: Gaussian sums, scores
                                               │       ▼
                                               │  2. /drip --math: forward process, ε = score
                                               │       ▼
                                               │  3. quiz 1.2: needed-now right, 80% of the rest
                                               │       ▼
                                               │  4. build: walk01.py, steps 0 to 11
 toy/rebuild/ reference, built ──────────────▶ │  5. compare: p1. numbers equal yours
                                               │       ▼
 math-scene, beside layer 2 ◀───────────────── │  6. scene
                                               │       ▼
 xref-cadence links the layer ◀─────────────── │  7. /drip-retro, then layer 2
```

## The steps

### 1. You: quiz 1.1

[Quiz 1.1](https://claude.ai/artifact/LsK1CitQkbAKZLNC3aB8u8) checks the college maths under this layer: √ᾱ x_0 + √(1 − ᾱ) ε as a Gaussian, the score of a Gaussian, and least squares.

⏱ you 20 to 40 min · stop: every *needed now* question right and at least 80% of the rest; below that, read the items your misses point at and retake.

### 2. You: the maths drip

You derive, one rung per message, how a clean point is noised and why the network's noise prediction is the score rescaled. Your quiz 1.1 misses become the first rungs.

<a href="../../../results/how-the-forward-process-dissolves-an-image/the-dog-at-five-steps.png"><img src="../../../results/how-the-forward-process-dissolves-an-image/the-dog-at-five-steps.png" width="420" alt="The same dog and the same noise draw at t = 0, 25, 50, 75 and 99 of the cosine schedule: the kept share ᾱ falls from 0.9994 to 2.43e-07, and the picture goes from a clear dog to pure noise"></a>

This is the forward process on a real picture: one dog and one noise draw, mixed as dog × √ᾱ_t plus noise × √(1 − ᾱ_t). By t = 50 the dog is still visible at weight 0.69; by t = 99 it is gone. The same arithmetic runs on 2D points in the build. It is also a scene: [Dissolving the Dog](https://claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9).

**Prompt:** [the maths drip](prompts/01-2-the-maths-drip.md)

⏱ you 45 to 90 min · AI ~10 min

**Outcome:** you can write q(x_t | x_0) = N(√ᾱ_t x_0, (1 − ᾱ_t) I), derive the sampled-noise identity ε = −√(1 − ᾱ_t) · ∇log q(x_t | x_0), and explain why squared-error training learns the rescaled marginal score, E[ε | x_t] = −√(1 − ᾱ_t) · ∇log q_t(x_t). Every later layer uses the marginal identity.

### 3. You: quiz 1.2

[Quiz 1.2](https://claude.ai/artifact/HMsmKatwrxvhUL2Q72uTWa) checks the layer itself: the forward process, the two identities, and the denoising loss, with banked problems.

⏱ you 45 to 90 min · stop: every *needed now* question right and at least 80% of the rest. Below that, the misses become the first rungs of a second `/drip --math`, and you retake before building.

### 4. You: the build

You type one 2D diffusion expert into `walk01.py` from an empty file, following [the step file](../01-one-expert-from-scratch.md) one step per message: the datasets, the ResNet, the energy, the schedule, training, and sampling. Claude's reference prints the same numbers.

<a href="../../../../toy/rebuild/out/phase1/datasets-against-trained-models.png"><img src="../../../../toy/rebuild/out/phase1/datasets-against-trained-models.png" width="420" alt="Ten 2D datasets in blue (rows 1 and 3) and 1,000 samples from each trained model in orange (rows 2 and 4), each model panel titled with its total variation and floor, from TV 0.10 against floor 0.09 for the horizontal bar to TV 0.18 against floor 0.05 for the ring"></a>

This is where the build ends up: each orange panel should have its blue panel's shape. The ring (top left) is the one that misses its floor, 0.18 against 0.05, because the sampler adds noise at the last step.

**Prompt:** [the build](prompts/01-4-the-build.md)

⏱ you 1.5 to 3 h (12 steps) · AI ~30 min, including the 5-minute full training run

**Outcome:** your own trained 2D expert. Its noise prediction equals 2x at initialisation.

### 5. You: compare your numbers with the reference

```bash
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m rebuild.walk_numbers | grep 'p1\.'
cd ~/Playground/PhD/simple-distributions-walk
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python walk01.py
```

⏱ you ~10 min · stop: the 41 `p1.` lines from the first command and the numbers your script prints agree to the digits shown, including `p1.params_total_torch: 565506`. A number that differs is either a lesson or a bug in one of the two files; find which before layer 2.

### 6. AI, in the background: the scene

A scene, in your notation, where the ring of eight's score field collapses into one sink as you raise the noise level. Paste it during layer 2's maths drip.

**Prompt:** [the scene](prompts/01-6-the-scene.md)

Scene: (URL lands here)

⏱ you ~20 min to operate it · AI ~45 min, beside layer 2's maths drip

### 7. You: close the layer

Run the `/drip-retro` command the build printed at its close (about 25 minutes). Then paste [the linking prompt](prompts/01-7-link-the-layer.md), which links what this layer wrote to the journey plans and notes that teach it.

⏱ you 2 min to paste · AI ~10 min (estimate)

**Skip:** the ResNet's internal layout, the moving average of weights, and the cosine schedule's derivation. Treat each as a given part.

---

⬅️ [Setup](00-setup.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 2: composing](02-composing-two-experts.md) ➡️
