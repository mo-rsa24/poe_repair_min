# ⭐ Layer 1: one diffusion expert

⬅️ [Setup](00-setup.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 2: composing](02-composing-two-experts.md) ➡️

**Why it is critical.** Everything later composes experts, so you need one expert you built and understand: what it learns (the score) and how it is trained (the denoising loss).

**Depth.** Build: the forward process, the ε-score identity, and training and sampling the expert. Link only: the Skip list below. You 3.3 to 6.7 h in all.

**Study first.** The forward process and the schedule: [the schedule and one reverse step](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/10-schedule-and-reverse-step.md) and the textbook's [forward process by hand](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/09-the-forward-process-by-hand.md). Noise prediction as the score: [noise prediction is the score](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/11-noise-prediction-is-the-score.md). The energy: [a scalar energy whose gradient is the noise prediction](../../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/34-a-scalar-energy-whose-gradient-is-the-score.md) and the vault note [The Energy Parameterisation](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/The%20Energy%20Parameterisation.md). The big picture: [Diffusion Models](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Diffusion%20Models.md) and [Multivariate Normal Distribution](../../../../../../Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Multivariate%20Normal%20Distribution.md). The [step file](../01-one-expert-from-scratch.md) links a textbook page and a timed video segment beside each fundamental it uses.

**Achieve.** `walk01.py`, run end to end: its noise prediction equals 2x at initialisation, and after the full 15,001 steps the bar dataset's total variation sits within 0.10 of its floor · on the whiteboard: one point x_0 noised at three values of ᾱ_t, with the direction ε points · worked: the 41 phase 1 numbers `walk_numbers` prints · problems: the ones quiz 1.2 banks · quizzes 1.1 and 1.2 passed.

## The flow

```text
 AI, in the background                         │  You, in order
───────────────────────────────────────────────┼───────────────────────────────────────────────
 quizzes 1.1 and 1.2, prepared ──────────────▶ │  1. quiz 1.1: Gaussian sums, scores
   (pasted at setup)                           │       ▼
                                               │  2. /drip --math: forward process, ε = score
                                               │       ▼
                                               │  3. quiz 1.2: needed-now right, 80% of the rest
                                               │       ▼
                                               │  4. build: walk01.py, steps 0 to 11
 rebuild/ reference, built ──────────────────▶ │  5. compare: p1. numbers equal yours
                                               │       ▼
 math-scene, beside layer 2 ◀───────────────── │  6. scene
                                               │       ▼
 xref-cadence links the layer ◀─────────────── │  7. /drip-retro, then layer 2
```

## The steps

### 1. You: quiz 1.1

Quiz 1.1: https://claude.ai/artifact/LsK1CitQkbAKZLNC3aB8u8

Built by the setup's background prompt. ⏱ you 20 to 40 min · stop: every *needed now* question right and at least 80% of the rest; below that, read the items your misses point at and retake.

### 2. You: the maths drip

```text
/drip --math the forward process, the cosine schedule, and why the noise prediction is the score
rescaled, as steps 4, 5 and 8 of
artifacts/drips/simple-distributions-rebuild/01-one-expert-from-scratch.md set them out

Starts from: quiz 1.1 passed; any question it marked missed becomes a first rung.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The forward process runs T = 100 steps; ᾱ_t = ∏(1 − β_j) from the cosine schedule with offset
  0.008, and q(x_t | x_0) = N(√ᾱ_t x_0, (1 − ᾱ_t) I). Code index t = 0 is the first noisy state, so
  ᾱ_t equals the schedule's boundary value C_{t+1} and ᾱ_0 = 0.9994 (step file 01, step 4).
- Two identities, kept apart. For one clean point x_0 and one sampled ε,
  ε = −√(1 − ᾱ_t)·∇log q(x_t | x_0): the conditional score. Averaged over the clean points given
  x_t, the squared-error optimum is E[ε | x_t] = −√(1 − ᾱ_t)·∇log q_t(x_t): the marginal score,
  which the trained network learns (step file 01, steps 5 and 8; the Codex mathematics review of
  2026-10-18).
- python -m rebuild.walk_numbers prints ᾱ_t and √(1 − ᾱ_t) at t = 0, 25, 50, 75 and 99 as the
  p1.abar_* and p1.sqrt_1m_abar_* keys; these are the worked numbers
  (rebuild/out/walk-numbers.json).
- The worked instance checks the conditional identity: clean point (0, 0.5), ε = (0.3, −1.2),
  t = 50, √(1 − ᾱ_50) = 0.7223, x_50 = (0.2167, −0.521); p1.example_score and
  p1.example_minus_eps_over_sigma are both (−0.4153, 1.6613) (rebuild/out/walk-numbers.json).
- The cosine schedule's derivation is out of scope; the schedule is given (this layer's Skip list).
Outputs: the drip's thread folder under artifacts/drips/, with dispatch.md recording my notation,
the worked numbers I used and what clicked.
Target: without notes, I derive the sampled-noise identity from q(x_t | x_0), then use posterior
averaging to derive the marginal identity E[ε | x_t] = −√(1 − ᾱ_t)·∇log q_t(x_t) for the trained
predictor, and reproduce the (−0.4153, 1.6613) conditional instance by hand; I judge.
Hand-off: the thread folder's path, which quiz 1.2, step 6's scene and step 7's cross-links read.
```

added: the schedule's constants and the worked instance (from `rebuild/out/walk-numbers.json` and step file 01) · quiz 1.1's misses as first rungs · step 8 and the conditional against marginal identities, with the index shift ᾱ_t = C_{t+1} (from the Codex mathematics review, numbers rerun)
dropped: nothing

⏱ you 45 to 90 min · AI ~10 min

**Outcome:** you can write q(x_t | x_0) = N(√ᾱ_t x_0, (1 − ᾱ_t) I), derive the sampled-noise identity ε = −√(1 − ᾱ_t) · ∇log q(x_t | x_0), and explain why squared-error training learns the rescaled marginal score, E[ε | x_t] = −√(1 − ᾱ_t) · ∇log q_t(x_t). Every later layer uses the marginal identity.

### 3. You: quiz 1.2

Quiz 1.2: https://claude.ai/artifact/HMsmKatwrxvhUL2Q72uTWa

⏱ you 45 to 90 min, the banked problems included · stop: every *needed now* question right and at least 80% of the rest. Below that, the misses become the first rungs of a second `/drip --math`, and you retake before building.

### 4. You: the build

```text
/drip-execute-plan artifacts/drips/simple-distributions-rebuild/01-one-expert-from-scratch.md

Task: walk the step file one step per message in its order, steps 0 to 11. I type every block myself
into ~/Playground/PhD/simple-distributions-walk/walk01.py and run it from that folder with
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python and
PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy.
Starts from: the setup's folder and environment; quiz 1.2 passed; the layer 1 --math drip folder
under artifacts/drips/, whose dispatch.md holds my notation.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The reference is rebuild/datasets.py, rebuild/model.py and rebuild/train.py; every worked number
  prints from python -m rebuild.walk_numbers, run from toy/, as 41 p1. keys
  (REBUILD_WALK.md, Pins; rebuild/out/walk-numbers.json).
- The ResNet has 565,506 parameters. The notebook prints 565,606 because it counts out_logvar, which
  the beta_forward variance never reads (REBUILD_WALK.md, brick "ResNet score network").
- With zero output layers, the noise prediction equals 2x at initialisation (same brick).
- The noise prediction is the gradient of the energy E = |net(x, t) − x|² (brick "Energy
  parameterisation"), and the schedule is computed in float64 (brick "Cosine schedule and DDPM").
- The full run is 15,001 steps of Adam at 1e-3, batch 1000, with a moving average of 0.999, about 5
  minutes on one card; step 9 builds a short version of 2,000 steps on the bar dataset (brick
  "Training"; step file, step 9).
- A trained expert passes when the total variation between 5,000 model and 5,000 data samples, on 40
  × 40 bins of [−1.5, 1.5]², is within 0.10 of the total variation between two data draws. 9 of 10
  datasets pass. The ring passes only without the last step's noise: 0.179, or 0.075 without it,
  against a floor of 0.049 (brick "Training"; step file, step 11).
- After 2,000 steps the bar dataset reads 0.239 against a floor of 0.098, a miss by design; the full
  run reads 0.115 against 0.095 (step file, step 11).
- A deepcopy of the moving-average model has its gradients switched off, so fine-tuning it trains
  nothing and raises no error (step file, step 9's broken run).
- Train on a card with under 1 GB in use, else on the CPU (step file, step 0; the GPU guard in
  ~/Playground/PhD/poe_repair_min/environment/hpc/execution-protocol.md).
Outputs:
- walk01.py: one script that runs end to end and prints each step's worked number;
- the step file's steps 0 to 11 ticked, and the walk map that holds my place between sittings;
- a samples figure from my trained expert beside datasets-1000-samples-each.png in toy/.
Target: every p1. number walk01.py prints equals walk_numbers to the digits shown; the parameter
count is 565,506; after the full 15,001 steps the bar dataset's total variation is within 0.10 of
its floor. I judge, from the printed lines.
Progress: step 1 prints the normal density values, step 4 the ᾱ_t values, step 6 the parameter
count, step 8 the initial loss on the bar dataset, step 11 the total-variation line.
Proof: python walk01.py prints "bar: TV model <x>  floor <y>" with x − y ≤ 0.10 after the full run,
and the comparison in step 5 of this layer shows no differing number.
Limits: never edit rebuild/ or the step file's code blocks; I type, I do not paste; no commit.
Hand-off: the retrospective the walk closes on, and the /drip-retro command it prints.
```

added: the parameter count and why the notebook's differs, the energy and float64 facts, the training settings, the pass rule with the ring exception, the 2,000-step miss by design, the broken moving-average copy and the GPU guard (all from `REBUILD_WALK.md` and the step file) · the full-run target in place of "samples match the figure", which the 2,000-step walk cannot meet
dropped: "my trained expert's samples match datasets-1000-samples-each.png" as the pass line (why: it has no tolerance; it stays as an output to look at)

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

```text
/math-scene the folder the layer 1 --math drip wrote under artifacts/drips/: the score field of the
ring of eight blurring into one sink as the noise level rises

Starts from: the layer 1 --math drip folder and its dispatch.md (step 2's hand-off).
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The ring of eight is toy_gmm in rebuild/datasets.py: eight Gaussian modes on a circle of radius
  0.5 (rebuild/datasets.py, ring_means).
- The noise levels to show are the ᾱ_t values walk_numbers prints at t = 0, 25, 50, 75 and 99, the
  p1.abar_* keys (rebuild/out/walk-numbers.json).
- Every number on the scene comes from walk_numbers or the drip's dispatch.md, never estimated.
Outputs: one published scene, in my notation from dispatch.md.
Hand-off: write the scene's URL into step 6 of
artifacts/drips/simple-distributions-rebuild/ladder/01-one-diffusion-expert.md, replacing "(URL
lands here)".
```

added: the ring's definition and the five noise levels (from `rebuild/datasets.py` and `walk-numbers.json`) · the URL write-back
dropped: nothing

Scene: (URL lands here)

⏱ you ~20 min to operate it · AI ~45 min, beside layer 2's maths drip

**Outcome:** a scene in your own notation where the ring of eight's score field collapses into one sink as you raise the noise level.

### 7. You: close the layer

Run the `/drip-retro` command the build printed at its close (⏱ about 25 minutes). Then paste this:

```text
In the background, dispatch xref-cadence over the files layer 1 of the composition ladder wrote: the
layer 1 --math drip folder under artifacts/drips/ and the retrospective the layer 1 walk closed on.

Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The reference sets to link against are the journey plans and vault notes on the Study first line
  of artifacts/drips/simple-distributions-rebuild/ladder/01-one-diffusion-expert.md, and the layer 1
  rows of the Bricks table in artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md.
- ~/Playground/PhD/simple-distributions-walk/ is scratch space; link nothing into it (CLAUDE.md, the
  playground is disposable).
Outputs: links written both ways between those files and the reference sets.
Hand-off: the agent's report, listing every link it wrote, read before layer 2 starts.
```

added: the reference sets, by path (from this layer's Study first line and `REBUILD_WALK.md`)
dropped: nothing

⏱ you 2 min to paste · AI ~10 min (estimate)

**Skip:** the ResNet's internal layout, the moving average of weights, and the cosine schedule's derivation. Treat each as a given part.

---

⬅️ [Setup](00-setup.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 2: composing](02-composing-two-experts.md) ➡️
