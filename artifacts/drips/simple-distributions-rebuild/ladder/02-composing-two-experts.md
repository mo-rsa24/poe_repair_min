# ⭐ Layer 2: composing two experts, and the samplers that reach the composition

⬅️ [Layer 1: one expert](01-one-diffusion-expert.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 3: ideal experts](03-ideal-experts-and-the-seven-kinds.md) ➡️

**Why it is critical.** The two constructions and the sampler problem both start here. Every later result reports which construction and which sampler it used.

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

Paste this when you start layer 1's build, so both quizzes are waiting here.

```text
In the background, prepare and build layer 2's two quizzes, bottom up, for the composition ladder
in artifacts/drips/simple-distributions-rebuild/: quiz 2.1 and quiz 2.2.

Task: for each quiz, first dispatch prep-cadence (quiz-prep-pulse --build) with the goal and minutes
below, then dispatch quiz-cadence (mode build, repo
/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain). The goal for both is "compose two experts and
use an accept-corrected sampler whose fixed kernel has the composition as its invariant density".
- Quiz 2.1, --minutes 40: the college maths under layer 2. Normalising constants, Markov chains and
  the Metropolis-Hastings accept ratio, total variation distance, a Langevin step as gradient ascent
  plus noise.
- Quiz 2.2, --minutes 130: layer 2 itself, from the step file 02-composing-and-sampling.md. The two
  constructions, why summed scores miss the product, MALA's proposal divisor 4·ss, the three numbers
  every composition reports.
Starts from: nothing from this layer; it runs during layer 1's build, so the layer 2 --math drip has
not run and the notation is the step file's. Quiz 2.1 takes its material from the Study first line
of artifacts/drips/simple-distributions-rebuild/ladder/02-composing-two-experts.md; quiz 2.2 takes
it from artifacts/drips/simple-distributions-rebuild/02-composing-and-sampling.md.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- quiz-cadence stops when a quiz has no coverage file in .quiz-prep/, so prep-cadence runs first for
  each quiz (quiz-cadence, "The preparation gate").
- The step file links a textbook page beside each fundamental it uses and carries timestamped video
  tables on the product of experts, the product of two Gaussians, total variation distance, Langevin
  sampling, Metropolis-Hastings and annealed Langevin; reuse them as nodes before searching again
  (step file 02; the guide, "When a step stalls on the maths").
- No timed video segment uses a Langevin step as the Metropolis-Hastings proposal (MALA) or reads
  the share of proposals accepted (step file 02, step 6's video table; the guide).
- Total variation distance has no passage that teaches it yet; plan 23, "what r_t costs the
  samples", only writes it from histograms, so quiz 2.1's preparation writes the missing passage
  (this layer's Study first line).
- The textbook for the accept ratio is Bishop, Pattern Recognition and Machine Learning 11.2.2,
  p557 (this layer's Study first line).
- pdftotext is not installed on mscluster, so
  ~/.claude/skills/index-source/scripts/find-in-textbooks.py fails; open the PDFs with PyMuPDF from
  the co3 environment (checked on mscluster109: import pymupdf works in co3, version 1.28.2, and
  import pypdf fails everywhere).
- The shelf index is ~/Obsidian/Mathematics/references/textbooks/Textbooks.md and the
  worked-examples index is ~/Obsidian/Mathematics/references/exercises/Worked Examples.md
  (quiz-prep-pulse, "The bank").
- The notation is the step file's: the product p(x) = p_A(x) p_B(x) / Z; one ULA move
  x' = x + ss·∇log p(x) + √(2ss)·z; the MALA proposal N(x'; x + ss·∇log p(x), 2ss·I), whose
  log-density divides by 4·ss (step file 02, steps 1, 5 and 6).
- Every worked number prints from python -m rebuild.walk_numbers as the 53 p2. keys in
  rebuild/out/walk-numbers.json; the 14 p2.step9_ keys are read from
  rebuild/out/phase2/step9.json, produced by python -m rebuild.phase2_step9. Among them are
  p2.prod_logZ -2.2655, p2.noised_product_mean 0.6931 against p2.product_of_noised_mean 0.2326,
  and p2.mala_fixed_log_alpha 0.045 against p2.mala_old_log_alpha -0.36
  (rebuild/out/walk-numbers.json).
- The three numbers every composition reports are the share in A∩B, the total variation to the
  exact composed density and the normaliser Z (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md,
  claim 5).
Outputs:
- two coverage files under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, each with a
  verdict per question: ready, blocked, or unprepared by choice;
- the missing total-variation passage, written where quiz-prep-pulse --build files passages;
- rows in the worked-examples index for every banked problem;
- two published quiz pages, each opening on a read-first list with minutes per item, a short path
  and a full path.
Target: no question on either page carries the verdict blocked; both quizzes carry banked problems,
about 4 to 6 per concept at the own depth (total variation distance) and 6 to 10 at the build
depth; every number in an answer traces to walk_numbers, the step file or a cited textbook page. I
judge, from the pages.
Progress: prep-cadence reports, per quiz, its question count, node count and verdicts; quiz-cadence
reports each URL as it publishes.
Proof: the two coverage files exist under .quiz-prep/ and list no question as blocked; the two
URLs open.
Limits: change nothing in the step files' prose or in rebuild/; in the ladder files, write only the
two URLs.
Hand-off: write quiz 2.1's URL into step 1 and quiz 2.2's URL into step 3 of
artifacts/drips/simple-distributions-rebuild/ladder/02-composing-two-experts.md, each replacing
"(URL lands here)".
```

added: every field the setup's quiz prompt carries, so this prompt stands alone instead of pointing at "the setup section of PROMPTS.md", which the guide no longer holds (the setup moved to `ladder/00-setup.md`) · the preparation gate (from the quiz-cadence agent definition) · the step file's video tables as ready nodes and the missing MALA segment (from step file 02 and the guide) · the missing total-variation passage (from this layer's Study first line) · the missing `pdftotext` and the PyMuPDF route (from the textbook pass) · the notation and the p2. keys (from step file 02 and `walk-numbers.json`) · the three numbers (from the `when-poe-reaches-a-and-b` idea map, claim 5) · the URL write-back · the goal names an accept-corrected sampler whose fixed kernel leaves the composition invariant, since no finite run is exact (from the Codex mathematics review) · the p2. key count, 53, after the review's numbers became keys (from the step 9 scoring fix and the port of the review's checks into walk_numbers) · the 14 p2.step9_ keys named as read from step9.json, produced by phase2_step9 (from the step 9 scoring fix)
dropped: "a sampler whose answer is exact" from the goal (from the Codex mathematics review: the accept step makes the target invariant for a fixed kernel; it does not make a finite run exact or guarantee mixing)

⏱ AI ~1.5 h (estimate), beside layer 1's build

### 1. You: quiz 2.1

Quiz 2.1: (URL lands here)

⏱ you 20 to 40 min · stop: the bar, every *needed now* question right and at least 80% of the rest; below that, read the items your misses point at and retake.

### 2. You: the maths drip

```text
/drip --math the two constructions p(x|A)p(x|B)/p(x) and p_A p_B, total variation distance, why
summed scores under reverse diffusion miss the product (Du et al. 2023, eqs. 11 and 12), and why
MALA's proposal divisor is 4·ss, as steps 1, 2 and 6 of
artifacts/drips/simple-distributions-rebuild/02-composing-and-sampling.md set them out

Starts from: quiz 2.1 passed; any question it marked missed becomes a first rung. Layer 1's --math
drip folder under artifacts/drips/ holds my notation for q(x_t | x_0) and the score.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- Write a(x) = p(x|A), b(x) = p(x|B) and r(x) for the background density of all the data.
  Construction (1), score(x|A) + score(x|B) − score(x), targets a·b/r, which is
  r(x)·p(A|x)·p(B|x) up to a constant and equals p(x|A,B) when A and B are conditionally
  independent given x; construction (2) targets a·b. They differ by the factor 1/r, so after
  normalising they agree only where r is constant on the region where a·b > 0. The idea maps
  write (1) as p(x|A)p(x|B)/p(x) with p(x) the background, while the step file's p(x) is the
  composed product (step file 02, step 1; artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md,
  claim 3; artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 3).
- The product of two Gaussians: N(0, 1)·N(2, 1) is N(1, 0.5) with log Z = log N(0; 2, 2) = -2.2655,
  the p2.prod_mean, p2.prod_var and p2.prod_logZ keys (step file 02, step 1;
  rebuild/out/walk-numbers.json).
- Summed noised scores target the product of the noised experts, not the noised product. With A =
  N(1, 0.1²) and B = N(−1, 1) at ᾱ = 0.5: clean product mean 0.9802, noised product mean 0.6931,
  product of the noised experts 0.2326 (step file 02, step 2; the p2.clean_product_mean,
  p2.noised_product_mean and p2.product_of_noised_mean keys; Du et al. 2023, p.4, eqs. 11 and 12).
- That error does not shrink with more steps or better networks: on the 16-pixel mosaic it leaves
  the head mean at 0.923 where the exact product has 0.980 (step file 02, step 2's broken run).
- Total variation is half the summed absolute difference between two binned distributions, and a
  sampler's value is read against the floor that exact samples of the same size reach, never
  against 0 (step file 03, step 4; the Wootters table under step file 02, step 2).
- The estimator behind the sampler TVs here: 40 × 40 bins on the canvas [−1.1, 1.1]², the exact
  density's mass per bin against each bin's sample count divided by the full sample count, never
  by the count that landed inside the canvas, so the result is a windowed histogram diagnostic
  and not the full-space TV (rebuild/ideal.py, tv; the kinds notebook, cell 1). Step 5's answer
  key uses 60 × 60 bins on [−1.5, 1.5]² and counts samples outside that box as error
  (artifacts/ideas/when-poe-reaches-a-and-b/runs/samplers-on-exact-experts/exact_toy.py,
  total_variation).
- The MALA proposal has variance 2·ss, so its log-density divides by 4·ss; the repository's code
  divided by 2·ss². One move on N(0, 1) from x = 1 with ss = 0.5 and z = 0.3 gives x' = 0.8 and a
  log-density change of 0.18; with 4·ss: log forward -0.045, log backward -0.18, log α 0.045,
  accept probability 1.0; with 2·ss²: -0.18, -0.72, log α -0.36, accept probability 0.6977 (step
  file 02, step 6; the p2.mala_* keys).
- Every composition reports three numbers together: the A∩B share, the total variation to the exact
  target and the normaliser Z. Reverse diffusion's share of 0.983 against the exact 0.989 hid a
  total variation 0.049 above the floor (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md,
  claim 5 and run 5).
Outputs: the drip's thread folder under artifacts/drips/, with dispatch.md recording my notation,
the worked numbers I used and what clicked.
Target: I state the four outcome sentences below without notes, and reproduce the divisor toy by
hand: log α 0.045 with 4·ss against -0.36 with 2·ss². I judge.
Hand-off: the thread folder's path, which step 6's scene and step 7's cross-links read.
```

added: the two constructions' targets (from both idea maps, claim 3) · the worked numbers for the product, the summed scores and the divisor (from step file 02 and `walk-numbers.json`) · the mosaic's broken run (from step file 02) · the floor rule for total variation (from step file 03, step 4) · the three numbers with reverse diffusion's hidden miss (from the `when-poe-reaches-a-and-b` idea map, claim 5 and run 5) · layer 1's drip folder as the notation source · the names a, b and r for the two constructions, the [conditional independence](../../../../../../Obsidian/Mathematics/Statistics/Probability/Conditional%20Independence.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/computer-vision/computer-vision-textbook.pdf#page=219 "Computer vision textbook, section 10.1, Works it on a joint table of three discrete variables taking 4, 3 and 2 values, 24 probabilities in all, where x₁ and..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=201 "Mathematics for machine learning, section 6.4.5, Definition 6.11, p(x, y ∣ z) = p(x ∣ z) p(y ∣ z) for every z, read as 'given knowledge about z, the distribu..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=389 "Pattern recognition and machine learning, section 8.2, The definition p(a ∣ b, c) = p(a ∣ c), then three-node graphs where observing c makes a and b independ..."), [▶1](https://www.youtube.com/watch?v=zRdE8A4UZes&t=4587s "CS229 lecture 5, 1:16:27 to 1:20:34, Makes the naive Bayes assumption that words are conditionally independent given the class, cutting the parameters from 2..."), [◆](../../../../../../goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/04-conditional-independence.md "Check independence inside each slice of a third variable, deep-learning/generative-models/03-sampling-from-a-graph, plan 04. Conditional independence checked...")</sub> that construction (1) needs, and the clash between the idea maps' p(x) and the step file's (from the Codex mathematics review) · the TV estimator's bins, window and denominators for both the kinds runs and the step 5 answer key (from the Codex mathematics review, with the window corrected against `rebuild/ideal.py` and `exact_toy.py`)
dropped: "the two agree only where p(x) is flat", replaced by the 1/r factor and its condition (from the Codex mathematics review)

⏱ you 55 to 110 min · AI ~15 min · stop: you state the four things below without notes

**Outcome:** four things you can state:

- construction (1) is construction (2) divided by a background expert;
- summed scores miss the product;
- for a fixed MALA step, the accept step makes the target a stationary law of the chain, which removes the proposal's step-size bias; a finite run can still miss through poor mixing or too few moves, and the tuned MALA here adapts while it [samples](../../../../../../Obsidian/Mathematics/Statistics/Sampling%20From%20A%20Distribution.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=539 "Pattern recognition and machine learning, chapter 11, Why expectations need samples, ancestral sampling of a directed graph (page 541), standard distribution..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=245 "Introduction to probability for data science, section 4.8, Generates random numbers for any PDF by running a uniform through the inverse CDF, in Python, with..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/deep-learning-textbook.pdf#page=606 "Deep learning textbook, section 17.1, Why machine learning samples at all (Monte Carlo estimates of sums and integrals, and sampling as the goal of a generat..."), [▶1](https://www.youtube.com/watch?v=9ixzzPQWuAY&t=474s "ritvikmath, Inverse Transform Sampling, 7:54 to 10:04, Inverts the exponential CDF and turns a uniform draw into an exponential one. Chapter-timed, from the ..."), [▶2](https://www.youtube.com/watch?v=OXDqjdVVePY&t=232s "ritvikmath, Accept-Reject Sampling, 3:52 to 7:45, An easy proposal g scaled by M into an envelope above f, then each draw accepted with probability f/(M g). ..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/textbook/l3-stochastic-processes/13-drawing-samples-by-hand.md "Drawing samples by hand: Monte Carlo, inverse transform and accept-reject, deep-learning/diffusion-models/00-from-scratch-to-sdxl, volume L3, section 13. Thr...")</sub>, so it is judged against the exact-sample floor;
- every composition reports three numbers together: the A∩B share, the total variation to the exact target, and the normaliser Z.

### 3. You: quiz 2.2

Quiz 2.2: (URL lands here)

⏱ you 65 to 130 min, the banked problems included · stop: the bar, then the build. Below the bar, the misses become the first rungs of a second `/drip --math`, and you retake before building.

### 4a. You: the build, walk02.py

```text
/drip-execute-plan artifacts/drips/simple-distributions-rebuild/02-composing-and-sampling.md, I type
every block myself into ~/Playground/PhD/simple-distributions-walk/walk02.py

Task: walk the step file one step per message in its order, steps 0 to 10. I type every block myself
into ~/Playground/PhD/simple-distributions-walk/walk02.py and run it from that folder with
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python and
PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy.
Starts from: the setup's folder and environment; quiz 2.2 passed; the layer 2 --math drip folder
under artifacts/drips/, whose dispatch.md holds my notation.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The reference is rebuild/compose.py, rebuild/samplers.py and the fixed anneal_samplers.py in
  toy/; every worked number prints from python -m rebuild.walk_numbers, run from toy/,
  as 53 p2. keys, of which the 14 p2.step9_ keys are read from rebuild/out/phase2/step9.json,
  produced by python -m rebuild.phase2_step9 (step file 02, opening; rebuild/out/walk-numbers.json).
- Step 0 imports the cosine schedule and the device guard from rebuild/ and the samplers from
  anneal_samplers.py, so walk02.py runs without walk01.py (step file 02, opening).
- Train or sample on a card with under 1 GB in use, else on the CPU (rebuild/device.py; the GPU
  guard in ~/Playground/PhD/poe_repair_min/environment/hpc/execution-protocol.md).
- anneal_samplers.py divides the MALA proposal's log-density by 4·ss and accepts each sample on its
  own; tests/test_anneal_samplers.py holds 6 tests and all pass (REBUILD_WALK.md, brick "MALA and
  CHA fix").
- The notebook's mixture is sharpened by a factor 3.5: E_mix = −logsumexp(−3.5E_A, −3.5E_B), 3.4991
  at E_A = 1, E_B = 3; its negation uses w = 0.3, and p_A^1.3·p_B^−0.3 = 0.3299 at p_A = 0.5,
  p_B = 2 (step file 02, steps 3 and 4).
- ULA at ss = 0.9 on N(0, 1) has stationary variance 2/(2 − ss) = 1.8182; 200,000 chains for 300
  moves give 1.806 (step file 02, step 5; p2.ula_stationary_var_ss0.9, p2.ula_simulated_var_ss0.9).
- With the fixed divisor the MALA variance comes out near 1 and with the old one about four times
  that: the step file's run printed 0.989 and 3.745, the test file's seed 0.989 and 4.026 (step file
  02, step 6's broken run).
- Per-sample acceptance on N(0, 1) at ss = 1.5, 4,000 chains, 10 moves: mean 0.656, sd 0.153 across
  samples; one accept decision for the whole batch would make all 4,000 entries equal, while one
  uniform shared but compared with each sample's own α would not (step file 02, step 7).
- leapfrog_step_c returns the tensor it was given, so the old CHA's accept test read the change in
  log p alone; the fixed class passes copies (step file 02, step 8; p2.cha_alias_same_tensor True).
  The leapfrog toy gives x' 1.105, v' 0.18425 and H 0.625 to 0.62749.
- Step 9's target is ring × bar as exact ideal experts, 400 points each, smoothed by σ = 0.05. The
  centres (0, 0.5) and (0, −0.5) are the only ring-mode centres inside the bar; the composition
  puts 0.933 of its mass within 0.2 of them and, because both experts are Gaussian smoothed, 0.065
  within 0.2 of the other six ring modes. Step sizes are 0.05·ν_t for ULA and MALA and 0.3·√ν_t
  for UHA, with ν_t = 1 − ᾱ_t + ᾱ_t·σ², 10 moves per level over 100 levels (step file 02, step 9).
- The last annealing level has ᾱ_0 = 0.9994, so the anneal ends at π_0, the composition smoothed
  further (variance 0.00313 per component against the clean 0.0025). π_0's exact share within 0.1
  of the two in-bar modes is 0.636 (p2.step9_exact_share_pi0); the clean composition's is 0.713
  (p2.step9_exact_share_clean). walk02.py's ULA, MALA and UHA shares are read against 0.636
  (step file 02, step 9).
- The step file's run printed shares on the two in-bar modes of exact 0.713 (the clean
  composition), 1000 perfect draws 0.718, ULA 0.356, MALA 0.370, UHA 0.563, reverse diffusion
  0.468. MALA at 0.05·ν_t accepts nearly every move and so behaves like ULA (step file 02, step 9).
- python -m rebuild.phase2_step9 reruns step 9 with the same seeds and adds a clean final level, 10
  moves on the clean composition after the anneal (ss = 0.05·σ² for ULA and MALA, h = 0.3·σ for
  UHA): ULA 0.407, MALA 0.411 and UHA 0.620, read against 0.713 (rebuild/out/phase2/step9.json;
  p2.step9_share_*_plus_clean_level).
- Step 10 runs python -m rebuild.phase2_compose on the phase 1 checkpoints in
  rebuild/out/phase1/ckpt/. The trained t = 0 energy splits the in-bar mass 0.10 and 0.90 where the
  data's product splits it 0.54 and 0.46, so 2D ground truth comes from ideal experts, never from a
  trained energy at t = 0 (step file 02, step 10; rebuild/out/phase2/phase2.json).
Outputs:
- walk02.py: one script that runs end to end and prints each step's worked number;
- step07-acceptance.png (one sample's acceptance over 10 moves) and step09-ring-times-bar.png (the
  four samplers side by side) in the typing folder;
- the step file's steps 0 to 10 ticked, and the walk map that holds my place between sittings.
Target: every p2. number walk02.py prints equals walk_numbers to the digits shown; step 7 prints a
per-sample acceptance whose sd is above 0; step 9 prints the exact share 0.713 and puts UHA's share
above ULA's and MALA's, with all three read against π_0's 0.636; step09-ring-times-bar.png read
beside rebuild/out/phase2/reference/product-0.png shows samples on the two in-bar modes. I judge,
from the printed lines and the picture.
Progress: step 1 prints mean 1.0, variance 0.5 and log Z -2.2655; step 2 prints 0.9802, 0.6931 and
0.2326; step 5 the ULA variance against 1.8182; step 6 the two stationary variances; step 7 the
acceptance mean and sd; step 8 "leapfrog returned the tensor it was given: True"; step 9 one share
per sampler.
Proof: python walk02.py prints "acceptance per sample: mean <m>  sd <s>" with s above 0, and the
comparison in step 5 of this layer shows no differing p2. number.
Limits: never edit rebuild/, anneal_samplers.py or the step file's code blocks; I type, I do not
paste; no commit.
Hand-off: the retrospective the walk closes on, and the /drip-retro command it prints.
```

added: the reference files and the 28 p2. keys, the device guard, the 6 sampler tests, the mixture's 3.5 and the negation's 0.3, the ULA bias, the two divisors' [variances](../../../../../../Obsidian/Mathematics/Statistics/Variance.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=150 "Introduction to probability for data science, section 3.4.4, Defines variance as E[(X − µ)²] and the standard deviation as its root (Definition 3.7), proves ..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=196 "Mathematics for machine learning, section 6.4.1, Variance as the covariance of a variable with itself (Definition 6.7), then the empirical mean and variance ..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/deep-learning-textbook.pdf#page=76 "Deep learning textbook, section 3.8, A short definition of variance and standard deviation beside expectation and covariance."), [▶1](https://www.youtube.com/watch?v=4-FysvtxQEU&t=1551s "Professor Leonard, Statistics 3.3, A standard deviation calculated by hand, 25:51 to 36:43, Works a standard deviation on a small data set by hand. Chapter-t..."), [▶2](https://www.youtube.com/watch?v=SzZ6GpcfoQY&t=236s "StatQuest, Calculating the Mean, Variance and Standard Deviation, Calculating population variance, 3:56 to 5:54, The population variance as the average squar..."), [◆](../../../../../../goal-setting/learning/deep-learning/transformers/textbook/06-show-that-variances-of-independent-terms-add.md "Show that variances of independent terms add, deep-learning/transformers, chapter 06. Predicts the spread of a dot product of random vectors by adding the va...")</sub>, the per-sample acceptance, the aliased leapfrog tensor, step 9's setup and printed shares, and step 10's lesson that ground truth comes from ideal experts (all from step file 02, `REBUILD_WALK.md`, `walk-numbers.json` and `phase2.json`) · a numeric target in place of "matches the paper's figure" · ring × bar's mass split (0.933 within 0.2 of the in-bar centres, 0.065 near the other six modes), the last annealing level's smoothing and its share 0.636, the name ν_t for the noise variance so it stays apart from step 8's momentum v, and the difference between one batch-wide decision and one shared uniform (from the Codex mathematics review) · the anneal's end at π_0 with exact share 0.636 as what walk02.py's annealed shares are read against, the reference's shares after a clean final level (ULA 0.407, MALA 0.411, UHA 0.620) against 0.713, and the p2. key count, 53 (from the step 9 scoring fix) · the 14 p2.step9_ keys named as read from step9.json, produced by phase2_step9 (from the step 9 scoring fix)
dropped: "bar: my ring × bar matches the paper's figure and my MALA reports acceptance per sample" from the first line (why: the figure half has no tolerance, so it stays as the picture to look at beside `product-0.png`, and the acceptance half is now the Target's sd above 0) · "the composition keeps only the two ring modes" and "one shared draw would make all 4,000 entries equal" (from the Codex mathematics review: the smoothed product keeps tails everywhere, and a shared uniform does not force one decision)

⏱ you 1.5 to 3 h (11 steps) · AI ~30 min · stop: the prompt's bar

**Outcome:** your own product, annealed ULA and per-sample MALA.

**Two findings from the idea walks, rebuilt here.**

### 4b. You: the first idea-walk rebuild, walk02b.py

```text
/drip-execute-plan --request rebuild sections 1 to 3 of
artifacts/ideas/poe-overlap-dial/what-an-expert-learns-and-kinds-of-composition.ipynb by hand in
~/Playground/PhD/simple-distributions-walk/walk02b.py; bar: the gradient difference is exactly 0.0
when only the labels change, construction (1) puts 1.00 of its mass in A∩B under "contains A"
against 0.08 under "exactly A", and a classifier trained on set membership guides sideways in A∩B,
(-0.38, -0.04) against (-0.14, +0.99) with true labels

Task: cut sections 1 to 3 of the notebook into 4 to 6 pieces, write the step file at the open, and
build, teach and test one piece per round. I type every block into walk02b.py and run it with
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python from
~/Playground/PhD/simple-distributions-walk/.
Starts from: walk02.py from step 4a; the notebook named above, whose printed outputs are the answer
key.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- No sampler appears: every density is an ideal expert, its training points smoothed by N(0, σ²I),
  computed exactly on a 241 × 241 grid over [−1.1, 1.1]², and A∩B is the top-left quadrant
  (notebook, cells 0 and 1).
- Section 1: A's set is (−0.5, −0.5), (−0.6, −0.4), (−0.5, 0.5) under two labellings. A tiny
  denoiser (Linear 3 to 32, SiLU, Linear 32 to 2, seed 1) takes one gradient step on each; the
  labels never enter the loss, so the largest gradient difference is exactly 0.0. At σ = 0.15 a
  third of A's mass sits in the top-left quadrant (notebook, cells 3 and 4).
- Section 2: the pool adds B's points (0.5, 0.5), (0.6, 0.4), (−0.4, 0.6), one of them secretly A.
  Under "contains A" every point whose label set includes A trains p(x|A); under "exactly A" a
  point labelled {A, B} drops out of p(x|A). Construction (1), p(x|A)p(x|B)/p(x) at σ = 0.15, puts
  1.00 of its mass in A∩B under "contains" and 0.08 under "exactly", where its leftover mass sits
  near the origin (notebook, cells 5 and 6; artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 1).
- Section 3: two sets of 400 points with 20% hidden in the other concept's half; a classifier (2 to
  64 to 64 to 1, SiLU, Adam 3e-3, weight decay 1e-3, 1,500 steps, seed 0) trained on set
  membership is unsure in A∩B, about 0.5, and its mean guidance direction there is (−0.38, −0.04);
  trained on the true labels x2 > 0 it is (−0.14, +0.99) (notebook, cells 7 and 8).
- The notebook draws its data from np.random.default_rng(0) with torch.manual_seed(0) in cell 1,
  so section 3's directions repeat only when the draws happen in the notebook's order (notebook,
  cell 1).
- The repaired claim: a density expert learns its training set's density, and labels matter only in
  the parts that read them, the conditioning scheme of construction (1) and any classifier
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 1).
Outputs:
- walk02b.py, printing the gradient difference, the two top-left masses and the two guidance
  directions;
- the step file the walk writes at its open, ticked piece by piece.
Target: the gradient difference prints exactly 0.0; the top-left masses print 1.00 and 0.08; the
two guidance directions print (−0.38, −0.04) and (−0.14, +0.99) to two decimals with the notebook's
seeds and order. I judge, from the printed lines.
Progress: piece 1 prints the gradient difference; piece 2 the masses under each scheme; piece 3 the
classifier's mean p(B|x) in A∩B and its two directions.
Proof: python walk02b.py prints "largest difference between the two versions' gradients: 0.0" and
the four numbers above, and step 5's comparison against the notebook shows no differing number.
Limits: never edit the notebook or rebuild/; I type, I do not paste; no commit.
Hand-off: the retrospective the walk closes on, and the /drip-retro command it prints.
```

added: the grid and the ideal-expert setup, the three sections' points, networks and seeds, and the repaired claim (from the notebook and the `poe-overlap-dial` idea map, claim 1) · the dependence of section 3's numbers on the draw order (from the notebook, cell 1)
dropped: nothing

⏱ you 1 to 2 h (one build concept) · AI ~35 min (the median over 7 measured `--request` walks is 33) · stop: the prompt's bar

**Outcome:** you can show three things:

- a [density](../../../../../../Obsidian/Mathematics/Statistics/Distributions/Probability%20Density%20Function.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=188 "Introduction to probability for data science, section 4.1, Builds the intuition of probability per unit length, defines the PDF and its properties (Definitio..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=186 "Mathematics for machine learning, section 6.2.2, The density and the CDF (Definitions 6.1 and 6.2), a uniform example (Example 6.3), and discrete against con..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=36 "Pattern recognition and machine learning, section 1.2.1, The density's two conditions (equations 1.25 and 1.26), its link to the CDF, and why a density picks..."), [▶1](https://www.youtube.com/watch?v=QKA4HNEw3aY&t=52s "The Organic Chemistry Tutor, Probability Density Functions, 0:52 to 3:39, Verifies that a given function is a density by checking that its integral equals 1...."), [▶2](https://www.youtube.com/watch?v=oI3hZJqXJuc&t=135s "StatQuest, The Main Ideas behind Probability Distributions, 2:15 to 5:14, A curve as a distribution, with probability read as area under it, not its height. ..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/02-densities-and-the-gaussian.md#21-a-density-can-exceed-1 "Chapter 2: Read a density as area and draw a Gaussian from its covariance, deep-learning/diffusion-models/01-poe-derivation-foundations, chapter 02, section ..."), [◈](https://claude.ai/artifact/DBmJtn6xy75sPtQwPEvQZU#nb-sec1 "Kinds of composition notebook, Setup: the ideal expert, The ideal expert's log-density on a grid, in four steps from the 2D Gaussian to its logsumexp form, t...")</sub> expert never reads its labels;
- the meaning of a label decides construction (1);
- a [classifier](../../../../../../Obsidian/Artificial%20Intelligence/Machine%20Learning/Classifier.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/natural-language-processing/hands-on-large-language-models.pdf#page=160 "Hands-on large language models, chapter 4, Works a classifier end to end on real data: 5,331 positive and 5,331 negative Rotten Tomatoes reviews labelled by ..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/probabilistic-machine-learning-an-introduction.pdf#page=30 "Probabilistic machine learning: an introduction, section 1.2.1, Classification on the iris data: 150 flowers with four measurements each and one of three spe..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/understanding-deep-learning.pdf#page=78 "Understanding deep learning, sections 5.4 to 5.5, A network becomes a binary classifier by predicting a Bernoulli probability through the logistic sigmoid, a..."), [▶1](https://www.youtube.com/watch?v=pdqofxJeBN8&t=995s "CS231n lecture 2, 16:35 to 20:12, Builds the nearest neighbour classifier: training memorises the data, prediction returns the label of the closest training ..."), [▶2](https://www.youtube.com/watch?v=DATnpGoGhM8&t=759s "CS229 lecture 1, 12:39 to 14:39, Separates regression (continuous y) from classification (discrete y) with a house against townhouse boundary, and frames nex..."), [◆](../../../../../../goal-setting/learning/machine-learning/classical-ml-from-scratch/MASTER_PLAN.md "Classical ML from Scratch, machine-learning/classical-ml-from-scratch. Goal 1 classifies a toy dataset with logistic regression written by hand, and goal 7 b..."), [◈](https://claude.ai/artifact/DBmJtn6xy75sPtQwPEvQZU#nb-sec4 "Kinds of composition notebook, A classifier learns 'not B', A classifier trained on the labels the data carries learns the hidden-B points as 'not B'.")</sub> trained on the labels the data carries reproduces a shortcut.

### 4c. You: the second idea-walk rebuild, walk02c.py

```text
/drip-execute-plan --request rebuild run 1 of
artifacts/ideas/poe-overlap-dial/kinds-of-composition-on-the-papers-toys.ipynb by hand in
~/Playground/PhD/simple-distributions-walk/walk02c.py; bar: MALA adapted to 0.6 acceptance takes
steps 18 to 82 times larger and reaches the floor on four kinds, while ULA at those steps diverges
(it keeps 1 of 2,000 samples on disjunction)

Task: cut run 1 into 4 to 6 pieces, write the step file at the open, and build, teach and test one
piece per round. I type the tuned MALA and the ULA comparison into walk02c.py by hand and run it
with /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python and
PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy.
Starts from: walk02.py from step 4a, whose per-sample MALA is the move run 1 tunes; the notebook
named above, whose printed run 1 table is the answer key.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- Run 1 runs on the seven kinds' exact targets: ideal experts with σ = 0.05, 800 points per set,
  2,000 samples, 100 levels, 10 moves per level from N(0, I) (notebook, cells 0 and 2;
  rebuild/kinds.py). Layer 3 builds those targets by hand; here import them from rebuild/kinds.py
  (build_kinds) and rebuild/ideal.py (exact_grid, composed_target, tv, V).
- In Figure 1, MALA at 0.05·ν_t accepted 0.99 to 1.00 of its proposals, so it equalled ULA and did
  not yet test the accept step (notebook, cells 8 and 10).
- Tuned MALA carries its step across levels scaled by ν_t and multiplies it by e^(acceptance − 0.6)
  after every move; the adaptation runs during sampling, so the chain is not exactly reversible
  while it adapts (notebook, cell 14; rebuild/samplers.py, tuned_mala).
- The notebook's run 1 table, kind: acceptance, final step over Figure 1's step, tuned TV, floor:
  intersection 0.60, 18.6x, 0.11, 0.11; disjunction 0.61, 82.4x, 0.22, 0.23; co-occurrence 0.61,
  68.9x, 0.11, 0.10; contention 0.60, 43.9x, 0.16, 0.10; dominance 0.60, 17.6x, 0.30, 0.11;
  interaction 0.61, 37.5x, 0.08, 0.07; negation 0.60, 28.7x, 0.21, 0.15 (notebook, cell 15).
- ULA at the same steps keeps 721, 1, 157, 1843, 777, 2000 and 446 of 2,000 samples in that kind
  order, with TV 0.33 to 0.50 where it survives (notebook, cells 15 and 16).
- Dominance stays at 0.30 against a floor of 0.11 even with the large steps: no local sampler
  moves mass between separated modes to match unequal weights (notebook, cell 16;
  artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, run 5).
- rebuild/phase3_kinds.py reproduces run 1 within 0.03 on every kind, its own tolerance for sampler
  numbers (rebuild/out/phase3/phase3.json, run1).
Outputs:
- walk02c.py, printing one row per kind: acceptance, step ratio, tuned TV, floor, ULA's TV and the
  samples ULA kept;
- the step file the walk writes at its open, ticked piece by piece.
Target: tuned acceptance prints 0.60 ± 0.03 on every kind; tuned TV is within 0.03 of the floor on
intersection, disjunction, co-occurrence and interaction; ULA at the same steps keeps under 2,000
samples on disjunction (the notebook kept 1). I judge, from the printed table.
Progress: piece 1 prints Figure 1's MALA acceptance per kind; the tuning piece prints each kind's
acceptance and step ratio; the last piece prints the ULA column.
Proof: python walk02c.py prints the seven-row table, and step 5's comparison with python -m
rebuild.phase3_kinds (Run 1) and the notebook's cell 15 shows each tuned TV within 0.03.
Limits: never edit the notebook, rebuild/ or anneal_samplers.py; I type, I do not paste; no
commit.
Hand-off: the retrospective the walk closes on, and the /drip-retro command it prints.
```

added: run 1's setup, the tuning rule and its reversibility caveat, the full run 1 table with ULA's kept samples, dominance's survival, and the reference's tolerance (from the kinds notebook, the `poe-overlap-dial` idea map, run 5, and `phase3.json`) · importing the kinds' exact targets from `rebuild/` here, since layer 3 is where they are built by hand · v_t renamed ν_t to match the step file (from the Codex mathematics review)
dropped: nothing

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

```text
/math-scene the folder the layer 2 --math drip wrote under artifacts/drips/: reverse diffusion with
summed scores leaving a faint ring between the two true modes of ring × bar, against annealed MALA
landing on them

Starts from: the layer 2 --math drip folder and its dispatch.md (step 2's hand-off).
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- Ring × bar is step file 02's step 9 target: the ring of eight Gaussian modes on a circle of radius
  0.5 (sd 0.03) and the bar uniform on |x1| < 0.2, |x2| < 1, 400 points each, smoothed by σ = 0.05.
  The centres (0, 0.5) and (0, −0.5) are the only ring-mode centres inside the bar. Because both
  experts are Gaussian smoothed, the product keeps tails everywhere: 0.933 of its mass lies within
  0.2 of those two centres and 0.065 within 0.2 of the other six ring modes (step file 02, step 9).
- On that target the exact share within 0.1 of the two in-bar modes is 0.713; reverse diffusion
  gives 0.468, ULA 0.356, MALA at 0.05·ν_t 0.370 and UHA 0.563 (step file 02, step 9). The
  annealed samplers end at π_0, whose exact share is 0.636; after a clean final level of 10 moves
  they give ULA 0.407, MALA 0.411 and UHA 0.620 against 0.713 (rebuild/out/phase2/step9.json).
- "Annealed MALA" in the scene is tuned MALA, adapted to 0.6 acceptance: at the fixed step 0.05·ν_t
  MALA accepts nearly every move and leaves the same stragglers as ULA (step file 02, step 9).
- The faint ring is the kinds notebook's intersection row (ring × a bar of half-width 0.05):
  reverse diffusion reaches TV 0.31 against a floor of 0.11, and tuned MALA reaches 0.11 against
  0.11 (artifacts/ideas/poe-overlap-dial/kinds-of-composition-on-the-papers-toys.ipynb, Figure 1 and
  run 1).
- With the trained phase 1 experts, the in-bar share is 0.928 for the trained composition, 0.580
  for reverse diffusion and 0.850 for tuned MALA (rebuild/out/phase2/phase2.json; step file 02,
  step 10).
- Every number on the scene comes from step file 02, the kinds notebook, phase2.json or the drip's
  dispatch.md, never estimated.
Outputs: one published scene, in my notation from dispatch.md.
Hand-off: write the scene's URL into step 6 of
artifacts/drips/simple-distributions-rebuild/ladder/02-composing-two-experts.md, replacing "(URL
lands here)".
```

added: ring × bar's definition and step 9's shares (from step file 02) · which MALA lands on the modes, since the fixed-step MALA does not (from step file 02, step 9) · the faint ring's numbers and tuned MALA's floor (from the kinds notebook) · the trained shares (from `phase2.json`) · the in-bar centres as the only ones inside the bar, with the product's tails near the other six modes (0.933 and 0.065 within 0.2), and ν_t for the noise variance (from the Codex mathematics review) · π_0's exact share 0.636 and the shares after a clean final level (from the step 9 scoring fix)
dropped: "the product keeps only the two ring modes" (from the Codex mathematics review: the smoothed product keeps nonzero mass elsewhere)

Scene: (URL lands here)

⏱ you ~20 min · AI ~45 min, beside layer 3's maths · stop: the scene is published and its numbers match your drip's

**Outcome:** a scene where reverse diffusion with summed scores leaves a faint ring between ring × bar's two modes, and annealed MALA lands on them.

### 7. You: close the layer

Run the `/drip-retro` command step 4a's build printed at its close (⏱ about 25 minutes). Then paste this:

```text
In the background, dispatch xref-cadence over the files layer 2 of the composition ladder wrote: the
layer 2 --math drip folder under artifacts/drips/, the step files the two --request walks of steps
4b and 4c wrote at their open, and the retrospectives the three layer 2 walks closed on.

Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The reference sets to link against are the journey plans, vault notes and scene on the Study
  first line of artifacts/drips/simple-distributions-rebuild/ladder/02-composing-two-experts.md; the
  layer 2 rows of the Bricks table in artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md
  ("MALA and CHA fix", "Product, mixture, negation wrappers", "Annealed ULA, MALA, UHA, MUHA");
  claims 1, 3 and 4 and runs 4 and 5 of artifacts/ideas/poe-overlap-dial/IDEA_MAP.md; and claims 3,
  4 and 5 and run 5 of artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md.
- ~/Playground/PhD/simple-distributions-walk/ is scratch space; link nothing into it (CLAUDE.md, the
  playground is disposable).
Outputs: links written both ways between those files and the reference sets.
Hand-off: the agent's report, listing every link it wrote, read before layer 3 starts.
```

added: the reference sets, by path (from this layer's Study first line, `REBUILD_WALK.md` and the two idea maps)
dropped: nothing

⏱ you 2 min to paste · AI ~10 min (estimate)

**Skip as build topics:** UHA and Hamiltonian moves, the CHA bug, and the mixture wrapper. UHA stays link-only here. Tuned MALA reaches the exact-sample floor on intersection, disjunction, co-occurrence and interaction in run 1, and stays above it on contention (`0.16` against `0.10`), dominance (`0.30` against `0.11`) and negation (`0.21` against `0.15`).

---

⬅️ [Layer 1: one expert](01-one-diffusion-expert.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 3: ideal experts](03-ideal-experts-and-the-seven-kinds.md) ➡️
