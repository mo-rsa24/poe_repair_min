# ⭐◇ Layer 3: ideal experts, and the seven kinds of composition

⬅️ [Layer 2: composing](02-composing-two-experts.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 4: the parts world](04-the-parts-world.md) ➡️

**Why it is half critical.** The ideal expert is critical: the training points smoothed by N(0, σ²I). Under a denoising objective taken over the whole distribution, with unlimited model capacity and a global optimum, its score is the best score field at that smoothing level. The rebuild then treats this smoothed set as the clean expert, which makes every later target exact, so a failure can be blamed on the sampler or the data rather than on a network's approximation. In the cited 2D run, trained networks matched it within `0.04` nats. Of the seven kinds, only co-occurrence, intersection, the chimera and the wet dog matter for the goal.

**Depth.** Build: the ideal expert, the exact target and its perfect-sample baseline, the four kinds that matter, and the weak-sampler rebuild. Link only: disjunction and dominance. You 4.3 to 8.7 h in all.

**Study first.** The kinds and their words: [Kinds Of Composition](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/Kinds%20Of%20Composition.md), [The Product As An Intersection](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/The%20Product%20As%20An%20Intersection.md), [What An Expert Learns From Its Training Set](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/What%20An%20Expert%20Learns%20From%20Its%20Training%20Set.md), and [the composition terms](../../../ideas/poe-overlap-dial/composition-terms.md). The exact noised mixture: [the exact score of a Gaussian mixture](../../../../../../goal-setting/learning/deep-learning/diffusion-models/06-diffusion-speciation-dynamics/plans/02-speciation/plans/02-exact-mixture-score.md). Why a product of noised experts differs from the noised product: [score additivity and the product density](../../../../../../goal-setting/learning/deep-learning/diffusion-models/06-diffusion-speciation-dynamics/plans/05-composition-and-the-experiment/plans/01-score-additivity-product-density.md). Smoothing by a Gaussian as a convolution: [summing out a joint, equation 2.3](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/textbook/02-joint-and-marginal-summing-out.md). Textbooks: Murphy, *Probabilistic Machine Learning* 16.3 (kernel density estimation, p605) and Chan, *Introduction to Probability for Data Science* 5.5 (the sum of two random variables, p296).

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

Paste this when you start layer 2's build.

```text
In the background, prepare and build layer 3's two quizzes, bottom up, for the composition ladder
in artifacts/drips/simple-distributions-rebuild/: quiz 3.1 and quiz 3.2.

Task: for each quiz, first dispatch prep-cadence (quiz-prep-pulse --build) with the goal and minutes
below, then dispatch quiz-cadence (mode build, repo
/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain). The goal for both is "build the ideal expert and
grade every composition against an exact target".
- Quiz 3.1, --minutes 40: the college maths under layer 3. Convolving a density with a Gaussian,
  kernel density estimates, softmax weights over training points.
- Quiz 3.2, --minutes 90: layer 3 itself, from the step file
  03-ideal-experts-and-the-seven-kinds.md. The ideal expert's score, why its composition is exact
  at every noise level, the perfect-sample baseline, co-occurrence, intersection, contention and
  interaction.
Starts from: nothing from this layer; it runs during layer 2's build, so the layer 3 --math drip has
not run and the notation is the step file's. Quiz 3.1 takes its material from the Study first line
of artifacts/drips/simple-distributions-rebuild/ladder/03-ideal-experts-and-the-seven-kinds.md;
quiz 3.2 takes it from
artifacts/drips/simple-distributions-rebuild/03-ideal-experts-and-the-seven-kinds.md.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- quiz-cadence stops when a quiz has no coverage file in .quiz-prep/, so prep-cadence runs first for
  each quiz (quiz-cadence, "The preparation gate").
- The step file links a textbook page beside each fundamental it uses and carries timestamped video
  tables on the kernel density estimate, the sum of two independent Gaussians, total variation
  distance and the product of experts; reuse them as nodes before searching again (step file 03;
  the guide, "When a step stalls on the maths").
- No listed creator works the variance rule Var(aX + bY) = a²Var(X) + b²Var(Y) for independent X
  and Y, and the kernel-density segment never says a Gaussian kernel density estimate is a Gaussian
  mixture with one equal-weight component per point (step file 03, steps 1 and 2, video tables).
- The textbooks are Murphy, Probabilistic Machine Learning 16.3 (kernel density estimation, p605)
  and Chan, Introduction to Probability for Data Science 5.5 (the sum of two random variables,
  p296); smoothing by a Gaussian as a convolution is equation 2.3 of the conditional-expectation
  textbook's chapter 02 (this layer's Study first line).
- The score of a Gaussian mixture is Σ r_i(x)(μ_i − x)/σ², with r_i the posterior softmax weights
  over the components (the plan "the exact score of a Gaussian mixture", linked on this layer's
  Study first line).
- pdftotext is not installed on mscluster, so
  ~/.claude/skills/index-source/scripts/find-in-textbooks.py fails; open the PDFs with PyMuPDF from
  the co3 environment (checked on mscluster109: import pymupdf works in co3, version 1.28.2, and
  import pypdf fails everywhere).
- The shelf index is ~/Obsidian/Mathematics/references/textbooks/Textbooks.md and the
  worked-examples index is ~/Obsidian/Mathematics/references/exercises/Worked Examples.md
  (quiz-prep-pulse, "The bank").
- The notation is the step file's: the ideal expert p_σ(x) = (1/N)Σ N(x; x_i, σ²I); its noised
  density q_t(x) = (1/N)Σ N(x; √ᾱ_t x_i, v_t I) with v_t = 1 − ᾱ_t + ᾱ_t σ²; σ = 0.05 and N = 800
  (step file 03, steps 0 to 2).
- Every worked number prints from python -m rebuild.walk_numbers as a p3. key in
  rebuild/out/walk-numbers.json. 7 of the 25 are the numbers walk03.py prints: the three-point
  log-density -2.6288 by formula and by scikit-learn, v_t = 1.0, 0.5229 and 0.00313 at t = 99, 50
  and 0, the cell area 8.403e-05 and 58081 grid points. The other 18 back the step file's
  derivations, among them the 200 perfect-draw redraws and dominance's mode weights
  (rebuild/out/walk-numbers.json).
- The perfect-sample baseline (the code prints it as floor TV) is the total variation, on 40 × 40
  bins, between n exact draws from the grid density and the density itself; a sampler is read
  against it, never against 0. It is random, not a lower bound: a sampler can score below one
  perfect draw by chance (step file 03, step 4).
- The exact measures of the four kinds the goal needs: intersection 0.71, co-occurrence 0.81,
  contention 0.67, interaction −0.41 (step file 03, steps 5, 7, 8 and 10).
- Trained 2D networks match the ideal experts within 0.04 nats
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, run 2).
Outputs:
- two coverage files under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, each with a
  verdict per question: ready, blocked, or unprepared by choice;
- rows in the worked-examples index for every banked problem;
- two published quiz pages, each opening on a read-first list with minutes per item, a short path
  and a full path.
Target: no question on either page carries the verdict blocked; both quizzes carry banked problems,
about 6 to 10 per concept at the build depth; every number in an answer traces to walk_numbers,
the step file or a cited textbook page. I judge, from the pages.
Progress: prep-cadence reports, per quiz, its question count, node count and verdicts; quiz-cadence
reports each URL as it publishes.
Proof: the two coverage files exist under .quiz-prep/ and list no question as blocked; the two
URLs open.
Limits: change nothing in the step files' prose or in rebuild/; in the ladder files, write only the
two URLs.
Hand-off: write quiz 3.1's URL into step 1 and quiz 3.2's URL into step 3 of
artifacts/drips/simple-distributions-rebuild/ladder/03-ideal-experts-and-the-seven-kinds.md, each
replacing "(URL lands here)".
```

added: every field the setup's quiz prompt carries, so this prompt stands alone instead of pointing at "the setup section of PROMPTS.md", which the guide no longer holds (the setup moved to `ladder/00-setup.md`) · the preparation gate (from the quiz-cadence agent definition) · the step file's video tables and their two gaps (from step file 03) · the textbooks (from this layer's Study first line) · the mixture score with softmax weights (from the exact-mixture-score plan) · the missing `pdftotext` and the PyMuPDF route (from the textbook pass) · the notation, the p3. keys, the floor and the four exact measures (from step file 03 and `walk-numbers.json`) · the 0.04 nats (from the `poe-overlap-dial` idea map, run 2) · the URL write-back · the floor renamed the perfect-sample baseline and marked as random rather than a lower bound (from the Codex mathematics review) · the p3. key count, 25, with the 7 walk03.py prints named apart from the 18 that back the derivations (from the port of the review's checks into walk_numbers)
dropped: nothing

⏱ AI ~1.5 h (estimate), beside layer 2's build

### 1. You: quiz 3.1

Quiz 3.1: (URL lands here)

⏱ you 20 to 40 min · stop: the bar, every *needed now* question right and at least 80% of the rest; below that, read the items your misses point at and retake.

### 2. You: the maths drip

```text
/drip --math why smoothing the training points by N(0, σ²I) is the best a diffusion model can learn
at that noise level, and under which assumptions, as steps 1 and 2 of
artifacts/drips/simple-distributions-rebuild/03-ideal-experts-and-the-seven-kinds.md set them out

Starts from: quiz 3.1 passed; any question it marked missed becomes a first rung. Layer 2's --math
drip folder under artifacts/drips/ holds my notation for the two constructions and the summed
scores.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The ideal expert of x_1, …, x_N at smoothing σ is p_σ(x) = (1/N)Σ N(x; x_i, σ²I), a Gaussian
  kernel density estimate with bandwidth σ (step file 03, step 1).
- Write the training set as p̂_0(x) = (1/N)Σ δ(x − x_i). Gaussian corruption of variance σ² gives
  p_σ(x) = (1/N)Σ N(x; x_i, σ²I). With unlimited capacity and the denoising objective taken over
  the whole distribution and minimised globally, the optimal learned score is ∇_x log p_σ(x). The
  rebuild chooses this kernel density estimate as its clean expert before applying the forward
  process. A network can do worse by underfitting or differently by extrapolating (step file 03,
  step 1).
- σ = 0.05 is a chosen smoothing, not the schedule's smallest noise: √(1 − ᾱ_0) = 0.0251
  (p1.sqrt_1m_abar_0). A model trained on the bare points would have component variance 1 − ᾱ_t at
  step t; the extra ᾱ_t σ² in v_t comes from noising p_σ (step file 03, steps 1 and 2).
- In run 2 of the idea map, trained 2D networks matched the ideal experts within 0.04 nats, so in
  that run a trained expert added nothing measurable past its smoothed data
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, run 2 and claim 7).
- The worked instance: three points (−0.5, −0.5), (−0.6, −0.4), (−0.5, 0.5) at σ = 0.15 give a
  log-density at (−0.5, 0) of -2.6288 by the formula and by scikit-learn's KernelDensity
  (p3.ideal_logdensity_formula, p3.ideal_logdensity_sklearn).
- Noising scales each kernel by √ᾱ_t and convolves it with N(0, (1 − ᾱ_t)I), and convolving two
  Gaussians adds their variances, so q_t(x) = (1/N)Σ N(x; √ᾱ_t x_i, v_t I) with
  v_t = 1 − ᾱ_t + ᾱ_t σ² (step file 03, step 2).
- With σ = 0.05, v_t is 1.0, 0.5229 and 0.00313 at t = 99, 50 and 0; at t = 50, ᾱ_t = 0.4783, and
  1 − 0.4783 + 0.4783 × 0.0025 = 0.5229 (the p3.noised_ideal_var_* and p1.abar_50 keys).
- The score of that mixture, derived. Put μ_i = √ᾱ_t x_i and φ_i(x) = N(x; μ_i, v_t I), so
  q_t = (1/N)Σ φ_i. Differentiating the log gives ∇ log q_t = Σ_i ∇φ_i / Σ_j φ_j. Writing
  ∇φ_i = φ_i ∇ log φ_i turns that into Σ_i r_i(x) ∇ log φ_i, with r_i = φ_i / Σ_j φ_j the softmax
  responsibilities over the training points. The score of an isotropic Gaussian is (μ_i − x)/v_t,
  so ∇ log q_t(x) = Σ_i r_i(x)(√ᾱ_t x_i − x)/v_t (step file 03, step 2; the plan "the exact score
  of a Gaussian mixture", linked on this layer's Study first line).
- Because each q_t is exact, every annealing target π_t = compose(log q^A_t, log q^B_t) and every
  score a sampler uses is exact (step file 03, step 2). Summed noised scores under reverse diffusion
  still target the product of the noised experts, not the noised product, so exact experts do not
  make reverse diffusion exact (step file 02, step 2).
Outputs: the drip's thread folder under artifacts/drips/, with dispatch.md recording my notation,
the worked numbers I used and what clicked.
Target: I derive q_t and v_t from the kernel density estimate without notes, reproduce v_t = 0.5229
at t = 50 and the -2.6288 instance by hand, and say which sampler the exactness helps and which it
does not. I judge.
Hand-off: the thread folder's path, which step 7's cross-links read.
```

added: the definition, why it is the optimum, and the 0.04 nats (from step file 03, step 1, and the `poe-overlap-dial` idea map, run 2) · the worked numbers and the check on v_t at t = 50 (from `walk-numbers.json`) · the mixture score with softmax weights (from the exact-mixture-score plan) · the line between an exact target and an exact sampler (from step files 02 and 03) · layer 2's drip folder as the notation source · the assumptions behind the optimum (whole-distribution objective, unlimited capacity, global optimum), the smoothed set as a chosen clean expert with σ separate from the schedule's 0.0251, and the step-by-step derivation of the mixture score (from the Codex mathematics review)
dropped: "in 2D a trained expert adds nothing past its smoothed data" as a general claim, narrowed to the one run that measured it (from the Codex mathematics review)

⏱ you 45 to 90 min · AI ~10 min · stop: you derive the Outcome below without notes

**Outcome:** you can derive the ideal expert's score and say why its composition is exact at every noise level.

### 3. You: quiz 3.2

Quiz 3.2: (URL lands here)

⏱ you 45 to 90 min, the banked problems included · stop: the bar, then the build. Below the bar, the misses become the first rungs of a second `/drip --math`, and you retake before building.

### 4a. You: the build, walk03.py

```text
/drip-execute-plan
artifacts/drips/simple-distributions-rebuild/03-ideal-experts-and-the-seven-kinds.md --depth
objective, I type every block myself into ~/Playground/PhD/simple-distributions-walk/walk03.py

Task: walk the step file one step per message in its order, steps 0 to 13, teaching all but steps 6
(disjunction) and 9 (dominance). I type every block myself into
~/Playground/PhD/simple-distributions-walk/walk03.py and run it from that folder with
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python and
PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy.
Starts from: the setup's folder and environment; walk02.py from layer 2; quiz 3.2 passed; the layer
3 --math drip folder under artifacts/drips/, whose dispatch.md holds my notation.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The reference is rebuild/ideal.py, rebuild/kinds.py and rebuild/phase3_kinds.py; every worked
  number prints from python -m rebuild.walk_numbers, run from toy/, as 25 p3. keys; the
  kinds notebook's printed numbers are in
  artifacts/ideas/poe-overlap-dial/kinds-of-composition-on-the-papers-toys.ipynb, and the rebuilt
  run beside them in rebuild/out/phase3/phase3.json (step file 03, opening).
- rebuild/phase3_kinds.py compares the reference with the notebook's printed numbers and never
  reads walk03.py: exact numbers to the notebook's printed precision, sampler numbers within 0.03
  (rebuild/phase3_kinds.py, docstring; REBUILD_WALK.md, brick "Ideal experts and the seven kinds").
- rng = np.random.default_rng(0) draws the training data in exactly the kinds notebook's order, and
  everything else random uses rng2, so typed in order the data equal the notebook's (step file 03,
  step 0). Steps 6 and 9 each draw training data from rng, so their show_kind lines are typed even
  though their teaching is skipped; leaving them out shifts every later kind's data off the
  notebook's (step file 03, steps 0, 4, 6 and 9).
- σ = 0.05, N = 800; the grid is 241 × 241 over [−1.1, 1.1]², cell area 8.403e-05, 58081 points
  (step file 03, steps 0 and 3; p3.grid_cell_area, p3.grid_points).
- Step 1's instance gives -2.6288 by formula and scikit-learn; step 2's v_t is 1.0, 0.5229 and
  0.00313 at t = 99, 50 and 0 (p3. keys).
- The perfect-sample baseline (printed as floor TV) is the total variation, on 40 × 40 bins, of n
  perfect draws from the grid density, each jittered inside its cell; for 2,000 points it is about
  0.10 (0.11 in the notebook's draw, 0.10 in the step file's run). It is random, with a standard
  deviation of 0.0062 to 0.0095 over 200 redraws (p3.perfect_tv_200_redraws_sd_range), so a
  sampler can score below it by chance (step file 03, steps 4 and 5).
- Exact measures: intersection 0.71 (mass within 0.08 of a ring mode, ring × a bar of half-width
  0.05), co-occurrence 0.81 (mass in the crossing), contention 0.67 (mass within 0.12 of the two
  apart positions, half of each set rotated), interaction −0.41 (mean x1 in the band at f = 0.3),
  negation 0.10 at w = 0.05 (step file 03, steps 5, 7, 8, 10 and 11).
- The dials: contention falls from 0.657 at 5% rotated to 0.002 at 0%, the chimera column of step
  8; negation at the paper's w = 0.3 leaves 0.863 of the mass away from every ring mode. Dials draw
  fresh data and move up to about 0.08 from the notebook's curves where a curve is steep (step file
  03, step 12).
- Step 13 on contention printed, against exact 0.67: UHA 0.62 with TV 0.14, tuned MALA 0.58 with TV
  0.18, reverse diffusion 0.60 with TV 0.24, and 1000 perfect draws at TV 0.13 (step file 03, step
  13).
- Dominance defeats every sampler (TV 0.29 to 0.58, against a perfect-draw TV of 0.11 in that
  run), because no local sampler moves mass between separated modes to match unequal weights; it
  is skipped here (step file 03, step 13; artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 4
  and run 4).
Outputs:
- walk03.py: one script that runs end to end and prints each step's worked number and each kind's
  "exact measure, floor TV" line;
- step11-seven-kinds-exact.png, the seven exact compositions in a row, in the typing folder;
- the step file's steps ticked, and the walk map that holds my place between sittings.
Target: every p3. number walk03.py prints equals walk_numbers to the digits shown; the exact
measures print 0.71, 0.81, 0.67 and −0.41 for intersection, co-occurrence, contention and
interaction, the notebook's and phase3_kinds' values to two decimals; the contention dial prints
0.002 at 0% rotated. I judge, from the printed lines.
Progress: step 1 prints -2.6288 twice; step 2 the three v_t values; step 3 the cell area and grid
points; steps 5, 7, 8 and 10 one "exact measure, floor TV" line each; step 12 the dial curves; step
13 one line per sampler on contention.
Proof: python walk03.py prints "intersection: exact measure 0.71", "co-occurrence: exact measure
0.81", "contention: exact measure 0.67" and "interaction: exact measure -0.41", and the comparison
in step 5 of this layer shows no differing number.
Limits: never edit rebuild/ or the step file's code blocks; I type, I do not paste; no commit.
Hand-off: the retrospective the walk closes on, and the /drip-retro command it prints.
```

added: the reference files, the 7 p3. keys, the draw-order rule and its consequence for the two skipped steps, the grid, the floor, the exact measures, the dials, step 13's printed numbers and dominance's failure (from step file 03, `walk-numbers.json`, `phase3_kinds.py`, `REBUILD_WALK.md` and the `poe-overlap-dial` idea map) · the step range, 0 to 13, with the two skipped teachings named · the floor renamed the perfect-sample baseline, with its spread over repeated draws, and dominance's 0.11 called a perfect-draw TV rather than a floor (from the Codex mathematics review) · the p3. key count, 25, and the redraw spread at four decimals, 0.0062 to 0.0095, in place of 0.006 to 0.010, which rounded 0.0095 up (from the port of the review's checks into walk_numbers)
dropped: "bar: python -m rebuild.phase3_kinds prints ok on co-occurrence, intersection, contention and interaction against my code" from the first line (why: `phase3_kinds` compares the reference with the notebook and never reads `walk03.py`, so its `ok` says nothing about my code; the same four exact numbers, printed by my script, are the bar instead)

⏱ you 1.5 to 3 h (12 of the 14 steps; disjunction and dominance are skipped) · AI ~30 min · stop: the prompt's bar

**Outcome:** your own ideal experts, plus four kinds of composition, each shown succeeding and failing.

### 4b. You: the idea-walk rebuild, walk03b.py

```text
/drip-execute-plan --request rebuild run 2 of
artifacts/ideas/poe-overlap-dial/kinds-of-composition-on-the-papers-toys.ipynb by hand in
~/Playground/PhD/simple-distributions-walk/walk03b.py; bar: with 5% of each set rotated, the exact
composition puts 0.75 of its mass on co-occurrence while reverse diffusion draws a chimera column
with 0.21, and at 2% overlap reverse diffusion, ULA and MALA add a dry-dog cluster that tuned MALA
and UHA do not

Task: cut run 2 into 4 to 6 pieces, write the step file at the open, and build, teach and test one
piece per round. I type every block into walk03b.py and run it with
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python and
PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy.
Starts from: walk03.py from step 4a, whose exact grid, perfect-sample baseline, kinds and dials run
2 reuses;
walk02c.py from layer 2, whose tuned MALA it reuses; the notebook named above, whose printed run 2
table is the answer key.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- Run 2 runs three kinds at the steep points of Figure 2, where a sampler's error and the data's
  limit could be mistaken for each other: co-occurrence with A's top edge at 0.4, contention with 5%
  of each set rotated, interaction at 2% overlap (notebook, cell 17).
- The samplers: perfect draws, reverse diffusion with the composed score, ULA and MALA at 0.05·v_t,
  tuned MALA adapted to 0.6 acceptance, and UHA at 0.3·√v_t with damping 0.5 and 3 leapfrog steps;
  2,000 samples each (notebook, cells 2 and 18).
- The notebook's run 2 table, measure and TV per sampler. Co-occurrence at top edge 0.4, exact 0.44:
  perfect 0.43 / 0.11, reverse diffusion 0.38 / 0.22, ULA 0.40 / 0.19, MALA 0.42 / 0.18, tuned MALA
  0.42 / 0.13, UHA 0.43 / 0.13. Contention at 5% rotated, exact 0.75: perfect 0.73 / 0.10, reverse
  diffusion 0.21 / 0.69, ULA 0.26 / 0.57, MALA 0.24 / 0.57, tuned MALA 0.53 / 0.30, UHA 0.53 /
  0.27. Interaction at 2% overlap, exact −0.38: perfect −0.38 / 0.05, reverse diffusion −0.35 /
  0.59, ULA −0.37 / 0.26, MALA −0.37 / 0.27, tuned MALA −0.38 / 0.08, UHA −0.38 / 0.08 (notebook,
  cell 18).
- At contention 5% rotated, reverse diffusion's samples form one column at x1 ≈ 0 between the bars,
  a chimera the exact composition does not contain; at interaction 2%, reverse diffusion, ULA and
  MALA put a second cluster at the dry position at the band's bottom edge (notebook, cell 19).
- A weak sampler near the edge of what the data supports draws the kind's failure picture, which
  without the exact composition beside it would be blamed on missing data; only the samplers that
  come closest to the perfect-sample baseline in 2D (tuned MALA, UHA) are read on image results
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 4 and run 6).
- rebuild/phase3_kinds.py reproduces run 2 within 0.05 of the notebook on every row, its own
  tolerance for these sampler numbers (rebuild/phase3_kinds.py, Run 2;
  rebuild/out/phase3/phase3.json, run2).
Outputs:
- walk03b.py, printing one block per steep setting: the exact measure, then measure and TV per
  sampler;
- a figure of the three settings with one panel per sampler, in the typing folder;
- the step file the walk writes at its open, ticked piece by piece.
Target: the exact measures print 0.44, 0.75 and −0.38; every sampler row's measure and TV are within
0.05 of the notebook's run 2 table, the tolerance phase3_kinds holds the reference to; the figure
shows reverse diffusion's chimera column at contention 5% and the dry-dog cluster at interaction
2%. I judge, from the printed table and the figure.
Progress: piece 1 prints the three exact measures; each sampler piece prints its row for all three
settings; the figure piece saves the panels.
Proof: python walk03b.py prints the three blocks, and step 5's comparison with python -m
rebuild.phase3_kinds (Run 2) shows each row within 0.05.
Limits: never edit the notebook or rebuild/; I type, I do not paste; no commit.
Hand-off: the retrospective the walk closes on, and the /drip-retro command it prints.
```

added: run 2's three settings, the sampler settings, the full run 2 table, the two failure pictures, the claim 4 repair and the reference's tolerance (from the kinds notebook, the `poe-overlap-dial` idea map, claim 4 and run 6, and `phase3_kinds.py`) · walk03.py and walk02c.py as the parts the rebuild reuses
dropped: "samplers that reach the floor", since one perfect draw is random and not a floor; now "the samplers that come closest to the perfect-sample baseline" (from the Codex mathematics review)

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

This layer has no scene, so the flow skips step 6. Run the `/drip-retro` command step 4a's build printed at its close (⏱ about 25 minutes). Then paste this:

```text
In the background, dispatch xref-cadence over the files layer 3 of the composition ladder wrote: the
layer 3 --math drip folder under artifacts/drips/, the step file the --request walk of step 4b
wrote at its open, and the retrospectives the two layer 3 walks closed on.

Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The reference sets to link against are the vault notes, journey plans and textbook chapter on
  the Study first line of
  artifacts/drips/simple-distributions-rebuild/ladder/03-ideal-experts-and-the-seven-kinds.md,
  including artifacts/ideas/poe-overlap-dial/composition-terms.md; the layer 3 row of the Bricks
  table in artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md ("Ideal experts and the
  seven kinds"); claims 4, 7 and 8 and runs 2, 3, 4 and 6 of
  artifacts/ideas/poe-overlap-dial/IDEA_MAP.md; and claim 4 of
  artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md.
- ~/Playground/PhD/simple-distributions-walk/ is scratch space; link nothing into it (CLAUDE.md, the
  playground is disposable).
Outputs: links written both ways between those files and the reference sets.
Hand-off: the agent's report, listing every link it wrote, read before layer 4 starts.
```

added: the reference sets, by path (from this layer's Study first line, `REBUILD_WALK.md` and the two idea maps)
dropped: nothing

⏱ you 2 min to paste · AI ~10 min (estimate)

**Skip:** disjunction and dominance. Dominance is unequal mode weights that no sampler here fixes (total variation `0.29` to `0.58`, against a perfect-draw total variation of `0.11` in that run). It is a sampler problem, not a composition one, and it stays open in `artifacts/ideas/poe-overlap-dial/IDEA_MAP.md`, claims 4 and 8.

---

⬅️ [Layer 2: composing](02-composing-two-experts.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 4: the parts world](04-the-parts-world.md) ➡️
