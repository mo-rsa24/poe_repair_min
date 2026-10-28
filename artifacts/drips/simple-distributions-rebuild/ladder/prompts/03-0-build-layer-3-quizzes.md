# Layer 3, step 0: build layer 3's quizzes

⬅️ [Layer 3, step 0](../03-ideal-experts-and-the-seven-kinds.md#0-ai-in-the-background-quizzes-31-and-32-one-layer-ahead)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`
when you start layer 2's build.

```text
In the background, prepare and build layer 3's two quizzes for the composition ladder, bottom up:
quiz 3.1 and quiz 3.2.

Background
The composition ladder in artifacts/drips/simple-distributions-rebuild/ walks from one 2D diffusion
expert to a correction for product-of-experts composition. Layer 3 builds the ideal expert, the
training points smoothed by a Gaussian, and grades every composition against an exact target. Each
layer opens on two quizzes, one on the college maths underneath it and one on the layer itself.
These two are built during layer 2's build, so the layer 3 maths drip has not run and the notation
is the step file's. The repo is /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain.

Objectives
1. For each quiz, dispatch prep-cadence (quiz-prep-pulse --build) with the goal "build the ideal
   expert and grade every composition against an exact target" and the minutes given below.
2. For each quiz, once its coverage file exists, dispatch quiz-cadence in build mode on the repo
   above.
3. Quiz 3.1, 40 minutes, covers the college maths under layer 3: convolving a density with a
   Gaussian, kernel density estimates, and softmax weights over training points.
4. Quiz 3.2, 90 minutes, covers layer 3 itself from the step file
   03-ideal-experts-and-the-seven-kinds.md: the ideal expert's score, why its composition is exact
   at every noise level, the perfect-sample baseline, co-occurrence, intersection, contention and
   interaction.
5. Write each published URL into its ladder step.

What to read
- artifacts/drips/simple-distributions-rebuild/ladder/03-ideal-experts-and-the-seven-kinds.md, its
  Study first line: the material for quiz 3.1.
- artifacts/drips/simple-distributions-rebuild/03-ideal-experts-and-the-seven-kinds.md: the material
  for quiz 3.2, with a textbook page and timed video segments beside each fundamental it uses.
- ~/Obsidian/Mathematics/references/textbooks/Textbooks.md: the shelf index.
- ~/Obsidian/Mathematics/references/exercises/Worked Examples.md: the worked-examples index.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- quiz-cadence stops when a quiz has no coverage file in .quiz-prep/, so prep-cadence runs first
  for each quiz (quiz-cadence, "The preparation gate").
- The step file links a textbook page beside each fundamental it uses and carries timestamped video
  tables on the kernel density estimate, the sum of two independent Gaussians, total variation
  distance and the product of experts, which are ready nodes to reuse before searching again (step
  file 03; the guide, "When a step stalls on the maths").
- No listed creator works the variance rule Var(aX + bY) = a²Var(X) + b²Var(Y) for independent X
  and Y (step file 03, steps 1 and 2, video tables).
- The kernel-density segment never says a Gaussian kernel density estimate is a Gaussian mixture
  with one equal-weight component per point (step file 03, steps 1 and 2, video tables).
- The textbooks are Murphy, Probabilistic Machine Learning 16.3 (kernel density estimation, p605)
  and Chan, Introduction to Probability for Data Science 5.5 (the sum of two random variables,
  p296) (layer 3's Study first line).
- Smoothing by a Gaussian as a convolution is equation 2.3 of the conditional-expectation
  textbook's chapter 02 (layer 3's Study first line).
- The score of a Gaussian mixture is Σ r_i(x)(μ_i − x)/σ², with r_i the posterior softmax weights
  over the components (the plan "the exact score of a Gaussian mixture", linked on layer 3's Study
  first line).
- pdftotext is not installed on mscluster, so
  ~/.claude/skills/index-source/scripts/find-in-textbooks.py fails there; open the PDFs with
  PyMuPDF from the co3 environment, where import pymupdf works (version 1.28.2, checked on
  mscluster109) and import pypdf fails everywhere.
- The shelf index and the worked-examples index are the two files under What to read
  (quiz-prep-pulse, "The bank").
- The notation is the step file's: the ideal expert p_σ(x) = (1/N)Σ N(x; x_i, σ²I); its noised
  density q_t(x) = (1/N)Σ N(x; √ᾱ_t x_i, v_t I) with v_t = 1 − ᾱ_t + ᾱ_t σ²; σ = 0.05 and N = 800
  (step file 03, steps 0 to 2).
- Every worked number prints from python -m rebuild.walk_numbers as a p3. key in
  toy/rebuild/out/walk-numbers.json, 25 in all (toy/rebuild/out/walk-numbers.json).
- 7 of the 25 are the numbers walk03.py prints: the three-point log-density -2.6288 by formula and
  by scikit-learn, v_t = 1.0, 0.5229 and 0.00313 at t = 99, 50 and 0, the cell area 8.403e-05 and
  58081 grid points (toy/rebuild/out/walk-numbers.json).
- The other 18 back the step file's derivations, among them the 200 perfect-draw redraws and
  dominance's mode weights (toy/rebuild/out/walk-numbers.json).
- The perfect-sample baseline (the code prints it as floor TV) is the total variation, on 40 × 40
  bins, between n exact draws from the grid density and the density itself, and a sampler is read
  against it, never against 0 (step file 03, step 4).
- The perfect-sample baseline is random, not a lower bound: a sampler can score below one perfect
  draw by chance (step file 03, step 4).
- The exact measures of the four kinds the goal needs are intersection 0.71, co-occurrence 0.81,
  contention 0.67 and interaction −0.41 (step file 03, steps 5, 7, 8 and 10).
- Trained 2D networks match the ideal experts within 0.04 nats
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, run 2).

Rules
- Change nothing in the step files' prose or in toy/rebuild/.
- In the ladder files, write only the two URLs.
- Report each quiz's question count, node count and verdicts from prep-cadence before its page is
  built.

What to hand back
- Two coverage files under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, each giving
  every question a verdict: ready, blocked, or unprepared by choice.
- A row in the worked-examples index for every banked problem.
- Two published quiz pages, each opening on a read-first list with minutes per item, a short path
  and a full path.
- Quiz 3.1's URL in step 1 and quiz 3.2's URL in step 3 of
  artifacts/drips/simple-distributions-rebuild/ladder/03-ideal-experts-and-the-seven-kinds.md, each
  replacing "(URL lands here)".
- The proof: the two coverage files exist under .quiz-prep/ and list no question as blocked, and
  the two URLs open.

What you will see
Along the way: per quiz, prep-cadence's question count, node count and verdicts, then each URL as
quiz-cadence publishes it.
At the end, a closing report: what was asked, in one sentence; what was done, with each coverage
file's path and each URL; PASS or MISS against no question blocked and about 6 to 10 banked
problems per concept at the build depth; what is still open; at most one question for me.

I judge, from the two pages: done when no question is blocked and every number in an answer traces to walk_numbers, the step file or a cited textbook page.
```

added: the seven sections, from the field-labelled version · a report of coverage counts before each page is built (from the old Progress line)
dropped: nothing (the earlier version had already renamed the floor the perfect-sample baseline and marked it random rather than a lower bound, from the Codex mathematics review)
