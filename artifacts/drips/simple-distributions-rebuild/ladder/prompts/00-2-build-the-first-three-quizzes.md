# Setup, step 2: build the first three quizzes

⬅️ [Setup, step 2](../00-setup.md#2-ai-in-the-background-quiz-0-and-layer-1s-quizzes)

Done once already: the three pages are linked from the setup and layer 1. Paste this only to rebuild
them, into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`.

```text
In the background, prepare and build three quizzes for the composition ladder: quiz 0, quiz 1.1 and
quiz 1.2.

Background
The composition ladder in artifacts/drips/simple-distributions-rebuild/ walks from one 2D diffusion
expert to a correction for product-of-experts composition. Each layer opens on two quizzes, one on
the college maths underneath it and one on the layer itself. These three are built before layer 1
starts, so they are waiting when I arrive. The repo is
/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain.

Objectives
1. For each quiz, dispatch prep-cadence (quiz-prep-pulse --build) with the goal "derive the ε-score
   identity and train one 2D expert" and the minutes given below.
2. For each quiz, once its coverage file exists, dispatch quiz-cadence in build mode.
3. Quiz 0, 90 minutes, covers the college maths under the whole ladder: logs and exponents, the
   chain rule, the Gaussian density and completing the square, mean and variance under scaling and
   under sums of independent variables, Bayes' rule, and the gradient of a log density.
4. Quiz 1.1, 40 minutes, covers the college maths under layer 1: √ᾱ x_0 + √(1 − ᾱ) ε as a Gaussian
   with its mean and variance, the score of a Gaussian −(x − μ)/σ², and least-squares regression.
5. Quiz 1.2, 90 minutes, covers layer 1 itself: the forward process, the sampled-noise identity
   ε = −√(1 − ᾱ_t)·∇log q(x_t | x_0) against the marginal one
   E[ε | x_t] = −√(1 − ᾱ_t)·∇log q_t(x_t) that the trained network learns, and the denoising loss.
6. Write each published URL into its ladder step.

What to read
- artifacts/drips/simple-distributions-rebuild/ladder/01-one-diffusion-expert.md and its Study
  first line: the material for quizzes 0 and 1.1.
- artifacts/drips/simple-distributions-rebuild/01-one-expert-from-scratch.md: the material for quiz
  1.2, with a textbook page and timed video segments beside each fundamental it uses.
- ~/Obsidian/Mathematics/references/textbooks/Textbooks.md: the shelf index.
- ~/Obsidian/Mathematics/references/exercises/Worked Examples.md: the worked-examples index.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- quiz-cadence stops when a quiz has no coverage file in .quiz-prep/, so prep-cadence runs first
  for each quiz (quiz-cadence, "The preparation gate").
- The step file's textbook links and video tables are ready nodes, to reuse before searching again
  (the guide, "When a step stalls on the maths").
- pdftotext is not installed on mscluster, so find-in-textbooks.py fails there; open the PDFs with
  PyMuPDF from the co3 environment, where import pymupdf works (version 1.28.2, checked on
  mscluster109) and import pypdf fails.
- The notation is the step file's: q(x_t | x_0) = N(√ᾱ_t x_0, (1 − ᾱ_t) I) over T = 100 steps of a
  cosine schedule, and every worked number prints from python -m rebuild.walk_numbers as the 41 p1.
  keys in toy/rebuild/out/walk-numbers.json (step file 01, steps 4 and 5).

Rules
- Change nothing in the step files' prose or in toy/rebuild/.
- In the ladder files, write only the three URLs.
- Report each quiz's coverage counts before its page is built.

What to hand back
- Three coverage files under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, each giving
  every question a verdict: ready, blocked, or unprepared by choice.
- A row in the worked-examples index for every banked problem.
- Three published quiz pages, each opening on a read-first list with minutes per item, a short path
  and a full path.
- Quiz 0's URL in step 3 of ladder/00-setup.md, and the URLs of quizzes 1.1 and 1.2 in steps 1 and 3
  of ladder/01-one-diffusion-expert.md, each replacing "(URL lands here)".
- The proof: the three coverage files list no question as blocked, and the three URLs open.

What you will see
Along the way: per quiz, prep-cadence's question count, node count and verdicts, then each URL as
quiz-cadence publishes it.
At the end, a closing report: what was asked, in one sentence; what was done, with each coverage
file's path and each URL; PASS or MISS against no question blocked and about 4 to 6 banked problems
per concept at the own depth and 6 to 10 at the build depth; what is still open; at most one
question for me.

I judge, from the three pages: done when no question is blocked and every number in an answer traces to walk_numbers, the step file or a cited textbook page.
```

added: the seven sections, from the field-labelled version · a report of coverage counts before each page is built (from the old Progress line)
dropped: nothing
