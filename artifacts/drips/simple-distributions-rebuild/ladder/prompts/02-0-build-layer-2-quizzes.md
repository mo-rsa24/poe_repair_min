# Layer 2, step 0: build layer 2's quizzes

⬅️ [Layer 2, step 0](../02-composing-two-experts.md#0-ai-in-the-background-quizzes-21-and-22-one-layer-ahead)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`
when you start layer 1's build, so both quizzes are waiting at layer 2.

```text
In the background, prepare and build layer 2's two quizzes for the composition ladder, bottom up:
quiz 2.1 and quiz 2.2.

Background
The composition ladder in artifacts/drips/simple-distributions-rebuild/ walks from one 2D diffusion
expert to a correction for product-of-experts composition. Layer 2 composes two experts and samples
the composition with an accept-corrected sampler. Each layer opens on two quizzes, one on the
college maths underneath it and one on the layer itself. These two are built during layer 1's
build, so the layer 2 maths drip has not run and the notation is the step file's. The repo is
/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain.

Objectives
1. For each quiz, dispatch prep-cadence (quiz-prep-pulse --build) with the goal "compose two
   experts and use an accept-corrected sampler whose fixed kernel has the composition as its
   invariant density" and the minutes given below.
2. For each quiz, once its coverage file exists, dispatch quiz-cadence in build mode on the repo
   above.
3. Quiz 2.1, 40 minutes, covers the college maths under layer 2: normalising constants, Markov
   chains and the Metropolis-Hastings accept ratio, total variation distance, and a Langevin step
   as gradient ascent plus noise.
4. Quiz 2.2, 130 minutes, covers layer 2 itself from the step file 02-composing-and-sampling.md:
   the two constructions, why summed scores miss the product, MALA's proposal divisor 4·ss, and
   the three numbers every composition reports.
5. Write the missing total-variation passage where quiz-prep-pulse --build files passages.
6. Write each published URL into its ladder step.

What to read
- artifacts/drips/simple-distributions-rebuild/ladder/02-composing-two-experts.md, its Study first
  line: the material for quiz 2.1.
- artifacts/drips/simple-distributions-rebuild/02-composing-and-sampling.md: the material for quiz
  2.2, with a textbook page and timed video segments beside each fundamental it uses.
- ~/Obsidian/Mathematics/references/textbooks/Textbooks.md: the shelf index.
- ~/Obsidian/Mathematics/references/exercises/Worked Examples.md: the worked-examples index.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- quiz-cadence stops when a quiz has no coverage file in .quiz-prep/, so prep-cadence runs first
  for each quiz (quiz-cadence, "The preparation gate").
- The step file links a textbook page beside each fundamental it uses and carries timestamped video
  tables on the product of experts, the product of two Gaussians, total variation distance,
  Langevin sampling, Metropolis-Hastings and annealed Langevin, which are ready nodes to reuse
  before searching again (step file 02; the guide, "When a step stalls on the maths").
- No timed video segment uses a Langevin step as the Metropolis-Hastings proposal (MALA) or reads
  the share of proposals accepted (step file 02, step 6's video table; the guide).
- Total variation distance has no passage that teaches it yet; plan 23, "what r_t costs the
  samples", only writes it from histograms, so quiz 2.1's preparation writes the missing passage
  (layer 2's Study first line).
- The textbook for the accept ratio is Bishop, Pattern Recognition and Machine Learning 11.2.2,
  p557 (layer 2's Study first line).
- pdftotext is not installed on mscluster, so
  ~/.claude/skills/index-source/scripts/find-in-textbooks.py fails there; open the PDFs with
  PyMuPDF from the co3 environment, where import pymupdf works (version 1.28.2, checked on
  mscluster109) and import pypdf fails everywhere.
- The shelf index and the worked-examples index are the two files under What to read
  (quiz-prep-pulse, "The bank").
- The notation is the step file's: the product p(x) = p_A(x) p_B(x) / Z; one ULA move
  x' = x + ss·∇log p(x) + √(2ss)·z; the MALA proposal N(x'; x + ss·∇log p(x), 2ss·I), whose
  log-density divides by 4·ss (step file 02, steps 1, 5 and 6).
- Every worked number prints from python -m rebuild.walk_numbers as the 53 p2. keys in
  toy/rebuild/out/walk-numbers.json (toy/rebuild/out/walk-numbers.json).
- The 14 p2.step9_ keys among them are read from toy/rebuild/out/phase2/step9.json, produced by
  python -m rebuild.phase2_step9 (toy/rebuild/out/walk-numbers.json).
- Among the keys are p2.prod_logZ -2.2655, p2.noised_product_mean 0.6931 against
  p2.product_of_noised_mean 0.2326, and p2.mala_fixed_log_alpha 0.045 against
  p2.mala_old_log_alpha -0.36 (toy/rebuild/out/walk-numbers.json).
- The three numbers every composition reports are the share in A∩B, the total variation to the
  exact composed density and the normaliser Z (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md,
  claim 5).

Rules
- Change nothing in the step files' prose or in toy/rebuild/.
- In the ladder files, write only the two URLs.
- Report each quiz's question count, node count and verdicts from prep-cadence before its page is
  built.

What to hand back
- Two coverage files under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, each giving
  every question a verdict: ready, blocked, or unprepared by choice.
- The missing total-variation passage, written where quiz-prep-pulse --build files passages.
- A row in the worked-examples index for every banked problem.
- Two published quiz pages, each opening on a read-first list with minutes per item, a short path
  and a full path.
- Quiz 2.1's URL in step 1 and quiz 2.2's URL in step 3 of
  artifacts/drips/simple-distributions-rebuild/ladder/02-composing-two-experts.md, each replacing
  "(URL lands here)".
- The proof: the two coverage files exist under .quiz-prep/ and list no question as blocked, and
  the two URLs open.

What you will see
Along the way: per quiz, prep-cadence's question count, node count and verdicts, then each URL as
quiz-cadence publishes it.
At the end, a closing report: what was asked, in one sentence; what was done, with each coverage
file's path, the total-variation passage's path and each URL; PASS or MISS against no question
blocked and about 4 to 6 banked problems per concept at the own depth (total variation distance)
and 6 to 10 at the build depth; what is still open; at most one question for me.

I judge, from the two pages: done when no question is blocked and every number in an answer traces to walk_numbers, the step file or a cited textbook page.
```

added: the seven sections, from the field-labelled version · a report of coverage counts before each page is built (from the old Progress line) · the notation's p2. key facts split into one sentence each
dropped: nothing (the old note on dropping "a sampler whose answer is exact" from the goal still holds: the goal names an accept-corrected sampler whose fixed kernel leaves the composition invariant, from the Codex mathematics review, since no finite run is exact)
