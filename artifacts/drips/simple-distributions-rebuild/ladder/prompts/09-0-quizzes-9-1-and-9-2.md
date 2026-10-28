# Layer 9, step 0: quizzes 9.1 and 9.2

⬅️ [Layer 9, step 0](../09-the-correction-learned-back.md#0-ai-in-the-background-the-layer-9-reference-and-quizzes-91-and-92-one-layer-ahead)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
when you start the prior-art read, beside the reference build prompt for this step.

```text
In the background, prepare and build layer 9's two quizzes, bottom up, for the composition ladder:
quiz 9.1 and quiz 9.2.

Background
The composition ladder in artifacts/drips/simple-distributions-rebuild/ walks from one 2D diffusion
expert to a correction for product-of-experts composition. Each layer opens on two quizzes, one on
the college maths underneath it and one on the layer itself. Layer 9 trains a correction and reads
the fair test. These quizzes are built during the prior-art read, beside the layer 9 reference, so
nothing from layer 9 has run and layer 9 has no step file yet. The repo is
/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain.

Objectives
1. For each quiz, dispatch prep-cadence (quiz-prep-pulse --build) with the goal "train a correction
   and read the fair test" and the minutes given below.
2. For each quiz, once its coverage file exists, dispatch quiz-cadence in build mode on this repo.
3. Quiz 9.1, 40 minutes, covers the college maths under layer 9: least-squares regression recovers
   a conditional expectation, and the gradient of a scalar function.
4. Quiz 9.2, 60 minutes, covers layer 9 itself: score matching as regression, why the score
   residual is ∇r_t, the three arms of the fair test, and why trained experts cost more under a
   better sampler.
5. Write each published URL into its step of layer 9's ladder file.

What to read
- artifacts/drips/simple-distributions-rebuild/ladder/09-the-correction-learned-back.md, its Study
  first line: the material for quiz 9.1.
- The same file's steps 2 and 4: material for quiz 9.2.
- artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, brick "The correction, learned
  back": material for quiz 9.2.
- artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claims 4 and 9 and run 7: material for quiz
  9.2.
- artifacts/ideas/when-poe-reaches-a-and-b/runs/sampler-by-expert-interaction/README.md: run 7's
  card, material for quiz 9.2.
- Layer 8's two quizzes and the layer 8 --math drip folder under artifacts/drips/: the notation for
  r(x) and r_t.
- ~/Obsidian/Mathematics/references/textbooks/Textbooks.md: the shelf index.
- ~/Obsidian/Mathematics/references/exercises/Worked Examples.md: the worked-examples index.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- quiz-cadence stops when a quiz has no coverage file in .quiz-prep/, so prep-cadence runs first
  for each quiz (quiz-cadence, "The preparation gate").
- The textbooks are Chen 6.3.3 (p126) and Goodfellow et al. 18.5 (p635), with the conditional-mean
  chapter and the training-loss chapter linked on the Study first line (this layer's Study first
  line).
- The timed video is Stanford CS236 lecture 13, 29:16 to 44:39 (this layer's Study first line).
- pdftotext is not installed on mscluster, so
  ~/.claude/skills/index-source/scripts/find-in-textbooks.py fails there (the textbook pass).
- Open the PDFs with PyMuPDF from the co3 environment, where import pymupdf works (version 1.28.2,
  checked on mscluster109) and import pypdf fails everywhere (the textbook pass).
- The shelf index is ~/Obsidian/Mathematics/references/textbooks/Textbooks.md, and the
  worked-examples index is ~/Obsidian/Mathematics/references/exercises/Worked Examples.md
  (quiz-prep-pulse, "The bank").
- The notation is q(x_t | x_0) = N(√ᾱ_t x_0, (1 − ᾱ_t) I) over T = 100 steps of a cosine schedule,
  with ε = −√(1 − ᾱ_t)·∇log q_t (ladder/01-one-diffusion-expert.md, step 2).
- r_t = log q_t(x|A,B) − [log q_t(x|A) + log q_t(x|B) − log q_t(x)], whose gradient is what the
  composed score misses of the joint's score (REBUILD_WALK.md, brick "The interaction term,
  exact").
- The fair test is a joint {A, B} condition trained on only the share r of the data that is A and
  B, against composition, against composition plus a small correction fine-tuned on the same few
  A∩B samples, at r = 0.05 and 0.001 (IDEA_MAP.md, claim 9; REBUILD_WALK.md, brick "The correction,
  learned back").
- A better sampler makes expert error cost more: trained experts add 0.005 TV under reverse
  diffusion and 0.059 under annealed MALA at r = 0.05, and 0.035 and 0.266 at r = 0.001 (the run
  card, "The answer").
- That is because the trained composed score is worst at low noise, with relative error 0.34 and
  1.30 at level 0 and under 0.04 from level 30 up, where MALA spends its last steps (the run card,
  "Score error").
- Every number an answer uses comes from run 7's card, the idea map, the p8. keys in
  rebuild/out/walk-numbers.json or a cited textbook page (this step).
- The p9. keys do not exist yet, because the layer 9 reference is built beside these quizzes (this
  step).

Rules
- Change nothing in the step files' prose, the idea maps, the run card or rebuild/.
- In the ladder files, write only the two URLs.
- Report each quiz's question count, node count and verdicts from prep-cadence before its page is
  built.

What to hand back
- Two coverage files under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, each giving
  every question a verdict: ready, blocked, or unprepared by choice.
- A row in the worked-examples index for every banked problem.
- Two published quiz pages, each opening on a read-first list with minutes per item, a short path
  and a full path.
- Quiz 9.1's URL in step 1 and quiz 9.2's URL in step 3 of
  artifacts/drips/simple-distributions-rebuild/ladder/09-the-correction-learned-back.md, each
  replacing "(URL lands here)".
- The proof: the two coverage files exist under .quiz-prep/ and list no question as blocked, and the
  two URLs open.

What you will see
Along the way: per quiz, prep-cadence's question count, node count and verdicts, then each URL as
quiz-cadence publishes it.
At the end, a closing report: what was asked, in one sentence; what was done, with each coverage
file's path and each URL; PASS or MISS against no question blocked on either page and quiz 9.2
carrying about 6 to 10 banked problems per concept at the build depth (score matching as
regression with ∇r_t as its target, the fair test); what is still open; at most one question for
me.

I judge, from the two pages: done when no question is blocked and every number in an answer traces to run 7's card, the idea map, the p8. keys or a cited textbook page.
```

added: the seven sections, from the field-labelled version · a report of coverage counts before each page is built (from the old Progress line) · the ladder and the layer's two-quiz pattern as Background (from the setup quiz prompt)
dropped: nothing (the earlier version had already replaced "plan PROMPTS.md (layer 9) with REBUILD_WALK.md" with quiz 9.2's material by path, since layer 9's prompts live in its own ladder file)
