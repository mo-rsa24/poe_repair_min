# Layer 8, step 0: quizzes 8.1 and 8.2

⬅️ [Layer 8, step 0](../08-the-interaction-term.md#0-ai-in-the-background-the-layer-8-reference-and-quizzes-81-and-82-one-layer-ahead)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
when you start layer 7's build, beside step 0's reference prompt.

```text
In the background, prepare and build layer 8's two quizzes, bottom up, for the composition ladder:
quiz 8.1 and quiz 8.2.

Background
The composition ladder in artifacts/drips/simple-distributions-rebuild/ walks from one 2D diffusion
expert to a correction for product-of-experts composition. Layer 8 defines the interaction term
exactly, the first half of the research goal. Each layer opens on two quizzes, one on the college
maths underneath it and one on the layer itself. These run during layer 7's build, so the layer 8
maths drip has not run and layer 8 has no step file yet; the layer 8 reference is built beside
them, so its p8. keys do not exist yet either. The repo is
/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain.

Objectives
1. For each quiz, dispatch prep-cadence (quiz-prep-pulse --build) with the goal "define the
   interaction term exactly and say where it lives" and the minutes given below.
2. For each quiz, once its coverage file exists, dispatch quiz-cadence in build mode on this repo.
3. Quiz 8.1, 40 minutes, covers the college maths under layer 8: Bayes' rule for two labels,
   conditional independence, and log-odds.
4. Quiz 8.2, 60 minutes, covers layer 8 itself: r(x) = log p(x|A,B) − [log p(x|A) + log p(x|B) −
   log p(x)], why it is constant when the labels are fixed functions of the image, its ε-space form
   ε_joint − ε_A − ε_B + ε_∅, and the label term for report-style labels.
5. Write the missing log-odds passage during quiz 8.1's preparation.
6. Write each published URL into its step of layer 8's ladder file.

What to read
- artifacts/drips/simple-distributions-rebuild/ladder/08-the-interaction-term.md, its Study first
  line: the material for quiz 8.1.
- The same file's steps 2, 4a and 4b: material for quiz 8.2.
- artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, brick row "The interaction term,
  exact": material for quiz 8.2.
- artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 7 (7.1 to 7.4): material for quiz 8.2.
- ~/Obsidian/Mathematics/references/textbooks/Textbooks.md: the shelf index.
- ~/Obsidian/Mathematics/references/exercises/Worked Examples.md: the worked-examples index.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- quiz-cadence stops when a quiz has no coverage file in .quiz-prep/, so prep-cadence runs first for
  each quiz (quiz-cadence, "The preparation gate").
- The textbook for Bayes is Deisenroth et al. 6.3, p189, and log-odds has no passage that teaches it
  yet beyond Murphy 10.2 (binary logistic regression, p377), so quiz 8.1's preparation writes the
  missing passage (this layer's Study first line).
- pdftotext is not installed on mscluster, so
  ~/.claude/skills/index-source/scripts/find-in-textbooks.py fails; open the PDFs with PyMuPDF from
  the co3 environment, where import pymupdf works (version 1.28.2, checked on mscluster109), and
  import pypdf fails everywhere (the textbook pass).
- The shelf index is ~/Obsidian/Mathematics/references/textbooks/Textbooks.md and the
  worked-examples index is ~/Obsidian/Mathematics/references/exercises/Worked Examples.md
  (quiz-prep-pulse, "The bank").
- By Bayes, r(x) = log p(A,B|x) − log p(A|x) − log p(B|x) + log[p(A)p(B)/p(A,B)], and when each
  label is a fixed function of the image the term is exactly zero, so the wet dog is a data
  condition (does either expert's data hold A as it looks under B), and the shift was found at 20%
  overlap and missed at 0% (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 7.1 and run 1;
  artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 9).
- With report-mined labels a label term survives: on claim 7.4's quadrant labels (A read with
  probability 0.9 when true, B read inside A∩B only when A was read) it is +0.105 nats for (A,B)
  inside A∩B, strongly negative for (¬A,B) there, and zero elsewhere, and +0.105 is −ln 0.9
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claims 7.3 and 7.4).
- A chest X-ray diagnosis is read from its findings with uncertainty, so diagnosis labels carry the
  label term while finding labels given the diagnosis do not
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 7, route 2).
- The notation is q(x_t | x_0) = N(√ᾱ_t x_0, (1 − ᾱ_t) I) over T = 100 steps of a cosine schedule,
  with ε = −√(1 − ᾱ_t)·∇log q_t, so ∇r_t = −(ε_joint − ε_A − ε_B + ε_∅)/√(1 − ᾱ_t)
  (ladder/01-one-diffusion-expert.md, step 2; REBUILD_WALK.md, brick "The interaction term,
  exact").
- Every number an answer uses comes from the idea maps, step files 04-the-parts-world.md and
  05-the-overlap-dial.md, rebuild/out/walk-numbers.json or a cited textbook page, because the p8.
  keys do not exist yet while the layer 8 reference is built beside these quizzes (layer 8, step
  0).

Rules
- Change nothing in the step files' prose, the idea maps or rebuild/.
- In the ladder files, write only the two URLs.
- Report each quiz's question count, node count and verdicts from prep-cadence before its page is
  built, and each URL as quiz-cadence publishes it.

What to hand back
- Two coverage files under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, each giving
  every question a verdict: ready, blocked, or unprepared by choice.
- The missing log-odds passage, written where quiz-prep-pulse --build files passages.
- A row in the worked-examples index for every banked problem.
- Two published quiz pages, each opening on a read-first list with minutes per item, a short path
  and a full path.
- Quiz 8.1's URL in step 1 and quiz 8.2's URL in step 3 of
  artifacts/drips/simple-distributions-rebuild/ladder/08-the-interaction-term.md, each replacing
  "(URL lands here)".
- The proof: the two coverage files exist under .quiz-prep/ and list no question as blocked, and the
  two URLs open.

What you will see
Along the way: per quiz, prep-cadence's question count, node count and verdicts, the log-odds
passage's path once written, then each URL as quiz-cadence publishes it.
At the end, a closing report: what was asked, in one sentence; what was done, with each coverage
file's path, the passage's path and each URL; PASS or MISS against no question blocked and about 6
to 10 banked problems per concept at the build depth on quiz 8.2 (r(x), its ε-space form, the label
term); what is still open; at most one question for me.

I judge, from the two pages: done when no question is blocked and every number in an answer traces to an idea map, step file 04 or 05, walk-numbers.json or a cited textbook page.
```

added: the seven sections, from the field-labelled version · the log-odds passage as an objective of its own (from the earlier Outputs) · the full path of the idea map behind the route 2 diagnosis fact (from the claim it cites)
dropped: nothing (the earlier version had already replaced "the setup section of PROMPTS.md" and "plan PROMPTS.md (layer 8) with REBUILD_WALK.md" with this file and its brick row, since the guide holds neither)
