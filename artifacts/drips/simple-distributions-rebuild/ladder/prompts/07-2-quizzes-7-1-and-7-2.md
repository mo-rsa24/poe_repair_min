# Layer 7, step 2: quizzes 7.1 and 7.2

⬅️ [Layer 7, step 2](../07-a-without-b.md#2-ai-in-the-background-quizzes-71-and-72)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
when you start layer 6's build, so both quizzes are waiting at layer 7.

```text
In the background, prepare and build layer 7's two quizzes, bottom up, for the composition ladder:
quiz 7.1 and quiz 7.2.

Background
The composition ladder in artifacts/drips/simple-distributions-rebuild/ walks from one 2D diffusion
expert to a correction for product-of-experts composition. Layer 7 is A without B: the medical
cells are A without B, and at the paper's own weight negation sends mass off every mode when the
experts have sharp tails. Each layer opens on two quizzes, one on the college maths underneath it
and one on the layer itself, built the way step 2 of
artifacts/drips/simple-distributions-rebuild/ladder/00-setup.md builds them for layer 1. Layer 7
has no step file yet; its build writes one (step 6 of the ladder file). The repo is
/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain.

Objectives
1. For each quiz, dispatch prep-cadence (quiz-prep-pulse --build) with the goal "say when A without
   B stays a density" and the minutes given below.
2. For each quiz, once its coverage file exists, dispatch quiz-cadence in build mode on this repo.
3. Quiz 7.1, 40 minutes, covers the college maths under layer 7: the quotient of two Gaussians, and
   when an integral over the real line converges.
4. Quiz 7.2, 60 minutes, covers layer 7 itself: when (1 + w) log p_A − w log p_B stays a proper
   density, and why a positive label p(no drain|x) ≤ 1 stays bounded.
5. Write each published URL into its step of layer 7's ladder file.

What to read
- artifacts/drips/simple-distributions-rebuild/ladder/07-a-without-b.md, its Study first line: the
  material for quiz 7.1, and with the rest of the file, for quiz 7.2.
- artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, the "A without B (negation)" row of
  the Bricks table: material for quiz 7.2.
- artifacts/ideas/poe-overlap-dial/IDEA_MAP.md (run 3) and
  artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md (claims 6 and 8): material for quiz 7.2.
- ~/Obsidian/Mathematics/references/textbooks/Textbooks.md: the shelf index.
- ~/Obsidian/Mathematics/references/exercises/Worked Examples.md: the worked-examples index.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- quiz-cadence stops when a quiz has no coverage file in .quiz-prep/, so prep-cadence runs first for
  each quiz (quiz-cadence, "The preparation gate", as step 2 of ladder/00-setup.md states).
- pdftotext is not installed on mscluster, so
  ~/.claude/skills/index-source/scripts/find-in-textbooks.py fails; open the PDFs with PyMuPDF from
  the co3 environment, where import pymupdf works (version 1.28.2, checked on mscluster109), and
  import pypdf fails everywhere (step 2 of ladder/00-setup.md).
- The shelf index is ~/Obsidian/Mathematics/references/textbooks/Textbooks.md and the
  worked-examples index is ~/Obsidian/Mathematics/references/exercises/Worked Examples.md
  (quiz-prep-pulse, "The bank").
- The layer's textbook passage is Anton et al., Calculus 7.8 (improper integrals, p571), and its
  vault notes are Products And Quotients Of Diagonal Gaussians and Product, Mixture And Negation
  (the Study first line).
- For Gaussians, q_a/q_b^α is a density only while 1/v_a − α/v_b > 0 (vault notes Product, Mixture
  And Negation and Powers Of Gaussians).
- Where p_B → 0, p_B^(−w) → ∞, so negation pushes mass toward wherever B is least likely (step file
  02, step 4).
- On the ring and bar, the exact mass left on the 2 in-bar ring modes is 0.146, 0.121, 0.085,
  0.037, 0.0, 0.0 at w = 0, 0.02, 0.05, 0.1, 0.3, 1.0, the mass away from every ring mode is 0.387,
  0.384, 0.391, 0.439, 0.863, 1.0, and negation works only near w ≈ 0.05
  (artifacts/ideas/poe-overlap-dial/kinds-of-composition-on-the-papers-toys.ipynb, Figure 2;
  artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, run 3).
- The medical cells wanted are pneumothorax without drain and drain without pneumothorax, a
  negation; 1,906 of 3,543 NEATX pneumothorax films have no drain, and what is missing is a drain
  label on negatives (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claims 6 and 8).
- Numbers for the quadrant toy, the parts world and the positive label come from the p7. keys of
  python -m rebuild.walk_numbers and from rebuild/out/phase7/phase7.json once step 1's reference has
  run, so before then a question asks for the derivation, never an invented number (layer 7, step
  1).

Rules
- Change nothing in rebuild/ or the step files.
- In the ladder file, write only the two URLs.
- Never invent a number the reference has not printed.
- Report each quiz's question count, node count and verdicts from prep-cadence before its page is
  built, and each URL as quiz-cadence publishes it.

What to hand back
- Two coverage files under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, each giving
  every question a verdict: ready, blocked, or unprepared by choice.
- A row in the worked-examples index for every banked problem.
- Two published quiz pages, each opening on a read-first list with minutes per item, a short path
  and a full path.
- Quiz 7.1's URL in step 3 and quiz 7.2's URL in step 5 of
  artifacts/drips/simple-distributions-rebuild/ladder/07-a-without-b.md, each replacing "(URL lands
  here)".
- The proof: both coverage files exist under .quiz-prep/ and list no question as blocked, and both
  URLs open.

What you will see
Along the way: per quiz, prep-cadence's question count, node count and verdicts, then each URL as
quiz-cadence publishes it.
At the end, a closing report: what was asked, in one sentence; what was done, with each coverage
file's path and each URL; PASS or MISS against no question blocked and about 6 to 10 banked problems
per concept at the build depth on quiz 7.2; what is still open; at most one question for me.

I judge, from the two pages: done when no question is blocked and every number in an answer traces to the kinds notebook, a p7. key or a cited textbook page.
```

added: the seven sections, from the field-labelled version · the layer's reason as Background (from the ladder file's "Why it is critical" line)
dropped: the garbled source "fails everywhere).md, step 2)" on the PyMuPDF fact (why: a broken edit; the fact now cites step 2 of `ladder/00-setup.md`, the source the fragment points at) · "the way the setup section of PROMPTS.md does" and "plan PROMPTS.md (layer 7)" stay dropped (why: the guide holds neither; the setup's quiz prompt is step 2 of `ladder/00-setup.md`, and the material is this ladder file and its brick row)
