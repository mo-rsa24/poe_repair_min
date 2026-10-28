# Layer 6, step 2: quizzes 6.1 and 6.2

⬅️ [Layer 6, step 2](../06-guidance-as-a-dial.md#2-ai-in-the-background-quizzes-61-and-62)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
when you start the reading stop, so both quizzes are waiting at layer 6.

```text
In the background, prepare and build layer 6's two quizzes, bottom up, for the composition ladder:
quiz 6.1 and quiz 6.2.

Background
The composition ladder in artifacts/drips/simple-distributions-rebuild/ walks from one 2D diffusion
expert to a correction for product-of-experts composition. Layer 6 makes guidance a dial, because
Stable Diffusion's cat-dog hybrid happens at weight 7.5 and everything before it ran at 1. Each
layer opens on two quizzes, one on the college maths underneath it and one on the layer itself,
built the way step 2 of artifacts/drips/simple-distributions-rebuild/ladder/00-setup.md builds them
for layer 1. Layer 6 has no step file yet; its build writes one (step 6 of the ladder file). The
repo is /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain.

Objectives
1. For each quiz, dispatch prep-cadence (quiz-prep-pulse --build) with the goal "write the guided
   target exactly and find where the blend appears" and the minutes given below.
2. For each quiz, once its coverage file exists, dispatch quiz-cadence in build mode on this repo.
3. Quiz 6.1, 40 minutes, covers the college maths under layer 6: raising a density to a power and
   renormalising, and Bayes' rule p(A|x) ∝ p(x|A)p(A).
4. Quiz 6.2, 60 minutes, covers layer 6 itself: classifier-free guidance as p(x)·p(A|x)^w·p(B|x)^w,
   what w > 1 does to the modes, and w = 1 as construction (1).
5. Write each published URL into its step of layer 6's ladder file.

What to read
- artifacts/drips/simple-distributions-rebuild/ladder/06-guidance-as-a-dial.md, its Study first
  line: the material for quiz 6.1, and with the rest of the file, for quiz 6.2.
- artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, the "Guidance as a tilted density"
  row of the Bricks table: material for quiz 6.2.
- artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md, claim 1, and
  artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 3 and run 1: material for quiz 6.2.
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
- The layer's textbook passage is Goodfellow et al., Deep Learning 17.5.1 (tempering, from p615),
  and its video is Umar Jamil on classifier-free guidance, 22:20 to 31:00 (the Study first line).
- Raising a Gaussian to the power w multiplies its precision by w (vault note Powers Of Gaussians).
- SDXL draws its cat-dog hybrid with one model and two prompts at w = 7.5, as
  ε_∅ + w(ε_A − ε_∅) + w(ε_B − ε_∅), and the 2D work measured only w = 1
  (artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md, claim 1).
- At w = 1 the target is construction (1), p(x|A)·p(x|B)/p(x), and on the four-quadrant toy with
  "contains" labels it puts 0.989 within 3 std of the top-right blob for every A∩B share r from 0.25
  to 0.001, with Z = r (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 3 and run 1).
- Numbers for w > 1 come from the p6. keys of python -m rebuild.walk_numbers and from
  rebuild/out/phase6/phase6.json once step 1's reference has run, so before then a question asks for
  the derivation, never an invented number (layer 6, step 1).

Rules
- Change nothing in rebuild/ or the step files.
- In the ladder file, write only the two URLs.
- Never invent a number for w > 1 that the reference has not printed.
- Report each quiz's question count, node count and verdicts from prep-cadence before its page is
  built, and each URL as quiz-cadence publishes it.

What to hand back
- Two coverage files under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, each giving
  every question a verdict: ready, blocked, or unprepared by choice.
- A row in the worked-examples index for every banked problem.
- Two published quiz pages, each opening on a read-first list with minutes per item, a short path
  and a full path.
- Quiz 6.1's URL in step 3 and quiz 6.2's URL in step 5 of
  artifacts/drips/simple-distributions-rebuild/ladder/06-guidance-as-a-dial.md, each replacing
  "(URL lands here)".
- The proof: both coverage files exist under .quiz-prep/ and list no question as blocked, and both
  URLs open.

What you will see
Along the way: per quiz, prep-cadence's question count, node count and verdicts, then each URL as
quiz-cadence publishes it.
At the end, a closing report: what was asked, in one sentence; what was done, with each coverage
file's path and each URL; PASS or MISS against no question blocked and about 6 to 10 banked problems
per concept at the build depth on quiz 6.2; what is still open; at most one question for me.

I judge, from the two pages: done when no question is blocked and every number in an answer traces to the answer key, a p6. key or a cited textbook page.
```

added: the seven sections, from the field-labelled version · the layer's reason as Background (from the ladder file's "Why it is critical" line)
dropped: the garbled source "fails everywhere).md, step 2)" on the PyMuPDF fact (why: a broken edit; the fact now cites step 2 of `ladder/00-setup.md`, the source the fragment points at) · "the way the setup section of PROMPTS.md does" and "plan PROMPTS.md (layer 6)" stay dropped (why: the guide holds neither; the setup's quiz prompt is step 2 of `ladder/00-setup.md`, and the material is this ladder file and its brick row)
