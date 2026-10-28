# Layer 12, step 1: build the two quizzes

⬅️ [Layer 12, step 1](../12-the-correction-at-scale.md#1-ai-in-the-background-layer-12s-quizzes)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`, when layer 9's build starts.

```text
In the background, prepare and build layer 12's two quizzes for the composition ladder, bottom up:
quiz 12.1 and quiz 12.2.

Background
The composition ladder in artifacts/drips/simple-distributions-rebuild/ walks from one 2D diffusion
expert to a correction for product-of-experts composition. Layer 12 is its last layer: the
correction at scale, read as a LoRA on SDXL. Each layer opens on two quizzes, one on the college
maths underneath it and one on the layer itself. These two are built during layer 9's build, so they
are waiting when I arrive. The repo is
/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain.

Objectives
1. For each quiz, dispatch prep-cadence (quiz-prep-pulse --build) with the goal "read the LoRA as a
   learned interaction term", the minutes and the topics below.
2. For each quiz, once its coverage file exists, dispatch quiz-cadence in build mode with the repo
   above, quiz 12.1 first so quiz 12.2 can rest on it.
3. Quiz 12.1, 40 minutes, covers the college maths under layer 12: matrix rank, a low-rank
   factorisation and its parameter count, and the singular value decomposition at gloss depth.
4. Quiz 12.2, 40 minutes, covers layer 12 itself: LoRA as a low-rank weight update, why its residual
   is the ε-space interaction term from layer 8, and compose rate as the measure.
5. Write each published URL into its ladder step.

What to read
- artifacts/drips/simple-distributions-rebuild/ladder/12-the-correction-at-scale.md, its Study first
  line: the material for quiz 12.1.
- The same file's steps 7 and 9: the material for quiz 12.2.
- artifacts/drips/simple-distributions-rebuild/ladder/08-the-interaction-term.md: layer 8's
  interaction term, which quiz 12.2 builds on.
- ~/Obsidian/Mathematics/references/textbooks/Textbooks.md: the shelf index.
- ~/Obsidian/Mathematics/references/exercises/Worked Examples.md: the worked-examples index.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- quiz-cadence stops when a quiz has no coverage file in .quiz-prep/, so prep-cadence runs first
  for each quiz (quiz-cadence, "The preparation gate").
- pdftotext is not installed on mscluster, so
  ~/.claude/skills/index-source/scripts/find-in-textbooks.py fails there; open the PDFs with PyMuPDF
  from the co3 environment, where import pymupdf works (version 1.28.2, checked on mscluster109) and
  import pypdf fails everywhere.
- The shelf index is ~/Obsidian/Mathematics/references/textbooks/Textbooks.md and the
  worked-examples index is ~/Obsidian/Mathematics/references/exercises/Worked Examples.md
  (quiz-prep-pulse, "The bank").
- The textbook passages are Anton and Rorres 4.8 (rank, p262) and 9.4 to 9.5 (the SVD and
  compression, p528 to p535), Deisenroth et al. 4.5 to 4.6 (p125 to p135), and Strang's MIT 18.065
  lecture 6 from 7:48 (layer 12's Study first line).
- A LoRA keeps a projection W frozen and trains a rank-r update BA, giving (W + BA)x with
  r·(in + out) parameters: 384 for the journey's toy (rank 2, 3 blocks) and 4,956,160 for the real
  case (rank 8, 210 modules) (the journey plan 08-lora-update.md, Description and Goal).
- poe_repair_min's correction is a rank-8 cross-attention LoRA trained on the cached guided residual
  r_t = ε̃_J − ε̃_PoE (Mission, in
  ~/Playground/PhD/poe_repair_min/plans/04-does-the-fix-reach-unseen-pairs/MASTER_PLAN.md).
- That residual is w·(ε_J − ε_A − ε_B + ε_∅) at guidance w = 7.5, written in the cache as
  gs*(eps_j − eps_a − eps_b + eps_uncond), and its bracket is layer 8's ε-space interaction term
  ε_joint − ε_A − ε_B + ε_∅ (the definition field of
  /datasets/mmolefe/poe_repair_min/outputs/showcase/interaction_bound/interaction_bound.json;
  REBUILD_WALK.md, brick "The interaction term, exact").
- The journey's real matrix is 440 × 65,536 cached correction rows, whose first singular value is
  133.3 (the journey plan 06-svd-of-the-real-matrix.md, Goal).
- A compose rate is the fraction of a pair's seeds where grounding-dino-tiny finds at least two
  distinct "animal" instances (NMS IoU under 0.5, confidence at least 0.30); it counts animals and
  does not identify them, so the 94% headline at full correction is an upper bound, with the true
  rate around 87% to 94% (~/Playground/PhD/poe_repair_min/context/world/compose-rate.md).

Rules
- Change nothing in poe_repair_min, the journey plans or rebuild/.
- In this ladder file, write only the two URLs.
- Report each quiz's question count, node count and verdicts before its page is built.

What to hand back
- Two coverage files under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, each giving
  every question a verdict: ready, blocked, or unprepared by choice.
- A row in the worked-examples index for every banked problem.
- Two published quiz pages, each opening on a read-first list with minutes per item, a short path
  and a full path.
- Quiz 12.1's URL in step 6 and quiz 12.2's URL in step 8 of
  artifacts/drips/simple-distributions-rebuild/ladder/12-the-correction-at-scale.md, each replacing
  "(URL lands here)".
- The proof: the two coverage files exist under .quiz-prep/ and list no question as blocked, and the
  two URLs open.

What you will see
Along the way: per quiz, prep-cadence's question count, node count and verdicts, then each URL as
quiz-cadence publishes it.
At the end, a closing report: what was asked, in one sentence; what was done, with each coverage
file's path and each URL; PASS or MISS against no question blocked on either page, about 4 to 6
banked problems for quiz 12.2's concept at the own depth, and every number in an answer traced to a
source above or a cited textbook page; what is still open; at most one question for me.

I judge, from the two pages: done when no question is blocked, quiz 12.2 carries about 4 to 6 banked problems at the own depth, and every number in an answer traces to a named source or textbook page.
```

added: the seven sections, from the field-labelled version · a report of coverage counts before each page is built (from the old Progress line)
dropped: "the way the setup section of PROMPTS.md does for layer 1" and "plan PROMPTS.md (layer 12)" as quiz 12.2's source (why: the guide holds no setup or layer 12 section; the setup is ladder/00-setup.md and quiz 12.2's material is named by path instead)
