# Layer 5a, step 0: quizzes R.1 and R.2

⬅️ [Layer 5a, step 0](../05a-reading-stop-projective-composition.md#0-ai-in-the-background-quizzes-r1-and-r2-one-layer-ahead)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
when you start layer 5's build, so both quizzes are waiting at the reading stop.

```text
In the background, prepare and build the reading stop's two quizzes, bottom up, for the composition
ladder: quiz R.1 and quiz R.2.

Background
The composition ladder in artifacts/drips/simple-distributions-rebuild/ walks from one 2D diffusion
expert to a correction for product-of-experts composition. Between layer 5 (the overlap dial) and
layer 6 (guidance) sits a reading stop on two Bradley papers, arXiv 2502.04549 and 2509.16447, which
prove when summed scores are exact. The stop opens on two quizzes, one on the college maths under
it and one on the stop itself. They are built during layer 5's build, so they are waiting when I
arrive. The repo is /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain.

Objectives
1. For each quiz, dispatch prep-cadence (quiz-prep-pulse --build) with the goal "state when summed
   scores are exact and place both toys against it" and the minutes given below.
2. For each quiz, once its coverage file exists, dispatch quiz-cadence in build mode on this repo.
3. Quiz R.1, 40 minutes, covers the college maths under the stop: conditional independence, and
   change of variables with a Jacobian.
4. Quiz R.2, 80 minutes, covers the stop itself from arXiv 2502.04549 and 2509.16447, papers in the
   "understand it" role: the factorized-conditionals condition (Theorem 5.3), exactness in a
   feature space (Theorem 6.1), and where the quadrant toy and the parts world satisfy or break the
   condition.
5. Write each published URL into its step of the reading stop's ladder file.

What to read
- artifacts/drips/simple-distributions-rebuild/ladder/05a-reading-stop-projective-composition.md,
  its Study first line: the material for quiz R.1.
- In the journey folder
  ~/goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/, the
  read-aheads sources/mechanisms-of-projective-composition/read-ahead/ and
  sources/local-mechanisms-of-compositional/read-ahead/: the material for quiz R.2, with the two
  papers' pages.
- The same journey's drips/read-mechanisms-of-projective-composition/READ_MAP.md: the earlier read
  of 2502.04549, its chapter pages and its slips.
- ~/Obsidian/Mathematics/references/textbooks/Textbooks.md: the shelf index.
- ~/Obsidian/Mathematics/references/exercises/Worked Examples.md: the worked-examples index.
- ~/goal-setting/learning/papers/Papers.md: the papers shelf index.

What is already known
- Nothing from this stop has run yet: the quizzes are built during layer 5's build, so the combined
  read of step 2 has not run and no COMBINE_MAP.md exists (this stop's step 0).
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- quiz-cadence stops when a quiz has no coverage file in .quiz-prep/, so prep-cadence runs first for
  each quiz (quiz-cadence, "The preparation gate").
- The textbook pages for quiz R.1 are Deisenroth et al., Mathematics for Machine Learning 6.7
  (change of variables, p220) and 6.4 to 6.5 (conditional independence, Definition 6.11, p201)
  (this stop's Study first line).
- pdftotext is not installed on mscluster, so
  ~/.claude/skills/index-source/scripts/find-in-textbooks.py fails, and the co3 environment has no
  pypdf module either, so open the PDFs read-only with PyMuPDF (import pymupdf) from
  /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python (checked 2026-10-18; quiz-prep-pulse,
  "The bank").
- The shelf index is ~/Obsidian/Mathematics/references/textbooks/Textbooks.md, the worked-examples
  index is ~/Obsidian/Mathematics/references/exercises/Worked Examples.md, and the papers shelf
  index is ~/goal-setting/learning/papers/Papers.md (quiz-prep-pulse, "The bank" and material kind
  "Papers").
- The papers' pages, with PDF page equal to printed page, are: 2502.04549 Definition 4.1 p5;
  Definitions 5.1 and 5.2 and Theorem 5.3 p6; Remark 5.5, the importance-of-background paragraph
  and Lemma 5.7 p7; Theorem 6.1 and Lemma 6.2 p8; 2509.16447 Definitions 1 and 2 p5; Lemma 1 and
  Corollary 1 p6 (Papers.md, both papers' ideas tables).
- Ready material to reuse as nodes before searching is the read-ahead chunk files 04, 05 and 06 of
  2502.04549 and 03 of 2509.16447, each chunk with its parts taken apart and an answer key in the
  paper's checks/ folder, and the chapter and word-by-word pages in READ_MAP.md's "Chapter pages by
  chunk" section, among them chapter 11 "The Paper's Rule"
  https://claude.ai/artifact/J4bpZuFmB6q7goAwmyio8h, "Theorem 5.3, Word by Word"
  https://claude.ai/artifact/6PZ6yPrEJE1NPyD2TcVihY, "Definition 5.2"
  https://claude.ai/artifact/EC2jKjuuii8UuvE1cFxvsG and chapter 12 "Feature Space"
  https://claude.ai/artifact/GZmUjXJMyCMhpKrdMNaTZu (READ_MAP.md of the earlier read).
- The earlier read of 2502.04549 stopped at abstract sentence 6 of 9, with 167 chunks not reached,
  so the learner has not walked Definition 4.1 or anything after it (READ_MAP.md, Position).
- Definition 4.1 says p̂ is a projective composition when Π_i♯p̂ = Π_i♯p_i for every i; it is a
  property, not a construction, there may be none or many, and p̂ may sit where no p_i has support
  (read-ahead 04, chunks 37 and 38).
- Definition 5.1 is C[p_b, p_1, ...](x) = p_b(x)·∏ p_i(x)/p_b(x) / Z, with score ∇log p_b +
  Σ(∇log p_i − ∇log p_b), and with p_b the unconditional it is the Bayes composition (read-ahead 05,
  chunks 42 and 43).
- Definition 5.2 is a partition M_b, M_1, ..., M_k with (1) x on M_i independent of the rest under
  p_i, (2) all blocks mutually independent under p_b, and (3) p_i equal to p_b off M_i (read-ahead
  05, chunk 45).
- Theorem 5.3 says that under Definition 5.2, reverse diffusion with the composed noised scores
  samples p̂(x) = p_b(x on M_b)·∏ p_i(x on M_i), a projective composition, and the sampling half
  rests on composing commuting with noising, C[N_t p] = N_t C[p] (read-ahead 05, chunks 47 and 49).
- Remark 5.5 says Definition 5.2 implies orthogonal score differences and is strictly stronger; on
  the toy, deltas confined to separate coordinates over a background with correlation 0.5 compose
  to N(2.1818, 0.2364) per coordinate against the target N(2, 0.25)
  (checks/separation_not_enough.out; read-ahead 05, chunks 45 and 51).
- The same experts compose with an empty background and fail with the unconditional as background
  (Figure 3, settings A and B) (read-ahead 05, chunk 53).
- Lemma 5.7 is false as printed, since on two binary pixels the KL is log(5/3) = 0.5108 against the
  bound log(4/3) = 0.2877, and Definition 5.2 prints as "Theorem 5.2" in three places (READ_MAP.md,
  Disagreements and slips; read-ahead 05 header).
- Theorem 6.1 says that if a C¹ diffeomorphism A makes (A♯p_b, A♯p_1, ...) satisfy Definition 5.2,
  then C[p] is projective with Π_i(x) = A(x) on M_i, with no sampling claim; Lemma 6.2 behind it is
  C[A♯p] = A♯C[p], and on the toy the ratio of the two orders is constant (max/min 1.0000) for C and
  varies by 17.0000 for the simple product (checks/reparam.out; read-ahead 06, chunks 63 and 66).
- In 2509.16447 a conditional projective composition (Definition 2) is the no-condition model on the
  uncontrolled pixels times each active condition's own block, Lemma 1 says it holds exactly when
  the score is a local conditional score (Definition 1), the approximation improves at higher noise,
  and Definition 2 drops Definition 5.2's condition 2 while Lemma 1's "only if" half needs it back
  (read-ahead 03 of 2509.16447, chunks 29, 32, 35 and 37).
- The layer 5 quadrant toy has A the left half, B the top half and A∩B the top-left quadrant; expert
  A trains on 800 points with x1 ~ U(−1, 0), a share f of them with x2 ~ U(0, 1) and the rest with
  x2 ~ U(−1, 0), B mirrors it, and the experts are ideal, smoothed by σ = 0.05, and composed as the
  plain product p_A·p_B (step file 05, steps 1 and 2).
- In 2D the background is uniform, so dividing by it changes the plain product only by a constant
  and the quadrant toy tests co-occurrence (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 2).
- The four-quadrant answer key is a second toy with one blob per quadrant at (±0.5, ±0.5), std 0.15,
  A is x1 > 0 and B is x2 > 0, and experts that model only their own coordinate and take the other
  from the background marginal of all four quadrants put 0.9999 (plain product) and 0.989
  (background-divided) in the top-right blob with no A∩B data
  (artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/README.md, rescue 2b).
- Separately fine-tuned experts at 0% overlap also learn "not B" and "not A", so their top-right
  share is at most 0.5 when the leftover density sits in the two adjacent quadrants, and leftover on
  the top-right alone gives 0.9997, ladder layer 5's corrected answer key
  (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 2a and run 2).
- The parts world is a 4 × 4 image of 16 pixels, cat +1 and dog −1, each expert a Gaussian mixture
  over placements with variance 0.01 on its animal's half, and the other half has mean 0 and
  variance 0.01 (an empty half) or 1.0 (a loose one) (step file 05, step 6).
- With the empty half at 0% overlap the product is a ghost with half-means (0.5, −0.5), 400 nats of
  mismatch and A∩B weight 0.000, from 0.1% overlap the two-animal pair wins by 386.2 nats and the
  weight is 1.000, and with the loose half it is 1.000 at every overlap including 0% (step file 05,
  step 6; p5.ghost_mismatch_nats and p5.matching_pair_advantage_nats in
  rebuild/out/walk-numbers.json).
- On the parts-world mosaic at ᾱ = 0.5, summed noised scores put the head mean at 0.2326 where the
  noised product has 0.6931 (step file 04, step 6).
- Construction (1), score(x|A) + score(x|B) − score(x), targets p(x|A)·p(x|B)/p(x), the composition
  operator with the unconditional as background, construction (2) is the plain product p_A·p_B, and
  they agree only where p(x) is flat (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim
  3).
- Across an invertible warp of the four-quadrant toy, construction (1) and negation change by 4e-17
  and 7e-17 total variation, and the plain product by 0.035, 0.080 and 0.126 at largest area
  stretches of 2.6, 9 and 49 (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, run 6).

Rules
- Change nothing in the step files, rebuild/, the read-ahead folders or either READ_MAP.md.
- In the ladder files, write only the two URLs.
- Report each quiz's question count, node count and verdicts from prep-cadence before its page is
  built, and each URL as quiz-cadence publishes it.

What to hand back
- Two coverage files under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, each giving
  every question a verdict: ready, blocked, or unprepared by choice.
- A row in the worked-examples index for every banked problem and every paper question.
- Two published quiz pages, each opening on a read-first list with minutes per item, a short path
  and a full path.
- Quiz R.1's URL in step 1 and quiz R.2's URL in step 3 of
  artifacts/drips/simple-distributions-rebuild/ladder/05a-reading-stop-projective-composition.md,
  each replacing "(URL lands here)".
- The proof: the two coverage files exist under .quiz-prep/ and list no question as blocked, and the
  two URLs open.

What you will see
Along the way: per quiz, prep-cadence's question count, node count and verdicts, then each URL as
quiz-cadence publishes it.
At the end, a closing report: what was asked, in one sentence; what was done, with each coverage
file's path and each URL; PASS or MISS against no question blocked, about 4 to 6 banked problems per
concept on quiz R.1, questions on both papers by their role with each source page on quiz R.2, and
at least one question placing each toy against Definition 5.2; what is still open; at most one
question for me.

I judge, from the two pages: done when no question is blocked and every number in an answer traces to a read-ahead answer key, walk_numbers, a step file or a cited page.
```

added: the seven sections, from the field-labelled version · a report of coverage counts before each page is built (from the old Progress line) · the stop's place in the ladder as Background (from the stop's opening lines)
dropped: nothing (the earlier version had already replaced "plan PROMPTS.md (this reading stop)" with this file and the two papers, since the guide holds no reading-stop section)
