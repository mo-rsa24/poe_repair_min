# ⭐ Reading stop: the two Bradley papers (the read is critical; the rest is optional)

⬅️ [Layer 5: the overlap dial](05-the-overlap-dial.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 6: guidance](06-guidance-as-a-dial.md) ➡️

**Why the read is critical.** Bradley et al. 2025a (arXiv 2502.04549) prove when summed scores are exact: under factorized conditionals, where each expert matches the background outside its own region (Theorem 5.3). They also prove when a composition is exact in a feature space (Theorem 6.1). That is the mathematical reason for layer 5's finding that an expert loose outside its own region needs no overlap. Bradley 2025b (arXiv 2509.16447) ties it to a local score.

**Depth.** Own: the factorized-conditionals condition, and exactness in a feature space. Later: the CLEVR reproduction and the Bradley rebuild, both optional below. You 1.7 to 3.3 h in all.

**Study first.** The 2025a paper walked chunk by chunk: [projective composition, Definition 4.1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/sources/mechanisms-of-projective-composition/read-ahead/04-projective-composition.md) and [the simple construction and factorized conditionals, Definition 5.2 and Theorem 5.3](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/sources/mechanisms-of-projective-composition/read-ahead/05-simple-construction.md), with the vault note [Composing Models With A Background](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/Composing%20Models%20With%20A%20Background.md). An earlier full read of the same paper stopped at chunk 6 of [its walk map](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/drips/read-mechanisms-of-projective-composition/READ_MAP.md). For quiz R.1: Deisenroth et al., *Mathematics for Machine Learning* 6.7 (change of variables, p220) and 6.4 to 6.5 on [conditional independence](../../../../../../Obsidian/Mathematics/Statistics/Probability/Conditional%20Independence.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/computer-vision/computer-vision-textbook.pdf#page=219 "Computer vision textbook, section 10.1, Works it on a joint table of three discrete variables taking 4, 3 and 2 values, 24 probabilities in all, where x₁ and..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=201 "Mathematics for machine learning, section 6.4.5, Definition 6.11, p(x, y ∣ z) = p(x ∣ z) p(y ∣ z) for every z, read as 'given knowledge about z, the distribu..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=389 "Pattern recognition and machine learning, section 8.2, The definition p(a ∣ b, c) = p(a ∣ c), then three-node graphs where observing c makes a and b independ..."), [▶1](https://www.youtube.com/watch?v=zRdE8A4UZes&t=4587s "CS229 lecture 5, 1:16:27 to 1:20:34, Makes the naive Bayes assumption that words are conditionally independent given the class, cutting the parameters from 2..."), [◆](../../../../../../goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/04-conditional-independence.md "Check independence inside each slice of a third variable, deep-learning/generative-models/03-sampling-from-a-graph, plan 04. Conditional independence checked...")</sub> (Definition 6.11, p201). The same condition drawn as a graph, where c₁ and c₂ are independent given x: [conditional independence inside each slice of a third variable](../../../../../../goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/04-conditional-independence.md), [reading it off with d-separation](../../../../../../goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/16-d-separation.md) and [the product-of-experts rule derived from the fork](../../../../../../goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/17-poe-rule-from-the-fork.md).

**Ideas to tie in.** From [the ideas that tie the ladder together](../../../ideas/ideas-that-tie-the-ladder-together.md): [conditional independence made visible as an interaction residual](../../../ideas/ideas-that-tie-the-ladder-together.md#1-make-conditional-independence-visible-as-an-interaction-residual) (needed, 2 to 3 h, no training). Each comes with an exact toy check.

**Achieve.** On the whiteboard: the quadrant toy with its background, marking where each expert matches the background and where it does not · quizzes R.1 and R.2 passed.

## The flow

```text
 AI, in the background                         │  You, in order
───────────────────────────────────────────────┼───────────────────────────────────────────────
 quizzes R.1 and R.2, prepared ──────────────▶ │  1. quiz R.1: independence, Jacobians
   (paste during layer 5's build)              │       ▼
                                               │  2. /drip-read --combine --lean
                                               │       ▼
                                               │  3. quiz R.2: needed-now right, 80% of the rest
                                               │       ▼
 concept-scene, beside layer 6 ◀────────────── │  4. scene
                                               │       ▼
 xref-cadence links the stop ◀──────────────── │  5. the next layer
```

## The steps

### 0. AI, in the background: quizzes R.1 and R.2, one layer ahead

Paste this when you start layer 5's build, so both quizzes are waiting here.

```text
In the background, prepare and build the reading stop's two quizzes, bottom up, for the composition
ladder in artifacts/drips/simple-distributions-rebuild/: quiz R.1 and quiz R.2.

Task: for each quiz, first dispatch prep-cadence (quiz-prep-pulse --build) with the goal and minutes
below, then dispatch quiz-cadence (mode build, repo
/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain). The goal for both is "state when summed scores
are exact and place both toys against it".
- Quiz R.1, --minutes 40: the college maths under the stop. Conditional independence, change of
  variables with a Jacobian.
- Quiz R.2, --minutes 80: the stop itself, from arXiv 2502.04549 and 2509.16447, papers in the
  "understand it" role. The factorized-conditionals condition (Theorem 5.3), exactness in a feature
  space (Theorem 6.1), where the quadrant toy and the parts world satisfy or break the condition.
Starts from: nothing from this stop; it runs during layer 5's build, so the combined read of step 2
has not run and no COMBINE_MAP.md exists yet. Quiz R.1 takes its material from the Study first line
of artifacts/drips/simple-distributions-rebuild/ladder/05a-reading-stop-projective-composition.md.
Quiz R.2 takes it from the two papers' pages and their read-aheads, which sit in the journey folder
~/goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/ as
sources/mechanisms-of-projective-composition/read-ahead/ and
sources/local-mechanisms-of-compositional/read-ahead/.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- quiz-cadence stops when a quiz has no coverage file in .quiz-prep/, so prep-cadence runs first for
  each quiz (quiz-cadence, "The preparation gate").
- The textbook pages for quiz R.1 are Deisenroth et al., Mathematics for Machine Learning 6.7
  (change of variables, p220) and 6.4 to 6.5 (conditional independence, Definition 6.11, p201)
  (this stop's Study first line).
- pdftotext is not installed on mscluster, so
  ~/.claude/skills/index-source/scripts/find-in-textbooks.py fails. The co3 environment has no pypdf
  module either; open the PDFs read-only with PyMuPDF (import pymupdf) from
  /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python (checked 2026-10-18; quiz-prep-pulse,
  "The bank").
- The shelf index is ~/Obsidian/Mathematics/references/textbooks/Textbooks.md, the worked-examples
  index is ~/Obsidian/Mathematics/references/exercises/Worked Examples.md (quiz-prep-pulse, "The
  bank"), and the papers shelf index is ~/goal-setting/learning/papers/Papers.md (quiz-prep-pulse,
  material kind "Papers").
- The papers' pages, PDF page equal to printed page: 2502.04549 Definition 4.1 p5; Definitions 5.1
  and 5.2 and Theorem 5.3 p6; Remark 5.5, the importance-of-background paragraph and Lemma 5.7 p7;
  Theorem 6.1 and Lemma 6.2 p8. 2509.16447 Definitions 1 and 2 p5; Lemma 1 and Corollary 1 p6
  (Papers.md, both papers' ideas tables).
- Ready material to reuse as nodes before searching: the read-ahead chunk files 04, 05 and 06 of
  2502.04549 and 03 of 2509.16447, each chunk with its parts taken apart and an answer key in the
  paper's checks/ folder; and the chapter and word-by-word pages listed in the READ_MAP.md
  "Chapter pages by chunk" section, among them chapter 11 "The Paper's Rule"
  https://claude.ai/artifact/J4bpZuFmB6q7goAwmyio8h, "Theorem 5.3, Word by Word"
  https://claude.ai/artifact/6PZ6yPrEJE1NPyD2TcVihY, "Definition 5.2"
  https://claude.ai/artifact/EC2jKjuuii8UuvE1cFxvsG and chapter 12 "Feature Space"
  https://claude.ai/artifact/GZmUjXJMyCMhpKrdMNaTZu (READ_MAP.md of the earlier read).
- The earlier read of 2502.04549 stopped at abstract sentence 6 of 9, with 167 chunks not reached,
  so the learner has not walked Definition 4.1 or anything after it (READ_MAP.md, Position).
- Definition 4.1: p̂ is a projective composition when Π_i♯p̂ = Π_i♯p_i for every i. It is a
  property, not a construction; there may be none or many, and p̂ may sit where no p_i has support
  (read-ahead 04, chunks 37 and 38).
- Definition 5.1: C[p_b, p_1, ...](x) = p_b(x)·∏ p_i(x)/p_b(x) / Z, with score ∇log p_b + Σ(∇log
  p_i − ∇log p_b); with p_b the unconditional it is the Bayes composition (read-ahead 05, chunks 42
  and 43).
- Definition 5.2: a partition M_b, M_1, ..., M_k with (1) x on M_i independent of the rest under
  p_i, (2) all blocks mutually independent under p_b, (3) p_i equal to p_b off M_i (read-ahead 05,
  chunk 45).
- Theorem 5.3: under Definition 5.2, reverse diffusion with the composed noised scores samples
  p̂(x) = p_b(x on M_b)·∏ p_i(x on M_i), a projective composition; the sampling half rests on
  composing commuting with noising, C[N_t p] = N_t C[p] (read-ahead 05, chunks 47 and 49).
- Remark 5.5: Definition 5.2 implies orthogonal score differences and is strictly stronger. On the
  toy, deltas confined to separate coordinates over a background with correlation 0.5 compose to
  N(2.1818, 0.2364) per coordinate against the target N(2, 0.25)
  (checks/separation_not_enough.out; read-ahead 05, chunks 45 and 51).
- The same experts compose with an empty background and fail with the unconditional as background
  (Figure 3, settings A and B) (read-ahead 05, chunk 53).
- Lemma 5.7 is false as printed: on two binary pixels the KL is log(5/3) = 0.5108 against the bound
  log(4/3) = 0.2877; and Definition 5.2 prints as "Theorem 5.2" in three places (READ_MAP.md,
  Disagreements and slips; read-ahead 05 header).
- Theorem 6.1: if a C¹ diffeomorphism A makes (A♯p_b, A♯p_1, ...) satisfy Definition 5.2, then
  C[p] is projective with Π_i(x) = A(x) on M_i, with no sampling claim. Lemma 6.2 behind it:
  C[A♯p] = A♯C[p]; on the toy the ratio of the two orders is constant (max/min 1.0000) for C and
  varies by 17.0000 for the simple product (checks/reparam.out; read-ahead 06, chunks 63 and 66).
- 2509.16447: a conditional projective composition (Definition 2) is the no-condition model on the
  uncontrolled pixels times each active condition's own block; Lemma 1 says it holds exactly when
  the score is a local conditional score (Definition 1); the approximation improves at higher noise.
  Definition 2 drops Definition 5.2's condition 2, and Lemma 1's "only if" half needs it back
  (read-ahead 03 of 2509.16447, chunks 29, 32, 35 and 37).
- The layer 5 quadrant toy: A is the left half, B the top half, A∩B the top-left quadrant. Expert A
  trains on 800 points with x1 ~ U(−1, 0), a share f of them with x2 ~ U(0, 1) and the rest with
  x2 ~ U(−1, 0); B mirrors it. Experts are ideal, smoothed by σ = 0.05, and composed as the plain
  product p_A·p_B (step file 05, steps 1 and 2).
- In 2D the background is uniform, so dividing by it changes the plain product only by a constant
  and the quadrant toy tests co-occurrence (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 2).
- The four-quadrant answer key is a second toy: one blob per quadrant at (±0.5, ±0.5), std 0.15, A
  is x1 > 0, B is x2 > 0. Experts that model only their own coordinate and take the other from the
  background marginal of all four quadrants put 0.9999 (plain product) and 0.989
  (background-divided) in the top-right blob with no A∩B data
  (artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/README.md, rescue 2b).
- Separately fine-tuned experts at 0% overlap also learn "not B" and "not A", so their top-right
  share is at most 0.5 when the leftover density sits in the two adjacent quadrants (leftover on the
  top-right alone gives 0.9997, ladder layer 5's corrected answer key)
  (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 2a and run 2).
- The parts world: a 4 × 4 image of 16 pixels, cat +1 and dog −1, each expert a Gaussian mixture
  over placements with variance 0.01 on its animal's half; the other half has mean 0 and variance
  0.01 (an empty half) or 1.0 (a loose one) (step file 05, step 6).
- With the empty half at 0% overlap the product is a ghost with half-means (0.5, −0.5), 400 nats
  of mismatch and A∩B weight 0.000; from 0.1% overlap the two-animal pair wins by 386.2 nats and
  the weight is 1.000. With the loose half it is 1.000 at every overlap including 0% (step file 05,
  step 6; p5.ghost_mismatch_nats and p5.matching_pair_advantage_nats in
  rebuild/out/walk-numbers.json).
- On the parts-world mosaic at ᾱ = 0.5, summed noised scores put the head mean at 0.2326 where the
  noised product has 0.6931 (step file 04, step 6).
- Construction (1), score(x|A) + score(x|B) − score(x), targets p(x|A)·p(x|B)/p(x), the composition
  operator with the unconditional as background; construction (2) is the plain product p_A·p_B;
  they agree only where p(x) is flat (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim
  3).
- Across an invertible warp of the four-quadrant toy, construction (1) and negation change by 4e-17
  and 7e-17 total variation, and the plain product by 0.035, 0.080 and 0.126 at largest area
  stretches of 2.6, 9 and 49 (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, run 6).
Outputs:
- two coverage files under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, each with a
  verdict per question: ready, blocked, or unprepared by choice;
- rows in the worked-examples index for every banked problem and every paper question;
- two published quiz pages, each opening on a read-first list with minutes per item, a short path
  and a full path.
Target: no question on either page carries the verdict blocked; quiz R.1 carries banked problems,
about 4 to 6 per concept; quiz R.2 carries questions on both papers by their role, each with its
source page, and at least one question placing each toy against Definition 5.2; every number in an
answer traces to a read-ahead answer key, walk_numbers, a step file or a cited page. I judge, from
the pages.
Progress: prep-cadence reports, per quiz, its question count, node count and verdicts; quiz-cadence
reports each URL as it publishes.
Proof: the two coverage files exist under .quiz-prep/ and list no question as blocked; the two
URLs open.
Limits: change nothing in the step files, rebuild/, the read-ahead folders or either READ_MAP.md; in
the ladder files, write only the two URLs.
Hand-off: write quiz R.1's URL into step 1 and quiz R.2's URL into step 3 of
artifacts/drips/simple-distributions-rebuild/ladder/05a-reading-stop-projective-composition.md, each
replacing "(URL lands here)".
```

added: every field the setup's quiz prompt carries, so this prompt stands alone instead of pointing at "the setup section of PROMPTS.md", which the guide no longer holds (the setup is `ladder/00-setup.md`) · the preparation gate (from the quiz-cadence agent definition) · the textbook pages (from this stop's Study first line) · the missing `pdftotext` and the PyMuPDF route, with `pypdf` absent from `co3` (checked on mscluster109, and quiz-prep-pulse's bank step) · the papers' pages (from `Papers.md`) · the read-ahead files and chapter pages as ready nodes, and where the earlier read stopped (from the two read-aheads and `READ_MAP.md`) · the definitions and theorems with their toy checks and the two printed slips (from read-ahead 04, 05 and 06 of 2502.04549, read-ahead 03 of 2509.16447 and `READ_MAP.md`) · both quadrant toys and the parts world with their numbers (from step files 04 and 05, `walk-numbers.json`, the answer-key card and both idea maps) · the URL write-back
dropped: "plan PROMPTS.md (this reading stop)" as quiz R.2's source (why: the guide holds no reading-stop section; the source is this file and the two papers)

⏱ AI ~1.5 h (estimate), beside layer 5's build

### 1. You: quiz R.1

Quiz R.1: (URL lands here)

⏱ you 20 to 40 min · stop: the bar, every *needed now* question right and at least 80% of the rest; below that, read the read-first items your misses point at and retake.

### 2. You: the combined read

```text
/drip-read --combine --lean "projective composition, factorized conditionals and local scores" with
arXiv 2502.04549 and 2509.16447, read against the quadrant toy and the parts world of
artifacts/drips/simple-distributions-rebuild/

Task: read the one idea across both papers in combine mode, one idea of the spine per round, each
passage as printed with its plain reading and the four comparison lines. My reason, for message 1:
to state when summed scores are exact and place the quadrant toy and the parts world against that
condition. Keep both papers. For "your plan says", offer the vault note "Composing Models With A
Background" and journey 06's plan, in
~/goal-setting/learning/deep-learning/diffusion-models/06-diffusion-speciation-dynamics/ as
plans/05-composition-and-the-experiment/plans/04-projective-composition-condition.md; I choose at
message 1.
Starts from: quiz R.1 passed, its misses as the first ideas to slow on. In the journey folder
~/goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/:
the read-aheads sources/mechanisms-of-projective-composition/read-ahead/ and
sources/local-mechanisms-of-compositional/read-ahead/, and the earlier read's map
drips/read-mechanisms-of-projective-composition/READ_MAP.md.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The PDFs are on the shelf under ~/goal-setting/learning/papers/, in
  deep-learning/diffusion-models/compositional-diffusion-models/compositional-generalization/ as
  mechanisms-of-projective-composition-of-diffusion-models.pdf and
  local-mechanisms-of-compositional-generalization-in-conditional-diffusion.pdf; PDF page equals
  printed page (Papers.md).
- Where the idea sits: 2502.04549 Definition 4.1 p5; Definitions 5.1 and 5.2 and Theorem 5.3 p6;
  Remark 5.5, the importance-of-background paragraph and Lemma 5.7 p7; Theorem 6.1 and Lemma 6.2
  p8. 2509.16447 Definitions 1 and 2 p5; Lemma 1, its relaxation and Corollary 1 p6 (Papers.md).
- The earlier read of 2502.04549 stopped at abstract sentence 6 of 9; Definition 4.1 onward is
  unread by me, so no round may assume it (READ_MAP.md, Position).
- Definition 4.1: p̂ is a projective composition when Π_i♯p̂ = Π_i♯p_i for every i; a property,
  with none or many solutions, allowed outside every p_i's support (read-ahead 04, chunks 37, 38).
- Definition 5.1: C[p_b, p_1, ...](x) = p_b(x)·∏ p_i(x)/p_b(x) / Z; with p_b the unconditional it
  is the Bayes composition (read-ahead 05, chunks 42 and 43).
- Definition 5.2: a partition M_b, M_1, ..., M_k with (1) x on M_i independent of the rest under
  p_i, (2) all blocks mutually independent under p_b, (3) p_i equal to p_b off M_i (read-ahead 05,
  chunk 45).
- Theorem 5.3: under Definition 5.2, reverse diffusion with the composed noised scores samples
  p̂(x) = p_b(x on M_b)·∏ p_i(x on M_i); sampling rests on C[N_t p] = N_t C[p] (read-ahead 05,
  chunks 47 and 49).
- Remark 5.5: Definition 5.2 implies orthogonal score differences and is strictly stronger; deltas
  confined to separate coordinates over a background with correlation 0.5 give N(2.1818, 0.2364)
  per coordinate against the target N(2, 0.25) (checks/separation_not_enough.out; read-ahead 05,
  chunks 45 and 51).
- The same experts compose with an empty background and fail with the unconditional (Figure 3, A
  against B) (read-ahead 05, chunk 53).
- Two printed slips: Lemma 5.7 is false as printed (KL log(5/3) = 0.5108 against the bound log(4/3)
  = 0.2877 on two binary pixels), and Definition 5.2 prints as "Theorem 5.2" in three places
  (READ_MAP.md, Disagreements and slips).
- Theorem 6.1: a C¹ diffeomorphism A that makes the pushed-forward inputs satisfy Definition 5.2
  makes C[p] projective with Π_i(x) = A(x) on M_i, with no sampling claim; Lemma 6.2, C[A♯p] =
  A♯C[p], holds for C and fails for the simple product (ratio max/min 1.0000 against 17.0000,
  checks/reparam.out) (read-ahead 06, chunks 63 and 66).
- 2509.16447: Lemma 1 says a conditional projective composition (Definition 2) holds exactly when
  the score is a local conditional score (Definition 1); Definition 2 drops Definition 5.2's
  condition 2, and the "only if" half needs it back; the approximation improves at higher noise
  (read-ahead 03 of 2509.16447, chunks 32, 35 and 37).
- The layer 5 quadrant toy: A the left half, B the top half; expert A's 800 points have x1 ~
  U(−1, 0), a share f with x2 ~ U(0, 1) and the rest x2 ~ U(−1, 0); B mirrors it; ideal experts
  smoothed by σ = 0.05, composed as the plain product (step file 05, steps 1 and 2).
- In 2D the background is uniform, so dividing by it changes the plain product only by a constant
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 2).
- The four-quadrant answer key (blobs at (±0.5, ±0.5), std 0.15): experts that model only their own
  coordinate and take the other from the background marginal put 0.9999 (plain) and 0.989
  (background-divided) in the top-right blob with no A∩B data
  (artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/README.md).
- The parts world: 16 pixels, cat +1, dog −1, variance 0.01 on the animal's half, the other half
  mean 0 with variance 0.01 (empty) or 1.0 (loose). Empty at 0% overlap: a ghost with half-means
  (0.5, −0.5) and A∩B weight 0.000, 1.000 from 0.1%. Loose: 1.000 at every overlap including 0%
  (step file 05, step 6; p5.ghost_mismatch_nats 400.0, p5.matching_pair_advantage_nats 386.2).
- On the parts-world mosaic at ᾱ = 0.5 the summed noised scores put the head mean at 0.2326 where
  the noised product has 0.6931 (step file 04, step 6).
- Construction (1) targets p(x|A)·p(x|B)/p(x), the operator with the unconditional as background;
  construction (2) is the plain product; they agree only where p(x) is flat; across an invertible
  warp of the four-quadrant toy, construction (1) changes by 4e-17 and the plain product by 0.035
  to 0.126 total variation (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 3 and run
  6).
Outputs:
- COMBINE_MAP.md under
  ~/goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/drips/
  in its combine-<idea-slug>/ folder, holding the Sources table, the idea spine with every idea
  walked, compared or tested, the symbol table and the differences table;
- combined-reading.md in the journey's Obsidian folder, written by the skill at the last idea;
- the retrospective the read closes on.
Target: with the map closed I state five things: Definition 5.2's three conditions, Theorem 5.3's
conclusion, Theorem 6.1's conclusion, and for each of the quadrant toy and the parts world whether
it satisfies or breaks the condition and why. 5 of 5 matched against the papers' statements; I
judge, from the read's closing check.
Progress: each round names one idea of the spine and ends on its four comparison lines; the
differences table gains a row per difference named; a `test` prints its numbers labelled ran.
Proof: the closing check prints its pass or miss line against the target, and every idea in
COMBINE_MAP.md's spine carries a state.
Limits: combine mode builds nothing, so no code, notebook or object; write nothing into either
READ_MAP.md or the read-ahead folders; never edit rebuild/ or the step files; no commit.
Hand-off: the COMBINE_MAP.md path, which step 4's scene and step 5's cross-links read, and the
retrospective's /drip-retro command.
```

added: the reason and the two candidate home sources for message 1 (from the combine reference and `READ_MAP.md`'s Terms table) · the shelf PDFs and their pages (from `Papers.md`) · where the earlier read stopped · the definitions, theorems, slips and toy checks (from the read-aheads and `READ_MAP.md`) · both quadrant toys, the parts world and the two constructions with their numbers (from step files 04 and 05, `walk-numbers.json`, the answer-key card and both idea maps) · the five-statement target in place of "you state the condition and place both toys against it", which had no count
dropped: nothing

⏱ you 40 to 80 min (the one measured `--lean` read, of a single paper, took 38 minutes over 6 sittings) · AI ~1 h · stop: you state the condition and place both toys against it

**Outcome:** you can state the factorized-conditionals condition and show where the quadrant toy and the parts world satisfy it or break it.

### 3. You: quiz R.2

Quiz R.2: (URL lands here)

⏱ you 40 to 80 min · stop: the bar, every *needed now* question right and at least 80% of the rest.

### 4. AI, in the background: the scene

```text
/concept-scene projective composition: the background-divided product and the
factorized-conditionals condition on the quadrant toy, with the background as a choice

Starts from: the combined read's COMBINE_MAP.md under
~/goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/drips/ in its
combine-<idea-slug>/ folder (step 2's hand-off), for my words and the differences the read named.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The scene's toy is the four-quadrant answer key: one Gaussian blob per quadrant at (±0.5, ±0.5),
  std 0.15, A is x1 > 0, B is x2 > 0, products on a 441 × 441 grid over [−1.1, 1.1]²
  (artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/README.md).
- Experts that model only their own coordinate and take the other from the background marginal of
  all four quadrants satisfy Definition 5.2; with no A∩B data they put 0.9999 (plain product) and
  0.989 (background-divided, p_A·p_B/p_bg) in the top-right blob (same card, rescue 2b).
- Experts trained from scratch on one quadrant each put 0.989 at the origin and 0.006 in the
  top-right blob, normaliser 7.9e-10 (same card, construction (2) at ε = 0).
- One conditional model with "contains" labels, composed as p(x|A)·p(x|B)/p(x), keeps 0.989 in the
  top-right blob for every A∩B share r from 0.25 down to 0.001, normaliser equal to r (same card,
  construction (1)).
- 0.989 recurs because one blob holds 1 − e^(−9/2) of its mass within 3 std (when-poe IDEA_MAP.md,
  run 4).
- Definition 5.2's three conditions: x on M_i independent of the rest under p_i; all blocks
  independent under p_b; p_i equal to p_b off M_i (read-ahead 05 of 2502.04549, chunk 45).
- The same experts compose with an empty background and fail with the unconditional as background
  (2502.04549 Figure 3, A and B; read-ahead 05, chunk 53).
- In 2D a uniform background changes the plain product only by a constant
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 2).
- Scenes already built, not to redraw differently: "Three ways a product reaches A-and-B"
  https://claude.ai/artifact/9eFnG2rWEciPsceMwdem4u and "Where the Product Lands"
  https://claude.ai/artifact/2A7YDnfKe8EPsS7yFDVcNT on this toy, and the 2502.04549 chapter "Divide
  Out the Wall" https://claude.ai/artifact/KS2mvPznS1VKNdWz7B6f85 (ladder/05-the-overlap-dial.md;
  when-poe IDEA_MAP.md, run 4; READ_MAP.md).
- Every number comes from answer_key.json, scene_key.py in
  artifacts/scenes/when-a-product-reaches-a-and-b/checks/, or a key script written before the build,
  never estimated.
Outputs: one published scene where the background is a control (empty, the four-quadrant marginal,
the unconditional) and each of Definition 5.2's three conditions shows as holding or breaking.
Target: every number on the scene equals the answer key or the scene's own key script; I judge.
Hand-off: write the scene's URL into step 4 of
artifacts/drips/simple-distributions-rebuild/ladder/05a-reading-stop-projective-composition.md,
replacing "(URL lands here)".
```

added: the four-quadrant answer key as the scene's toy, since its background-divided numbers exist and layer 5's square toy has none (from the answer-key card) · the answer-key numbers and why 0.989 recurs (from the card and the when-poe idea map) · Definition 5.2 and Figure 3 (from read-ahead 05) · the three scenes not to redraw (from layer 5's ladder file, the when-poe idea map and `READ_MAP.md`) · the URL write-back
dropped: nothing

Scene: (URL lands here)

⏱ you ~20 min · AI ~45 min, beside layer 6's maths · stop: the scene is published

**Outcome:** a scene where you change the background on the quadrant toy and watch the factorized-conditionals condition hold or break.

### 5. You: close the stop

Run the `/drip-retro` command the read printed at its close (⏱ about 25 minutes). Then paste this:

```text
In the background, dispatch xref-cadence over the files the reading stop of the composition ladder
wrote: the combined read's COMBINE_MAP.md and combined-reading.md, and the retrospective the read
closed on.

Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- COMBINE_MAP.md sits under
  ~/goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/drips/ in its
  combine-<idea-slug>/ folder (step 2's hand-off).
- The reference sets to link against are: the read-ahead files and the vault note on the Study first
  line of
  artifacts/drips/simple-distributions-rebuild/ladder/05a-reading-stop-projective-composition.md;
  the layer 5 rows of the Bricks table in
  artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md ("Overlap dial" and "Fine-tune
  variant"), since the stop has no brick row of its own; claims 2 and 3 of
  artifacts/ideas/poe-overlap-dial/IDEA_MAP.md; and claims 2 (child 2b), 3 and 10 of
  artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md.
- Each READ_MAP.md is written only by its walking session, so neither earlier read's map takes a
  link (READ_MAP.md, header).
- ~/Playground/PhD/simple-distributions-walk/ is scratch space; link nothing into it (CLAUDE.md, the
  playground is disposable).
Outputs: links written both ways between those files and the reference sets.
Hand-off: the agent's report, listing every link it wrote, read before layer 6 starts.
```

added: the reference sets, by path (from this stop's Study first line, `REBUILD_WALK.md` and the claims the stop rests on in both idea maps) · the READ_MAP write rule (from `READ_MAP.md`)
dropped: nothing

⏱ you 2 min to paste · AI ~10 min (estimate)

### ◇ Optional, expensive: the paper's CLEVR figures

```text
/drip-execute-plan --reproduce 2502.04549

Task: reproduce the paper's CLEVR figures on mscluster, one target per round, picking at the open
only what the 2D toys and the parts world cannot stand in for.
Starts from: this stop's combined read closed; for the target list, the read-ahead in the journey
folder ~/goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/ at
sources/mechanisms-of-projective-composition/read-ahead/; the repo
/home-mscluster/mmolefe/Playground/PhD/projective-composition and its
runbook/training-edm2-on-clevr.md.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The authors released no code; their CLEVR models were built on NVIDIA's EDM2, cloned at
  reference/edm2 in the repo at commit 4bf8162 (READ_MAP.md, Position; the repo's README.md).
- The paper trained edm2-img64-xs (124M weights) at batch 2048 on 128 × 2^20 images, 16 hours on
  32 A100s, without classifier-free guidance, sampling with a 32-step Heun sampler (runbook, quoting
  the paper's appendix).
- On mscluster: CLEVR v1.0 at /datasets/mmolefe/projective-composition/data/CLEVR_v1.0; two 20,000
  image subsets at 64 × 64, clevr-color-20k-64 (one random object labelled by colour) and
  clevr-none-20k-64 (unlabelled); EDM2-xs checkpoints clevr-color-20k-64-xs-8Mi (conditional) and
  clevr-none-20k-64-xs-8Mi (background) under /datasets/mmolefe/projective-composition/checkpoints/,
  last snapshot network-snapshot-0008388 (runbook steps 2 and 3; the checkpoint listing).
- scripts/compose_clevr.py composes two conditional denoisers minus the background with EDM2's
  Heun sampler; it has run only on untrained step-0 snapshots (runbook step 4).
- Training runs in co3_bw on a Blackwell node (mscluster110); the cluster's EDM2 copy needs the
  InfiniteSampler fix on torch 2.11; two jobs together run at 6.43 s per thousand images each
  (runbook steps 1 and 3).
- The official CLEVR holds 3 to 10 objects per scene (70,000 training images, 17.7 GiB); a subset
  reaches the cluttered composition (Figure 3 C, Table 1's cluttered rows), and the single-object
  and empty-background experiments need renders nobody released (READ_MAP.md, the object section).
- The paper's measure is a hand count of 100 samples per method; two "Single-object Bayes" rows of
  Table 1 sum to 109 and 43; Appendix B's 65 network calls in 32 Heun steps is 63 in EDM2's sampler
  (runbook step 5; READ_MAP.md, Disagreements and slips).
- The CLEVR grids in artifacts/results/joint-against-poe-by-kind-of-composition/ come from Liu et
  al.'s clevr_pos and clevr_rel checkpoints, seed 0, a 100-step ancestral sampler, so they stand in
  for no EDM2 figure (that folder's README.md).
Outputs: per target, the paper's value quoted beside ours as ran with a verdict, under
artifacts/results/<the question the target answers>/; a target table marking each figure ran,
disagrees or out of reach with its reason.
Target: each chosen target meets the bar the open writes for it before it runs (for Figure 3, the
hand count on 100 samples per method against the paper's); I judge.
Progress: the open prints the target table with each target's cost; each round prints one target's
two values and its verdict.
Proof: the target table, every chosen row carrying a verdict or an out-of-reach reason.
Limits: spend cluster hours only on targets I pick, per figure, at the open; write nothing into the
read-ahead or READ_MAP.md; no commit.
Hand-off: the retrospective the walk closes on, and the /drip-retro command it prints.
```

added: every field, from the projective-composition repo's README and training runbook, the checkpoint listing on `/datasets`, `READ_MAP.md` and the CLEVR grids' README · the two trained EDM2 checkpoints and the untried composition script, which make Figure 3 C reachable without new training
dropped: nothing

⏱ you ~1 h across its opening and checks (estimate) · cluster hours: listed per figure by its opening · box: set per figure you pick

**Outcome:** the paper's CLEVR figures on the cluster. Its opening lists each figure and its cost, so pick only what the toys cannot stand in for. The CLEVR grids already generated in `artifacts/results/joint-against-poe-by-kind-of-composition/` may cover some of them.

### ◇ Optional, recommended if the read raised questions: the Bradley rebuild

```text
/unbuild --walk "projective composition from arXiv 2502.04549 and local scores from arXiv
2509.16447, rebuilt on the quadrant toy and the parts world (background choice, factorized
conditionals, a local score), then in a learned invertible feature space as layer 10's VAE step"

Task: rebuild both papers' constructions by hand in build order, one brick per round, checked
against the read-aheads' answer keys. I type every block into a new folder,
~/Playground/PhD/simple-distributions-walk/bradley/, run with
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python and
PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy.
Starts from: this stop's combined read closed (COMBINE_MAP.md, step 2's hand-off); layer 5's
walk05.py; the answer keys in the journey folder
~/goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/, at
sources/mechanisms-of-projective-composition/checks/ and
sources/local-mechanisms-of-compositional/checks/.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The answer keys run with uv run --no-project --with numpy --with scipy python checks/<piece>.py
  from the paper's folder; the 2509.16447 keys import the earlier paper's through checks/_earlier.py
  (READ_MAP.md, Position; read-ahead 00-INDEX.md of 2509.16447).
- Factorized conditionals on the split and shared toy pairs: "split ... FC holds: True", "shared ...
  FC holds: False" (checks/fc_check.out).
- Composing commutes with noising on the split pair and not the shared one at ᾱ = 0.5 (shared:
  noise-then-compose mean −0.8485 against compose-then-noise −1.0607) (checks/noise_commute.out).
- The local conditional score matches the exact score to 0.0 on the split pair at ᾱ = 1.0, 0.5 and
  0.0058; on the shared pair it misses by up to 0.5 at ᾱ = 1.0, falling to 0.0381 at ᾱ = 0.0058
  (checks/local_score.out).
- Composing in a feature space: identity, rotation and shear all map the composition back to
  (2.0, 2.0) (checks/fc_experts.out); the operator's two orders agree to max/min 1.0000 and the
  simple product's vary by 17.0000 (checks/reparam.out).
- Four-quadrant answer key: experts that model only their own coordinate reach 0.9999 (plain) and
  0.989 (background-divided) in the top-right blob with no A∩B data
  (artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/answer_key.json).
- The parts world with a loose half (variance 1.0) puts A∩B weight 1.000 at 0% overlap and with an
  empty half 0.000 (step file 05, step 6).
- Across an invertible warp of the four-quadrant toy, construction (1) changes by 4e-17 and the
  plain product by 0.035, 0.080 and 0.126 at area stretches 2.6, 9 and 49
  (artifacts/ideas/when-poe-reaches-a-and-b/runs/compose-before-or-after-a-warp/README.md).
- Layer 10's latent brick, rebuild/latent.py, is not built yet; the learned invertible feature space
  is its first piece (REBUILD_WALK.md, brick "Composing in a latent space").
Outputs: one script per brick in ~/Playground/PhD/simple-distributions-walk/bradley/, each printing
its check beside the answer key's line; the walk's REBUILD_WALK.md in that folder with one row per
brick and its check.
Target: every brick's printed number equals its answer key to the digits shown, and the local score
is computed on both toys (the split pair to 0.0, the shared pair to the miss values above); I judge.
Progress: each round ends on a picture of what the brick produced and its check line.
Proof: the rebuild folder's REBUILD_WALK.md table with every row marked matched.
Limits: never edit rebuild/, the step files, the read-aheads or their checks/ folders; I type, I do
not paste; no commit.
Hand-off: the brick table in the rebuild folder's REBUILD_WALK.md, which layer 10 reads, and the
walk's retrospective.
```

added: every field · the rebuild folder and interpreter (from the setup and layer 1's build) · the answer keys with their printed lines, and how to run them (from the read-ahead `checks/` outputs, `READ_MAP.md` and the 2509.16447 index) · the answer-key, parts-world and warp numbers (from the answer-key run, step file 05 and the warp card) · layer 10's unbuilt brick (from `REBUILD_WALK.md`)
dropped: nothing

⏱ you 2 to 4 h (two build concepts) · AI ~1 h · stop: the local score computed on both toys

**Outcome:** projective composition and the local score rebuilt by hand on the quadrant toy and the parts world.

---

⬅️ [Layer 5: the overlap dial](05-the-overlap-dial.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 6: guidance](06-guidance-as-a-dial.md) ➡️
