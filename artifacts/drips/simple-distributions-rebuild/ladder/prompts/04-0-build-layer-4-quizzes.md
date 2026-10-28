# Layer 4, step 0: build layer 4's quizzes

⬅️ [Layer 4, step 0](../04-the-parts-world.md#0-ai-in-the-background-quizzes-41-and-42)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`
when you start layer 3's build.

```text
In the background, prepare and build layer 4's two quizzes for the composition ladder, bottom up:
quiz 4.1 and quiz 4.2.

Background
The composition ladder in artifacts/drips/simple-distributions-rebuild/ walks from one 2D diffusion
expert to a correction for product-of-experts composition. Layer 4 is the parts world, the only
place the cat-dog product is exact in more than two dimensions: 16-pixel images and diagonal
Gaussian and mixture experts. Each layer opens on two quizzes, one on the college maths underneath
it and one on the layer itself, built during the previous layer's build. Every path below is from
the repo root, /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain, except code
paths, which are under toy/.

Objectives
1. For each quiz, dispatch prep-cadence (quiz-prep-pulse --build) with the goal "multiply Gaussian
   and mixture experts exactly in 16 dimensions" and the minutes and topics given below.
2. For each quiz, once its coverage file exists, dispatch quiz-cadence in build mode on the repo
   above.
3. Quiz 4.1, 40 minutes, covers the product of two 1D Gaussians (precisions add, the mean is
   precision-weighted), diagonal covariance, and Gaussian mixtures and their weights.
4. Quiz 4.2, 60 minutes, covers the plan 04-the-parts-world.md: the mosaic and the blend, the
   mixture-product weight w_ij ∝ π_i ρ_j N(μ_i; ν_j, Σ_i + Λ_j), and the summed-score error at
   ᾱ = 0.5.
5. Write each published URL into its ladder step.

What to read
- The miss ledger (quiz-ledger) of every earlier quiz on this ladder.
- artifacts/drips/simple-distributions-rebuild/ladder/04-the-parts-world.md and its Study first
  line: the material for quiz 4.1.
- artifacts/drips/simple-distributions-rebuild/04-the-parts-world.md, steps 0 to 8: the material for
  quiz 4.2.
- artifacts/ideas/poe-overlap-dial/example-9-parts-world/numbers.json and
  toy/rebuild/out/phase4/phase4.json: the reference numbers.
- The two published quizzes named under What is already known, before writing.
- ~/Obsidian/Mathematics/references/textbooks/Textbooks.md: the shelf index.
- ~/Obsidian/Mathematics/references/exercises/Worked Examples.md: the worked-examples index.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- quiz-cadence stops when a quiz has no coverage file in
  artifacts/drips/simple-distributions-rebuild/.quiz-prep/, so prep-cadence runs first for each
  quiz (quiz-cadence, "The preparation gate").
- The step file links a textbook page beside each fundamental it uses: pp. 230, 242, 299 and 307 of
  introduction-to-probability-for-data-science.pdf, pp. 43 and 449 of
  pattern-recognition-and-machine-learning.pdf, p. 42 of matrix-cookbook.pdf and p. 110 of
  algebra-handbook.pdf (step file 04, steps 2, 3, 5, 6 and 7).
- The step file carries three tables of timestamped video segments (the product of two Gaussians,
  the product of two mixtures, total variation distance), which are ready nodes to reuse before
  searching again (step file 04, steps 2, 3, 5, 6 and 7).
- pdftotext is not installed on mscluster, so
  ~/.claude/skills/index-source/scripts/find-in-textbooks.py fails (checked on mscluster109,
  2026-10-18).
- pypdf does not import in any environment there either; open the PDFs with PyMuPDF (import
  pymupdf, version 1.28.2) from /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python (checked on
  mscluster109, 2026-10-18).
- The shelf index and the worked-examples index are the two files under What to read
  (quiz-prep-pulse, "The bank").
- The notation is the step file's: an image is x ∈ ℝ^16 drawn 4 × 4 row by row, pixels 0 to 7 the
  head and 8 to 15 the body, the cat +1 and the dog −1 (step file 04, step 1).
- An expert is N(m, diag v), tight at v = 0.01 and loose at v = 1 (step file 04, step 2).
- The product per pixel, for inputs N(m_1, diag v_1) and N(m_2, diag v_2): precisions add,
  1/v = 1/v_1 + 1/v_2; the mean is m = v(m_1/v_1 + m_2/v_2); the constant is
  Z = N(m_1; m_2, diag(v_1 + v_2)), derived by completing the square (step file 04, step 3).
- Every answer number traces to numbers.json in artifacts/ideas/poe-overlap-dial/example-9-parts-world/
  or to toy/rebuild/out/phase4/phase4.json (numbers.json; step file 04, steps 3 to 6).
- The mosaic's head mean is 0.9802 and its log Z −46.47 (numbers.json; step file 04, step 3).
- The blend has mean 0 and sd 0.0707, 10 sds from each expert, and log Z −1583.4 (numbers.json;
  step file 04, step 4).
- Side-by-side pairs sit at unnormalised log weight −24.09 and same-half pairs at −803.21, a gap
  Δ = 779.12 (numbers.json; step file 04, step 5).
- Exactly, w_side = 1/(2(1 + e^−Δ)) is a hair below 0.5 and w_same = e^−Δ w_side, which float64
  underflows to 0.0, so the printed weights are 0.5 and 0.0 (numbers.json; step file 04, step 5).
- At ᾱ = 0.5 the head mean is 0.2326 for the product of the noised experts against 0.6931 for the
  noised product (numbers.json; step file 04, step 6).
- Sampler scores use 2,000 exact draws as a finite-sample baseline, not a lower bound: in the
  co-occurrence setting every sampler's weight error (0.009 to 0.014) is below the exact draws'
  0.026 (phase4.json; step file 04, step 7).
- Two published quizzes already hold parts-world questions: the diagonal Gaussian quiz,
  https://claude.ai/artifact/Vmd1FBg7oWwZq1onu9Jofa, section "Products of Gaussian mixtures", and
  the diffusion models quiz, https://claude.ai/artifact/MyTdAaVfZW3wiqhbLKg4wu, section "Mosaic,
  blend or two animals from one product", so the new pages extend them (overlap-dial idea map, run
  7).
- The bar a learner meets is every question marked needed now right and at least 80% of the rest,
  and a missed question returns in the next quiz (PROMPTS.md, "The bar for every quiz").

Rules
- Change nothing in the step file's prose, in toy/rebuild/ or in the example-9 folder.
- In the ladder file, write only the two URLs.
- Read the two published quizzes before writing a question.
- Report each quiz's question count, node count and verdicts from prep-cadence before its page is
  built.

What to hand back
- Two coverage files under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, each giving
  every question a verdict: ready, blocked, or unprepared by choice.
- A row in the worked-examples index for every banked problem.
- Two published quiz pages, each opening on a read-first list with minutes per item, a short path
  and a full path.
- Quiz 4.1's URL in step 1 and quiz 4.2's URL in step 3 of
  artifacts/drips/simple-distributions-rebuild/ladder/04-the-parts-world.md, each replacing "(URL
  lands here)".
- The proof: the two coverage files exist under .quiz-prep/ and list no question as blocked, and
  the two URLs open.

What you will see
Along the way: per quiz, prep-cadence's question count, node count and verdicts, then each URL as
quiz-cadence publishes it.
At the end, a closing report: what was asked, in one sentence; what was done, with each coverage
file's path and each URL; PASS or MISS against no question blocked and about 6 to 10 banked
problems per concept on quiz 4.2 for its three build-depth concepts (the diagonal product, the
mixture product, the summed-score error); what is still open; at most one question for me.

I judge, from the two pages: done when no question is blocked and every number in an answer traces to numbers.json, phase4.json, the step file or a cited textbook page.
```

added: the seven sections, from the field-labelled version · a report of coverage counts before each page is built (from the old Progress line) · reading the two published quizzes as a rule (from the settled fact that said "read them before writing")
dropped: nothing (the earlier version had already dropped "weight 0.5 each" and "e^−779 relative" as exact values, from the Codex mathematics review: 0.5 and 0.0 are float64 roundings of the exact weights)
