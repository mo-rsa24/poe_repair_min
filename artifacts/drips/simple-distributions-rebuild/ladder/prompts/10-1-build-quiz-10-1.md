# Layer 10, step 1: build quiz 10.1

⬅️ [Layer 10, step 1](../10-latent-space-and-vaes.md#1-ai-in-the-background-quiz-101)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
when you start layer 10, so the quiz is waiting after the build.

```text
In the background, prepare and build one quiz for the composition ladder: quiz 10.1, on warp
invariance.

Background
The composition ladder in artifacts/drips/simple-distributions-rebuild/ walks from one 2D diffusion
expert to a correction for product-of-experts composition. Layer 10 adds one variable, the latent:
it asks whether a composition survives an invertible change of space, and whether a VAE's latent
behaves like one. Quiz R.1 at the reading stop already covered change of variables, so this layer
has this one quiz. The repo is /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain.

Objectives
1. Dispatch prep-cadence (quiz-prep-pulse --build) with the goal "say why a composition whose
   exponents sum to 1 survives an invertible warp", 40 minutes, and the topics below.
2. The topics: a density pushed through an invertible map x = g(z) picks up the factor |det J|⁻¹.
3. A weighted product ∏ p_i^(w_i) then picks up |det J|^(−Σw_i).
4. Construction (1), p(x|A)·p(x|B)/p(x), has exponents 1 + 1 − 1 = 1, and negation p_A^(1+w)/p_B^w
   has (1 + w) − w = 1, so both are unchanged.
5. The plain product p_A·p_B has exponents summing to 2 and is reweighted by how much the map
   stretches each region.
6. The last topic is what a non-invertible VAE decoder changes.
7. Once the coverage file exists, dispatch quiz-cadence in build mode.
8. Write the published URL into step 5 of the layer 10 page.

What to read
- artifacts/drips/simple-distributions-rebuild/ladder/10-latent-space-and-vaes.md and its Study
  first line: the layer and its reading.
- artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 10: the warp claim.
- artifacts/ideas/when-poe-reaches-a-and-b/runs/compose-before-or-after-a-warp/README.md: the run
  card, with warp_check.py and warp_check.json beside it.
- ~/Obsidian/Mathematics/references/textbooks/Textbooks.md: the shelf index.
- ~/Obsidian/Mathematics/references/exercises/Worked Examples.md: the worked-examples index.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- Nothing before this quiz runs in layer 10 (the layer 10 page, the flow).
- quiz-cadence stops when a quiz has no coverage file in .quiz-prep/, so prep-cadence runs first
  (quiz-cadence, "The preparation gate").
- Quiz R.1 at the reading stop already covers change of variables with a Jacobian, so this quiz
  starts from |det J| as known and asks only what layer 10 adds (the layer 10 page, Depth line).
- pdftotext is not installed on mscluster, so
  ~/.claude/skills/index-source/scripts/find-in-textbooks.py fails; open the PDFs with PyMuPDF from
  the co3 environment (checked on mscluster109: import pymupdf works in co3, version 1.28.2, and
  import pypdf fails everywhere).
- The shelf index and the worked-examples index are the two files named above (quiz-prep-pulse,
  "The bank").
- The textbook passages are Deisenroth et al. 6.7 (p220) and Chen 7.1.1 (p140), and the paper is
  Bradley et al. 2025a (arXiv 2502.04549), Theorem 6.1 with its proof in appendix I (the layer 10
  page, Study first line).
- The warp is x1 = z1 + a·tanh(2 z1), x2 = z2 + a·tanh(2 z2) + 0.4 z1 (warp_check.py).
- Its Jacobian is lower triangular, so det J = (1 + 2a·sech²(2 z1))(1 + 2a·sech²(2 z2)) (warp_check.py).
- det J is largest at z = 0, where it is (1 + 2a)²: 2.56, 9 and 49 at a = 0.3, 1 and 3
  (warp_check.py; warp_check.json).
- Across that warp, construction (1) agrees to 4e-17 total variation and negation to 7e-17, which
  is rounding (the run card; claim 10).
- The plain product differs by 0.035, 0.080 and 0.126 at largest area stretches 2.6, 9 and 49 (the
  run card; claim 10).
- A VAE decoder is not invertible; its analogue of |det J| is the volume factor √det(JᵀJ) of the
  pullback metric, marked "likely, verify", so questions on it are marked as such (claim 10).

Rules
- Change nothing in the idea map, the run folder or rebuild/.
- In this ladder file, write only the URL.
- Report prep-cadence's question count, node count and verdicts before the page is built.

What to hand back
- One coverage file under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, with a verdict
  per question: ready, blocked, or unprepared by choice.
- A row in the worked-examples index for every banked problem.
- One published quiz page, opening on a read-first list with minutes per item, a short path and a
  full path.
- The URL in step 5 of artifacts/drips/simple-distributions-rebuild/ladder/10-latent-space-and-vaes.md,
  replacing "(URL lands here)".
- The proof: the coverage file exists under .quiz-prep/ and lists no question as blocked, and the
  URL opens.

What you will see
Along the way: prep-cadence's question count, node count and verdicts, then the URL as quiz-cadence
publishes it.
At the end, a closing report: what was asked, in one sentence; what was done, with the coverage
file's path and the URL; PASS or MISS against no question blocked and about 4 to 6 banked problems
for the concept at the own depth; what is still open; at most one question for me.

I judge, from the page: done when no question is blocked and every number in an answer traces to the run card, warp_check.json or a cited textbook page.
```

added: the seven sections, from the field-labelled version · the report of coverage counts before the page is built (from the old Progress line) · the topics split into one objective each
dropped: nothing (the earlier version had already dropped "ask for it with the layer 8 prompt's shape", since a fresh session cannot read a shape from another file)
