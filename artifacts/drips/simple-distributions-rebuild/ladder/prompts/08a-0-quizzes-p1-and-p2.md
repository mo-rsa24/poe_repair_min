# Layer 8a, step 0: quizzes P.1 and P.2

⬅️ [Layer 8a, step 0](../08a-correction-prior-art.md#0-ai-in-the-background-quizzes-p1-and-p2-one-layer-ahead)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
when you start layer 8's build, so both quizzes are waiting at the prior-art stop.

```text
In the background, prepare and build the prior-art stop's two quizzes, bottom up, for the
composition ladder: quiz P.1 and quiz P.2.

Background
The composition ladder in artifacts/drips/simple-distributions-rebuild/ walks from one 2D diffusion
expert to a correction for product-of-experts composition. Between layer 8 (the interaction term)
and layer 9 (the correction, learned back) sits a stop on two published methods that already
correct a composition: CO3 and Feynman-Kac correctors, read through Soiffer et al. 2026. The stop
opens on two quizzes, one on the college maths under it and one on the stop itself. They are built
during layer 8's build, so the combined read of the stop's step 2 has not run and no COMBINE_MAP.md
exists yet. The repo is /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain.

Objectives
1. For each quiz, dispatch prep-cadence (quiz-prep-pulse --build) with the goal "place my
   correction against CO3 and Feynman-Kac correctors" and the minutes given below.
2. For each quiz, once its coverage file exists, dispatch quiz-cadence in build mode on this repo.
3. Quiz P.1, 40 minutes, covers the college maths under the stop: importance weights as a ratio of
   two densities, and sliced Wasserstein-2 at gloss depth.
4. Quiz P.2, 80 minutes, covers the stop itself from arXiv 2509.25940, 2606.23920 and 2503.02819,
   papers in the "understand it" role: CO3's p(x|C)/∏p(x|c_i), Feynman-Kac reweighting, and why the
   correction helps with exact scores and hurts with learned ones (Soiffer et al. Table 1).
5. Write each published URL into its step of the stop's ladder file.

What to read
- artifacts/drips/simple-distributions-rebuild/ladder/08a-correction-prior-art.md, its Study first
  line: the material for quiz P.1.
- The three papers' pages on the shelf, found through ~/goal-setting/learning/papers/Papers.md: the
  material for quiz P.2.
- Layer 8's --math drip folder under artifacts/drips/ and its dispatch.md: my notation for r(x).
- ~/Obsidian/Mathematics/references/textbooks/Textbooks.md: the shelf index.
- ~/Obsidian/Mathematics/references/exercises/Worked Examples.md: the worked-examples index.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- quiz-cadence stops when a quiz has no coverage file in .quiz-prep/, so prep-cadence runs first
  for each quiz (quiz-cadence, "The preparation gate").
- Quiz P.1's material is Goodfellow et al. 17.2 (importance sampling, p608), the annealed importance
  sampling chapter in
  ~/goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/textbook/, and plan
  26 of ~/goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/
  (this stop's Study first line).
- Sliced Wasserstein-2 has no passage that teaches it, and Chen, The Little Book of Generative AI
  Foundations 8.2.2 (p168) covers plain Wasserstein, which is enough for its gloss, so the quiz asks
  for two plain lines and no derivation (this stop's Study first line and Depth line).
- pdftotext is not installed on mscluster, so
  ~/.claude/skills/index-source/scripts/find-in-textbooks.py fails there (quiz-prep-pulse, "The
  bank").
- The co3 environment has no pypdf module either, so open the PDFs read-only with PyMuPDF (import
  pymupdf) from /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python (checked 2026-10-18;
  quiz-prep-pulse, "The bank").
- The shelf index is ~/Obsidian/Mathematics/references/textbooks/Textbooks.md, and the
  worked-examples index is ~/Obsidian/Mathematics/references/exercises/Worked Examples.md
  (quiz-prep-pulse, "The bank").
- The papers shelf index is ~/goal-setting/learning/papers/Papers.md (quiz-prep-pulse, material
  kind "Papers").
- In all three papers the PDF page equals the printed page (Papers.md).
- CO3 (Dutta et al., arXiv 2509.25940) has equation 1 on p2, equations 12 and 13 on p4, and
  Proposition 1 on p5 (Papers.md).
- Feynman-Kac correctors (Skreta et al., arXiv 2503.02819) has the weighted SDE on p3, and the
  product of experts in section 3.4 with Proposition 3.3 on p5 (Papers.md).
- Soiffer et al. 2026 (arXiv 2606.23920) has Definition 2 and Theorem 3 on p7 and Table 1 on p10
  (Papers.md).
- In plan 26, unnormalised weights 2.0, 0.5, 4.0 and 0.1 normalise to 0.30303, 0.07576, 0.60606 and
  0.01515, an effective sample size of 2.150 out of 4 (plan 26, toy line and engagement gate).
- In plan 26, experts N(−1, 0.5) and N(1, 0.8) have the product mean −0.230769 and variance
  0.307692 (plan 26, toy line and engagement gate).
- In plan 26, one reweighting with 20 stages, 20,000 particles and seed 0 ends at mean −0.2295,
  variance 0.3019, and lowest effective size 16,907 (plan 26, toy line and engagement gate).
- CO3 samples a corrector distribution p(x|C)/∏p(x|c_i): it keeps the joint-prompt model as its
  main term and divides out the single-concept models (Papers.md, CO3's ideas table;
  artifacts/ideas/poe-overlap-dial/composition-terms.md).
- CO3 composes Tweedie means with weights w_0 > 0 and w_k < 0; weights summing to 1 keep a valid
  classifier-free guidance form (a corrector), and weights summing to 0 give a resampler (Papers.md,
  CO3's ideas table; composition-terms.md).
- CO3 runs on SDXL with 50 DDIM steps and guidance 5.0, in the first 20% of steps (Papers.md, CO3's
  ideas table; composition-terms.md).
- Feynman-Kac correctors derive a per-sample weight that makes the product path q¹_t·q²_t/Z_t exact,
  and resample particles by it (composition-terms.md; Papers.md, the Feynman-Kac ideas table).
- At β = 1 only the inner product of the two scores remains in the Feynman-Kac weight
  (composition-terms.md; Papers.md, the Feynman-Kac ideas table).
- Soiffer et al. Table 1 is sliced Wasserstein-2 between 5,000 target samples and 5,000 generated
  ones, averaged over 30 training runs, for K particles and N training samples (Table 1, p10).
- Out of distribution with N = 10,000 learned scores, factorized case, Table 1 goes from 0.3948 at
  K = 1 to 2.8210 at K = 256 (Table 1, panel c, p10).
- With exact analytic scores, non-factorized case, Table 1 goes from 0.6375 to 0.0820 (Table 1,
  panel d, p10).
- With analytic scores, factorized case, Table 1 stays at 0.1122 to 0.1140 (Table 1, panels c and
  d, p10).
- The stop's two quoted numbers come from different panels: 0.64 to 0.08 is panel d (exact scores,
  non-factorized) and 0.39 to 2.82 is panel c (learned scores, factorized), so the gap between them
  mixes the score source with the factorization (Table 1, p10).
- Under factorized conditionals the Feynman-Kac weights vanish, so estimation noise alone drives the
  resampling (Soiffer et al. Theorem 3, p7; Papers.md).
- Causes acting on separate mechanisms give exactly the Bayes form P_a(x)·P_a′(x)/P_0(x), and when
  the targets overlap the product form is wrong, which is the field's interaction (Soiffer et al.
  Appendix B; composition-terms.md).
- No source on the 17-paper shortlist adds an interaction factor to a product of single-concept
  experts at sampling time (composition-terms.md, "does any source add a term beyond the product").
- Layer 8's term is r(x) = log p(x|A,B) − [log p(x|A) + log p(x|B) − log p(x)], in ε-space
  ε_joint − ε_A − ε_B + ε_∅, the residual poe_repair_min caches for 25 pairs at all 50 steps
  (artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, brick "The interaction term,
  exact").
- The project's own version of Table 1 crosses exact or trained experts with reverse diffusion or
  annealed MALA, with interactions of +0.054 ± 0.016 total variation at r = 0.05 and +0.232 ± 0.046
  at r = 0.001 (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 4 and run 7).
- In that run, trained MALA reaches the right top-right share 0.989 at the wrong place, total
  variation 0.391 (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 4 and run 7).

Rules
- Change nothing in the step files, rebuild/ or plan 26.
- In the ladder files, write only the two URLs.
- Report each quiz's question count, node count and verdicts from prep-cadence before its page is
  built.

What to hand back
- Two coverage files under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, each giving
  every question a verdict: ready, blocked, or unprepared by choice.
- A row in the worked-examples index for every banked problem and every paper question.
- Two published quiz pages, each opening on a read-first list with minutes per item, a short path
  and a full path.
- Quiz P.1's URL in step 1 and quiz P.2's URL in step 3 of
  artifacts/drips/simple-distributions-rebuild/ladder/08a-correction-prior-art.md, each replacing
  "(URL lands here)".
- The proof: the two coverage files exist under .quiz-prep/ and list no question as blocked, and the
  two URLs open.

What you will see
Along the way: per quiz, prep-cadence's question count, node count and verdicts, then each URL as
quiz-cadence publishes it.
At the end, a closing report: what was asked, in one sentence; what was done, with each coverage
file's path and each URL; PASS or MISS against no question blocked on either page, quiz P.1
carrying about 4 to 6 banked problems per concept at the own depth and none on sliced Wasserstein-2
beyond its gloss, quiz P.2 carrying questions on all three papers by their role with each source
page, and at least one question naming which Table 1 panel each quoted number comes from; what is
still open; at most one question for me.

I judge, from the two pages: done when no question is blocked and every number in an answer traces to plan 26, Table 1 or a cited page.
```

added: the seven sections, from the field-labelled version · a report of coverage counts before each page is built (from the old Progress line) · the stop's place in the ladder as Background (from the stop's opening lines) · layer 8's drip folder under What to read (from the old Starts from line)
dropped: nothing (the earlier version had already replaced "plan PROMPTS.md (this stop)" with this file and the three papers, since the guide holds no prior-art section)
