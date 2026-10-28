# Layer 8a, step 2: the combined read

⬅️ [Layer 8a, step 2](../08a-correction-prior-art.md#2-you-the-combined-read)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
after quiz P.1 is passed.

```text
/drip-read --combine --lean "correcting a composition: a joint model divided by its parts, and
Feynman-Kac reweighting" with arXiv 2509.25940 (CO3), 2606.23920 (Soiffer et al. 2026) and
2503.02819 (Feynman-Kac correctors), read against layer 8's interaction term

Background
This is the prior-art stop of the composition ladder in artifacts/drips/simple-distributions-rebuild/,
between layer 8 (the interaction term) and layer 9 (the correction, learned back). Two published
methods already correct a composition, and one of them gets worse with learned scores, so layer 9's
learned correction has to be placed against both before it is built. I have passed quiz P.1, and
its misses are the first ideas to slow on.

Objectives
1. Read the one idea across the three papers in combine mode, one idea of the spine per round.
2. Give each passage as printed, with its plain reading and the four comparison lines.
3. Open message 1 on my reason: to place layer 9's learned correction against CO3 and Feynman-Kac
   correctors before building it.
4. Keep the three papers, and use plan 26 as the "your plan says" source, which makes four
   sources, the cap.
5. Close the map with the read's closing check of my seven statements: for each of CO3 and
   Feynman-Kac correctors, what it divides or reweights, what it needs (a joint model, or exact
   scores), and the Table 1 panel that shows it helping or hurting, and then how layer 9's learned
   correction differs from both.

What to read
- The three PDFs on the shelf, by the paths and pages below.
- ~/goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/26-feynman-kac-correctors.md:
  plan 26, the Feynman-Kac weights derived and run on a toy.
- Quiz P.1's results: the misses to slow on.
- Layer 8's --math drip folder under artifacts/drips/ and its dispatch.md: my notation for r(x).
- ~/Playground/PhD/simple-distributions-walk/walk08.py and walk08b.py: layer 8's walks.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The PDFs are on the shelf under
  ~/goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/
  (Papers.md).
- CO3 is region-and-binding-failures/steer-away-from-mode-collisions.pdf there (Papers.md).
- Soiffer et al. is compositional-generalization/catastrophic-compositional-generation.pdf there
  (Papers.md).
- Feynman-Kac correctors is sampling-composed-models/feynman-kac-correctors-in-diffusion.pdf there
  (Papers.md).
- In all three PDFs the PDF page equals the printed page (Papers.md).
- In CO3 the idea sits at equation 1 on p2, equations 12 and 13 on p4, and Proposition 1 on p5
  (Papers.md).
- In Feynman-Kac correctors it sits at equations 7 to 10 on p3, and section 3.4 with Proposition 3.3
  on p5 (Papers.md).
- In Soiffer et al. it sits at Definition 2 and Theorem 3 on p7, Table 1 on p10, and Appendix B on
  p20 (Papers.md).
- CO3 samples p(x|C)/∏p(x|c_i), keeping the joint-prompt model as its main term and dividing out
  the single-concept models (Papers.md; composition-terms.md).
- In CO3, weights summing to 1 make a corrector and weights summing to 0 a resampler (Papers.md;
  composition-terms.md).
- CO3 runs on SDXL with 50 DDIM steps and guidance 5.0, in the first 20% of steps (Papers.md;
  composition-terms.md).
- Feynman-Kac correctors weight each particle so the product path q¹_t·q²_t/Z_t is exact, and
  resample by the weights (Papers.md; artifacts/ideas/poe-overlap-dial/composition-terms.md).
- At β = 1 only the inner product of the two scores remains in the Feynman-Kac weight (Papers.md;
  artifacts/ideas/poe-overlap-dial/composition-terms.md).
- Plan 26 runs one reweighting under three target sequences on experts N(−1, 0.5) and N(1, 0.8)
  (plan 26, toy line and task 2.3).
- In plan 26 the product ends at mean −0.2295 and variance 0.3019 against the closed form −0.230769
  and 0.307692, with lowest effective size 16,907 of 20,000 (plan 26, toy line and task 2.3).
- In plan 26, on the temperature path, resampling moves the final mean by under 0.01 (plan 26, toy
  line and task 2.3).
- Soiffer et al. Table 1 is sliced Wasserstein-2 on 5,000 samples, the mean of 30 training runs
  (Table 1, p10).
- Out of distribution, with learned scores from N = 10,000, factorized, Table 1 goes from 0.3948 at
  K = 1 particle to 2.8210 at K = 256 (Table 1, panel c, p10).
- With exact analytic scores, non-factorized, it goes from 0.6375 to 0.0820 (Table 1, panel d,
  p10).
- With analytic scores, factorized, it stays at 0.1122 to 0.1140 (Table 1, panels c and d, p10).
- The stop's Outcome quotes 0.64 to 0.08 from panel d and 0.39 to 2.82 from panel c, so the two
  numbers differ in the score source and in the factorization (Table 1, p10).
- Under factorized conditionals the Feynman-Kac weights vanish and estimation noise drives the
  resampling (Soiffer et al. Theorem 3, p7).
- Causes on separate mechanisms give exactly the Bayes form P_a(x)·P_a′(x)/P_0(x), and overlapping
  targets break it, which is the field's interaction (Soiffer et al. Appendix B;
  composition-terms.md).
- No source on the 17-paper shortlist adds an interaction factor to a product at sampling time
  (composition-terms.md).
- Layer 8's r(x) = log p(x|A,B) − [log p(x|A) + log p(x|B) − log p(x)], in ε-space ε_joint − ε_A −
  ε_B + ε_∅ (artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, brick 8).
- Layer 9 trains a correction on the gradient of r and runs the fair test at overlap 0.05 and
  0.001 (artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, brick 9).
- The project's own 2 × 2 of exact or trained experts by reverse diffusion or annealed MALA has
  interactions of +0.054 ± 0.016 total variation at r = 0.05 and +0.232 ± 0.046 at r = 0.001
  (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, run 7).
- The interaction term's lower bound on SDXL did not track which of 25 pairs blend
  (REBUILD_WALK.md, brick "The correction at scale").

Rules
- Combine mode builds nothing, so no code, notebook or object.
- Never edit rebuild/, plan 26 or the step files.
- No commit.
- Each round names one idea of the spine and ends on its four comparison lines.
- The differences table gains a row per difference named.
- A test prints its numbers labelled ran.

What to hand back
- COMBINE_MAP.md in the journey folder
  ~/goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/,
  under drips/ in its combine-<idea-slug>/ folder, holding the Sources table, the idea spine with
  every idea walked, compared or tested, the symbol table and the differences table.
- combined-reading.md in the journey's Obsidian folder, written by the skill at the last idea.
- The retrospective the read closes on, and its /drip-retro command.
- The COMBINE_MAP.md path, which step 4's cross-links and layer 9's walk read.
- The proof: the closing check prints its pass or miss line against the seven statements, and every
  idea in COMBINE_MAP.md's spine carries a state.

What you will see
Along the way: one idea of the spine per round, each ending on its four comparison lines, and a row
added to the differences table for each difference named.
At the end, a closing report: what was asked, in one sentence; what was done, with the paths of
COMBINE_MAP.md, combined-reading.md and the retrospective; PASS or MISS against 7 of 7 statements
matched against the papers; what is still unclear to me; at most one question.

I judge, from the read's closing check: done when all 7 statements on CO3, Feynman-Kac correctors and layer 9's correction match the papers and every spine idea carries a state.
```

added: the seven sections, from the field-labelled version · the stop's place in the ladder as Background (from the stop's opening lines) · walk08.py and walk08b.py placed in the playground folder (from layer 8's build, where every walk file lives) · the seven-statement target, which replaced "you place your correction against both methods below" because that had no count
dropped: nothing
