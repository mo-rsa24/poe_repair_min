# ⭐ Correction prior art (read before layer 9)

⬅️ [Layer 8: the interaction term](08-the-interaction-term.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 9: the correction](09-the-correction-learned-back.md) ➡️

**Why it is critical.** Two published methods already correct a composition, and one of them gets worse with learned scores. Layer 9's correction has to be placed against both.

**Depth.** Own: CO3's division of a joint model by its parts, and Feynman-Kac reweighting. Gloss: sliced Wasserstein-2. You 2 to 4 h in all (rough).

**Study first.** [Importance weights](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/probabilistic-machine-learning-advanced-topics.pdf#page=518) as ratios of densities: [annealed importance sampling and its proof](../../../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/textbook/l3-stochastic-processes/10-annealing-on-three-axes-and-annealed-importance-sampling.md). [The Feynman-Kac weights, derived and run on a toy](../../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/26-feynman-kac-correctors.md). Textbook: Goodfellow et al. 17.2 (importance [sampling](../../../../../../Obsidian/Mathematics/Statistics/Sampling%20From%20A%20Distribution.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=539 "Pattern recognition and machine learning, chapter 11, Why expectations need samples, ancestral sampling of a directed graph (page 541), standard distribution..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=245 "Introduction to probability for data science, section 4.8, Generates random numbers for any PDF by running a uniform through the inverse CDF, in Python, with..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/deep-learning-textbook.pdf#page=606 "Deep learning textbook, section 17.1, Why machine learning samples at all (Monte Carlo estimates of sums and integrals, and sampling as the goal of a generat..."), [▶1](https://www.youtube.com/watch?v=9ixzzPQWuAY&t=474s "ritvikmath, Inverse Transform Sampling, 7:54 to 10:04, Inverts the exponential CDF and turns a uniform draw into an exponential one. Chapter-timed, from the ..."), [▶2](https://www.youtube.com/watch?v=OXDqjdVVePY&t=232s "ritvikmath, Accept-Reject Sampling, 3:52 to 7:45, An easy proposal g scaled by M into an envelope above f, then each draw accepted with probability f/(M g). ..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/textbook/l3-stochastic-processes/13-drawing-samples-by-hand.md "Drawing samples by hand: Monte Carlo, inverse transform and accept-reject, deep-learning/diffusion-models/00-from-scratch-to-sdxl, volume L3, section 13. Thr...")</sub>, p608). Sliced Wasserstein-2 has no passage that teaches it; Chen, *The Little Book of Generative AI Foundations* 8.2.2 (p168) covers plain Wasserstein, enough for its gloss.

**Achieve.** On the whiteboard: one toy with CO3's division and Feynman-Kac's weights drawn side by side · quizzes P.1 and P.2 passed.

## The flow

```text
 AI, in the background                         │  You, in order
───────────────────────────────────────────────┼───────────────────────────────────────────────
 quizzes P.1 and P.2, prepared ──────────────▶ │  1. quiz P.1: importance weights
   (paste during layer 8's build)              │       ▼
                                               │  2. /drip-read --combine --lean, 3 papers
                                               │       ▼
                                               │  3. quiz P.2: needed-now right, 80% of the rest
                                               │       ▼
 xref-cadence links the stop ◀──────────────── │  4. layer 9
```

## The steps

### 0. AI, in the background: quizzes P.1 and P.2, one layer ahead

Paste this when you start layer 8's build, so both quizzes are waiting here.

```text
In the background, prepare and build the prior-art stop's two quizzes, bottom up, for the
composition ladder in artifacts/drips/simple-distributions-rebuild/: quiz P.1 and quiz P.2.

Task: for each quiz, first dispatch prep-cadence (quiz-prep-pulse --build) with the goal and minutes
below, then dispatch quiz-cadence (mode build, repo
/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain). The goal for both is "place my correction against
CO3 and Feynman-Kac correctors".
- Quiz P.1, --minutes 40: the college maths under the stop. Importance weights, a ratio of two
  densities, sliced Wasserstein-2 at gloss depth.
- Quiz P.2, --minutes 80: the stop itself, from arXiv 2509.25940, 2606.23920 and 2503.02819, papers
  in the "understand it" role. CO3's p(x|C)/∏p(x|c_i), Feynman-Kac reweighting, why the correction
  helps with exact scores and hurts with learned ones (Soiffer et al. Table 1).
Starts from: nothing from this stop; it runs during layer 8's build, so the combined read of step 2
has not run and no COMBINE_MAP.md exists yet. Layer 8's --math drip folder under artifacts/drips/
exists by then, and its dispatch.md holds my notation for r(x). Quiz P.1 takes its material from
the Study first line of
artifacts/drips/simple-distributions-rebuild/ladder/08a-correction-prior-art.md; quiz P.2 from the
three papers' pages on the shelf.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- quiz-cadence stops when a quiz has no coverage file in .quiz-prep/, so prep-cadence runs first for
  each quiz (quiz-cadence, "The preparation gate").
- Quiz P.1's material: Goodfellow et al. 17.2 (importance sampling, p608); the annealed importance
  sampling chapter in
  ~/goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/textbook/; plan 26
  of ~/goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/
  (this stop's Study first line).
- Sliced Wasserstein-2 has no passage that teaches it; Chen, The Little Book of Generative AI
  Foundations 8.2.2 (p168) covers plain Wasserstein, enough for its gloss, so the quiz asks for two
  plain lines, not a derivation (this stop's Study first line and Depth line).
- pdftotext is not installed on mscluster, so
  ~/.claude/skills/index-source/scripts/find-in-textbooks.py fails. The co3 environment has no pypdf
  module either; open the PDFs read-only with PyMuPDF (import pymupdf) from
  /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python (checked 2026-10-18; quiz-prep-pulse,
  "The bank").
- The shelf index is ~/Obsidian/Mathematics/references/textbooks/Textbooks.md, the worked-examples
  index is ~/Obsidian/Mathematics/references/exercises/Worked Examples.md (quiz-prep-pulse, "The
  bank"), and the papers shelf index is ~/goal-setting/learning/papers/Papers.md (quiz-prep-pulse,
  material kind "Papers").
- The papers' pages, PDF page equal to printed page: CO3 (Dutta et al., arXiv 2509.25940) equation
  1 p2, equations 12 and 13 p4, Proposition 1 p5; Feynman-Kac correctors (Skreta et al., arXiv
  2503.02819) the weighted SDE p3, the product of experts in section 3.4 with Proposition 3.3 p5;
  Soiffer et al. 2026 (arXiv 2606.23920) Definition 2 and Theorem 3 p7, Table 1 p10 (Papers.md).
- Plan 26's worked numbers: unnormalised weights 2.0, 0.5, 4.0 and 0.1 normalise to 0.30303,
  0.07576, 0.60606 and 0.01515, an effective sample size of 2.150 out of 4. Experts N(−1, 0.5) and
  N(1, 0.8) have the product mean −0.230769 and variance 0.307692; one reweighting with 20 stages,
  20,000 particles and seed 0 ends at mean −0.2295, variance 0.3019, lowest effective size 16,907
  (plan 26, toy line and engagement gate).
- CO3 samples a corrector distribution p(x|C)/∏p(x|c_i): it keeps the joint-prompt model as its main
  term and divides out the single-concept models, composing Tweedie means with weights w_0 > 0 and
  w_k < 0; weights summing to 1 keep a valid classifier-free guidance form (a corrector) and summing
  to 0 give a resampler; it runs on SDXL with 50 DDIM steps, guidance 5.0, in the first 20% of steps
  (Papers.md, CO3's ideas table; artifacts/ideas/poe-overlap-dial/composition-terms.md).
- Feynman-Kac correctors derive a per-sample weight that makes the product path q¹_t·q²_t/Z_t exact
  and resample particles by it; at β = 1 only the inner product of the two scores remains in the
  weight (composition-terms.md; Papers.md, the Feynman-Kac ideas table).
- Soiffer et al. Table 1 is sliced Wasserstein-2 between 5,000 target samples and 5,000 generated
  ones, averaged over 30 training runs, for K particles and N training samples. Out of
  distribution with N = 10,000 learned scores, factorized case, it goes from 0.3948 at K = 1 to
  2.8210 at K = 256; with exact analytic scores, non-factorized case, from 0.6375 to 0.0820; with
  analytic scores, factorized case, it stays at 0.1122 to 0.1140 (Table 1, panels c and d, p10).
- The stop's two quoted numbers come from different panels: 0.64 to 0.08 is panel d (exact scores,
  non-factorized) and 0.39 to 2.82 is panel c (learned scores, factorized), so the gap between them
  mixes the score source with the factorization (Table 1, p10).
- Under factorized conditionals the Feynman-Kac weights vanish, so estimation noise alone drives the
  resampling (Soiffer et al. Theorem 3, p7; Papers.md).
- Causes acting on separate mechanisms give exactly the Bayes form P_a(x)·P_a′(x)/P_0(x); when the
  targets overlap the product form is wrong, which is the field's interaction (Soiffer et al.
  Appendix B; composition-terms.md).
- No source on the 17-paper shortlist adds an interaction factor to a product of single-concept
  experts at sampling time (composition-terms.md, "does any source add a term beyond the product").
- Layer 8's term: r(x) = log p(x|A,B) − [log p(x|A) + log p(x|B) − log p(x)], in ε-space
  ε_joint − ε_A − ε_B + ε_∅, the residual poe_repair_min caches for 25 pairs at all 50 steps
  (artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, brick "The interaction term,
  exact").
- The project's own version of Table 1: exact or trained experts by reverse diffusion or annealed
  MALA, with interactions of +0.054 ± 0.016 total variation at r = 0.05 and +0.232 ± 0.046 at
  r = 0.001; trained MALA reaches the right top-right share 0.989 at the wrong place (total
  variation 0.391) (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 4 and run 7).
Outputs:
- two coverage files under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, each with a
  verdict per question: ready, blocked, or unprepared by choice;
- rows in the worked-examples index for every banked problem and every paper question;
- two published quiz pages, each opening on a read-first list with minutes per item, a short path
  and a full path.
Target: no question on either page carries the verdict blocked; quiz P.1 carries banked problems,
about 4 to 6 per concept at the own depth and none on sliced Wasserstein-2 beyond its gloss; quiz
P.2 carries questions on all three papers by their role, each with its source page, and at least
one that names which Table 1 panel each quoted number comes from; every number in an answer traces
to plan 26, Table 1 or a cited page. I judge, from the pages.
Progress: prep-cadence reports, per quiz, its question count, node count and verdicts; quiz-cadence
reports each URL as it publishes.
Proof: the two coverage files exist under .quiz-prep/ and list no question as blocked; the two
URLs open.
Limits: change nothing in the step files, rebuild/ or plan 26; in the ladder files, write only the
two URLs.
Hand-off: write quiz P.1's URL into step 1 and quiz P.2's URL into step 3 of
artifacts/drips/simple-distributions-rebuild/ladder/08a-correction-prior-art.md, each replacing
"(URL lands here)".
```

added: every field the setup's quiz prompt carries, so this prompt stands alone instead of pointing at "the setup section of PROMPTS.md", which the guide no longer holds (the setup is `ladder/00-setup.md`) · the preparation gate (from the quiz-cadence agent definition) · quiz P.1's material and the sliced Wasserstein-2 gap (from this stop's Study first line) · the missing `pdftotext` and the PyMuPDF route, with `pypdf` absent from `co3` (checked on mscluster109, and quiz-prep-pulse's bank step) · the papers' pages (from `Papers.md`) · plan 26's worked numbers · what CO3 and Feynman-Kac correctors compute (from `Papers.md` and the composition terms) · Table 1's panels, read on its page, and which panel each quoted number comes from · Theorem 3, Appendix B and the shortlist search (from `Papers.md` and the composition terms) · layer 8's term (from `REBUILD_WALK.md`) and the project's own 2 × 2 (from the when-poe idea map, run 7) · the URL write-back
dropped: "plan PROMPTS.md (this stop)" as quiz P.2's source (why: the guide holds no prior-art section; the source is this file and the three papers)

⏱ AI ~1.5 h (estimate), beside layer 8's build

### 1. You: quiz P.1

Quiz P.1: (URL lands here)

⏱ you 20 to 40 min · stop: the bar, every *needed now* question right and at least 80% of the rest; below that, read the read-first items your misses point at and retake.

### 2. You: the combined read

```text
/drip-read --combine --lean "correcting a composition: a joint model divided by its parts, and
Feynman-Kac reweighting" with arXiv 2509.25940 (CO3), 2606.23920 (Soiffer et al. 2026) and
2503.02819 (Feynman-Kac correctors), read against layer 8's interaction term

Task: read the one idea across the three papers in combine mode, one idea of the spine per round,
each passage as printed with its plain reading and the four comparison lines. My reason, for
message 1: to place layer 9's learned correction against CO3 and Feynman-Kac correctors before
building it. Keep the three papers; "your plan says" is plan 26, in
~/goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/ as
plans/26-feynman-kac-correctors.md, which makes four sources, the cap.
Starts from: quiz P.1 passed, its misses as the first ideas to slow on; layer 8's --math drip
folder under artifacts/drips/, whose dispatch.md holds my notation for r(x); layer 8's walk08.py
and walk08b.py.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The PDFs are on the shelf under
  ~/goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/:
  CO3 as region-and-binding-failures/steer-away-from-mode-collisions.pdf, Soiffer et al. as
  compositional-generalization/catastrophic-compositional-generation.pdf, and Feynman-Kac
  correctors as sampling-composed-models/feynman-kac-correctors-in-diffusion.pdf; PDF page equals
  printed page (Papers.md).
- Where the idea sits: CO3 equation 1 p2, equations 12 and 13 p4, Proposition 1 p5; Feynman-Kac
  correctors equations 7 to 10 p3, section 3.4 with Proposition 3.3 p5; Soiffer et al. Definition 2
  and Theorem 3 p7, Table 1 p10, Appendix B p20 (Papers.md).
- CO3 samples p(x|C)/∏p(x|c_i), keeping the joint-prompt model as its main term and dividing out the
  single-concept models; weights summing to 1 make a corrector, summing to 0 a resampler; SDXL, 50
  DDIM steps, guidance 5.0, the first 20% of steps (Papers.md; composition-terms.md).
- Feynman-Kac correctors weight each particle so the product path q¹_t·q²_t/Z_t is exact and
  resample by the weights; at β = 1 only the inner product of the two scores remains in the weight
  (Papers.md; artifacts/ideas/poe-overlap-dial/composition-terms.md).
- Plan 26 runs one reweighting under three target sequences on experts N(−1, 0.5) and N(1, 0.8):
  the product ends at mean −0.2295 and variance 0.3019 against the closed form −0.230769 and
  0.307692, lowest effective size 16,907 of 20,000; on the temperature path, resampling moves the
  final mean by under 0.01 (plan 26, toy line and task 2.3).
- Soiffer et al. Table 1: sliced Wasserstein-2, 5,000 samples, mean of 30 training runs. Out of
  distribution, learned scores from N = 10,000, factorized: 0.3948 at K = 1 particle to 2.8210 at
  K = 256. Exact analytic scores, non-factorized: 0.6375 to 0.0820. Analytic, factorized: 0.1122 to
  0.1140 (Table 1, panels c and d, p10).
- The stop's Outcome quotes 0.64 to 0.08 from panel d and 0.39 to 2.82 from panel c, so the two
  numbers differ in the score source and in the factorization (Table 1, p10).
- Under factorized conditionals the Feynman-Kac weights vanish and estimation noise drives the
  resampling (Soiffer et al. Theorem 3, p7).
- Causes on separate mechanisms give exactly the Bayes form P_a(x)·P_a′(x)/P_0(x); overlapping
  targets break it, which is the field's interaction (Soiffer et al. Appendix B;
  composition-terms.md).
- No source on the 17-paper shortlist adds an interaction factor to a product at sampling time
  (composition-terms.md).
- Layer 8's r(x) = log p(x|A,B) − [log p(x|A) + log p(x|B) − log p(x)], in ε-space ε_joint − ε_A −
  ε_B + ε_∅; layer 9 trains a correction on its gradient and runs the fair test at overlap 0.05 and
  0.001 (artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, bricks 8 and 9).
- The project's own 2 × 2 of exact or trained experts by reverse diffusion or annealed MALA has
  interactions of +0.054 ± 0.016 total variation at r = 0.05 and +0.232 ± 0.046 at r = 0.001
  (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, run 7).
- The interaction term's lower bound on SDXL did not track which of 25 pairs blend
  (REBUILD_WALK.md, brick "The correction at scale").
Outputs:
- COMBINE_MAP.md in the journey folder
  ~/goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/
  under drips/ in its combine-<idea-slug>/ folder, holding the Sources table, the idea spine with
  every idea walked, compared or tested, the symbol table and the differences table;
- combined-reading.md in the journey's Obsidian folder, written by the skill at the last idea;
- the retrospective the read closes on.
Target: with the map closed I state, for each of CO3 and Feynman-Kac correctors, what it divides or
reweights, what it needs (a joint model, or exact scores), and the Table 1 panel that shows it
helping or hurting, and then how layer 9's learned correction differs from both. 7 of 7 statements
matched against the papers; I judge, from the read's closing check.
Progress: each round names one idea of the spine and ends on its four comparison lines; the
differences table gains a row per difference named; a `test` prints its numbers labelled ran.
Proof: the closing check prints its pass or miss line against the target, and every idea in
COMBINE_MAP.md's spine carries a state.
Limits: combine mode builds nothing, so no code, notebook or object; never edit rebuild/, plan 26
or the step files; no commit.
Hand-off: the COMBINE_MAP.md path, which step 4's cross-links and layer 9's walk read, and the
retrospective's /drip-retro command.
```

added: the reason and plan 26 as the home source, within the four-source cap (from the combine reference and this stop's Study first line) · the shelf PDFs and their pages (from `Papers.md`) · what CO3 and Feynman-Kac correctors compute (from `Papers.md` and the composition terms) · plan 26's toy result · Table 1's panels read on its page, and which panel each quoted number comes from · Theorem 3, Appendix B and the shortlist search · layers 8 and 9 and the 25-pair caveat (from `REBUILD_WALK.md`) · the project's own 2 × 2 (from the when-poe idea map, run 7) · the seven-statement target in place of "you place your correction against both methods below", which had no count
dropped: nothing

⏱ rough: you 60 to 120 min (three papers) · AI ~1.5 h · stop: you place your correction against both methods below

**Outcome:** you can place your correction against two existing ones:

- CO3 divides a joint model by its parts, p(x|C)/∏p(x|c_i).
- Soiffer et al.'s Table 1 shows Feynman-Kac correction improving the fit with exact scores (sliced Wasserstein-2 from `0.64` to `0.08`) and worsening it with learned scores (`0.39` to `2.82`).

The two pairs come from different panels, so they differ in two things at once: `0.64` to `0.08` is analytic scores on a target that does not factorise (panel d), and `0.39` to `2.82` is scores learned from 10,000 [samples](../../../../../../Obsidian/Mathematics/Statistics/Sampling%20From%20A%20Distribution.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=539 "Pattern recognition and machine learning, chapter 11, Why expectations need samples, ancestral sampling of a directed graph (page 541), standard distribution..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=245 "Introduction to probability for data science, section 4.8, Generates random numbers for any PDF by running a uniform through the inverse CDF, in Python, with..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/deep-learning-textbook.pdf#page=606 "Deep learning textbook, section 17.1, Why machine learning samples at all (Monte Carlo estimates of sums and integrals, and sampling as the goal of a generat..."), [▶1](https://www.youtube.com/watch?v=9ixzzPQWuAY&t=474s "ritvikmath, Inverse Transform Sampling, 7:54 to 10:04, Inverts the exponential CDF and turns a uniform draw into an exponential one. Chapter-timed, from the ..."), [▶2](https://www.youtube.com/watch?v=OXDqjdVVePY&t=232s "ritvikmath, Accept-Reject Sampling, 3:52 to 7:45, An easy proposal g scaled by M into an envelope above f, then each draw accepted with probability f/(M g). ..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/textbook/l3-stochastic-processes/13-drawing-samples-by-hand.md "Drawing samples by hand: Monte Carlo, inverse transform and accept-reject, deep-learning/diffusion-models/00-from-scratch-to-sdxl, volume L3, section 13. Thr...")</sub> on a target that does (panel c). Read each on its own, never the gap between them as the effect of learned scores alone. That is the same lesson as layer 9's trained experts.

### 3. You: quiz P.2

Quiz P.2: (URL lands here)

⏱ you 40 to 80 min · stop: the bar, every *needed now* question right and at least 80% of the rest.

### 4. You: close the stop

Run the `/drip-retro` command the read printed at its close (⏱ about 25 minutes). Then paste this:

```text
In the background, dispatch xref-cadence over the files the correction prior-art stop of the
composition ladder wrote: the combined read's COMBINE_MAP.md and combined-reading.md, and the
retrospective the read closed on.

Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- COMBINE_MAP.md sits in the journey folder
  ~/goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/
  under drips/ in its combine-<idea-slug>/ folder (step 2's hand-off).
- The reference sets to link against are: the textbook chapter and plan 26 on the Study first line
  of artifacts/drips/simple-distributions-rebuild/ladder/08a-correction-prior-art.md; the layer 8
  and layer 9 rows of the Bricks table in
  artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md ("The interaction term, exact" and
  "The correction, learned back"), since the stop has no brick row of its own; claim 4 of
  artifacts/ideas/poe-overlap-dial/IDEA_MAP.md; and claims 4 (child 4.1) and 9 of
  artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md.
- ~/Playground/PhD/simple-distributions-walk/ is scratch space; link nothing into it (CLAUDE.md, the
  playground is disposable).
Outputs: links written both ways between those files and the reference sets.
Hand-off: the agent's report, listing every link it wrote, read before layer 9 starts.
```

added: the reference sets, by path (from this stop's Study first line, `REBUILD_WALK.md` and the claims the stop rests on in both idea maps)
dropped: nothing

⏱ you 2 min to paste · AI ~10 min (estimate)

---

⬅️ [Layer 8: the interaction term](08-the-interaction-term.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 9: the correction](09-the-correction-learned-back.md) ➡️
