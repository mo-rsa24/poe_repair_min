# ⭐ Layer 6: guidance as a dial

⬅️ [Reading stop: Bradley](05a-reading-stop-projective-composition.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 7: A without B](07-a-without-b.md) ➡️

**Why it is critical.** Stable Diffusion's cat-dog hybrid happens at guidance weight 7.5, and everything above ran at 1. The correction in layer 9 has to work at the weight the real model uses.

**Depth.** Build: a [density](../../../../../../Obsidian/Mathematics/Statistics/Distributions/Probability%20Density%20Function.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=188 "Introduction to probability for data science, section 4.1, Builds the intuition of probability per unit length, defines the PDF and its properties (Definitio..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=186 "Mathematics for machine learning, section 6.2.2, The density and the CDF (Definitions 6.1 and 6.2), a uniform example (Example 6.3), and discrete against con..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=36 "Pattern recognition and machine learning, section 1.2.1, The density's two conditions (equations 1.25 and 1.26), its link to the CDF, and why a density picks..."), [▶1](https://www.youtube.com/watch?v=QKA4HNEw3aY&t=52s "The Organic Chemistry Tutor, Probability Density Functions, 0:52 to 3:39, Verifies that a given function is a density by checking that its integral equals 1...."), [▶2](https://www.youtube.com/watch?v=oI3hZJqXJuc&t=135s "StatQuest, The Main Ideas behind Probability Distributions, 2:15 to 5:14, A curve as a distribution, with probability read as area under it, not its height. ..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/02-densities-and-the-gaussian.md#21-a-density-can-exceed-1 "Chapter 2: Read a density as area and draw a Gaussian from its covariance, deep-learning/diffusion-models/01-poe-derivation-foundations, chapter 02, section ..."), [◈](https://claude.ai/artifact/DBmJtn6xy75sPtQwPEvQZU#nb-sec1 "Kinds of composition notebook, Setup: the ideal expert, The ideal expert's log-density on a grid, in four steps from the 2D Gaussian to its logsumexp form, t...")</sub> [raised to a power](../../../../../../goal-setting/learning/mathematics/textbooks/algebra/algebra-handbook.pdf#page=109), and the guided target on both toys. You 2.3 to 4.7 h in all (rough).

**Study first.** A density raised to a power: [what is annealed](../../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/24-what-is-annealed.md) and [The Tempered Density And Its Score](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/The%20Tempered%20Density%20And%20Its%20Score.md). The guided target: [guidance as a tilted density, section 8.4](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/08-assembling-the-poe-rule.md), [classifier-free guidance trained on MNIST](../../../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/plans/08-cfg-trained-on-mnist.md) and [Powers Of Gaussians](../../../../../../Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Powers%20Of%20Gaussians.md). Textbook: Goodfellow et al., *Deep Learning* 17.5.1 (tempering, from p615). Video: Umar Jamil on [classifier-free guidance](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Classifier-Free%20Guidance.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/probabilistic-machine-learning-advanced-topics.pdf#page=909 "Probabilistic machine learning: advanced topics, section 25.6.3, Classifier-free guidance, Derives the implicit classifier from Bayes' rule, log p(c∣x) = log..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/understanding-deep-learning.pdf#page=379 "Understanding deep learning, section 18.6.3, Conditional generation, Feeds the class into the U-Net the way the time step is added and trains on conditional ..."), [3](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/sampling-composed-models/compositional-visual-generation-with-composable-diffusion-models.pdf#page=7 "Liu and others 2022, Compositional visual generation with composable diffusion models, section 4.2, Composes concepts as p(x) ∏ p(c_i∣x) and notes that with ..."), [▶1](https://www.youtube.com/watch?v=Edr4uZFh4EE&t=2730s "CS231n lecture 14, 45:30 to 50:05, Drops the label to a null value half the time in training and combines v_cfg = (1 + w) v_y − w v_null at sampling, where w..."), [▶2](https://www.youtube.com/watch?v=WUUq6TVAu8U&t=5665s "CME296 lecture 4, 1:34:25 to 1:40:38, Derives the implicit classifier from the conditional and unconditional predictions with a weight w of about 3, the cond..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/plans/08-cfg-trained-on-mnist.md "CFG trained on MNIST, a w sweep, deep-learning/diffusion-models/00-from-scratch-to-sdxl, plan 08. Trains a guided model on MNIST and sweeps the guidance weig..."), [◈](https://claude.ai/artifact/1GayZthQkRL4FmukdCV9WW#nb-poe "The SDXL notebook in ten parts, PoE composition and CFG inside it, Two prompts composed as a product of experts, a run that shows the composition failing, an...")</sub>, [22:20 to 31:00](https://www.youtube.com/watch?v=ZBKpAp_6TGI&t=1340s).

**Ideas to tie in.** From [the ideas that tie the ladder together](../../../ideas/ideas-that-tie-the-ladder-together.md): [guidance as tempering, and negation tested for normalisability](../../../ideas/ideas-that-tie-the-ladder-together.md#4-treat-guidance-as-tempering-and-test-negation-for-normalisability) (needed, 2 to 4 h, no training). Each comes with an exact toy check.

**Achieve.** `walk06.py` printing A∩B share, [total variation](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/probabilistic-machine-learning-advanced-topics.pdf#page=94) and Z for each w from 1 to 7.5 · on the whiteboard: a two-mode 1D density raised to w = 1, 2 and 7.5 · problems: the ones quiz 6.2 banks · quizzes 6.1 and 6.2 passed.

## The flow

```text
 AI, in the background                         │  You, in order
───────────────────────────────────────────────┼───────────────────────────────────────────────
 2. quizzes 6.1 and 6.2, prepared ───────────▶ │  3. quiz 6.1: powers of densities, Bayes
   (paste during the reading stop)             │       ▼
                                               │  4. /drip --math: the guided target
                                               │       ▼
                                               │  5. quiz 6.2: needed-now right, 80% of the rest
                                               │       ▼
                                               │  6. build: walk06.py (--request)
 1. builds rebuild/phase6_guidance.py ───────▶ │  7. compare: p6. numbers equal yours
                                               │       ▼
 math-scene, beside layer 7 ◀───────────────── │  8. scene
                                               │       ▼
 xref-cadence links the layer ◀─────────────── │  9. /drip-retro, then the next layer
```

## The steps

**AI, one layer ahead.** Paste steps 1 and 2 when you start the reading stop. The first builds the reference your compare step needs; the second builds the quizzes.

### 1. AI, in the background: the layer 6 reference

```text
In the background, build the layer 6 reference: rebuild/phase6_guidance.py, the guidance dial for
w = 1 to 7.5 on the quadrant toy and the parts world, starting from rebuild/ideal.py and
rebuild/parts_world.py, printing A∩B share, total variation and Z per w and the w where the
parts-world blend appears. Add those numbers to rebuild/walk_numbers.py as phase6() with keys p6.*,
and run it once with the co3 python.

Task: in /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy, on the current branch, write
rebuild/guidance.py (the guided target as a function of w, for both toys) and
rebuild/phase6_guidance.py (the dial), run from toy/ as python -m rebuild.phase6_guidance
with /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python. Write the guided target
p(x)·p(A|x)^w·p(B|x)^w in code as (1 − 2w)·log p(x) + w·log p(x, A) + w·log p(x, B).
- Quadrant toy: the four-quadrant answer-key toy, in closed form on ideal.py's 241 × 241 grid
  (ideal.grid, ideal.CELL, ideal.grid_density), at A∩B shares r = 0.05 and 0.001.
- Parts world: p(x), p(x|A) and p(x|B) as parts_world.Mixture objects over the placements
  phase5_dial.pw_experts builds, with p(x) the mixture of every placement either set holds, at both
  backgrounds. Its Z and A∩B share at w ≠ 1 have no closed form: estimate them by self-normalised
  importance sampling from exact draws of p(x) (Mixture.sample), whose weights p(A|x)^w·p(B|x)^w lie
  in [0, 1], and print each with its standard error.
- Layer 5's toy: p(x) as make_sets' two sets pooled, and the guided target's A∩B share per w at
  f = 0.01, 0.05 and 0.2, expected near 1.0 at every w; it checks the guided target's code.
- Samplers, per toy and w: tuned MALA, UHA and reverse diffusion on the noised guided target
  (samplers.Target), 2,000 samples each, with the settings phase5_dial.sample_2d uses, beside 2,000
  draws from the exact target as the floor (ideal.perfect_samples on the grid; exact draws in the
  parts world), scored by ideal.tv and parts_world.tv_projected.
- A w grid from 1 to 7.5 that includes both ends.
Starts from: nothing before it in this layer. The reference for layers 1 to 5 is in rebuild/ and its
outputs in rebuild/out/phase1 to phase5.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The brick: the target p(x)·p(A|x)^w·p(B|x)^w computed exactly on the quadrant toy and the parts
  world for w = 1 to 7.5, with A∩B share, total variation and Z per w, in rebuild/guidance.py and
  rebuild/phase6_guidance.py (REBUILD_WALK.md, brick "Guidance as a tilted density").
- SDXL composes "a cat" and "a dog" with one model and two prompts at w = 7.5 over 50 DDIM steps, as
  ε_∅ + w(ε_A − ε_∅) + w(ε_B − ε_∅); the tilted target is inferred from the cache's residual formula
  gs*(eps_j − eps_a − eps_b + eps_uncond), and the 2D work measured construction (1) only at w = 1
  (artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md, claim 1 and Sources).
- At w = 1 the target is construction (1), p(x|A)·p(x|B)/p(x), which equals construction (2)'s
  p_A·p_B only where p(x) is flat (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 3).
- The four-quadrant toy has one Gaussian blob per quadrant at (±0.5, ±0.5), std 0.15; A is x1 > 0
  and B is x2 > 0. With "contains" labels and a share r in the top-right, construction (1) puts
  0.989 within 3 std of (0.5, 0.5) for every r from 0.25 to 0.001, with normaliser Z = r, and at
  r = 0 it puts 0.970 at the origin
  (artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/README.md, answer_key.py).
- 0.989 is 1 − e^(−9/2), the mass one blob holds within 3 std (when-poe-reaches-a-and-b idea map,
  run 4).
- Layer 5's quadrant sets (phase5_dial.make_sets) have an unconditional p(x): the two sets pooled,
  which is what one model trained on both with the labels dropped learns. With overlap share f, A is
  uniform on the left half with f of its mass top-left and B uniform on the top half with f
  top-left, so p(x) is (1 − f)/2 bottom-left, f top-left, (1 − f)/2 top-right and 0 bottom-right.
  The guided target p(x)^(1 − 2w)·p_A^w·p_B^w is then f top-left and 0 elsewhere: its A∩B share is 1
  at every w for any f > 0, up to the ideal experts' smoothing at the square edges
  (rebuild/phase5_dial.py, make_sets; Bayes' rule).
- ideal.logq weights every data point equally, so blobs weighted r and (1 − r)/3 need a weighted
  form. A Gaussian noised to step t is N(√ᾱ_t μ, (ᾱ_t v + 1 − ᾱ_t) I), so each blob's noised
  variance is ᾱ_t·0.15² + 1 − ᾱ_t (rebuild/ideal.py; parts_world.Mixture.noised).
- p(A|x) and p(B|x) are at most 1, so the guided target never exceeds p(x) and stays a density for
  every w ≥ 0 when all three densities come from one labelled set (Bayes' rule).
- Raising a Gaussian to the power w multiplies its precision by w (vault note Powers Of Gaussians).
  A power of a mixture is not a mixture, and parts_world.mixture_product's closed form covers the
  plain product only (rebuild/parts_world.py, module docstring).
- Success is three numbers reported together: the share in A∩B, the total variation to the exact
  composed density, and Z. Reverse diffusion's share 0.983 against 0.989 hid a total variation 0.049
  above the floor (when-poe-reaches-a-and-b idea map, claim 5 and run 5).
- With exact experts at r = 0.05, annealed ULA, fixed MALA and UHA sit at the exact-sample floor
  (total variation 0.124 to 0.132 against 0.133 ± 0.006) and reverse diffusion with summed scores at
  0.182 (when-poe-reaches-a-and-b idea map, run 5).
- Parts world, layer 5: with an empty background (variance 0.01) the product at 0% overlap is a
  ghost with half-means (0.5, −0.5) and A∩B weight 0.000, and 1.000 from 0.1% up; with a loose
  background (variance 1.0) it is 1.000 at every overlap (step file 05, step 6).
- The parts-world blend is example 9(b): both experts tight on every pixel, product mean 0, 10 sds
  from each expert, log Z −1583.4 (step file 04, step 4). No source shows that guidance produces it
  at any w.
- On the mosaic, reverse diffusion leaves the head mean at 0.923 where the product has 0.980, a
  projected total variation of 0.732 against a floor of 0.013 (step file 04, step 7).
- walk_numbers.py holds worked-example numbers, each written by put(key, value); run results
  (masses and total variations per sampler) are read from rebuild/out/phaseN/ instead. Its __main__
  runs (phase1, phase2, phase3, phase5) and writes rebuild/out/walk-numbers.json by a relative path,
  so it runs from toy/ (rebuild/walk_numbers.py).
- Tolerances: an exact number within 0.005, a sampler number within 0.03
  (rebuild/phase3_kinds.py).
- rebuild.device.pick_device takes the first card with under 1 GB in use, else the CPU
  (rebuild/device.py; REBUILD_WALK.md, Pins).
Outputs:
- rebuild/guidance.py and rebuild/phase6_guidance.py; the phase script prints a w = 1 check line,
  marked matches or differs against the answer key, then one row per toy and w;
- rebuild/out/phase6/phase6.json with every row, and a figure of the guided target per w on both
  toys;
- phase6() in rebuild/walk_numbers.py, added to the __main__ tuple: the w grid as p6.w_grid, the
  exact share and Z per toy and w as p6.<toy>_<measure>_w<w>, and the blend's w as p6.blend_w (None
  when no w up to 7.5 gives one);
- rebuild/out/walk-numbers.json rewritten, its 84 earlier keys unchanged.
Target: at w = 1 the quadrant toy's top-right share is within 0.005 of 0.989 and Z within 0.005·r
of r, at both values of r; every sampler row carries its total variation beside the floor; every
estimated parts-world number says so, with its standard error. I judge, from the printed rows.
Progress: the w = 1 check prints first; then one row per w for the quadrant toy and then for the
parts world, each with the share and Z, and per sampler the share and total variation.
Proof: python -m rebuild.phase6_guidance prints "w = 1 check: matches"; then
python -m rebuild.walk_numbers | grep 'p6\.' lists every p6. key; the evidence is
rebuild/out/phase6/phase6.json.
Limits: add the two files and phase6() only; leave ideal.py, parts_world.py, kinds.py and the phase
1 to 5 scripts as they are; write nothing in the ladder, the step files or REBUILD_WALK.md; no
commit.
Hand-off: a report naming the w grid, every p6. key with its value and the brick row's new mark;
steps 6 and 7 of artifacts/drips/simple-distributions-rebuild/ladder/06-guidance-as-a-dial.md read
the keys and phase6.json.
```

added: `rebuild/guidance.py` beside the phase script (from the brick row in `REBUILD_WALK.md`) · the four-quadrant answer-key toy, its w = 1 numbers as the check, and layer 5's pooled p(x), on which the dial is flat (from `answer_key.py`, its card and `rebuild/phase5_dial.py`) · the bound that keeps the guided target a [density](../../../../../../Obsidian/Mathematics/Statistics/Distributions/Probability%20Density%20Function.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=188 "Introduction to probability for data science, section 4.1, Builds the intuition of probability per unit length, defines the PDF and its properties (Definitio..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=186 "Mathematics for machine learning, section 6.2.2, The density and the CDF (Definitions 6.1 and 6.2), a uniform example (Example 6.3), and discrete against con..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=36 "Pattern recognition and machine learning, section 1.2.1, The density's two conditions (equations 1.25 and 1.26), its link to the CDF, and why a density picks..."), [▶1](https://www.youtube.com/watch?v=QKA4HNEw3aY&t=52s "The Organic Chemistry Tutor, Probability Density Functions, 0:52 to 3:39, Verifies that a given function is a density by checking that its integral equals 1...."), [▶2](https://www.youtube.com/watch?v=oI3hZJqXJuc&t=135s "StatQuest, The Main Ideas behind Probability Distributions, 2:15 to 5:14, A curve as a distribution, with probability read as area under it, not its height. ..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/02-densities-and-the-gaussian.md#21-a-density-can-exceed-1 "Chapter 2: Read a density as area and draw a Gaussian from its covariance, deep-learning/diffusion-models/01-poe-derivation-foundations, chapter 02, section ..."), [◈](https://claude.ai/artifact/DBmJtn6xy75sPtQwPEvQZU#nb-sec1 "Kinds of composition notebook, Setup: the ideal expert, The ideal expert's log-density on a grid, in four steps from the 2D Gaussian to its logsumexp form, t...")</sub> and the importance-sampling route for the parts world (from [Bayes' rule](../../../../../../Obsidian/Mathematics/Statistics/Probability/Bayes%20Theorem.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=105 "Introduction to probability for data science, section 2.4.3, Works with numbers: the theorem with its proof and the law of total probability, then the chance..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=31 "Pattern recognition and machine learning, section 1.2, Works with numbers: a red box of 2 apples and 6 oranges and a blue box of 3 apples and 1 orange, the s..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-statistics.pdf#page=74 "Introduction to statistics (Wits lecture notes), sections 3.7.7 and 3.7.8, Probability tree diagrams, the law of total probability, then Bayes' theorem for a..."), [▶1](https://www.youtube.com/watch?v=HZGCoVF3YvM&t=0s "3Blue1Brown, Bayes theorem, the geometry of changing beliefs, 0:00 to 4:09, The video's opening example, which the next chapter turns into the formula. Chapt..."), [▶2](https://www.youtube.com/watch?v=HZGCoVF3YvM&t=249s "3Blue1Brown, Bayes theorem, the geometry of changing beliefs, 4:09 to 10:13, From a worked example to the general formula. Chapter-timed, from the video less..."), [◆](../../../../../../goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/02-bayes-rule.md "Turn a conditional around with Bayes' rule, deep-learning/generative-models/03-sampling-from-a-graph, plan 02. Bayes' rule worked on the fuel gauge's table.")</sub> and `rebuild/parts_world.py`) · the samplers and the floor (from `rebuild/phase5_dial.py` and run 5 of the when-poe map) · SDXL's weight and the inferred tilt (from the hybrid-testbed map, claim 1) · what `walk_numbers.py` holds and how it runs, and the two tolerances (from `rebuild/walk_numbers.py` and `rebuild/phase3_kinds.py`)
dropped: the assumption that a parts-world blend appears at some w up to 7.5 (why: no source shows one does, so the reference records `None` when none does)

⏱ you 2 min to paste · the AI's time is in step 2's line

### 2. AI, in the background: quizzes 6.1 and 6.2

```text
In the background, prepare and build layer 6's two quizzes, bottom up, the way step 2 of
artifacts/drips/simple-distributions-rebuild/ladder/00-setup.md does for layer 1. Goal: "write the
guided target exactly and find where the blend appears".

Task: for each quiz, first dispatch prep-cadence (quiz-prep-pulse --build) with the goal and minutes
below, then dispatch quiz-cadence (mode build, repo
/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain).
- Quiz 6.1, --minutes 40: the college maths under layer 6. Raising a density to a power and
  renormalising, Bayes' rule p(A|x) ∝ p(x|A)p(A).
- Quiz 6.2, --minutes 60: layer 6 itself. Classifier-free guidance as p(x)·p(A|x)^w·p(B|x)^w, what
  w > 1 does to the modes, w = 1 as construction (1).
Starts from: quiz 6.1 takes its material from the Study first line of
artifacts/drips/simple-distributions-rebuild/ladder/06-guidance-as-a-dial.md. Quiz 6.2 takes it
from that file, the "Guidance as a tilted density" row of the Bricks table in
artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, and the idea-map claims below. Layer
6 has no step file yet; its build writes one (step 6 of the ladder file).
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- quiz-cadence stops when a quiz has no coverage file in .quiz-prep/, so prep-cadence runs first for
  each quiz (quiz-cadence, "The preparation gate", as step 2 of ladder/00-setup.md states).
- pdftotext is not installed on mscluster, so
  ~/.claude/skills/index-source/scripts/find-in-textbooks.py fails; open the PDFs with PyMuPDF from
  the co3 environment (checked on mscluster109: import pymupdf works in co3, version 1.28.2, and
  import pypdf fails everywhere).md, step 2).
- The shelf index is ~/Obsidian/Mathematics/references/textbooks/Textbooks.md and the
  worked-examples index is ~/Obsidian/Mathematics/references/exercises/Worked Examples.md
  (quiz-prep-pulse, "The bank").
- The layer's textbook passage is Goodfellow et al., Deep Learning 17.5.1 (tempering, from p615),
  and its video is Umar Jamil on classifier-free guidance, 22:20 to 31:00 (the Study first line).
- Raising a Gaussian to the power w multiplies its precision by w (vault note Powers Of Gaussians).
- SDXL draws its cat-dog hybrid with one model and two prompts at w = 7.5, as
  ε_∅ + w(ε_A − ε_∅) + w(ε_B − ε_∅), and the 2D work measured only w = 1
  (artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md, claim 1).
- At w = 1 the target is construction (1), p(x|A)·p(x|B)/p(x); on the four-quadrant toy with
  "contains" labels it puts 0.989 within 3 std of the top-right blob for every A∩B share r from 0.25
  to 0.001, with Z = r (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 3 and run 1).
- Numbers for w > 1 come from the p6. keys of python -m rebuild.walk_numbers and from
  rebuild/out/phase6/phase6.json once step 1's reference has run; before then a question asks for
  the derivation, never an invented number.
Outputs:
- two coverage files under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, each with a
  verdict per question: ready, blocked, or unprepared by choice;
- rows in the worked-examples index for every banked problem;
- two published quiz pages, each opening on a read-first list with minutes per item, a short path
  and a full path.
Target: no question on either page carries the verdict blocked; quiz 6.2 carries banked problems,
about 6 to 10 per concept at the build depth; every number in an answer traces to the answer key, a
p6. key or a cited textbook page. I judge, from the pages.
Progress: prep-cadence reports, per quiz, its question count, node count and verdicts; quiz-cadence
reports each URL as it publishes.
Proof: both coverage files exist under .quiz-prep/ and list no question as blocked; both URLs open.
Limits: change nothing in rebuild/ or the step files; in the ladder file, write only the two URLs.
Hand-off: write quiz 6.1's URL into step 3 and quiz 6.2's into step 5 of
artifacts/drips/simple-distributions-rebuild/ladder/06-guidance-as-a-dial.md, each replacing "(URL
lands here)".
```

added: the two-dispatch order, the preparation gate, the missing `pdftotext` and the PyMuPDF route, and the two indexes (from step 2 of the setup file) · the layer's textbook and video (from the Study first line) · the facts each quiz tests, with the answer key's w = 1 numbers (from the hybrid-testbed and when-poe maps) · the rule against numbers that do not exist yet · the URL write-back
dropped: "the way the setup section of PROMPTS.md does" (why: that quiz prompt is step 2 of `ladder/00-setup.md`; the guide holds none) · "plan PROMPTS.md (layer 6)" (why: the guide has no layer 6 section; the material is this ladder file and its brick row)

⏱ AI ~2 h for both (estimate), beside the reading stop

### 3. You: quiz 6.1

Quiz 6.1: (URL lands here)

Built by step 2. ⏱ you 20 to 40 min · stop: every *needed now* question right and at least 80% of the rest; below that, read the items your misses point at and retake.

### 4. You: the maths drip

```text
/drip --math raising a density to a power and what it does to the modes, ending at classifier-free
guidance as p(x)·p(A|x)^w·p(B|x)^w, worked on the quadrant toy's ideal experts

Starts from: quiz 6.1 passed; any question it marked missed becomes a first rung.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- Raising a Gaussian to the power w multiplies its precision by w, so q_a^(w_a)·q_b^(w_b) is a
  Gaussian with precision w_a/v_a + w_b/v_b while that stays positive (vault note Powers Of
  Gaussians).
- By Bayes' rule p(A|x) = p(x|A)p(A)/p(x), so p(x)·p(A|x)^w·p(B|x)^w is
  p(x)^(1 − 2w)·p(x|A)^w·p(x|B)^w up to a constant; at w = 1 it is construction (1),
  p(x|A)·p(x|B)/p(x) (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 3).
- SDXL composes two prompts at w = 7.5 as ε_∅ + w(ε_A − ε_∅) + w(ε_B − ε_∅), and the 2D work
  measured only w = 1 (artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md, claim 1).
- The quadrant toy here is the four-quadrant answer-key toy: blobs at (±0.5, ±0.5), std 0.15, A is
  x1 > 0, B is x2 > 0. At w = 1 with "contains" labels and an A∩B share r it puts 0.989 within 3
  std of (0.5, 0.5) for every r from 0.25 to 0.001, with Z = r
  (artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/README.md).
- 0.989 is 1 − e^(−9/2), the mass one blob holds within 3 std (when-poe-reaches-a-and-b idea map,
  run 4).
- p(A|x) and p(B|x) are at most 1, so the guided target never exceeds p(x) (Bayes' rule).
- Numbers for w > 1 are the p6. keys step 1's reference writes; the drip computes its own and never
  quotes one that does not exist yet.
Outputs: the drip's thread folder under artifacts/drips/, with dispatch.md recording my notation,
the worked numbers I used and what clicked.
Target: I write the guided target from Bayes' rule without notes, rewrite it as
p(x)^(1 − 2w)·p(x|A)^w·p(x|B)^w, and show by hand that at w = 1 on the four-quadrant toy Z = r; I
judge.
Hand-off: the thread folder's path, which quiz 6.2, step 8's scene and step 9's cross-links read.
```

added: the precision rule, the Bayes rewrite and SDXL's weight (from the vault note and the two idea maps) · which quadrant toy, with its w = 1 numbers (from the answer-key card) · quiz 6.1's misses as first rungs · the Z = r derivation as the target
dropped: nothing

⏱ rough: you 30 to 60 min · AI ~10 min · stop: you write the guided target without notes

**Outcome:** you can say what w > 1 does to a density, and write the guided target exactly. Factored [classifier-free guidance](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Classifier-Free%20Guidance.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/probabilistic-machine-learning-advanced-topics.pdf#page=909 "Probabilistic machine learning: advanced topics, section 25.6.3, Classifier-free guidance, Derives the implicit classifier from Bayes' rule, log p(c∣x) = log..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/understanding-deep-learning.pdf#page=379 "Understanding deep learning, section 18.6.3, Conditional generation, Feeds the class into the U-Net the way the time step is added and trains on conditional ..."), [3](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/sampling-composed-models/compositional-visual-generation-with-composable-diffusion-models.pdf#page=7 "Liu and others 2022, Compositional visual generation with composable diffusion models, section 4.2, Composes concepts as p(x) ∏ p(c_i∣x) and notes that with ..."), [▶1](https://www.youtube.com/watch?v=Edr4uZFh4EE&t=2730s "CS231n lecture 14, 45:30 to 50:05, Drops the label to a null value half the time in training and combines v_cfg = (1 + w) v_y − w v_null at sampling, where w..."), [▶2](https://www.youtube.com/watch?v=WUUq6TVAu8U&t=5665s "CME296 lecture 4, 1:34:25 to 1:40:38, Derives the implicit classifier from the conditional and unconditional predictions with a weight w of about 3, the cond..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/plans/08-cfg-trained-on-mnist.md "CFG trained on MNIST, a w sweep, deep-learning/diffusion-models/00-from-scratch-to-sdxl, plan 08. Trains a guided model on MNIST and sweeps the guidance weig..."), [◈](https://claude.ai/artifact/1GayZthQkRL4FmukdCV9WW#nb-poe "The SDXL notebook in ten parts, PoE composition and CFG inside it, Two prompts composed as a product of experts, a run that shows the composition failing, an...")</sub> (Xia and Glocker 2025) is construction (1) with one weight per condition, already used on MIMIC-CXR.

### 5. You: quiz 6.2

Quiz 6.2: (URL lands here)

⏱ you 30 to 60 min, the banked problems included · stop: every *needed now* question right and at least 80% of the rest, then the build. Below that, the misses become the first rungs of a second `/drip --math`, and you retake before building.

### 6. You: the build

```text
/drip-execute-plan --request the guidance dial for w = 1 to 7.5 on the quadrant toy and the parts
world, starting from rebuild/ideal.py and rebuild/parts_world.py, reporting A∩B share, total
variation and Z per w, as layer 6 of artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md; I
type every block myself into ~/Playground/PhD/simple-distributions-walk/walk06.py; bar: w = 1
reproduces construction (1) exactly, and the walk finds the w where the parts-world blend appears

Task: cut the layer into 4 to 6 pieces and write them at the walk's open as the step file
artifacts/drips/simple-distributions-rebuild/06-guidance-as-a-dial.md, in the shape of the step
files for layers 1 to 5; then build, teach and test one piece per round. I run walk06.py from
~/Playground/PhD/simple-distributions-walk/ with
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python and
PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy.
Starts from: quiz 6.2 passed; the layer 6 --math drip folder under artifacts/drips/, whose
dispatch.md holds my notation (step 4); rebuild/guidance.py, rebuild/phase6_guidance.py, the p6.
keys and rebuild/out/phase6/phase6.json (step 1).
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The w grid, the p6. keys and phase6.json are step 1's; walk06.py uses the same grid (from step 1).
- The brick: the target p(x)·p(A|x)^w·p(B|x)^w computed exactly on the quadrant toy and the parts
  world for w = 1 to 7.5, with A∩B share, total variation and Z per w (REBUILD_WALK.md, brick
  "Guidance as a tilted density").
- SDXL's hybrid comes from one model and two prompts at w = 7.5 over 50 DDIM steps; the tilted
  target is inferred from the cache's residual formula, and the 2D work measured only w = 1
  (artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md, claim 1 and Sources).
- At w = 1 the target is construction (1), p(x|A)·p(x|B)/p(x), which equals p_A·p_B only where p(x)
  is flat (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 3).
- The quadrant toy is the four-quadrant answer-key toy: blobs at (±0.5, ±0.5), std 0.15, A is
  x1 > 0, B is x2 > 0. At w = 1 with "contains" labels it puts 0.989 within 3 std of (0.5, 0.5) for
  every A∩B share r from 0.25 to 0.001, with Z = r
  (artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/README.md, answer_key.py).
- Layer 5's quadrant sets (phase5_dial.make_sets) have an unconditional p(x), the two sets pooled;
  on them the guided target's A∩B share is near 1 at every w for any f > 0 (step 1's reference).
- p(A|x) and p(B|x) are at most 1, so the guided target never exceeds p(x) and stays a density for
  every w ≥ 0 (Bayes' rule).
- A power of a mixture is not a mixture, so in the parts world Z and the A∩B share at w ≠ 1 are
  estimates, with a standard error (rebuild/parts_world.py; from step 1).
- Success is three numbers together: the share in A∩B, the total variation to the exact density,
  and Z; reverse diffusion's share 0.983 against 0.989 hid a total variation 0.049 above the floor
  (when-poe-reaches-a-and-b idea map, claim 5 and run 5).
- With exact experts at r = 0.05, annealed ULA, fixed MALA and UHA sit at the floor (0.124 to 0.132
  against 0.133 ± 0.006) and reverse diffusion at 0.182 (when-poe-reaches-a-and-b idea map, run 5).
- Parts world, layer 5: an empty background (variance 0.01) gives a ghost at 0% overlap and A∩B
  weight 1.000 from 0.1% up; a loose background (variance 1.0) gives 1.000 at every overlap (step
  file 05, step 6).
- The parts-world blend is example 9(b): product mean 0, 10 sds from each expert, log Z −1583.4
  (step file 04, step 4). No source shows that guidance produces it at any w.
- Tolerances: an exact number within 0.005, a sampler number within 0.03
  (rebuild/phase3_kinds.py).
- Phases 1 to 5 are imported from the reference, as step file 05 does (step file 05, "How to walk
  it").
Outputs:
- the step file artifacts/drips/simple-distributions-rebuild/06-guidance-as-a-dial.md, its pieces
  ticked, and the walk map that holds my place between sittings;
- walk06.py: one script that prints, per toy and w, the A∩B share and Z, and per sampler the share
  and total variation beside the floor;
- on the whiteboard, a two-mode 1D density raised to w = 1, 2 and 7.5 (this layer's Achieve line).
Target: at w = 1 on the quadrant toy, the top-right share is within 0.005 of 0.989 and Z within
0.005·r of r; every exact number equals its p6. key to the digits shown; every sampler number is
within 0.03 of phase6.json; the walk names the w where the parts-world blend appears, or that none
up to 7.5 does. I judge, from the printed lines.
Progress: each piece prints its number as it lands: the w = 1 check first, then the quadrant toy's
share and Z per w, then the samplers' total variation, then the parts world.
Proof: python walk06.py prints the w = 1 check within those tolerances, and step 7's comparison
shows no differing number.
Limits: never edit rebuild/ or the ladder files; I type, I do not paste; no commit.
Hand-off: the step file's path, the retrospective the walk closes on, and the /drip-retro command
it prints.
```

added: where the step file lands and how `walk06.py` runs (from the guide and step file 05) · the reference's outputs as the starting point (from step 1) · the brick, the SDXL weight, construction (1) and the answer key's w = 1 numbers (from `REBUILD_WALK.md` and the two idea maps) · layer 5's pooled p(x), on which the dial is flat (from `rebuild/phase5_dial.py`) · the density bound, the estimated parts-world numbers, the three success numbers and the sampler floor (from Bayes' rule, `rebuild/parts_world.py` and the when-poe map) · the layer 5 parts-world results and the blend's definition (from step files 04 and 05) · the tolerances (from `rebuild/phase3_kinds.py`)
dropped: "the walk finds the w where the parts-world blend appears" as a certain outcome (why: no source shows guidance produces the blend; the bar now accepts the answer that none up to 7.5 does)

⏱ rough: you 1 to 2 h (about 5 pieces) · AI ~35 min · stop: the prompt's bar

**Outcome:** the A∩B share, total variation and Z for each w from 1 to 7.5 on both toys, and the w where the parts-world blend appears.

### 7. You: compare your numbers with the reference

**Compare.** `python -m rebuild.walk_numbers` prints the `p6.` keys the AI lane added; your `walk06.py` prints the same numbers.

```bash
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m rebuild.walk_numbers | grep 'p6\.'
cd ~/Playground/PhD/simple-distributions-walk
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python walk06.py
```

⏱ you ~10 min (estimate) · stop: every exact number your script prints equals its `p6.` line to the digits shown, and every sampler number sits within 0.03 of `rebuild/out/phase6/phase6.json`. A number that differs is either a lesson or a bug in one of the two files; find which before layer 7.

### 8. AI, in the background: the scene

```text
/math-scene the folder the layer 6 --math drip wrote under artifacts/drips/: the guided target on
the quadrant toy and the parts world as w moves from 1 to 7.5

Starts from: the layer 6 --math drip folder and its dispatch.md (step 4's hand-off); the p6. keys
and rebuild/out/phase6/phase6.json (step 1).
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The quadrant toy is the four-quadrant answer-key toy: blobs at (±0.5, ±0.5), std 0.15, A is
  x1 > 0, B is x2 > 0; at w = 1 it puts 0.989 within 3 std of the top-right blob, with Z = r
  (artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/README.md).
- SDXL's hybrid appears at w = 7.5, the slider's right end
  (artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md, claim 1).
- The w grid and every number per w are the p6. keys and phase6.json rows (from step 1); parts-world
  numbers at w ≠ 1 are estimates and carry their standard error (from step 1).
- Every number on the scene comes from those keys, the answer key or the drip's dispatch.md, never
  estimated by the scene.
Outputs: one published scene with w as a slider, in my notation from dispatch.md.
Target: every number the scene shows equals its p6. key or phase6.json row.
Hand-off: write the scene's URL into step 8 of
artifacts/drips/simple-distributions-rebuild/ladder/06-guidance-as-a-dial.md, replacing "(URL
lands here)".
```

added: the toy's definition and its w = 1 numbers (from the answer-key card) · SDXL's weight as the slider's end (from the hybrid-testbed map) · the reference's keys as the only number source · the URL write-back
dropped: nothing

Scene: (URL lands here)

⏱ you ~20 min · AI ~45 min, beside layer 7's maths · stop: the scene is published and its numbers match your walk's

**Outcome:** a scene with w as a slider, showing the guided target on both toys.

### 9. You: close the layer

Run the `/drip-retro` command the build printed at its close (⏱ about 25 minutes). Then paste this:

```text
In the background, dispatch xref-cadence over the files layer 6 of the composition ladder wrote: the
layer 6 --math drip folder under artifacts/drips/, the step file
artifacts/drips/simple-distributions-rebuild/06-guidance-as-a-dial.md, and the retrospective the
layer 6 walk closed on.

Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The reference sets to link against are the journey plans and vault notes on the Study first line
  of artifacts/drips/simple-distributions-rebuild/ladder/06-guidance-as-a-dial.md; the "Guidance as
  a tilted density" row of the Bricks table in
  artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md; claim 1 of
  artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md; and claims 3 and 5 of
  artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md.
- ~/Playground/PhD/simple-distributions-walk/ is scratch space; link nothing into it (CLAUDE.md, the
  playground is disposable).
Outputs: links written both ways between those files and the reference sets.
Hand-off: the agent's report, listing every link it wrote, read before layer 7 starts.
```

added: the reference sets, by path (from this layer's Study first line, `REBUILD_WALK.md` and the claims the layer rests on)
dropped: nothing

⏱ you 2 min to paste · AI ~10 min (estimate)

---

⬅️ [Reading stop: Bradley](05a-reading-stop-projective-composition.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 7: A without B](07-a-without-b.md) ➡️
