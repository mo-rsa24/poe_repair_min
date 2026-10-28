# ⭐ Layer 8: the interaction term, defined and shown

⬅️ [Layer 7: A without B](07-a-without-b.md) · 🏠 [the guide](../PROMPTS.md) · [Correction prior art](08a-correction-prior-art.md) ➡️

**Why it is critical.** This is the first half of the research goal. It is defined exactly here before any learned version of it is trusted.

**Depth.** Build: r(x) in log-density form, its ε-space form, and the label term. You 3.3 to 6.7 h in all (rough).

**Study first.** Bayes and conditional independence: [the assumption the product rests on](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/02-the-assumption-poe-rests-on.md), worked in [joint tables and independence](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/01-joint-tables-and-independence.md), and [the route through p(c1, c2 | x), section 8.2](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/08-assembling-the-poe-rule.md). The interaction term: [Kinds Of Composition](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/Kinds%20Of%20Composition.md), [where independence fails](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/15-where-independence-fails.md) and its ε-space form in [a corrector on the cached trajectories, equation 27.1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/27-a-corrector-on-the-cached-trajectories.md). Textbook: Deisenroth et al. 6.3 (Bayes, p189). Log-odds has no passage that teaches it yet beyond Murphy 10.2 (binary logistic regression, p377); quiz 8.1's preparation writes the missing passage. The interaction term on a four-image toy, where a hidden cause breaks the fork and the product's error is measured: [break the fork with a hidden cause](../../../../../../goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/18-poe-fork-built.md).

**Ideas to tie in.** From [the ideas that tie the ladder together](../../../ideas/ideas-that-tie-the-ladder-together.md): [conditional independence made visible as an interaction residual](../../../ideas/ideas-that-tie-the-ladder-together.md#1-make-conditional-independence-visible-as-an-interaction-residual) (needed, 2 to 3 h, no training); [the interaction score as the object a correction can learn](../../../ideas/ideas-that-tie-the-ladder-together.md#5-learn-the-identifiable-object-which-is-the-interaction-score) (needed, 2 to 3 h, no training); [the interaction's own diffusion-time trajectory](../../../ideas/ideas-that-tie-the-ladder-together.md#6-give-the-interaction-its-own-diffusion-time-trajectory) (needed, 2 to 4 h, no training). Each comes with an exact toy check.

**Achieve.** `walk08.py` plotting r with the wet rule off and on at every noise level, and `walk08b.py` printing the label term · on the whiteboard: the two-by-two table of labels with r in each cell · problems: the ones quiz 8.2 banks · quizzes 8.1 and 8.2 passed.

## The flow

```text
 AI, in the background                         │  You, in order
───────────────────────────────────────────────┼───────────────────────────────────────────────
 quizzes 8.1 and 8.2, prepared ──────────────▶ │  1. quiz 8.1: Bayes for two labels
   (paste during layer 7's build)              │       ▼
                                               │  2. /drip --math: r(x) and its ε form
                                               │       ▼
                                               │  3. quiz 8.2: needed-now right, 80% of the rest
                                               │       ▼
                                               │  4. build: walk08.py, walk08b
 builds rebuild/phase8_interaction.py ───────▶ │  5. compare: p8. numbers equal yours
                                               │       ▼
 math-scene, beside prior art ◀─────────────── │  6. scene
                                               │       ▼
 xref-cadence links the layer ◀─────────────── │  7. /drip-retro, then the next layer
```

## The steps

### 0. AI, in the background: the layer 8 reference and quizzes 8.1 and 8.2, one layer ahead

**AI, one layer ahead.** Paste both when you start layer 7's build.

```text
In the background, build the layer 8 reference: rebuild/phase8_interaction.py, the interaction term
in the parts world with a wet rule (the cat half shifts when a dog is present), computed exactly
from the Gaussian mixtures, and r_t with its gradient at each noise level. Print r with the rule off
and on, per noise level. Add those numbers to rebuild/walk_numbers.py as phase8() with keys p8.*,
and run it once with the co3 python.

Task: in /home-mscluster/mmolefe/PhD/reduce-reuse-recycle, write rebuild/interaction.py (the
functions) and rebuild/phase8_interaction.py (the run, python -m rebuild.phase8_interaction --out
rebuild/out/phase8), the split the brick row names, and add phase8() to rebuild/walk_numbers.py.
Run everything from the repo root with /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python.
Starts from: rebuild/ as layers 1 to 5 left it; nothing from layer 8's walk.
Settled facts:
- The brick row: in the parts world with a "wet" rule (the cat half shifts when a dog is present),
  r(x) = log p(x|A,B) − [log p(x|A) + log p(x|B) − log p(x)] in closed form, expected constant
  with the rule off and varying with it on; then the noised r_t and its gradient, written also in
  ε-space as ε_joint − ε_A − ε_B + ε_∅, the residual poe_repair_min already caches for 25 pairs
  at all 50 steps; files rebuild/interaction.py and rebuild/phase8_interaction.py; not built
  (artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, brick "The interaction term,
  exact").
- When each label is a fixed function of the image, the interaction term is exactly zero, so
  interaction is no third operator: it is a same-variable product whose A∩B appearance must be in
  the training data. The wet-dog shift was found at 20% overlap and missed at 0% (mean x1 −0.79
  against −0.48) (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 7.1 and run 1).
- By Bayes, r(x) = log p(A,B|x) − log p(A|x) − log p(B|x) + log[p(A)p(B)/p(A,B)], so with clean
  labels r is one constant wherever the joint has mass, wet rule or not; the brick row's "varying
  with it on" can then only appear in r_t at t > 0, off the joint's support, or for experts whose
  training sets lack the wet A∩B images (claim 7.1;
  artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claims 3 and 9). Compute and report all
  three; never tune the world to force a difference at t = 0.
- "Interaction" splits in two: appearance interaction (a data condition, no term) and label
  dependence given the image (a real term, zero for clean labels, nonzero for report-mined labels)
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 7.3).
- No paper adds an interaction term to a product; the clean-label argument is Bradley et al.
  2025a's Bayes composition, p.4 (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 7, route 1).
- rebuild/parts_world.py: a sample is 16 pixels, HEAD 0 to 7, BODY 8 to 15, LEFT and RIGHT the
  column halves; Mixture(w, m, v) normalises its weights, noised(abar) gives N(√ᾱ m, ᾱ v + 1 − ᾱ)
  per component, responsibilities(x) is float64 NumPy, and torch() casts to float32, so keep the
  closed-form checks out of torch() and logp_torch(); placed(value, half, var_in=0.01,
  var_out=1.0, bg=0.0) builds one placement; mixture_product(A, B) builds every pair.
- rebuild/phase5_dial.py: pw_experts(f, bg_var) builds layer 5's experts, cat-only on either half
  at weight (1 − f)/2 each and cat-and-dog at f/2 each (B likewise with the dog at −1), object
  variance 0.01, background mean 0 with variance 0.01 or 1.0; pw_ab(x) calls a sample A∩B when
  one half's mean is above 0.75 and the other's below −0.75.
- Layer 5's parts-world dial: with an empty background the product at 0% overlap is a ghost with
  half-means (0.5, −0.5) and A∩B weight 0.000, and the matching pair wins by 386.2 nats from 0.1%;
  with a loose background (variance 1.0) the A∩B weight is 1.000 at every overlap including 0%
  (step file 05-the-overlap-dial.md, step 6; p5.matching_pair_advantage_nats in
  rebuild/out/walk-numbers.json).
- The parts world is exact: rebuild/phase4_parts_world.py matches example 9's numbers.json to a
  largest relative difference of 2.87e-16 (step file 04-the-parts-world.md, step 8).
- Noise levels come from the cosine schedule, rebuild.ideal.ABAR over T = 100; t = −1 means clean
  in parts_world.composed_target; ᾱ_t is 0.9994, 0.8356, 0.4783, 0.1335 and 0.0 at t = 0, 25,
  50, 75 and 99 (the p1.abar_* keys in rebuild/out/walk-numbers.json).
- ε = −√(1 − ᾱ_t)·∇log q_t (layer 1), so ∇r_t = −(ε_joint − ε_A − ε_B + ε_∅)/√(1 − ᾱ_t), and
  ∇r_t is exactly what the composed score s_A + s_B − s_∅ misses of the joint's score (brick row
  above; ladder/01-one-diffusion-expert.md, step 2).
- SDXL's cache under /datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache/ holds ε_A,
  ε_B, ε_joint and ε_∅ at all 50 steps for 25 pairs, with the residual gs*(eps_j − eps_a − eps_b +
  eps_uncond) at guidance 7.5; its lower bound on the interaction did not track which pairs blend
  (artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md, claim 1 and Sources).
- Claim 7.4's label term: points uniform on [−1, 1]², A true when x1 < 0 and B true when x2 > 0
  (the A and B of rebuild/phase5_dial.py), A read with probability 0.9 when true, and inside A∩B,
  B read only when A was read; the term log p(A,B|x) − log p(A|x) − log p(B|x) is +0.105 nats
  for (A,B) inside A∩B, strongly negative for (¬A,B) there, and zero elsewhere
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 7.4). +0.105 is −ln 0.9, whatever the
  probability of reading B once A was read.
- rebuild/walk_numbers.py prints through put(key, value), its __main__ loop runs phase1, phase2,
  phase3 and phase5 and writes rebuild/out/walk-numbers.json; it holds worked numbers, and run
  results are read from rebuild/out/phase*/ (walk_numbers.py docstring).
- Every phase is one command from the repo root, python -m rebuild.<script>, with outputs in
  rebuild/out/phaseN; pick the device with rebuild.device.pick_device, the GPU guard
  (REBUILD_WALK.md, Pins).
Outputs:
- rebuild/interaction.py: r_t and ∇r_t, in float64, for any four Mixture objects (joint, A, B,
  unconditional) at a noise level t, in any dimension, since layer 9 applies it to a 2D toy;
- rebuild/phase8_interaction.py: one world of image kinds (cat-only, dog-only, cat-and-dog) with
  prior weights, its true conditionals derived by "contains" labels, the wet rule off and on, and
  the experts that never saw a cat-and-dog image from pw_experts(0.0, 1.0); it prints r's mean and
  spread over 2,000 joint draws and 2,000 p(x) draws at t = −1, 0, 25, 50, 75 and 99, ∇r_t at one
  fixed image, and the exact label term of claim 7.4 in each label cell;
- rebuild/out/phase8/phase8.json with every chosen setting beside each number, and
  rebuild/out/phase8/interaction-term.png: r's spread on the joint's draws (y) against the noise
  level (x), one line per setting, rule off and on;
- phase8() in rebuild/walk_numbers.py, added to its __main__ loop, putting p8. keys: the chosen
  settings (p8.wet_cat_value, the world's weights, the background variance), r and ∇r_t at the
  fixed image per setting and level, the ε-space residual there, and p8.label_term_* per cell;
- rebuild/out/walk-numbers.json rewritten with the p8. keys added.
Target: the composed score plus interaction.py's ∇r_t equals the joint's exact score, and ∇r_t
equals the four exact scores combined, both within 1e-10 at every printed level; r from the true
conditionals has a spread on the joint's draws at t = −1 and t = 0 at float64 precision, rule off
and on, or the run stops and reports it against claim 7.1; the label term for (A,B) inside A∩B
prints 0.1054; every p1., p2., p3. and p5. value is unchanged. I judge, from the printed lines.
Progress: the identity check prints first, then one line per setting and level, then the label
cells, then the p8. keys.
Proof: python -m rebuild.phase8_interaction --out rebuild/out/phase8 prints "identity max
difference <x>" with x under 1e-10, and python -m rebuild.walk_numbers | grep 'p8\.' prints the
keys; git diff rebuild/out/walk-numbers.json shows only added p8. lines.
Limits: change no existing function in rebuild/ and no p1. to p5. value; write nothing under
artifacts/; no commit.
Hand-off: rebuild/out/phase8/phase8.json and the p8. keys, which step 4a's walk, step 5's
comparison, step 6's scene and layer 9's reference read; at the top of phase8_interaction.py's
docstring, one paragraph saying where r moved with the wet rule and where it did not.
```

added: the function and run split, the run command and the output folder (from the brick row and the Pins) · claim 7.1, its Bayes form, and the instruction to report rather than force the brick row's contrast (from the `poe-overlap-dial` idea map, claims 7.1 and 7.3, and the `when-poe-reaches-a-and-b` idea map, claims 3 and 9) · the real functions in `parts_world.py` and `phase5_dial.py`, and the float32 cast to avoid (from the code) · layer 5's ghost and loose-background results (from step file 05 and `walk-numbers.json`) · the noise levels, the ε-space gradient and SDXL's cache (from `walk-numbers.json`, layer 1 and the `sdxl-shaped-hybrid-testbed` idea map) · the exact label term, so step 4b has a reference (from claim 7.4) · how `walk_numbers.py` is built (from its docstring)
dropped: "rule off (constant)" as a printed expectation (why: claim 7.1 says r from clean labels is constant with the rule on too, so the run prints the spread and reports which holds)

```text
In the background, prepare and build layer 8's two quizzes, bottom up, for the composition ladder
in artifacts/drips/simple-distributions-rebuild/: quiz 8.1 and quiz 8.2.

Task: for each quiz, first dispatch prep-cadence (quiz-prep-pulse --build) with the goal and minutes
below, then dispatch quiz-cadence (mode build, repo
/home-mscluster/mmolefe/PhD/reduce-reuse-recycle). The goal for both is "define the interaction
term exactly and say where it lives".
- Quiz 8.1, --minutes 40: the college maths under layer 8. Bayes' rule for two labels, conditional
  independence, log-odds.
- Quiz 8.2, --minutes 60: layer 8 itself. r(x) = log p(x|A,B) − [log p(x|A) + log p(x|B) −
  log p(x)], why it is constant when the labels are fixed functions of the image, its ε-space form
  ε_joint − ε_A − ε_B + ε_∅, the label term for report-style labels.
Starts from: nothing from this layer; it runs during layer 7's build, so the layer 8 --math drip
has not run and layer 8 has no step file yet. Quiz 8.1 takes its material from the Study first
line of artifacts/drips/simple-distributions-rebuild/ladder/08-the-interaction-term.md; quiz 8.2
takes it from that file's steps 2, 4a and 4b, the brick row "The interaction term, exact" in
artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, and claim 7 (7.1 to 7.4) of
artifacts/ideas/poe-overlap-dial/IDEA_MAP.md.
Settled facts:
- quiz-cadence stops when a quiz has no coverage file in .quiz-prep/, so prep-cadence runs first for
  each quiz (quiz-cadence, "The preparation gate").
- The textbook for Bayes is Deisenroth et al. 6.3, p189; log-odds has no passage that teaches it
  yet beyond Murphy 10.2 (binary logistic regression, p377), so quiz 8.1's preparation writes the
  missing passage (this layer's Study first line).
- pdftotext is not installed on mscluster, so
  ~/.claude/skills/index-source/scripts/find-in-textbooks.py fails; open the PDFs with PyMuPDF from
  the co3 environment (checked on mscluster109: import pymupdf works in co3, version 1.28.2, and
  import pypdf fails everywhere).
- The shelf index is ~/Obsidian/Mathematics/references/textbooks/Textbooks.md and the
  worked-examples index is ~/Obsidian/Mathematics/references/exercises/Worked Examples.md
  (quiz-prep-pulse, "The bank").
- By Bayes, r(x) = log p(A,B|x) − log p(A|x) − log p(B|x) + log[p(A)p(B)/p(A,B)]; when each label
  is a fixed function of the image the term is exactly zero, so the wet dog is a data condition
  (does either expert's data hold A as it looks under B), and the shift was found at 20% overlap
  and missed at 0% (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 7.1 and run 1;
  artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 9).
- With report-mined labels a label term survives; on claim 7.4's quadrant labels (A read with
  probability 0.9 when true, B read inside A∩B only when A was read) it is +0.105 nats for (A,B)
  inside A∩B, strongly negative for (¬A,B) there, and zero elsewhere; +0.105 is −ln 0.9
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claims 7.3 and 7.4).
- A chest X-ray diagnosis is read from its findings with uncertainty, so diagnosis labels carry the
  label term while finding labels given the diagnosis do not (claim 7, route 2).
- The notation: q(x_t | x_0) = N(√ᾱ_t x_0, (1 − ᾱ_t) I) over T = 100 steps of a cosine schedule,
  ε = −√(1 − ᾱ_t)·∇log q_t, so ∇r_t = −(ε_joint − ε_A − ε_B + ε_∅)/√(1 − ᾱ_t)
  (ladder/01-one-diffusion-expert.md, step 2; REBUILD_WALK.md, brick "The interaction term,
  exact").
- Every number an answer uses comes from the idea maps, step files 04-the-parts-world.md and
  05-the-overlap-dial.md, rebuild/out/walk-numbers.json or a cited textbook page; the p8. keys do
  not exist yet, because the layer 8 reference is built beside these quizzes.
Outputs:
- two coverage files under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, each with a
  verdict per question: ready, blocked, or unprepared by choice;
- the missing log-odds passage, written where quiz-prep-pulse --build files passages;
- rows in the worked-examples index for every banked problem;
- two published quiz pages, each opening on a read-first list with minutes per item, a short path
  and a full path.
Target: no question on either page carries the verdict blocked; quiz 8.2 carries banked problems,
about 6 to 10 per concept at the build depth (r(x), its ε-space form, the label term); every number
in an answer traces to a source named above. I judge, from the pages.
Progress: prep-cadence reports, per quiz, its question count, node count and verdicts; quiz-cadence
reports each URL as it publishes.
Proof: the two coverage files exist under .quiz-prep/ and list no question as blocked; the two
URLs open.
Limits: change nothing in the step files' prose, the idea maps or rebuild/; in the ladder files,
write only the two URLs.
Hand-off: write quiz 8.1's URL into step 1 and quiz 8.2's URL into step 3 of
artifacts/drips/simple-distributions-rebuild/ladder/08-the-interaction-term.md, each replacing
"(URL lands here)".
```

added: every field the setup's quiz prompt carries, so this prompt stands alone instead of pointing at "the setup section of PROMPTS.md", which the guide no longer holds (the setup is `ladder/00-setup.md`) · quiz 8.2's material named by path in place of "plan PROMPTS.md (layer 8) with REBUILD_WALK.md", since layer 8's prompts live in this file · the preparation gate (from the quiz-cadence agent definition) · the Bayes and log-odds textbooks and the missing passage (from this layer's Study first line) · the missing `pdftotext` and the PyMuPDF route (from the textbook pass) · claim 7's facts and the route 2 diagnosis line (from the `poe-overlap-dial` idea map) · the notation (from layer 1 and the brick row) · that the p8. keys do not exist yet · the URL write-back
dropped: nothing

⏱ AI ~2 h for both (estimate), beside layer 7's build

### 1. You: quiz 8.1

Quiz 8.1: (URL lands here)

⏱ you 20 to 40 min · stop: the bar, every *needed now* question right and at least 80% of the rest; below that, read the items your misses point at and retake.

### 2. You: the maths drip

```text
/drip --math Bayes' rule for two labels and conditional independence, ending at r(x) = log p(x|A,B)
- [log p(x|A) + log p(x|B) - log p(x)] and its ε-space form ε_joint - ε_A - ε_B + ε_∅

Starts from: quiz 8.1 passed; any question it marked missed becomes a first rung. Layer 1's --math
drip folder under artifacts/drips/ holds my notation for q(x_t | x_0) and the score.
Settled facts:
- By Bayes, log p(x|A) = log p(A|x) + log p(x) − log p(A), so r(x) = log p(A,B|x) − log p(A|x) −
  log p(B|x) + log[p(A)p(B)/p(A,B)]: r is the label term plus a constant (the route through
  p(c1, c2 | x), section 8.2 of the textbook on this layer's Study first line;
  artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 9).
- Construction (1), score(x|A) + score(x|B) − score(x), targets p(x|A)p(x|B)/p(x), which is
  p(x)·p(A|x)·p(B|x) up to a constant, and assumes A and B independent given x
  (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 3 and "What the words are").
- When each label is a fixed function of the image, the term is exactly zero, so the wet dog is a
  data condition rather than a third operator; the shift was found at 20% overlap and missed at 0%
  (mean x1 −0.79 against −0.48) (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 7.1 and run
  1).
- With report-mined labels the term survives: +0.105 nats for (A,B) inside A∩B on claim 7.4's
  labels, which is −ln 0.9 (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claims 7.3 and 7.4).
- ε = −√(1 − ᾱ_t)·∇log q_t, so ∇r_t = −(ε_joint − ε_A − ε_B + ε_∅)/√(1 − ᾱ_t), the gap between
  the joint's score and the composed score (ladder/01-one-diffusion-expert.md, step 2;
  REBUILD_WALK.md, brick "The interaction term, exact").
- poe_repair_min caches that residual for 25 pairs at all 50 steps, at guidance 7.5
  (artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md, claim 1 and Sources).
Outputs: the drip's thread folder under artifacts/drips/, with dispatch.md recording my notation,
the worked numbers I used and what clicked.
Target: I derive r and its ε-space form without notes, say why r is constant when the labels are
fixed functions of the image, and reproduce +0.105 = −ln 0.9 by hand. I judge.
Hand-off: the thread folder's path, which quiz 8.2, step 6's scene and step 7's cross-links read.
```

added: the Bayes rewrite and construction (1)'s target (from the textbook on the Study first line and the `when-poe-reaches-a-and-b` idea map, claims 3 and 9) · claim 7.1 with run 1's numbers and the label term (from the `poe-overlap-dial` idea map) · the ε-space gradient (from layer 1 and the brick row) · SDXL's cache (from the `sdxl-shaped-hybrid-testbed` idea map) · layer 1's drip folder as the notation source
dropped: nothing

⏱ rough: you 30 to 60 min · AI ~10 min · stop: you derive r and its ε-space form without notes

**Outcome:** you can derive r, say why it is constant when the labels are fixed functions of the image, and write its noise-prediction form, the residual `poe_repair_min` already caches for 25 pairs.

### 3. You: quiz 8.2

Quiz 8.2: (URL lands here)

⏱ you 30 to 60 min, the banked problems included · stop: the bar, then the build. Below the bar, the misses become the first rungs of a second `/drip --math`, and you retake before building.

### 4a. You: the build, walk08.py

```text
/drip-execute-plan --request the interaction term in the parts world with a wet rule (the cat half
shifts when a dog is present), computed exactly from the Gaussian mixtures and plotted, then r_t and
its gradient at each noise level, as layer 8 of
artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md; I type every block myself into
~/Playground/PhD/simple-distributions-walk/walk08.py; bar: every p8. number walk08.py prints
equals the reference's, and r from clean labels is constant on the joint's draws at t = 0 with the
rule off and on, with the rule's effect shown where claim 7.1 puts it

Task: cut the layer into 4 to 6 pieces, write the step file at the open, and build, teach and test
one piece per round. I type every block into walk08.py and run it with
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python from
~/Playground/PhD/simple-distributions-walk/, with
PYTHONPATH=/home-mscluster/mmolefe/PhD/reduce-reuse-recycle.
Starts from: the layer 8 reference from step 0, rebuild/interaction.py,
rebuild/phase8_interaction.py, rebuild/out/phase8/phase8.json and the p8. keys in
rebuild/out/walk-numbers.json, whose settings (p8.wet_cat_value, the world's weights, the
background variance) I use; quiz 8.2 passed; the layer 8 --math drip folder under artifacts/drips/,
whose dispatch.md holds my notation.
Settled facts:
- The brick row expects r constant with the rule off and varying with it on, and the noised r_t
  and its gradient in ε-space as ε_joint − ε_A − ε_B + ε_∅ (REBUILD_WALK.md, brick "The
  interaction term, exact").
- When each label is a fixed function of the image, r is exactly constant, so the wet rule alone
  does not move r at t = 0; it shows in r_t at t > 0, off the joint's support, and for experts
  whose training sets lack the wet A∩B images (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim
  7.1; step 0's reference, the docstring of rebuild/phase8_interaction.py).
- The parts world is rebuild/parts_world.py: 16 pixels, HEAD 0 to 7 and BODY 8 to 15, LEFT and
  RIGHT halves, a cat +1 and a dog −1 on their pixels with variance 0.01; Mixture.noised(abar)
  gives N(√ᾱ m, ᾱ v + 1 − ᾱ) per component, and a product of mixtures is a mixture over every
  pair (step file 04-the-parts-world.md, steps 1, 5 and 6).
- rebuild/phase5_dial.py's pw_experts(f, bg_var) gives layer 5's experts; at 0% overlap an empty
  background gives a ghost at half-means (0.5, −0.5), and a loose one (variance 1.0) reaches A∩B
  with weight 1.000 (step file 05-the-overlap-dial.md, step 6).
- ᾱ_t is 0.9994, 0.8356, 0.4783, 0.1335 and 0.0 at t = 0, 25, 50, 75 and 99 (the p1.abar_* keys in
  rebuild/out/walk-numbers.json).
- Work in float64: Mixture.torch() casts to float32, which hides a spread at float64 precision
  (rebuild/parts_world.py).
- ∇r_t = −(ε_joint − ε_A − ε_B + ε_∅)/√(1 − ᾱ_t) is what the composed score misses of the joint's
  score (layer 1's ε = −√(1 − ᾱ_t)·∇log q_t; brick row above).
Outputs:
- walk08.py: one script that prints r's mean and spread per setting and noise level, ∇r_t at the
  reference's fixed image, and the identity check, and saves the plot of r with the rule off and
  on at every noise level;
- the step file the walk writes at its open, ticked piece by piece.
Target: every p8. number walk08.py prints equals the reference's to the digits shown; the identity
check prints under 1e-10; r from the true conditionals has a spread at float64 precision on the
joint's draws at t = 0, rule off and on. I judge, from the printed lines.
Progress: piece 1 prints the world's weights and the four densities at one image; piece 2 r at
t = 0, rule off and on; piece 3 r_t per noise level; piece 4 ∇r_t and the identity check; piece 5
the plot.
Proof: python walk08.py prints "identity max difference <x>" with x under 1e-10, and step 5's
comparison shows no differing p8. number.
Limits: never edit rebuild/ or the idea maps; I type, I do not paste; no commit.
Hand-off: the retrospective the walk closes on, and the /drip-retro command it prints.
```

added: the reference's outputs and its settings as the starting point (from step 0) · claim 7.1 and where the wet rule can show (from the `poe-overlap-dial` idea map) · the parts world, layer 5's two backgrounds, the noise levels and the float32 cast (from step files 04 and 05, `walk-numbers.json` and `parts_world.py`) · the ε-space gradient (from layer 1 and the brick row)
dropped: "r is constant with the rule off and structured with it on, at every noise level" as the bar (why: by claim 7.1, r from clean labels is constant at t = 0 with the rule on as well, and r_t at intermediate noise levels varies with the rule off too; the bar now asks for the reference's numbers and claim 7.1's constancy)

⏱ rough: you 1 to 2 h (about 5 pieces) · AI ~35 min · stop: the prompt's bar

**Outcome:** r(x) in the parts world, plotted with the wet rule off and on, and r_t with its gradient at every noise level.

### 4b. You: the label term, walk08b.py

```text
/drip-execute-plan --request the label term with report-style labels on the quadrant toy: A is read
with probability 0.9 when true, and inside A∩B, B is read only when A was read; a 4-way classifier
over the label pairs against two separate classifiers gives log p(A,B|x) - log p(A|x) - log p(B|x);
I type every block myself into ~/Playground/PhD/simple-distributions-walk/walk08b.py; bar: +0.105
nats for (A,B) inside A∩B, zero elsewhere, strongly negative for (not A, B)

Task: cut the label-term toy into 4 to 6 pieces, write the step file at the open, and build, teach
and test one piece per round. I type every block into walk08b.py and run it with
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python from
~/Playground/PhD/simple-distributions-walk/.
Starts from: walk08.py from step 4a; the p8.label_term_* keys in rebuild/out/walk-numbers.json from
step 0's reference, which are the exact values.
Settled facts:
- The toy: points uniform on [−1, 1]², A true when x1 < 0 and B true when x2 > 0 (the A and B of
  rebuild/phase5_dial.py), so A∩B is the top-left quadrant
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 7.4; rebuild/phase5_dial.py docstring).
- Claim 7.4's expected label term: +0.105 nats for (A,B) inside A∩B, very negative for (¬A,B)
  there, zero elsewhere (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 7.4).
- Inside A∩B, P(A,B|x) / [P(A|x)·P(B|x)] = 0.9q / (0.9 · 0.9q) = 1/0.9 for any probability q of
  reading B once A was read, so +0.105 = −ln 0.9; (¬A,B) has probability 0 there, so its term is
  −∞ and a classifier reports it as strongly negative; outside A∩B at most one label is uncertain,
  so the term is 0 (the arithmetic of claim 7.4's labelling).
- The term is zero for clean labels and nonzero for report-mined ones, and it is a different thing
  from the wet dog of step 4a (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 7.3).
- On chest X-rays it matters most for the negation cells, pneumothorax without drain and drain
  without pneumothorax; 20% to 54% of NIH pneumothoraces have no drain
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 6 and "The idea, as it stands").
- A margin of 0.15 inside a quadrant keeps boundary points out of a region count
  (rebuild/phase5_dial.py, MARGIN).
Outputs:
- walk08b.py: one script that trains the 4-way classifier and the two binary classifiers on the
  same points and prints the label term's mean per label cell, inside A∩B and outside it, beside
  the exact p8.label_term_* values;
- the step file the walk writes at its open, ticked piece by piece.
Target: the classifier's (A,B) term inside A∩B (points at least 0.15 inside) is within 0.05 nats of
+0.105; outside A∩B (points at least 0.15 from both boundaries) it is within 0.05 of 0; (¬A,B)
inside A∩B is below −2. I judge, from the printed lines.
Progress: piece 1 prints the label counts per cell; piece 2 the exact term per cell; pieces 3 and 4
each classifier's accuracy; piece 5 the estimated term beside the exact one.
Proof: python walk08b.py prints "(A,B) inside A∩B: <x> nats, exact 0.1054" with x within 0.05 of
0.1054.
Limits: never edit rebuild/ or the idea maps; I type, I do not paste; no commit.
Hand-off: the retrospective the walk closes on, and the /drip-retro command it prints.
```

added: the toy's geometry (from claim 7.4 and `phase5_dial.py`) · the arithmetic behind +0.105 and the −∞ cell (worked from claim 7.4's labelling) · claim 7.3's split and claim 6's drain rates (from the `poe-overlap-dial` idea map) · the margin (from `phase5_dial.py`) · the reference's exact values as the answer key (from step 0) · the tolerances 0.05 nats and −2, which are estimates: change them here if a classifier on a few thousand points cannot reach them
dropped: nothing

⏱ rough: you 1 to 2 h (one build concept) · AI ~35 min · stop: the prompt's bar

**Outcome:** the interaction term in both places it lives. In the pixels it is the wet dog. In the labels it is zero for clean labels and real for report-mined ones like NIH's, and it hits the negation cells hardest.

**What it looks like at scale (look by hand).** Open `artifacts/results/joint-against-poe-by-kind-of-composition/README.md` and its `interaction-grids/`. On Stable Diffusion v1-4, the product fuses one cat-dog in 4 of 4 seeds, and it paints rain streaks over a dry dog in 4 of 4. Those are the two things the term you just computed leaves out, on a real model.

### 5. You: compare your numbers with the reference

**Compare.** `python -m rebuild.walk_numbers` prints the `p8.` keys; your `walk08.py` prints the same numbers.

```bash
cd /home-mscluster/mmolefe/PhD/reduce-reuse-recycle
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m rebuild.walk_numbers | grep 'p8\.'
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m rebuild.phase8_interaction --out ~/Playground/PhD/simple-distributions-walk/phase8-reference | grep 'identity'
cd ~/Playground/PhD/simple-distributions-walk
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python walk08.py
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python walk08b.py
```

⏱ you ~10 min (`--out` keeps the reference run from rewriting `rebuild/out/phase8/`) · stop: every `p8.` line and the number your scripts print agree to the digits shown; both identity lines read under `1e-10`; the label term for (A,B) inside A∩B reads `0.1054` from the reference and within 0.05 of it from `walk08b.py`. A number that differs is either a lesson or a bug in one of the two files; find which before the prior-art read.

### 6. AI, in the background: the scene

```text
/math-scene the folder the layer 8 --math drip wrote under artifacts/drips/: the interaction term in
the parts world, with the wet rule as a toggle and the noise level as a slider

Starts from: the layer 8 --math drip folder and its dispatch.md (step 2's hand-off); the layer 8
reference's rebuild/out/phase8/phase8.json and its p8. keys (step 0).
Settled facts:
- The parts world: 16 pixels drawn as a 4 × 4 image, a cat +1 and a dog −1 on their pixels with
  variance 0.01, each expert a mixture of diagonal Gaussians, so every noised density is exact
  (step file 04-the-parts-world.md, steps 1, 5 and 6).
- The noise levels are t = 0, 25, 50, 75 and 99, where ᾱ_t is 0.9994, 0.8356, 0.4783, 0.1335 and
  0.0 (the p1.abar_* keys in rebuild/out/walk-numbers.json).
- With clean labels r is constant at t = 0 whether the wet rule is on or off; the toggle changes r
  only where phase8.json shows it does, and the scene shows that, not the brick row's expectation
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 7.1; the docstring of
  rebuild/phase8_interaction.py).
- The wet dog at scale: on Stable Diffusion v1-4 the product paints rain streaks over a dry dog
  where the joint prompt gives wet fur in 4 of 4 seeds
  (artifacts/results/joint-against-poe-by-kind-of-composition/five-interactions.md).
- Every number on the scene comes from phase8.json, the p8. keys or the drip's dispatch.md, never
  estimated.
Outputs: one published scene, in my notation from dispatch.md.
Hand-off: write the scene's URL into step 6 of
artifacts/drips/simple-distributions-rebuild/ladder/08-the-interaction-term.md, replacing "(URL
lands here)".
```

added: the reference's outputs as the scene's numbers (from step 0) · the parts world and the noise levels (from step file 04 and `walk-numbers.json`) · that the toggle shows what the reference found (from claim 7.1) · the wet dog at scale (from the results card) · the URL write-back
dropped: nothing

Scene: (URL lands here)

⏱ you ~20 min · AI ~45 min, beside the correction prior-art read · stop: the scene is published and its numbers match your walk's

**Outcome:** a scene of the parts world's interaction term, with the wet rule as a toggle and the noise level as a slider.

### 7. You: close the layer

Run the `/drip-retro` commands the two builds printed at their close (⏱ about 25 minutes each). Then paste this:

```text
In the background, dispatch xref-cadence over the files layer 8 of the composition ladder wrote: the
layer 8 --math drip folder under artifacts/drips/, the step files the two --request walks of steps
4a and 4b wrote at their open, and the retrospectives the two layer 8 walks closed on.

Settled facts:
- The reference sets to link against are the journey plans, textbook chapters and vault note on
  the Study first line of
  artifacts/drips/simple-distributions-rebuild/ladder/08-the-interaction-term.md; the layer 8 row
  of the Bricks table in artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md ("The
  interaction term, exact") and the vault note it links, What An Expert Learns From Its Training
  Set; and claim 7 (7.1 to 7.4) with run 1 of artifacts/ideas/poe-overlap-dial/IDEA_MAP.md.
- ~/Playground/PhD/simple-distributions-walk/ is scratch space; link nothing into it (CLAUDE.md, the
  playground is disposable).
Outputs: links written both ways between those files and the reference sets.
Hand-off: the agent's report, listing every link it wrote, read before the prior-art read starts.
```

added: the reference sets, by path (from this layer's Study first line, `REBUILD_WALK.md` and the `poe-overlap-dial` idea map, claim 7)
dropped: nothing

⏱ you 2 min to paste · AI ~10 min (estimate)

---

⬅️ [Layer 7: A without B](07-a-without-b.md) · 🏠 [the guide](../PROMPTS.md) · [Correction prior art](08a-correction-prior-art.md) ➡️
