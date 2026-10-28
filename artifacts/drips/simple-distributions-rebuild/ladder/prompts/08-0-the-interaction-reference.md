# Layer 8, step 0: the layer 8 reference

⬅️ [Layer 8, step 0](../08-the-interaction-term.md#0-ai-in-the-background-the-layer-8-reference-and-quizzes-81-and-82-one-layer-ahead)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
when you start layer 7's build, beside step 0's quiz prompt.

```text
In the background, build the layer 8 reference: rebuild/phase8_interaction.py, the interaction term
in the parts world with a wet rule (the cat half shifts when a dog is present), computed exactly
from the Gaussian mixtures, and r_t with its gradient at each noise level.

Background
The composition ladder in artifacts/drips/simple-distributions-rebuild/ reaches the first half of
the research goal in layer 8: the interaction term, defined exactly here before any learned version
of it is trusted. Claude writes the reference code in toy/rebuild/ one layer ahead, so that my own
walk08.py and walk08b.py have numbers to match. It starts from rebuild/ as layers 1 to 5 left it,
and from nothing of layer 8's walk. Work in
/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy.

Objectives
1. Write rebuild/interaction.py: r_t and ∇r_t, in float64, for any four Mixture objects (joint, A,
   B, unconditional) at a noise level t, in any dimension, since layer 9 applies it to a 2D toy.
2. Write rebuild/phase8_interaction.py, run as python -m rebuild.phase8_interaction --out
   rebuild/out/phase8, which builds one world of image kinds (cat-only, dog-only, cat-and-dog) with
   prior weights, its true conditionals derived by "contains" labels, the wet rule off and on, and
   the experts that never saw a cat-and-dog image from pw_experts(0.0, 1.0).
3. Print an identity check first: the composed score plus ∇r_t against the joint's exact score, and
   ∇r_t against the four exact scores combined.
4. Print r's mean and spread over 2,000 joint draws and 2,000 p(x) draws at t = −1, 0, 25, 50, 75
   and 99, with the rule off and on, and ∇r_t at one fixed image.
5. Compute and report all three places the wet rule can show (r_t at t > 0, off the joint's support,
   and for experts whose training sets lack the wet A∩B images), without tuning the world to force a
   difference at t = 0.
6. Print the exact label term of claim 7.4 in each label cell.
7. Add phase8() to rebuild/walk_numbers.py's __main__ loop with keys p8.*, and run everything from
   toy/ with /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python.
8. Open phase8_interaction.py's docstring with one paragraph saying where r moved with the wet rule
   and where it did not.

What to read
- rebuild/parts_world.py and rebuild/phase5_dial.py: the parts world's mixtures and layer 5's
  experts.
- rebuild/walk_numbers.py: how worked numbers are written.
- artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, brick "The interaction term, exact",
  and its Pins.
- artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 7 (7.1 to 7.4) and run 1.
- artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claims 3 and 9.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The brick row says: in the parts world with a "wet" rule (the cat half shifts when a dog is
  present), r(x) = log p(x|A,B) − [log p(x|A) + log p(x|B) − log p(x)] in closed form, expected
  constant with the rule off and varying with it on, then the noised r_t and its gradient, written
  also in ε-space as ε_joint − ε_A − ε_B + ε_∅, the residual poe_repair_min already caches for 25
  pairs at all 50 steps, in files rebuild/interaction.py and rebuild/phase8_interaction.py, not yet
  built (artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, brick "The interaction term,
  exact").
- When each label is a fixed function of the image, the interaction term is exactly zero, so
  interaction is no third operator but a same-variable product whose A∩B appearance must be in the
  training data, and the wet-dog shift was found at 20% overlap and missed at 0% (mean x1 −0.79
  against −0.48) (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 7.1 and run 1).
- By Bayes, r(x) = log p(A,B|x) − log p(A|x) − log p(B|x) + log[p(A)p(B)/p(A,B)], so with clean
  labels r is one constant wherever the joint has mass, wet rule or not, and the brick row's
  "varying with it on" can then only appear in r_t at t > 0, off the joint's support, or for experts
  whose training sets lack the wet A∩B images (claim 7.1;
  artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claims 3 and 9).
- "Interaction" splits in two: appearance interaction (a data condition, no term) and label
  dependence given the image (a real term, zero for clean labels, nonzero for report-mined labels)
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 7.3).
- No paper adds an interaction term to a product, and the clean-label argument is Bradley et al.
  2025a's Bayes composition, p.4 (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 7, route 1).
- In rebuild/parts_world.py a sample is 16 pixels, HEAD 0 to 7, BODY 8 to 15, LEFT and RIGHT the
  column halves; Mixture(w, m, v) normalises its weights, noised(abar) gives N(√ᾱ m, ᾱ v + 1 − ᾱ)
  per component, responsibilities(x) is float64 NumPy, and torch() casts to float32, so the
  closed-form checks stay out of torch() and logp_torch(); placed(value, half, var_in=0.01,
  var_out=1.0, bg=0.0) builds one placement, and mixture_product(A, B) builds every pair
  (rebuild/parts_world.py).
- In rebuild/phase5_dial.py, pw_experts(f, bg_var) builds layer 5's experts, cat-only on either half
  at weight (1 − f)/2 each and cat-and-dog at f/2 each (B likewise with the dog at −1), object
  variance 0.01, background mean 0 with variance 0.01 or 1.0, and pw_ab(x) calls a sample A∩B when
  one half's mean is above 0.75 and the other's below −0.75 (rebuild/phase5_dial.py).
- In layer 5's parts-world dial, with an empty background the product at 0% overlap is a ghost with
  half-means (0.5, −0.5) and A∩B weight 0.000, the matching pair wins by 386.2 nats from 0.1%, and
  with a loose background (variance 1.0) the A∩B weight is 1.000 at every overlap including 0% (step
  file 05-the-overlap-dial.md, step 6; p5.matching_pair_advantage_nats in
  rebuild/out/walk-numbers.json).
- The parts world is exact: rebuild/phase4_parts_world.py matches example 9's numbers.json to a
  largest relative difference of 2.87e-16 (step file 04-the-parts-world.md, step 8).
- Noise levels come from the cosine schedule, rebuild.ideal.ABAR over T = 100, t = −1 means clean
  in parts_world.composed_target, and ᾱ_t is 0.9994, 0.8356, 0.4783, 0.1335 and 0.0 at t = 0, 25,
  50, 75 and 99 (the p1.abar_* keys in rebuild/out/walk-numbers.json).
- ε = −√(1 − ᾱ_t)·∇log q_t (layer 1), so ∇r_t = −(ε_joint − ε_A − ε_B + ε_∅)/√(1 − ᾱ_t), and ∇r_t is
  exactly what the composed score s_A + s_B − s_∅ misses of the joint's score (the brick row;
  ladder/01-one-diffusion-expert.md, step 2).
- SDXL's cache under /datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache/ holds ε_A,
  ε_B, ε_joint and ε_∅ at all 50 steps for 25 pairs, with the residual gs*(eps_j − eps_a − eps_b +
  eps_uncond) at guidance 7.5, and its lower bound on the interaction did not track which pairs
  blend (artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md, claim 1 and Sources).
- Claim 7.4's label term uses points uniform on [−1, 1]², A true when x1 < 0 and B true when x2 > 0
  (the A and B of rebuild/phase5_dial.py), A read with probability 0.9 when true, and inside A∩B, B
  read only when A was read, so the term log p(A,B|x) − log p(A|x) − log p(B|x) is +0.105 nats for
  (A,B) inside A∩B, strongly negative for (¬A,B) there, and zero elsewhere
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 7.4).
- +0.105 is −ln 0.9, whatever the probability of reading B once A was read (the arithmetic of claim
  7.4's labelling).
- rebuild/walk_numbers.py prints through put(key, value), its __main__ loop runs phase1, phase2,
  phase3 and phase5 and writes rebuild/out/walk-numbers.json, and it holds worked numbers while run
  results are read from rebuild/out/phase*/ (walk_numbers.py docstring).
- Every phase is one command from toy/, python -m rebuild.<script>, with outputs in
  rebuild/out/phaseN, and the device is picked with rebuild.device.pick_device, the GPU guard
  (REBUILD_WALK.md, Pins).

Rules
- Change no existing function in rebuild/ and no p1. to p5. value.
- Write nothing under artifacts/.
- No commit.
- Never tune the world to force a difference in r at t = 0; if r from the true conditionals shows a
  spread on the joint's draws at t = −1 or t = 0 above float64 precision, stop and report it against
  claim 7.1.
- The identity check prints first, then one line per setting and level, then the label cells, then
  the p8. keys.

What to hand back
- rebuild/interaction.py and rebuild/phase8_interaction.py.
- rebuild/out/phase8/phase8.json with every chosen setting beside each number, and
  rebuild/out/phase8/interaction-term.png: r's spread on the joint's draws (y) against the noise
  level (x), one line per setting, rule off and on.
- phase8() in rebuild/walk_numbers.py, added to its __main__ loop, putting p8. keys: the chosen
  settings (p8.wet_cat_value, the world's weights, the background variance), r and ∇r_t at the fixed
  image per setting and level, the ε-space residual there, and p8.label_term_* per cell.
- rebuild/out/walk-numbers.json rewritten with the p8. keys added.
- At the top of phase8_interaction.py's docstring, one paragraph saying where r moved with the wet
  rule and where it did not.
- phase8.json and the p8. keys are read by step 4a's walk, step 5's comparison, step 6's scene and
  layer 9's reference.
- The proof: python -m rebuild.phase8_interaction --out rebuild/out/phase8 prints "identity max
  difference <x>" with x under 1e-10, python -m rebuild.walk_numbers | grep 'p8\.' prints the keys,
  and git diff rebuild/out/walk-numbers.json shows only added p8. lines.

What you will see
Along the way: the identity check first, then one line per setting and level, then the label cells.
At the end, a closing report: what was asked, in one sentence; what was done, with each file's
path; PASS or MISS against both identities within 1e-10 at every printed level, r from the true
conditionals showing a spread at float64 precision on the joint's draws at t = −1 and t = 0 with the
rule off and on, the label term for (A,B) inside A∩B printing 0.1054, and every p1., p2., p3. and
p5. value unchanged; where r moved with the wet rule and where it did not; what is still open; at
most one question.

I judge, from the printed lines: done when both identities hold within 1e-10, r stays constant to float64 precision at t = −1 and 0 or the run has stopped and reported it, the (A,B) label term prints 0.1054, and no earlier key changed.
```

added: the seven sections, from the field-labelled version · the layer's reason as Background (from the ladder file's "Why it is critical" line) · the three places the wet rule can show as an objective, and the stop-and-report as a rule (both from the earlier Settled facts and Target)
dropped: nothing (the earlier version had already dropped "rule off (constant)" as a printed expectation, since claim 7.1 says r from clean labels is constant with the rule on too, so the run prints the spread and reports which holds)
