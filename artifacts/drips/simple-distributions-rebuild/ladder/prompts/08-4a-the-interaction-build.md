# Layer 8, step 4a: the build, walk08.py

⬅️ [Layer 8, step 4a](../08-the-interaction-term.md#4a-you-the-build-walk08py)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
after quiz 8.2.

```text
/drip-execute-plan --request the interaction term in the parts world with a wet rule (the cat half
shifts when a dog is present), computed exactly from the Gaussian mixtures and plotted, then r_t and
its gradient at each noise level, as layer 8 of
artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md; I type every block myself into
~/Playground/PhD/simple-distributions-walk/walk08.py

Background
This is layer 8's first build in the composition ladder (artifacts/drips/simple-distributions-rebuild/):
I type the interaction term from an empty file, one piece per round. The interaction term is the
first half of the research goal, and it is defined exactly here before any learned version of it is
trusted. Claude's reference is rebuild/interaction.py and rebuild/phase8_interaction.py, built in
step 0; my file and the reference print the same numbers, and a difference is a lesson or a bug. I
have passed quiz 8.2, and the layer 8 maths drip folder under artifacts/drips/ holds my notation in
its dispatch.md.

Objectives
1. At the walk's open, cut the layer into 4 to 6 pieces and write them as the step file, then
   build, teach and test one piece per round.
2. I type every block into walk08.py and run it with
   /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python from
   ~/Playground/PhD/simple-distributions-walk/, with
   PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy.
3. Use the reference's settings from the p8. keys: p8.wet_cat_value, the world's weights and the
   background variance.
4. Piece by piece: the world's weights and the four densities at one image; r at t = 0, rule off and
   on; r_t per noise level; ∇r_t and the identity check; the plot of r with the rule off and on at
   every noise level.
5. Show r from clean labels constant on the joint's draws at t = 0 with the rule off and on, and the
   rule's effect where claim 7.1 puts it.
6. Tick each piece in the step file as it lands.

What to read
- rebuild/interaction.py, rebuild/phase8_interaction.py and its docstring, rebuild/out/phase8/phase8.json
  and the p8. keys in rebuild/out/walk-numbers.json (step 0): the reference and where r moved.
- The layer 8 maths drip folder under artifacts/drips/ and its dispatch.md (step 2): my notation.
- artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, brick "The interaction term,
  exact".
- artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 7.1.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The brick row expects r constant with the rule off and varying with it on, and the noised r_t and
  its gradient in ε-space as ε_joint − ε_A − ε_B + ε_∅ (REBUILD_WALK.md, brick "The interaction
  term, exact").
- When each label is a fixed function of the image, r is exactly constant, so the wet rule alone
  does not move r at t = 0, and it shows in r_t at t > 0, off the joint's support, and for experts
  whose training sets lack the wet A∩B images (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim
  7.1; step 0's reference, the docstring of rebuild/phase8_interaction.py).
- The parts world is rebuild/parts_world.py: 16 pixels, HEAD 0 to 7 and BODY 8 to 15, LEFT and RIGHT
  halves, a cat +1 and a dog −1 on their pixels with variance 0.01; Mixture.noised(abar) gives
  N(√ᾱ m, ᾱ v + 1 − ᾱ) per component, and a product of mixtures is a mixture over every pair (step
  file 04-the-parts-world.md, steps 1, 5 and 6).
- rebuild/phase5_dial.py's pw_experts(f, bg_var) gives layer 5's experts, and at 0% overlap an empty
  background gives a ghost at half-means (0.5, −0.5) while a loose one (variance 1.0) reaches A∩B
  with weight 1.000 (step file 05-the-overlap-dial.md, step 6).
- ᾱ_t is 0.9994, 0.8356, 0.4783, 0.1335 and 0.0 at t = 0, 25, 50, 75 and 99 (the p1.abar_* keys in
  rebuild/out/walk-numbers.json).
- Work in float64, because Mixture.torch() casts to float32, which hides a spread at float64
  precision (rebuild/parts_world.py).
- ∇r_t = −(ε_joint − ε_A − ε_B + ε_∅)/√(1 − ᾱ_t) is what the composed score misses of the joint's
  score (layer 1's ε = −√(1 − ᾱ_t)·∇log q_t; the brick row).

Rules
- I type; I do not paste.
- Never edit rebuild/ or the idea maps.
- No commit.
- Piece 1 prints the world's weights and the four densities at one image; piece 2 r at t = 0, rule
  off and on; piece 3 r_t per noise level; piece 4 ∇r_t and the identity check; piece 5 the plot;
  show each before moving on.

What to hand back
- walk08.py: one script that prints r's mean and spread per setting and noise level, ∇r_t at the
  reference's fixed image, and the identity check, and saves the plot of r with the rule off and on
  at every noise level.
- The step file the walk writes at its open, ticked piece by piece.
- The retrospective the walk closes on, and the /drip-retro command it prints.
- The proof: python walk08.py prints "identity max difference <x>" with x under 1e-10, and step 5's
  comparison shows no differing p8. number.

What you will see
Along the way: one piece per round, each ending on the number it printed and a yes to move on.
At the end, a closing report: what was asked, in one sentence; what was done, with walk08.py's path,
the plot's and the step file's; PASS or MISS against every p8. number walk08.py prints equal to the
reference's to the digits shown, the identity check under 1e-10, and r from the true conditionals
showing a spread at float64 precision on the joint's draws at t = 0, rule off and on; where the wet
rule's effect showed; what is still open; at most one question.

I judge, from the printed lines: done when every p8. number matches the reference, the identity check reads under 1e-10, and r is constant to float64 precision on the joint's draws at t = 0 with the rule off and on.
```

added: the seven sections, from the field-labelled version · the reference's docstring paragraph on where r moved as reading (from step 0's hand-off)
dropped: nothing (the earlier version had already replaced the bar "r is constant with the rule off and structured with it on, at every noise level", because by claim 7.1 r from clean labels is constant at t = 0 with the rule on as well, and r_t at intermediate noise levels varies with the rule off too)
