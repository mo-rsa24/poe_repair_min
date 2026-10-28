# Layer 3, step 4a: the build, walk03.py

⬅️ [Layer 3, step 4a](../03-ideal-experts-and-the-seven-kinds.md#4a-you-the-build-walk03py)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`.

```text
/drip-execute-plan
artifacts/drips/simple-distributions-rebuild/03-ideal-experts-and-the-seven-kinds.md --depth
objective, I type every block myself into ~/Playground/PhD/simple-distributions-walk/walk03.py

Background
This is layer 3's build in the composition ladder (artifacts/drips/simple-distributions-rebuild/):
I type my own ideal experts and four kinds of composition from the step file, each shown succeeding
and failing, one step per message. Claude's reference code is under toy/ in this repo; my file and
the reference print the same numbers, and a difference is a lesson or a bug. I start from the
setup's folder and environment and walk02.py from layer 2, with quiz 3.2 passed, and the layer 3
maths drip folder under artifacts/drips/ holds my notation in its dispatch.md.

Objectives
1. Walk the step file's steps 0 to 13, one per message, in its order, teaching all but steps 6
   (disjunction) and 9 (dominance).
2. I type every block into ~/Playground/PhD/simple-distributions-walk/walk03.py and run it from that
   folder with /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python and
   PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy.
3. Type steps 6 and 9's show_kind lines even though their teaching is skipped.
4. Save step11-seven-kinds-exact.png, the seven exact compositions in a row, in the typing folder.
5. Tick each step in the step file as it lands, and keep my place in the walk map between sittings.

What to read
- artifacts/drips/simple-distributions-rebuild/03-ideal-experts-and-the-seven-kinds.md: the step
  file.
- toy/rebuild/ideal.py, toy/rebuild/kinds.py and toy/rebuild/phase3_kinds.py: the reference.
- artifacts/ideas/poe-overlap-dial/kinds-of-composition-on-the-papers-toys.ipynb: the notebook's
  printed numbers.
- toy/rebuild/out/phase3/phase3.json: the rebuilt run beside them.
- The layer 3 maths drip's dispatch.md: my notation.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The reference is rebuild/ideal.py, rebuild/kinds.py and rebuild/phase3_kinds.py (step file 03,
  opening).
- Every worked number prints from python -m rebuild.walk_numbers, run from toy/, as 25 p3. keys
  (step file 03, opening).
- The kinds notebook's printed numbers are in
  artifacts/ideas/poe-overlap-dial/kinds-of-composition-on-the-papers-toys.ipynb, and the rebuilt
  run beside them is in toy/rebuild/out/phase3/phase3.json (step file 03, opening).
- rebuild/phase3_kinds.py compares the reference with the notebook's printed numbers and never reads
  walk03.py: exact numbers to the notebook's printed precision, sampler numbers within 0.03
  (toy/rebuild/phase3_kinds.py, docstring; REBUILD_WALK.md, brick "Ideal experts and the seven
  kinds").
- rng = np.random.default_rng(0) draws the training data in exactly the kinds notebook's order, and
  everything else random uses rng2, so typed in order the data equal the notebook's (step file 03,
  step 0).
- Steps 6 and 9 each draw training data from rng, so their show_kind lines are typed even though
  their teaching is skipped, and leaving them out shifts every later kind's data off the notebook's
  (step file 03, steps 0, 4, 6 and 9).
- σ = 0.05 and N = 800; the grid is 241 × 241 over [−1.1, 1.1]², cell area 8.403e-05, 58081 points
  (step file 03, steps 0 and 3; p3.grid_cell_area, p3.grid_points).
- Step 1's instance gives -2.6288 by formula and by scikit-learn (p3. keys).
- Step 2's v_t is 1.0, 0.5229 and 0.00313 at t = 99, 50 and 0 (p3. keys).
- The perfect-sample baseline (printed as floor TV) is the total variation, on 40 × 40 bins, of n
  perfect draws from the grid density, each jittered inside its cell (step file 03, steps 4 and 5).
- For 2,000 points the perfect-sample baseline is about 0.10 (0.11 in the notebook's draw, 0.10 in
  the step file's run) (step file 03, steps 4 and 5).
- The perfect-sample baseline is random, with a standard deviation of 0.0062 to 0.0095 over 200
  redraws (p3.perfect_tv_200_redraws_sd_range), so a sampler can score below it by chance (step
  file 03, steps 4 and 5).
- The exact measures are intersection 0.71 (mass within 0.08 of a ring mode, ring × a bar of
  half-width 0.05), co-occurrence 0.81 (mass in the crossing), contention 0.67 (mass within 0.12 of
  the two apart positions, half of each set rotated), interaction −0.41 (mean x1 in the band at
  f = 0.3) and negation 0.10 at w = 0.05 (step file 03, steps 5, 7, 8, 10 and 11).
- Contention falls from 0.657 at 5% rotated to 0.002 at 0%, the chimera column of step 8 (step file
  03, step 12).
- Negation at the paper's w = 0.3 leaves 0.863 of the mass away from every ring mode (step file 03,
  step 12).
- The dials draw fresh data and move up to about 0.08 from the notebook's curves where a curve is
  steep (step file 03, step 12).
- Step 13 on contention printed, against exact 0.67: UHA 0.62 with TV 0.14, tuned MALA 0.58 with TV
  0.18, reverse diffusion 0.60 with TV 0.24, and 1000 perfect draws at TV 0.13 (step file 03, step
  13).
- Dominance defeats every sampler (TV 0.29 to 0.58, against a perfect-draw TV of 0.11 in that run)
  because no local sampler moves mass between separated modes to match unequal weights, so it is
  skipped here (step file 03, step 13; artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 4 and
  run 4).

Rules
- I type; nothing is pasted into walk03.py.
- Never edit toy/rebuild/ or the step file's code blocks.
- No commit.
- Show each printed number before moving on: step 1 prints -2.6288 twice; step 2 the three v_t
  values; step 3 the cell area and grid points; steps 5, 7, 8 and 10 one "exact measure, floor TV"
  line each; step 12 the dial curves; step 13 one line per sampler on contention.

What to hand back
- walk03.py, one script that runs end to end and prints each step's worked number and each kind's
  "exact measure, floor TV" line.
- step11-seven-kinds-exact.png in the typing folder.
- The step file's steps ticked, and the walk map that holds my place between sittings.
- The retrospective the walk closes on, and the /drip-retro command it prints.
- The proof: python walk03.py prints "intersection: exact measure 0.71", "co-occurrence: exact
  measure 0.81", "contention: exact measure 0.67" and "interaction: exact measure -0.41", and the
  comparison in step 5 of layer 3 shows no differing number.

What you will see
Along the way: one step per message, each ending on the number it printed and a yes to move on.
At the end, a closing report: what was asked, in one sentence; what was done, with walk03.py's path
and the figure's; PASS or MISS against every p3. number equal to walk_numbers to the digits shown,
exact measures of 0.71, 0.81, 0.67 and −0.41 for intersection, co-occurrence, contention and
interaction (the notebook's and phase3_kinds' values to two decimals), and the contention dial at
0.002 at 0% rotated; what is still open; at most one question.

I judge, from the printed lines: done when every p3. number matches, the four exact measures read 0.71, 0.81, 0.67 and −0.41, and the contention dial prints 0.002 at 0% rotated.
```

added: the seven sections, from the field-labelled version · typing the skipped steps' show_kind lines as an objective (from the draw-order fact it rests on)
dropped: nothing (the earlier version had already dropped "phase3_kinds prints ok against my code" as the pass line, since `phase3_kinds` compares the reference with the notebook and never reads `walk03.py`; the four exact numbers, printed by my script, are the target instead)
