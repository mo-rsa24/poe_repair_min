# Layer 7, step 6: the build

⬅️ [Layer 7, step 6](../07-a-without-b.md#6-you-the-build)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
after quiz 7.2.

```text
/drip-execute-plan --request negation on the quadrant toy and the parts world, and the same cell
built from a positive "not B" label, as layer 7 of
artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md; I type every block myself into
~/Playground/PhD/simple-distributions-walk/walk07.py

Background
This is layer 7's build in the composition ladder (artifacts/drips/simple-distributions-rebuild/): I
type negation from an empty file, one piece per round. The medical cells are A without B, and at
the paper's own weight negation sends mass off every mode when the experts have sharp tails, so this
layer measures the escaping mass and builds the same cell from a positive "not B" label beside it.
Claude's reference is rebuild/negation.py and rebuild/phase7_negation.py, built in step 1; my file
and the reference print the same numbers, and a difference is a lesson or a bug. I have passed quiz
7.2, and the layer 7 maths drip folder under artifacts/drips/ holds my notation in its dispatch.md.

Objectives
1. At the walk's open, cut the layer into 4 to 6 pieces and write them as the step file
   artifacts/drips/simple-distributions-rebuild/07-a-without-b.md, in the shape of the step files
   for layers 1 to 5.
2. Build, teach and test one piece per round, in this order: the dial, the quadrant toy's
   A-without-B mass per w, the parts world, the positive label, then the samplers.
3. I type every block into walk07.py and run it from ~/Playground/PhD/simple-distributions-walk/
   with /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python and
   PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy.
4. Reproduce 0.863 of the mass off the ring at w = 0.3 first, importing the reference's replay
   function and using its w grid.
5. Find the w where A-without-B peaks on each toy.
6. Show the positive label staying bounded.
7. Tick each piece in the step file as it lands, and keep my place in the walk map between sittings.

What to read
- The layer 7 maths drip folder under artifacts/drips/ and its dispatch.md (step 4): my notation.
- rebuild/negation.py, rebuild/phase7_negation.py, the p7. keys and rebuild/out/phase7/phase7.json
  (step 1): the reference.
- artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, brick "A without B (negation)".
- artifacts/drips/simple-distributions-rebuild/05-the-overlap-dial.md: the step-file shape and "How
  to walk it".

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The w grid, the p7. keys, phase7.json and the function in rebuild/negation.py that returns the
  replayed ring and bar are step 1's, and walk07.py imports that function and uses the same grid
  (from step 1).
- The brick is the pneumothorax target p(x|PTX)^(1+w)/p(x|drain)^w, with exact mass in the
  A-without-B cell (quadrant: bottom-left; parts world: cat with no dog) against w, per sampler,
  reproducing first in-bar mass 0.146 at w = 0, 0.037 at w = 0.1, and 0.863 off every ring mode at
  w = 0.3 (REBUILD_WALK.md, brick "A without B (negation)").
- The whole dial at w = 0, 0.02, 0.05, 0.1, 0.3, 1.0 is in-bar 0.146, 0.121, 0.085, 0.037, 0.0, 0.0
  and away from every ring mode 0.387, 0.384, 0.391, 0.439, 0.863, 1.0 (kinds notebook, Figure 2).
- "In-bar" is the mass within 0.08 of the two ring modes inside the bar, and "away" the mass more
  than 0.08 from every ring mode, so the A-without-B mass on the six other modes is 1 minus the two
  (rebuild/kinds.py, MEASURES).
- Fresh draws move a dial by up to about 0.08 where its curve is steep, so the twelve values
  reproduce only with the replayed draw (step file 03, step 12), and an earlier draw put 0.891 off
  every mode at w = 0.3 (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, run 3).
- Negation works only near w ≈ 0.05, where the exact in-bar mass is 0.10 (overlap-dial idea map, run
  3; step file 03, step 11).
- At w = 0.05, reverse diffusion doubles the in-bar mass (0.20 against 0.10), and tuned MALA reaches
  total variation 0.21 against the floor's 0.15 (kinds notebook, Figure 1 and run 1).
- The notebook's NegationEBMDiffusionModel is w = 0.3, and kinds.negation(w) is (1 + w)·a − w·b on
  log-densities (rebuild/compose.py; rebuild/kinds.py).
- Where p_B → 0, p_B^(−w) → ∞, so negation pushes mass toward wherever B is least likely (step file
  02, step 4), and for Gaussians q_a/q_b^α is a density only while 1/v_a − α/v_b > 0 (vault note
  Product, Mixture And Negation).
- p(¬B|x) is at most 1, so p(x|A)·p(¬B|x)^w never exceeds p(x|A) and its normaliser is at most 1
  (Bayes' rule).
- In layer 5's quadrant toy A is the left half, B the top half and A without B the bottom-left, and
  in layer 5's parts world A is "contains a cat", B "contains a dog", with background variance 0.01
  or 1.0 (rebuild/phase5_dial.py).
- A power of a mixture is not a mixture, so the parts world's numbers are importance-sampling
  estimates with a standard error (rebuild/parts_world.py; from step 1).
- Negation's exponents sum to 1, so it is unchanged by an invertible warp (total variation 7e-17)
  (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 10 and run 6).
- The medical cells are a negation, 1,906 of 3,543 NEATX pneumothorax films have no drain, and a
  drain label on negatives is missing (when-poe-reaches-a-and-b idea map, claims 6 and 8).
- The tolerances are a dial value to the 3 decimals printed and a sampler number within 0.03
  (rebuild/phase3_kinds.py).
- Phases 1 to 5 are imported from the reference, as step file 05 does (step file 05, "How to walk
  it").

Rules
- I type; I do not paste.
- Never edit rebuild/ or the ladder files.
- No commit.
- Each piece prints its number as it lands: the dial first, then the quadrant toy's A-without-B mass
  per w, then the parts world, then the positive label, then the samplers; show each before moving
  on.

What to hand back
- The step file artifacts/drips/simple-distributions-rebuild/07-a-without-b.md, its pieces ticked,
  and the walk map that holds my place between sittings.
- walk07.py: one script that prints the dial, the A-without-B mass per toy and w with its peak, the
  positive label's mass and normaliser per w, and per sampler the cell's mass and total variation
  beside the floor.
- On the whiteboard, N(0, 1) divided by N(0, 0.5)^w with its tails for two values of w (this
  layer's Achieve line).
- The step file's path, the retrospective the walk closes on, and the /drip-retro command it
  prints.
- The proof: python walk07.py prints the dial with 0.146 at w = 0, 0.037 at w = 0.1 and 0.863 away
  at w = 0.3, and step 7's comparison shows no differing number.

What you will see
Along the way: one piece per round, each ending on the number it printed and a yes to move on.
At the end, a closing report: what was asked, in one sentence; what was done, with walk07.py's path
and the step file's; PASS or MISS against the twelve dial values matching to 3 decimals (0.863 at
w = 0.3 among them), the peak w per toy equal to its p7. key, the positive label's normaliser at
most 1 at every w, every other exact number equal to its p7. key to the digits shown, and every
sampler number within 0.03 of phase7.json; what is still open; at most one question.

I judge, from the printed lines: done when the twelve dial values match to 3 decimals, each peak w equals its p7. key, the positive label's normaliser never exceeds 1, and every sampler number is within 0.03 of phase7.json.
```

added: the seven sections, from the field-labelled version · the step file's place-keeping as an objective (from the layer 1 build prompt's shape)
dropped: nothing
