# Layer 7, step 1: the layer 7 reference

⬅️ [Layer 7, step 1](../07-a-without-b.md#1-ai-in-the-background-the-layer-7-reference)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
when you start layer 6's build, so the reference is ready for layer 7's compare step.

```text
In the background, build the layer 7 reference: rebuild/phase7_negation.py, negation on the quadrant
toy and the parts world, and the same cell built from a positive "not B" label.

Background
The composition ladder in artifacts/drips/simple-distributions-rebuild/ reaches A without B in
layer 7. The medical cells the project cares about (pneumothorax without drain) are A without B, and
at the paper's own weight, negation sends mass off every mode when the experts have sharp tails.
Claude writes the reference code in toy/rebuild/ one layer ahead, so that my own walk07.py has
numbers to match; the reference for layers 1 to 5 is in rebuild/ with outputs in
rebuild/out/phase1 to phase5, and layer 6's reference (rebuild/phase6_guidance.py) may already have
added p6. keys. Work in /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy, on
the current branch.

Objectives
1. Write rebuild/negation.py (the negation target and the positive-label target for each toy) and
   rebuild/phase7_negation.py (the dial), run from toy/ as python -m rebuild.phase7_negation with
   /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python.
2. Reproduce the kinds notebook's negation dial on the ring and the bar, exact on the grid
   (ideal.exact_grid with kinds.negation(w)), at w = 0, 0.02, 0.05, 0.1, 0.3 and 1.0, by replaying
   that dial's data draw, and expose the replayed ring and bar as a function of rebuild/negation.py
   so walk07.py can import it; the mass off the ring at w = 0.3 is expected at 0.863.
3. On layer 5's quadrant toy (phase5_dial.make_sets, data seed np.random.default_rng(100) as phase 5
   seeds its first run), compute the exact bottom-left mass of p_A^(1+w)/p_B^w against w, at the
   overlaps of phase5_dial.OVERLAPS_2D.
4. In the parts world (phase5_dial.pw_experts, both backgrounds), the cat-with-no-dog mass against w
   has no closed form, because a power of a mixture is not a mixture, so estimate it by
   self-normalised importance sampling from exact draws of p_A (Mixture.sample), with
   weights (p_A/p_B)^w, and print the standard error and the effective sample size beside it.
5. Build the positive label: the same cells as p(x|A)·p(¬B|x)^w, with p(¬B|x) read from the pooled
   data of both sets (an ideal expert on the pooled points without B over one on all pooled
   points), with its mass and normaliser per w, and print the pooled set and the label rule in the
   script's header.
6. Per toy and w, run tuned MALA, UHA and reverse diffusion on the noised target (samplers.Target),
   2,000 samples each, with the settings phase5_dial.sample_2d uses, scored by the cell's mass and
   by ideal.tv or parts_world.tv_projected beside a 2,000-draw floor.
7. Use the kinds dial's w grid for every toy, with points added where the A-without-B mass peaks,
   and name the w where it peaks per toy.
8. Add those numbers to rebuild/walk_numbers.py as phase7() with keys p7.*, add it to the __main__
   tuple, and run it once with the co3 python.

What to read
- rebuild/kinds.py, rebuild/phase3_kinds.py and rebuild/compose.py: the dial, its measures, its
  draw order and the notebook's weight.
- artifacts/ideas/poe-overlap-dial/kinds-of-composition-on-the-papers-toys.ipynb, Figures 1 and 2:
  the dial's values.
- rebuild/ideal.py, rebuild/parts_world.py and rebuild/phase5_dial.py: the grid, the mixtures and
  layer 5's toys.
- rebuild/walk_numbers.py: how worked numbers are written.
- artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, brick "A without B (negation)", and
  its Pins.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The brick is the pneumothorax target p(x|PTX)^(1+w)/p(x|drain)^w, with exact mass in the
  A-without-B cell (quadrant: bottom-left; parts world: cat with no dog) against w, per sampler, in
  rebuild/negation.py and rebuild/phase7_negation.py, reproducing first in-bar mass 0.146 at w = 0,
  0.037 at w = 0.1, and 0.863 off every ring mode at the paper's w = 0.3 (REBUILD_WALK.md, brick "A
  without B (negation)").
- The kinds dial at w = 0, 0.02, 0.05, 0.1, 0.3, 1.0 has mass on the 2 in-bar modes 0.146, 0.121,
  0.085, 0.037, 0.0, 0.0 and mass away from every ring mode 0.387, 0.384, 0.391, 0.439, 0.863, 1.0
  (artifacts/ideas/poe-overlap-dial/kinds-of-composition-on-the-papers-toys.ipynb, Figure 2;
  REF_FIG2 in rebuild/phase3_kinds.py).
- A is the ring, kinds.toy_gmm, 8 modes on a circle of radius 0.5, std 0.03, 800 points, and B is
  the bar, kinds.vbar(rng, 800, 0.0, 0.2), half-width 0.2, both ideal experts smoothed by 0.05;
  "in-bar" is the mass within 0.08 of the two ring modes inside the bar (kinds.RING[[2, 6]]), and
  "away" is the mass more than 0.08 from every ring mode, so the A-without-B mass on the six other
  modes is 1 minus the two (rebuild/kinds.py, MEASURES; rebuild/phase3_kinds.py).
- rebuild/phase3_kinds.main draws that ring and bar from np.random.default_rng(0) after
  kinds.build_kinds, one ideal.perfect_samples call of 2,000 per kind in Figure 1, and the
  co-occurrence, contention and interaction dials, in that order (rebuild/phase3_kinds.py).
- Fresh draws move a dial by up to about 0.08 where its curve is steep, and python -m
  rebuild.phase3_kinds draws the notebook's own data and matches every printed value (step file 03,
  step 12); an earlier draw put 0.891 off every mode at w = 0.3 (overlap-dial idea map, run 3).
- Negation works only near w ≈ 0.05, where the exact in-bar mass is 0.10
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, run 3; step file 03, step 11).
- At w = 0.05 on the ring and bar, measure and total variation per sampler are perfect 0.09 and
  0.15, reverse diffusion 0.20 and 0.26, ULA 0.13 and 0.25, MALA 0.11 and 0.25, UHA 0.12 and 0.21,
  tuned MALA reaches 0.21 against the floor's 0.15 at acceptance 0.60, and ULA at the tuned steps
  keeps 446 of 2,000 samples (kinds notebook, Figure 1 and run 1; overlap-dial idea map, runs 4 and
  5).
- The notebook's NegationEBMDiffusionModel uses 1.3·E1 − 0.3·E2, which is w = 0.3, and
  kinds.negation(w) is (1 + w)·a − w·b on log-densities (rebuild/compose.py; rebuild/kinds.py),
  while Du et al.'s own negation experiments use α = 0.5 in q^0/(q^1)^α (vault note Product,
  Mixture And Negation).
- Where p_B → 0, p_B^(−w) → ∞, so negation pushes mass toward wherever B is least likely, not just
  out of B (step file 02, step 4).
- For Gaussians, q_a/q_b^α is a density only while 1/v_a − α/v_b > 0 (vault notes Product, Mixture
  And Negation and Powers Of Gaussians).
- Negation's exponents sum to 1, (1 + w) − w, so it is unchanged by an invertible warp (total
  variation 7e-17), while the plain product changes by 0.035 to 0.126
  (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 10 and run 6).
- In layer 5's quadrant toy, A is the left half and B the top half, so A without B is the
  bottom-left, and at overlap f a share f of each set sits in the top-left (rebuild/phase5_dial.py,
  docstring).
- In layer 5's parts world, A is "contains a cat" and B "contains a dog", a sample is A∩B when one
  half's mean is above 0.75 and the other's below −0.75 (phase5_dial.pw_ab), and the background
  variance is 0.01 or 1.0 (rebuild/phase5_dial.py).
- p(¬B|x) is at most 1, so p(x|A)·p(¬B|x)^w never exceeds p(x|A) and its normaliser is at most 1
  (Bayes' rule).
- walk_numbers.py holds worked-example numbers, each written by put(key, value), while run results
  (masses and total variations per sampler) are read from rebuild/out/phaseN/ instead; its __main__
  runs (phase1, phase2, phase3, phase5) and writes rebuild/out/walk-numbers.json by a relative path,
  so it runs from toy/ (rebuild/walk_numbers.py).
- The tolerances are a dial value to the 3 decimals printed, as rebuild/phase3_kinds.py checks
  Figure 2, and a sampler number within 0.03 (rebuild/phase3_kinds.py).
- rebuild.device.pick_device takes the first card with under 1 GB in use, else the CPU
  (rebuild/device.py; REBUILD_WALK.md, Pins).

Rules
- Add the two files and phase7() only.
- Leave ideal.py, parts_world.py, kinds.py, compose.py, the phase 1 to 6 scripts and phase6() as
  they are.
- Write nothing in the ladder, the step files or REBUILD_WALK.md.
- No commit.
- The dial and its matches line print first; then the quadrant toy's rows, the parts world's, the
  positive label's, and the samplers' per toy.

What to hand back
- rebuild/negation.py and rebuild/phase7_negation.py; the phase script first prints the dial beside
  the notebook's twelve values, marked matches or differs, then one row per toy and w.
- rebuild/out/phase7/phase7.json with every row, and a figure of the exact target per w on the ring
  and bar and on the quadrant toy.
- phase7() in rebuild/walk_numbers.py, added to the __main__ tuple: the w grid as p7.w_grid, the
  dial as p7.kinds_inbar_w<w> and p7.kinds_away_w<w>, the exact or estimated A-without-B mass and
  the positive label's mass and normaliser per toy and w as p7.<toy>_<measure>_w<w>, and the peak
  per toy as p7.<toy>_peak_w.
- rebuild/out/walk-numbers.json rewritten, its 84 earlier keys and any p6. keys unchanged.
- A report naming the w grid, the replay function's name, every p7. key with its value, and the
  brick row's new mark; steps 6 and 7 of
  artifacts/drips/simple-distributions-rebuild/ladder/07-a-without-b.md read the keys and
  phase7.json.
- The proof: python -m rebuild.phase7_negation prints "negation dial: matches"; then
  python -m rebuild.walk_numbers | grep 'p7\.' lists every p7. key; the evidence is
  rebuild/out/phase7/phase7.json.

What you will see
Along the way: the dial beside the notebook's twelve values with its matches line, then the
quadrant toy's rows, the parts world's with standard errors, the positive label's, and the
samplers'.
At the end, a closing report: what was asked, in one sentence; what was done, with each file's
path; PASS or MISS against all twelve dial values matching to 3 decimals (0.863 at w = 0.3 among
them), the peak w named per toy with the grid points on either side, the positive label's
normaliser at most 1 at every w, every sampler row carrying its total variation beside the floor,
and every estimated parts-world number marked as an estimate with its standard error; what is still
open; at most one question.

I judge, from the printed rows: done when all twelve dial values match to 3 decimals, each toy's peak w is named, the positive label's normaliser never exceeds 1, and every estimate carries its standard error.
```

added: the seven sections, from the field-labelled version · the layer's reason as Background (from the ladder file's "Why it is critical" line)
dropped: nothing
