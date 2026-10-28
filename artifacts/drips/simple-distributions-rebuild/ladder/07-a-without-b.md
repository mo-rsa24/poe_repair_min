# ⭐ Layer 7: A without B

⬅️ [Layer 6: guidance](06-guidance-as-a-dial.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 8: the interaction term](08-the-interaction-term.md) ➡️

**Why it is critical.** The medical cells are A without B. At the paper's own weight, negation sends mass off every mode when the experts have sharp tails.

**Depth.** Build: a quotient of Gaussians and when it stays a density, and the positive label's bound. You 2.3 to 4.7 h in all (rough).

**Study first.** [Products And Quotients Of Diagonal Gaussians](../../../../../../Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Products%20And%20Quotients%20Of%20Diagonal%20Gaussians.md) (a quotient is a density only while its precision stays positive) and [Product, Mixture And Negation](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/Product%2C%20Mixture%20And%20Negation.md) (negation's score, and when it stays a density). Textbook: Anton et al., *Calculus* 7.8 (improper integrals, p571).

**Ideas to tie in.** From [the ideas that tie the ladder together](../../../ideas/ideas-that-tie-the-ladder-together.md): [guidance as tempering, and negation tested for normalisability](../../../ideas/ideas-that-tie-the-ladder-together.md#4-treat-guidance-as-tempering-and-test-negation-for-normalisability) (needed, 2 to 4 h, no training); [whether the chest target is observational or counterfactual](../../../ideas/ideas-that-tie-the-ladder-together.md#8-specify-whether-the-chest-target-is-observational-or-counterfactual) (needed before the chest experiment, 2 to 4 h). Each comes with an exact toy check.

**Achieve.** `walk07.py` showing the escaping mass, the w where A-without-B peaks, and the bounded positive label · on the whiteboard: N(0, 1) divided by N(0, 0.5)^w, with its tails for two values of w · problems: the ones quiz 7.2 banks · quizzes 7.1 and 7.2 passed.

## The flow

```text
 AI, in the background                         │  You, in order
───────────────────────────────────────────────┼───────────────────────────────────────────────
 2. quizzes 7.1 and 7.2, prepared ───────────▶ │  3. quiz 7.1: quotients, convergence
   (paste during layer 6's build)              │       ▼
                                               │  4. /drip --math: when A without B is a density
                                               │       ▼
                                               │  5. quiz 7.2: needed-now right, 80% of the rest
                                               │       ▼
                                               │  6. build: walk07.py (--request)
 1. builds rebuild/phase7_negation.py ───────▶ │  7. compare: p7. numbers equal yours
                                               │       ▼
 math-scene, beside layer 8 ◀───────────────── │  8. scene
                                               │       ▼
 xref-cadence links the layer ◀─────────────── │  9. /drip-retro, then the next layer
```

## The steps

**AI, one layer ahead.** Paste steps 1 and 2 when you start layer 6's build.

### 1. AI, in the background: the layer 7 reference

```text
In the background, build the layer 7 reference: rebuild/phase7_negation.py, negation on the quadrant
toy and the parts world and the same cell built from a positive "not B" label, printing the mass off
the ring at w = 0.3 (0.863 expected), the w where A-without-B peaks, and the positive label's mass
per w. Add those numbers to rebuild/walk_numbers.py as phase7() with keys p7.*, and run it once with
the co3 python.

Task: in /home-mscluster/mmolefe/PhD/reduce-reuse-recycle, on the current branch, write
rebuild/negation.py (the negation target and the positive-label target for each toy) and
rebuild/phase7_negation.py (the dial), run from the repo root as python -m rebuild.phase7_negation
with /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python. In this order:
1. Reproduce the kinds notebook's negation dial on the ring and the bar, exact on the grid
   (ideal.exact_grid with kinds.negation(w)), at w = 0, 0.02, 0.05, 0.1, 0.3 and 1.0, by replaying
   that dial's data draw, and expose the replayed ring and bar as a function of rebuild/negation.py
   so walk07.py can import it.
2. The quadrant toy of layer 5 (phase5_dial.make_sets, data seed np.random.default_rng(100) as phase
   5 seeds its first run): the exact bottom-left mass of p_A^(1+w)/p_B^w against w, at the overlaps
   of phase5_dial.OVERLAPS_2D.
3. The parts world (phase5_dial.pw_experts, both backgrounds): the cat-with-no-dog mass against w.
   A power of a mixture is not a mixture, so estimate it by self-normalised importance sampling from
   exact draws of p_A (Mixture.sample), with weights (p_A/p_B)^w, and print the standard error and
   the effective sample size beside it.
4. The positive label: the same cells as p(x|A)·p(¬B|x)^w, with p(¬B|x) read from the pooled data
   of both sets (an ideal expert on the pooled points without B over one on all pooled points), its
   mass and normaliser per w. Print the pooled set and the label rule in the script's header.
5. Per toy and w: tuned MALA, UHA and reverse diffusion on the noised target (samplers.Target),
   2,000 samples each, with the settings phase5_dial.sample_2d uses, scored by the cell's mass and
   by ideal.tv or parts_world.tv_projected beside a 2,000-draw floor.
Use the kinds dial's w grid for every toy, with points added where the A-without-B mass peaks.
Starts from: nothing before it in this layer. The reference for layers 1 to 5 is in rebuild/ and its
outputs in rebuild/out/phase1 to phase5; layer 6's reference (rebuild/phase6_guidance.py) may
already have added p6. keys.
Settled facts:
- The brick: the pneumothorax target p(x|PTX)^(1+w)/p(x|drain)^w; exact mass in the A-without-B
  cell (quadrant: bottom-left; parts world: cat with no dog) against w, per sampler, in
  rebuild/negation.py and rebuild/phase7_negation.py; reproduce first: in-bar mass 0.146 at w = 0,
  0.037 at w = 0.1, 0.863 off every ring mode at the paper's w = 0.3 (REBUILD_WALK.md, brick "A
  without B (negation)").
- The kinds dial at w = 0, 0.02, 0.05, 0.1, 0.3, 1.0: mass on the 2 in-bar modes 0.146, 0.121,
  0.085, 0.037, 0.0, 0.0; mass away from every ring mode 0.387, 0.384, 0.391, 0.439, 0.863, 1.0
  (artifacts/ideas/poe-overlap-dial/kinds-of-composition-on-the-papers-toys.ipynb, Figure 2;
  REF_FIG2 in rebuild/phase3_kinds.py).
- A is the ring, kinds.toy_gmm: 8 modes on a circle of radius 0.5, std 0.03, 800 points; B is the
  bar, kinds.vbar(rng, 800, 0.0, 0.2), half-width 0.2; both are ideal experts smoothed by 0.05.
  "In-bar" is the mass within 0.08 of the two ring modes inside the bar (kinds.RING[[2, 6]]), and
  "away" is the mass more than 0.08 from every ring mode, so the A-without-B mass on the six other
  modes is 1 minus the two (rebuild/kinds.py, MEASURES; rebuild/phase3_kinds.py).
- rebuild/phase3_kinds.main draws that ring and bar from np.random.default_rng(0) after
  kinds.build_kinds, one ideal.perfect_samples call of 2,000 per kind in Figure 1, and the
  co-occurrence, contention and interaction dials, in that order (rebuild/phase3_kinds.py).
- Fresh draws move a dial by up to about 0.08 where its curve is steep, and python -m
  rebuild.phase3_kinds draws the notebook's own data and matches every printed value (step file 03,
  step 12). An earlier draw put 0.891 off every mode at w = 0.3 (overlap-dial idea map, run 3).
- Negation works only near w ≈ 0.05, where the exact in-bar mass is 0.10
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, run 3; step file 03, step 11).
- At w = 0.05 on the ring and bar, measure and total variation per sampler: perfect 0.09 and 0.15,
  reverse diffusion 0.20 and 0.26, ULA 0.13 and 0.25, MALA 0.11 and 0.25, UHA 0.12 and 0.21; tuned
  MALA reaches 0.21 against the floor's 0.15 at acceptance 0.60, and ULA at the tuned steps keeps
  446 of 2,000 samples (kinds notebook, Figure 1 and run 1; overlap-dial idea map, runs 4 and 5).
- The notebook's NegationEBMDiffusionModel uses 1.3·E1 − 0.3·E2, which is w = 0.3, and
  kinds.negation(w) is (1 + w)·a − w·b on log-densities (rebuild/compose.py; rebuild/kinds.py).
  Du et al.'s own negation experiments use α = 0.5 in q^0/(q^1)^α (vault note Product, Mixture And
  Negation).
- Where p_B → 0, p_B^(−w) → ∞, so negation pushes mass toward wherever B is least likely, not just
  out of B (step file 02, step 4).
- For Gaussians, q_a/q_b^α is a density only while 1/v_a − α/v_b > 0 (vault notes Product, Mixture
  And Negation and Powers Of Gaussians).
- Negation's exponents sum to 1, (1 + w) − w, so it is unchanged by an invertible warp (total
  variation 7e-17), while the plain product changes by 0.035 to 0.126
  (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 10 and run 6).
- Layer 5's quadrant toy: A is the left half, B the top half, so A without B is the bottom-left; at
  overlap f a share f of each set sits in the top-left (rebuild/phase5_dial.py, docstring).
- Layer 5's parts world: A is "contains a cat", B "contains a dog"; a sample is A∩B when one half's
  mean is above 0.75 and the other's below −0.75 (phase5_dial.pw_ab); background variance 0.01 or
  1.0 (rebuild/phase5_dial.py).
- p(¬B|x) is at most 1, so p(x|A)·p(¬B|x)^w never exceeds p(x|A) and its normaliser is at most 1.
- walk_numbers.py holds worked-example numbers, each written by put(key, value); run results
  (masses and total variations per sampler) are read from rebuild/out/phaseN/ instead. Its __main__
  runs (phase1, phase2, phase3, phase5) and writes rebuild/out/walk-numbers.json by a relative path,
  so it runs from the repo root (rebuild/walk_numbers.py).
- Tolerances: a dial value to the 3 decimals printed, as rebuild/phase3_kinds.py checks Figure 2; a
  sampler number within 0.03 (rebuild/phase3_kinds.py).
- rebuild.device.pick_device takes the first card with under 1 GB in use, else the CPU
  (rebuild/device.py; REBUILD_WALK.md, Pins).
Outputs:
- rebuild/negation.py and rebuild/phase7_negation.py; the phase script first prints the dial beside
  the notebook's twelve values, marked matches or differs, then one row per toy and w;
- rebuild/out/phase7/phase7.json with every row, and a figure of the exact target per w on the ring
  and bar and on the quadrant toy;
- phase7() in rebuild/walk_numbers.py, added to the __main__ tuple: the w grid as p7.w_grid, the
  dial as p7.kinds_inbar_w<w> and p7.kinds_away_w<w>, the exact or estimated A-without-B mass and
  the positive label's mass and normaliser per toy and w as p7.<toy>_<measure>_w<w>, and the peak
  per toy as p7.<toy>_peak_w;
- rebuild/out/walk-numbers.json rewritten, its 84 earlier keys and any p6. keys unchanged.
Target: all twelve dial values match to 3 decimals, 0.863 at w = 0.3 among them; the peak w is
named per toy with the grid points on either side; the positive label's normaliser prints at most 1
at every w; every sampler row carries its total variation beside the floor; every estimated
parts-world number says so, with its standard error. I judge, from the printed rows.
Progress: the dial and its matches line print first; then the quadrant toy's rows, the parts
world's, the positive label's, and the samplers' per toy.
Proof: python -m rebuild.phase7_negation prints "negation dial: matches"; then
python -m rebuild.walk_numbers | grep 'p7\.' lists every p7. key; the evidence is
rebuild/out/phase7/phase7.json.
Limits: add the two files and phase7() only; leave ideal.py, parts_world.py, kinds.py, compose.py,
the phase 1 to 6 scripts and phase6() as they are; write nothing in the ladder, the step files or
REBUILD_WALK.md; no commit.
Hand-off: a report naming the w grid, the replay function's name, every p7. key with its value and
the brick row's new mark; steps 6 and 7 of
artifacts/drips/simple-distributions-rebuild/ladder/07-a-without-b.md read the keys and phase7.json.
```

added: `rebuild/negation.py` beside the phase script, and the five pieces in order (from the brick row in `REBUILD_WALK.md`) · the whole dial, the toy and measure definitions, and the draw order that reproduces them (from the kinds notebook, `rebuild/kinds.py` and `rebuild/phase3_kinds.py`) · why fresh draws differ and the earlier 0.891 (from step file 03 and run 3 of the overlap-dial map) · the samplers on negation (from the kinds notebook and runs 4 and 5) · the notebook's weight against the paper's α (from `rebuild/compose.py` and the vault note) · the precision condition, the warp invariance and the positive label's bound (from the vault notes, claim 10 of the when-poe map and p(¬B|x) ≤ 1) · layer 5's toy definitions (from `rebuild/phase5_dial.py`) · the importance-sampling route for the parts world · what `walk_numbers.py` holds and the tolerances
dropped: nothing

⏱ you 2 min to paste · the AI's time is in step 2's line

### 2. AI, in the background: quizzes 7.1 and 7.2

```text
In the background, prepare and build layer 7's two quizzes, bottom up, the way step 2 of
artifacts/drips/simple-distributions-rebuild/ladder/00-setup.md does for layer 1. Goal: "say when A
without B stays a density".

Task: for each quiz, first dispatch prep-cadence (quiz-prep-pulse --build) with the goal and minutes
below, then dispatch quiz-cadence (mode build, repo
/home-mscluster/mmolefe/PhD/reduce-reuse-recycle).
- Quiz 7.1, --minutes 40: the college maths under layer 7. The quotient of two Gaussians, when an
  integral over the real line converges.
- Quiz 7.2, --minutes 60: layer 7 itself. When (1 + w) log p_A − w log p_B stays a proper density,
  why a positive label p(no drain|x) ≤ 1 stays bounded.
Starts from: quiz 7.1 takes its material from the Study first line of
artifacts/drips/simple-distributions-rebuild/ladder/07-a-without-b.md. Quiz 7.2 takes it from that
file, the "A without B (negation)" row of the Bricks table in
artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, and the idea-map claims below. Layer
7 has no step file yet; its build writes one (step 6 of the ladder file).
Settled facts:
- quiz-cadence stops when a quiz has no coverage file in .quiz-prep/, so prep-cadence runs first for
  each quiz (quiz-cadence, "The preparation gate", as step 2 of ladder/00-setup.md states).
- pdftotext is not installed on mscluster, so
  ~/.claude/skills/index-source/scripts/find-in-textbooks.py fails; open the PDFs with PyMuPDF from
  the co3 environment (checked on mscluster109: import pymupdf works in co3, version 1.28.2, and
  import pypdf fails everywhere).md, step 2).
- The shelf index is ~/Obsidian/Mathematics/references/textbooks/Textbooks.md and the
  worked-examples index is ~/Obsidian/Mathematics/references/exercises/Worked Examples.md
  (quiz-prep-pulse, "The bank").
- The layer's textbook passage is Anton et al., Calculus 7.8 (improper integrals, p571), and its
  vault notes are Products And Quotients Of Diagonal Gaussians and Product, Mixture And Negation
  (the Study first line).
- For Gaussians, q_a/q_b^α is a density only while 1/v_a − α/v_b > 0 (vault notes Product, Mixture
  And Negation and Powers Of Gaussians).
- Where p_B → 0, p_B^(−w) → ∞, so negation pushes mass toward wherever B is least likely (step file
  02, step 4).
- On the ring and bar, the exact mass left on the 2 in-bar ring modes is 0.146, 0.121, 0.085,
  0.037, 0.0, 0.0 at w = 0, 0.02, 0.05, 0.1, 0.3, 1.0, and the mass away from every ring mode is
  0.387, 0.384, 0.391, 0.439, 0.863, 1.0; negation works only near w ≈ 0.05
  (artifacts/ideas/poe-overlap-dial/kinds-of-composition-on-the-papers-toys.ipynb, Figure 2;
  artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, run 3).
- The medical cells wanted are pneumothorax without drain and drain without pneumothorax, a
  negation; 1,906 of 3,543 NEATX pneumothorax films have no drain, and what is missing is a drain
  label on negatives (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claims 6 and 8).
- Numbers for the quadrant toy, the parts world and the positive label come from the p7. keys of
  python -m rebuild.walk_numbers and from rebuild/out/phase7/phase7.json once step 1's reference
  has run; before then a question asks for the derivation, never an invented number.
Outputs:
- two coverage files under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, each with a
  verdict per question: ready, blocked, or unprepared by choice;
- rows in the worked-examples index for every banked problem;
- two published quiz pages, each opening on a read-first list with minutes per item, a short path
  and a full path.
Target: no question on either page carries the verdict blocked; quiz 7.2 carries banked problems,
about 6 to 10 per concept at the build depth; every number in an answer traces to the kinds
notebook, a p7. key or a cited textbook page. I judge, from the pages.
Progress: prep-cadence reports, per quiz, its question count, node count and verdicts; quiz-cadence
reports each URL as it publishes.
Proof: both coverage files exist under .quiz-prep/ and list no question as blocked; both URLs open.
Limits: change nothing in rebuild/ or the step files; in the ladder file, write only the two URLs.
Hand-off: write quiz 7.1's URL into step 3 and quiz 7.2's into step 5 of
artifacts/drips/simple-distributions-rebuild/ladder/07-a-without-b.md, each replacing "(URL lands
here)".
```

added: the two-dispatch order, the preparation gate, the missing `pdftotext` and the PyMuPDF route, and the two indexes (from step 2 of the setup file) · the layer's textbook and vault notes (from the Study first line) · the facts each quiz tests: the precision condition, the escaping mass, the dial's numbers and the medical counts (from the vault notes, step file 02, the kinds notebook and the two idea maps) · the rule against numbers that do not exist yet · the URL write-back
dropped: "the way the setup section of PROMPTS.md does" (why: that quiz prompt is step 2 of `ladder/00-setup.md`; the guide holds none) · "plan PROMPTS.md (layer 7)" (why: the guide has no layer 7 section; the material is this ladder file and its brick row)

⏱ AI ~2 h for both (estimate), beside layer 6's build

### 3. You: quiz 7.1

Quiz 7.1: (URL lands here)

Built by step 2. ⏱ you 20 to 40 min · stop: every *needed now* question right and at least 80% of the rest; below that, read the items your misses point at and retake.

### 4. You: the maths drip

```text
/drip --math quotients of Gaussians, why dividing by an expert whose density falls off like e^(-d²)
pushes mass away without limit, and why a positive label ("no drain", a factor p(no drain|x) ≤ 1)
stays bounded where dividing by the drain expert does not

Starts from: quiz 7.1 passed; any question it marked missed becomes a first rung.
Settled facts:
- For Gaussians, q_a/q_b^α is a density only while 1/v_a − α/v_b > 0: a negative power subtracts
  precision (vault notes Product, Mixture And Negation and Powers Of Gaussians).
- Where p_B → 0, p_B^(−w) → ∞, so negation pushes mass toward wherever B is least likely, not just
  out of B (step file 02, step 4).
- On the ring and bar (8 ring modes of std 0.03 at radius 0.5, a bar of half-width 0.2, ideal
  experts smoothed by 0.05), the exact mass on the 2 in-bar modes is 0.146, 0.121, 0.085, 0.037,
  0.0, 0.0 at w = 0, 0.02, 0.05, 0.1, 0.3, 1.0, and the mass away from every ring mode is 0.387,
  0.384, 0.391, 0.439, 0.863, 1.0 (kinds notebook, Figure 2; rebuild/kinds.py).
- Negation works only near w ≈ 0.05; the notebook's own weight is w = 0.3, the 1.3 and −0.3 of
  NegationEBMDiffusionModel (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, run 3 and "What the words
  are").
- p(no drain|x) is at most 1, so p(x|PTX)·p(no drain|x)^w never exceeds p(x|PTX).
- Negation's exponents sum to 1, so it is unchanged by an invertible warp (total variation 7e-17)
  (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 10 and run 6).
- 1,906 of 3,543 NEATX pneumothorax films have no drain; what is missing is a drain label on
  negatives, so a positive "no drain" label needs one (when-poe-reaches-a-and-b idea map, claims 6
  and 8).
- Report-mined labels add a label term that matters most for these negation cells
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, "The idea, as it stands").
Outputs: the drip's thread folder under artifacts/drips/, with dispatch.md recording my notation,
the worked numbers I used and what clicked.
Target: I say without notes when (1 + w) log p_A − w log p_B stays a proper density, work N(0, 1)
divided by N(0, 0.5)^w at two values of w by hand, and say why the positive label stays bounded; I
judge.
Hand-off: the thread folder's path, which quiz 7.2, step 8's scene and step 9's cross-links read.
```

added: the precision condition and the escaping-mass argument (from the vault notes and step file 02) · the dial's numbers and the working weight against the notebook's (from the kinds notebook and the overlap-dial map) · the positive label's bound, the warp invariance and the medical counts (from p(no drain|x) ≤ 1 and the when-poe map) · the label term (from the overlap-dial map) · the whiteboard example as part of the target (from this layer's Achieve line) · quiz 7.1's misses as first rungs
dropped: nothing

⏱ rough: you 30 to 60 min · AI ~10 min · stop: you say without notes when the weighted difference stays a proper density

**Outcome:** you can say when (1 + w) log p_A − w log p_B stays a proper density, and why a positive label is the safer choice wherever one exists.

### 5. You: quiz 7.2

Quiz 7.2: (URL lands here)

⏱ you 30 to 60 min, the banked problems included · stop: every *needed now* question right and at least 80% of the rest, then the build. Below that, the misses become the first rungs of a second `/drip --math`, and you retake before building.

### 6. You: the build

```text
/drip-execute-plan --request negation on the quadrant toy and the parts world, and the same cell
built from a positive "not B" label, as layer 7 of
artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md; I type every block myself into
~/Playground/PhD/simple-distributions-walk/walk07.py; bar: reproduce 0.863 of the mass off the ring
at w = 0.3, then find the w where A-without-B peaks, and show the positive label staying bounded

Task: cut the layer into 4 to 6 pieces and write them at the walk's open as the step file
artifacts/drips/simple-distributions-rebuild/07-a-without-b.md, in the shape of the step files for
layers 1 to 5; then build, teach and test one piece per round. I run walk07.py from
~/Playground/PhD/simple-distributions-walk/ with
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python and
PYTHONPATH=/home-mscluster/mmolefe/PhD/reduce-reuse-recycle.
Starts from: quiz 7.2 passed; the layer 7 --math drip folder under artifacts/drips/, whose
dispatch.md holds my notation (step 4); rebuild/negation.py, rebuild/phase7_negation.py, the p7.
keys and rebuild/out/phase7/phase7.json (step 1).
Settled facts:
- The w grid, the p7. keys, phase7.json and the function in rebuild/negation.py that returns the
  replayed ring and bar are step 1's; walk07.py imports that function and uses the same grid (from
  step 1).
- The brick: the pneumothorax target p(x|PTX)^(1+w)/p(x|drain)^w; exact mass in the A-without-B
  cell (quadrant: bottom-left; parts world: cat with no dog) against w, per sampler; reproduce
  first: in-bar mass 0.146 at w = 0, 0.037 at w = 0.1, 0.863 off every ring mode at w = 0.3
  (REBUILD_WALK.md, brick "A without B (negation)").
- The whole dial at w = 0, 0.02, 0.05, 0.1, 0.3, 1.0: in-bar 0.146, 0.121, 0.085, 0.037, 0.0, 0.0;
  away from every ring mode 0.387, 0.384, 0.391, 0.439, 0.863, 1.0 (kinds notebook, Figure 2).
- "In-bar" is the mass within 0.08 of the two ring modes inside the bar, "away" the mass more than
  0.08 from every ring mode, so the A-without-B mass on the six other modes is 1 minus the two
  (rebuild/kinds.py, MEASURES).
- Fresh draws move a dial by up to about 0.08 where its curve is steep, so the twelve values
  reproduce only with the replayed draw (step file 03, step 12). An earlier draw put 0.891 off
  every mode at w = 0.3 (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, run 3).
- Negation works only near w ≈ 0.05, where the exact in-bar mass is 0.10 (overlap-dial idea map,
  run 3; step file 03, step 11).
- At w = 0.05, reverse diffusion doubles the in-bar mass (0.20 against 0.10), and tuned MALA reaches
  total variation 0.21 against the floor's 0.15 (kinds notebook, Figure 1 and run 1).
- The notebook's NegationEBMDiffusionModel is w = 0.3, and kinds.negation(w) is (1 + w)·a − w·b on
  log-densities (rebuild/compose.py; rebuild/kinds.py).
- Where p_B → 0, p_B^(−w) → ∞, so negation pushes mass toward wherever B is least likely (step file
  02, step 4); for Gaussians q_a/q_b^α is a density only while 1/v_a − α/v_b > 0 (vault note
  Product, Mixture And Negation).
- p(¬B|x) is at most 1, so p(x|A)·p(¬B|x)^w never exceeds p(x|A) and its normaliser is at most 1.
- Layer 5's quadrant toy: A the left half, B the top half, A without B the bottom-left; layer 5's
  parts world: A "contains a cat", B "contains a dog", background variance 0.01 or 1.0
  (rebuild/phase5_dial.py).
- A power of a mixture is not a mixture, so the parts world's numbers are importance-sampling
  estimates with a standard error (rebuild/parts_world.py; from step 1).
- Negation's exponents sum to 1, so it is unchanged by an invertible warp (total variation 7e-17)
  (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 10 and run 6).
- The medical cells are a negation; 1,906 of 3,543 NEATX pneumothorax films have no drain, and a
  drain label on negatives is missing (when-poe-reaches-a-and-b idea map, claims 6 and 8).
- Tolerances: a dial value to the 3 decimals printed; a sampler number within 0.03
  (rebuild/phase3_kinds.py).
- Phases 1 to 5 are imported from the reference, as step file 05 does (step file 05, "How to walk
  it").
Outputs:
- the step file artifacts/drips/simple-distributions-rebuild/07-a-without-b.md, its pieces ticked,
  and the walk map that holds my place between sittings;
- walk07.py: one script that prints the dial, the A-without-B mass per toy and w with its peak, the
  positive label's mass and normaliser per w, and per sampler the cell's mass and total variation
  beside the floor;
- on the whiteboard, N(0, 1) divided by N(0, 0.5)^w with its tails for two values of w (this
  layer's Achieve line).
Target: the twelve dial values match to 3 decimals, 0.863 at w = 0.3 among them; the peak w per toy
equals its p7. key; the positive label's normaliser is at most 1 at every w; every other exact
number equals its p7. key to the digits shown; every sampler number is within 0.03 of phase7.json.
I judge, from the printed lines.
Progress: each piece prints its number as it lands: the dial first, then the quadrant toy's
A-without-B mass per w, then the parts world, then the positive label, then the samplers.
Proof: python walk07.py prints the dial with 0.146 at w = 0, 0.037 at w = 0.1 and 0.863 away at
w = 0.3, and step 7's comparison shows no differing number.
Limits: never edit rebuild/ or the ladder files; I type, I do not paste; no commit.
Hand-off: the step file's path, the retrospective the walk closes on, and the /drip-retro command
it prints.
```

added: where the step file lands and how `walk07.py` runs (from the guide and step file 05) · the reference's outputs and its replay function as the starting point (from step 1) · the whole dial, its measures, and why it reproduces only with the replayed draw (from the kinds notebook, `rebuild/kinds.py` and step file 03) · the working weight, the samplers on negation, the notebook's weight and the escaping-mass argument (from the overlap-dial map, the kinds notebook, `rebuild/compose.py` and step file 02) · the positive label's bound, layer 5's toys, the parts world's estimates, the warp invariance and the medical counts · the tolerances (from `rebuild/phase3_kinds.py`)
dropped: nothing

⏱ rough: you 1 to 2 h (about 5 pieces) · AI ~35 min · stop: the prompt's bar

**Outcome:** negation's escaping mass on both toys, the w where A-without-B peaks, and the positive "not B" label staying bounded beside it.

### 7. You: compare your numbers with the reference

**Compare.** `python -m rebuild.walk_numbers` prints the `p7.` keys; your `walk07.py` prints the same numbers.

```bash
cd /home-mscluster/mmolefe/PhD/reduce-reuse-recycle
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m rebuild.walk_numbers | grep 'p7\.'
cd ~/Playground/PhD/simple-distributions-walk
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python walk07.py
```

⏱ you ~10 min (estimate) · stop: every exact number your script prints equals its `p7.` line to the digits shown, including `p7.kinds_away_w0.3` at 0.863, and every sampler number sits within 0.03 of `rebuild/out/phase7/phase7.json`. A number that differs is either a lesson or a bug in one of the two files; find which before layer 8.

### 8. AI, in the background: the scene

```text
/math-scene the folder the layer 7 --math drip wrote under artifacts/drips/: the negation weight as
a slider, the in-bar modes vanishing and the mass escaping, beside the bounded positive label

Starts from: the layer 7 --math drip folder and its dispatch.md (step 4's hand-off); the p7. keys
and rebuild/out/phase7/phase7.json (step 1).
Settled facts:
- The ring and bar: 8 ring modes of std 0.03 at radius 0.5, a bar of half-width 0.2, ideal experts
  smoothed by 0.05 (rebuild/kinds.py).
- On that pair the exact mass on the 2 in-bar modes is 0.146, 0.121, 0.085, 0.037, 0.0, 0.0 at
  w = 0, 0.02, 0.05, 0.1, 0.3, 1.0, and the mass away from every ring mode is 0.387, 0.384, 0.391,
  0.439, 0.863, 1.0 (kinds notebook, Figure 2).
- p(¬B|x) is at most 1, so the positive label's target never exceeds p(x|A); its mass per w is the
  p7. keys (from step 1).
- Every number on the scene comes from the dial, the p7. keys or the drip's dispatch.md, never
  estimated by the scene; parts-world numbers carry their standard error (from step 1).
Outputs: one published scene with the negation weight as a slider, in my notation from dispatch.md.
Target: every number the scene shows equals its dial value, p7. key or phase7.json row; at
w = 0.3 the scene shows 0.863 away from every ring mode.
Hand-off: write the scene's URL into step 8 of
artifacts/drips/simple-distributions-rebuild/ladder/07-a-without-b.md, replacing "(URL lands
here)".
```

added: the ring and bar's definition and the whole dial (from `rebuild/kinds.py` and the kinds notebook) · the positive label's bound and the reference's keys as the only number source · the URL write-back
dropped: nothing

Scene: (URL lands here)

⏱ you ~20 min · AI ~45 min, beside layer 8's maths · stop: the scene is published and its numbers match your walk's

**Outcome:** a scene where you raise the negation weight and watch the in-bar modes vanish and the mass escape, beside the bounded positive label.

### 9. You: close the layer

Run the `/drip-retro` command the build printed at its close (⏱ about 25 minutes). Then paste this:

```text
In the background, dispatch xref-cadence over the files layer 7 of the composition ladder wrote: the
layer 7 --math drip folder under artifacts/drips/, the step file
artifacts/drips/simple-distributions-rebuild/07-a-without-b.md, and the retrospective the layer 7
walk closed on.

Settled facts:
- The reference sets to link against are the vault notes on the Study first line of
  artifacts/drips/simple-distributions-rebuild/ladder/07-a-without-b.md; the "A without B
  (negation)" row of the Bricks table in
  artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md; claim 6 and run 3 of
  artifacts/ideas/poe-overlap-dial/IDEA_MAP.md; and claims 6, 8 and 10 of
  artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md.
- ~/Playground/PhD/simple-distributions-walk/ is scratch space; link nothing into it (CLAUDE.md, the
  playground is disposable).
Outputs: links written both ways between those files and the reference sets.
Hand-off: the agent's report, listing every link it wrote, read before layer 8 starts.
```

added: the reference sets, by path (from this layer's Study first line, `REBUILD_WALK.md` and the claims the layer rests on)
dropped: nothing

⏱ you 2 min to paste · AI ~10 min (estimate)

---

⬅️ [Layer 6: guidance](06-guidance-as-a-dial.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 8: the interaction term](08-the-interaction-term.md) ➡️
