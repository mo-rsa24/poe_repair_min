# ⭐ Layer 5: the overlap dial

⬅️ [Layer 4: the parts world](04-the-parts-world.md) · 🏠 [the guide](../PROMPTS.md) · [Reading stop: Bradley](05a-reading-stop-projective-composition.md) ➡️

**Why it is critical.** It answers the original question: how much A-and-B data a product needs, and whether pretraining supplies it.

**Depth.** Build: why the product weights shared places most, and measuring A∩B in the right region (the dial's deep square, the answer key's blobs) with the answer-key numbers. Gloss: what fine-tuning does to inherited support. Own, optional: how Z moves with expert B's blob. You 2.4 to 4.8 h in all, plus 30 to 60 minutes if you take the optional Z.

**Study first.** The question itself: [When Neither Training Set Holds A And B](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/When%20Neither%20Training%20Set%20Holds%20A%20And%20B.md) and [What An Expert Learns From Its Training Set](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/What%20An%20Expert%20Learns%20From%20Its%20Training%20Set.md). Why a background matters: [Composing Models With A Background](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/Composing%20Models%20With%20A%20Background.md). Why samplers cannot reweight separated modes: [the MCMC correction](../../../../../../goal-setting/learning/deep-learning/diffusion-models/06-diffusion-speciation-dynamics/plans/05-composition-and-the-experiment/plans/03-mcmc-correction.md) and [a mixture needs heights](../../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/33-a-mixture-needs-heights.md). How sure a sampled share is: an A∩B share is k of n samples, so [the exact interval for a proportion](../../../../../../goal-setting/learning/mathematics/statistics/sampling/textbook/07-the-exact-interval.md) bounds it, and [the percentile bootstrap](../../../../../../goal-setting/learning/mathematics/statistics/sampling/plans/13-the-percentile-bootstrap-derived.md) says which units to draw again when the spread comes from data seeds.

**Achieve.** `walk05.py` printing deep A∩B mass per overlap, exact and sampled · on the whiteboard: two blobs overlapping by 5%, shading where the product's mass goes · worked: `0.30` deep mass at 5% overlap · problems: the ones quiz 5.2 banks · quizzes 5.1 and 5.2 passed.

## The flow

```text
 AI, in the background                         │  You, in order
───────────────────────────────────────────────┼───────────────────────────────────────────────
 quizzes 5.1 and 5.2, prepared ──────────────▶ │  1. quiz 5.1: mass in a region, normalisers
   (paste during layer 4's build)              │       ▼
                                               │  2. /drip --math: shared places, fine-tuning
                                               │       ▼
                                               │  3. quiz 5.2: needed-now right, 80% of the rest
                                               │       ▼
                                               │  4. build: walk05.py
 rebuild/ reference, built ──────────────────▶ │  5. compare: p5. numbers equal yours
                                               │       ▼
 results-scene, beside the read ◀───────────── │  6. scene
                                               │       ▼
 xref-cadence links the layer ◀─────────────── │  7. /drip-retro, then the next layer
```

## The steps

### 0. AI, in the background: quizzes 5.1 and 5.2

**AI, one layer ahead.** Paste this when you start layer 4's build.

```text
In the background, prepare and build layer 5's two quizzes, bottom up, for the composition ladder
in artifacts/drips/simple-distributions-rebuild/. Goal: "say how much A-and-B data a product
needs, with the numbers". Quiz 5.1, --minutes 40: integrating a density over a region, the mass
share of a region, ratios of normalising constants. Quiz 5.2, plan 05-the-overlap-dial.md,
--minutes 60: why 5% overlap puts 0.30 of the product's mass deep in A∩B, the three answer-key
numbers, what fine-tuning does to inherited support (at gloss depth), and how Z moves with expert
B's blob.

Task: for each quiz, first dispatch prep-cadence (quiz-prep-pulse --build) with the goal, minutes
and topics above, then dispatch quiz-cadence (mode build, repo
/home-mscluster/mmolefe/PhD/reduce-reuse-recycle). Every path below is from that repo's root.
Starts from: the miss ledger (quiz-ledger) of every earlier quiz on this ladder. Quiz 5.1 takes its
material from artifacts/drips/simple-distributions-rebuild/ladder/05-the-overlap-dial.md and its
Study first line; quiz 5.2 takes it from
artifacts/drips/simple-distributions-rebuild/05-the-overlap-dial.md, steps 0 to 8, and from step 4
of the ladder file (the answer key) and its optional Z step.
Settled facts:
- quiz-cadence stops when a quiz has no coverage file in
  artifacts/drips/simple-distributions-rebuild/.quiz-prep/, so prep-cadence runs first for each
  quiz (quiz-cadence, "The preparation gate").
- The step file links a textbook page beside each fundamental it uses (p. 230 of
  introduction-to-probability-for-data-science.pdf for the normal CDF, p. 42 of matrix-cookbook.pdf
  for the constant of a product of Gaussians) and carries three tables of timestamped video
  segments (the kernel density estimate, annealed Langevin, the product of two mixtures); reuse
  them as nodes before searching again (step file 05, steps 1, 2, 5 and 6).
- pdftotext is not installed on mscluster, so
  ~/.claude/skills/index-source/scripts/find-in-textbooks.py fails. pypdf does not import in any
  environment there either; open the PDFs with PyMuPDF (import pymupdf, version 1.28.2) from
  /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python (checked on mscluster109, 2026-10-18).
- The shelf index is ~/Obsidian/Mathematics/references/textbooks/Textbooks.md and the
  worked-examples index is ~/Obsidian/Mathematics/references/exercises/Worked Examples.md
  (quiz-prep-pulse, "The bank").
- Two toys carry this layer, in two notations. The overlap dial's: A is the left half, B the top
  half, A∩B the top-left quadrant, each expert 800 points smoothed by σ = 0.05, a share f of them
  in A∩B, and "deep" means x1 < −0.15 and x2 > 0.15 (step file 05, steps 1 and 3). The answer
  key's: one Gaussian blob of std 0.15 per quadrant at (±0.5, ±0.5), A is x1 > 0, B is x2 > 0,
  A∩B the top-right quadrant, shares counted within 3 std of a blob's centre (answer-key-quadrants
  README, in artifacts/ideas/when-poe-reaches-a-and-b/runs/).
- Overlap-dial numbers, 3 data seeds: deep A∩B mass 0.00 at 0% (rounded: it is below 1.1e-6),
  0.30 ± 0.10 at 5% (about 40 shared points per set on average, since each point is shared with
  probability f), 0.59 at 20%, 0.76 at 100%; the whole quadrant reads 0.20 ± 0.02 at 0%, all of
  it the sliver around the corner; tuned MALA reaches 0.17 at 5% (python -m rebuild.phase5_tables;
  step file 05, steps 2 to 5).
- Answer-key numbers: one conditional model with "contains" labels keeps 0.989 in A∩B for every
  A∩B share r from 0.25 down to 0.001; separately fine-tuned experts whose leftover sits only on
  the two quadrants next to their own hold 0.5, fixed by that construction's x → −x symmetry (it
  is not a ceiling for every pair of separately fine-tuned experts); experts over separate parts
  reach 0.9999 with no A∩B data; 0.989 is 1 − e^(−9/2), the mass of one blob within 3 std, and
  0.9999 is 1 − e^(−9), the same radius around a blob squared to variance σ²/2
  (p5.mass_within_3sd_2d and p5.mass_within_3sd_2d_half_variance; answer_key.json;
  the three-ways scene's README; when-poe-reaches-a-and-b idea map, run 4; ladder file 05, step 4).
- Fine-tuning: pretrained support falls from 0.159 deep mass to 0.001 within 30 fine-tune steps on
  the exact grid; tuned MALA and UHA still read 0.056 and 0.068 at step 30, 0.004 and 0.0055 at
  step 100, and at most 0.001 from step 300; experts trained from scratch give 0.000 (step file 05,
  step 7; rebuild/out/phase5/finetune.json).
- Z with expert A's blob in the bottom-right corner and B's blob moved: 7.9 × 10⁻¹⁰ opposite,
  5.3 × 10⁻⁵ next door, 3.54 on top of A (scene_key.json in
  artifacts/scenes/three-ways-a-product-reaches-a-and-b/, keys support.TL.Z, support.TR.Z and
  support.BR.Z). By hand, two normalised 2D Gaussians of std σ with centres d apart multiply to a
  total mass Z = exp(−d²/(4σ²)) / (4πσ²) (p5.Z_prefactor_sd0.15, p5.Z_next_door_by_hand and
  p5.Z_opposite_by_hand), which with σ = 0.15 and d² = 2, 1, 0 gives the same
  three values (ladder file 05, step 5b).
- The diffusion models quiz, https://claude.ai/artifact/MyTdAaVfZW3wiqhbLKg4wu, already holds
  sections "When neither training set holds A and B" and "The four-quadrant answer key" (24
  questions on the answer key and the three rescues, when-poe-reaches-a-and-b idea map, run 3).
  Read them before writing, so the new pages extend them.
- The bar a learner meets: every question marked needed now right and at least 80% of the rest; a
  missed question returns in the next quiz (PROMPTS.md, "The bar for every quiz").
Outputs:
- two coverage files under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, each with a
  verdict per question: ready, blocked, or unprepared by choice;
- rows in the worked-examples index for every banked problem;
- two published quiz pages, each opening on a read-first list with minutes per item, a short path
  and a full path.
Target: no question on either page carries the verdict blocked; quiz 5.2 carries banked problems,
about 6 to 10 per concept for its two build-depth concepts (why the product weights shared places
most, measuring A∩B in the right region) and 4 to 6 for the own-depth Z; every number in an answer
traces to phase5_tables, answer_key.json, scene_key.json, the step file or a cited textbook page.
I judge, from the pages.
Progress: prep-cadence reports, per quiz, its question count, node count and verdicts; quiz-cadence
reports each URL as it publishes.
Proof: the two coverage files exist under .quiz-prep/ and list no question as blocked; the two URLs
open.
Limits: change nothing in the step file's prose, in rebuild/ or in the answer-key and scene
folders; in the ladder file, write only the two URLs.
Hand-off: write quiz 5.1's URL into step 1 and quiz 5.2's URL into step 3 of
artifacts/drips/simple-distributions-rebuild/ladder/05-the-overlap-dial.md, each replacing "(URL
lands here)".
```

added: the task's two dispatches and their order, the preparation gate (from the quiz-cadence agent definition) · the step file's textbook pages and video tables as ready nodes (from step file 05) · the missing `pdftotext`, and PyMuPDF in place of pypdf, which does not import on mscluster109 (checked 2026-10-18) · the shelf and worked-examples indexes (from the setup's quiz prompt) · the two toys and their notations (from step file 05 and the answer-key README) · the overlap-dial, answer-key, fine-tune and Z numbers (from `phase5_tables`, `answer_key.json`, `finetune.json` and `scene_key.json`) · the quiz sections already published (from the when-poe-reaches-a-and-b idea map, run 3) · the quiz bar (from `PROMPTS.md`) · outputs, target, progress, proof, limits and the URL write-back · from the Codex mathematics review: the 40 shared points as an average of a random count, the 0% deep mass as rounded, the 0.5 as a property of the adjacent-leftover construction rather than a ceiling for every fine-tuned pair, where 0.9999 comes from, the samplers' fine-tune values at steps 30, 100 and 300 (0.0055 for UHA at step 100 exceeds the 0.005 the prompt stated), the hand formula for Z, and "the right region" in place of "inside each blob", since the dial toy has a deep square rather than blobs (checked against `answer_key.json`, `finetune.json`, `overlap-dial.json` and `scene_key.json`) · the key names for the closed forms 1 − e^(−9/2), 1 − e^(−9) and the by-hand Z (from the port of the review's checks into walk_numbers)
dropped: "the way the setup section of artifacts/drips/simple-distributions-rebuild/PROMPTS.md does for layer 1" (why: `PROMPTS.md` has no setup section any more; the setup's quiz prompt is step 2 of [the setup](00-setup.md), and this prompt now carries everything it carried)

⏱ AI ~1.5 h (estimate), beside layer 4's build

### 1. You: quiz 5.1

Quiz 5.1: (URL lands here)

Built by step 0's background prompt. ⏱ you 20 to 40 min · stop: the bar, every *needed now* question right and at least 80% of the rest; below that, read the items your misses point at and retake.

### 2. You: the maths drip

```text
/drip --math steps 4 and 7 of artifacts/drips/simple-distributions-rebuild/05-the-overlap-dial.md:
why the product weights shared places most, and what fine-tuning does to inherited support

Starts from: quiz 5.1 passed; any question it marked missed becomes a first rung.
Settled facts:
- The quadrant toy: A is the left half, B the top half, A∩B the top-left quadrant; each expert
  holds 800 points, a share f of them in A∩B, smoothed by σ = 0.05 (step file 05, step 1). Each
  point lands in A∩B independently with probability f, so A has 8 points there on average at 1%
  and 40 at 5%, and one data seed's count differs (walk_numbers, p5.overlap_points_of_800_at_1pct
  and _5pct, which are 800f).
- The product's mass at a point is the product of both experts' densities there. Outside A∩B at
  least one expert is near zero almost everywhere, so a few points shared by both sets carry a
  large share of the product. Averaged over the data, the deep mass is
  M_D(f) = f²C_D / (f²C + f(1−f)C_1 + (1−f)²C_0), with corner overlap C_0 = 0.0008, edge overlaps
  C_1 = 0.053, the square C = 0.890 and its deep part C_D = 0.675, which gives 0.31 at 5% and 0.76
  at 100% (step file 05, step 4; p5.deep_mass_C0_corner, p5.deep_mass_C1_edges,
  p5.deep_mass_C_square, p5.deep_mass_CD_deep_square and p5.deep_mass_formula_MD).
- Deep A∩B means x1 < −0.15 and x2 > 0.15, 3σ inside; at 100% overlap a uniform density would put
  0.85 × 0.85 = 0.7225 there, a benchmark the product exceeds, since its exact mass there is 0.76
  (step file 05, step 3; p5.interior_area_share_margin0.15).
- At 0% the two squares touch only at the origin. Along each axis the product is
  Φ(−x/σ)Φ(x/σ) = u(1 − u) with u = Φ(x/σ), largest at 0 where it equals 0.25, with half its mass
  within 0.0302 of the boundary; the whole quadrant reads 0.20 ± 0.02 and the deep region rounds
  to 0.00 (step file 05, steps 2 and 3; p5.corner_profile_peak_at_0,
  p5.corner_profile_half_mass_within).
- The exact deep mass over 3 data seeds: 0.00 at 0%, 0.05 ± 0.05 at 1%, 0.09 at 2%, 0.30 ± 0.10
  at 5%, 0.51 at 10%, 0.59 at 20%, 0.71 at 50%, 0.76 at 100% (step file 05, step 4;
  python -m rebuild.phase5_tables).
- Fine-tuning: one model pretrained on all four quadrants, two copies fine-tuned at 0% overlap
  with Adam at 1e-4. Grid / tuned MALA / UHA deep mass: 0.159 / 0.158 / 0.163 at step 0, 0.089 /
  0.104 / 0.122 at step 10, 0.001 / 0.056 / 0.068 at step 30, 0.005 / 0.004 / 0.0055 at step 100,
  and at most 0.001 from step 300. Experts trained from scratch give 0.000 (step file 05, step 7;
  finetune.json).
- Rescue would mean the pretrained support survives long enough for fine-tune length to be a dial;
  erasure at once means pretraining cannot stand in for overlap data (step file 05, step 7).
- Claim 2 of the overlap-dial idea map, repaired: the product puts samples in A∩B only where some
  expert carries density there, from overlap in the training data or from an expert loose outside
  its concept's region; support inherited from pretraining does not survive fine-tuning
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 2).
Outputs: the drip's thread folder under artifacts/drips/, with dispatch.md recording my notation,
the worked numbers I used and what clicked.
Target: I give both reasons in this layer's Outcome without notes, and work 0.25 and 0.7225 by hand;
I judge.
Hand-off: the thread folder's path, which quiz 5.2, the build in step 4, the scene in step 6 and
step 7's cross-links read.
```

added: the toy, the reason a few shared points dominate, the deep region and its ceiling, the corner sliver, the 3-seed curve and the fine-tune numbers (from step file 05, `walk-numbers.json`, `phase5_tables` and `finetune.json`) · what would count as rescue (from step file 05) · claim 2's repaired wording (from the overlap-dial idea map) · quiz 5.1's misses as first rungs · outputs, target and hand-off · from the Codex mathematics review: the 8 and 40 as averages of a random count, the normalised deep-mass formula with its four constants, 0.7225 as a uniform-area benchmark rather than a ceiling (0.76 exceeds it), the u(1 − u) form of the corner profile, the 0% deep mass as rounded, and every fine-tune checkpoint to step 300 (UHA's 0.0055 at step 100 exceeds the 0.005 the prompt stated) · the key names for the deep-mass constants and curve (from the port of the review's checks into walk_numbers)
dropped: nothing

⏱ you 35 to 65 min · AI ~10 min · stop: you give both reasons in the Outcome below without notes

**Outcome:** you can say why 5% overlap puts `0.30` of the product's mass deep in A∩B, more than 5%, and why pretraining support is gone within 30 to 100 fine-tune steps.

### 3. You: quiz 5.2

Quiz 5.2: (URL lands here)

⏱ you 30 to 60 min, the banked problems included · stop: the bar, then the build. Below the bar, the misses become the first rungs of a second `/drip --math`, and you retake before building.

### 4. You: the build: walk05.py

```text
/drip-execute-plan artifacts/drips/simple-distributions-rebuild/05-the-overlap-dial.md, I type every
block myself into ~/Playground/PhD/simple-distributions-walk/walk05.py, measuring the dial's A∩B in
its deep square and the answer key's within 3 std of each blob, with the whole quadrant only as a
spill check; bar: deep A∩B mass 0.00 at 0% and 0.30 at 5% overlap (exact), and tuned MALA's 0.17
at 5%

Task: walk the step file one step per message in its order, steps 0 to 8, then rebuild the three
answer-key numbers listed under step 4 of
artifacts/drips/simple-distributions-rebuild/ladder/05-the-overlap-dial.md inside the same walk. I
run walk05.py from ~/Playground/PhD/simple-distributions-walk/ with
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python and
PYTHONPATH=/home-mscluster/mmolefe/PhD/reduce-reuse-recycle.
Starts from: the setup's folder and environment; walk04.py from layer 4; quiz 5.2 passed; the
layer 5 --math drip folder under artifacts/drips/, whose dispatch.md holds my notation.
Settled facts:
- The reference is rebuild/phase5_dial.py and rebuild/phase5_finetune.py; their results are
  rebuild/out/phase5/overlap-dial.json and finetune.json, printed as tables by
  python -m rebuild.phase5_tables, and the worked numbers are p5. keys of
  python -m rebuild.walk_numbers: 24 of them, 8 for walk05.py's own toys and 16 for the step and
  ladder files' derivations (step file 05, "Where the numbers come from").
- From step 0 on, phases 1 to 4 are imported from the reference (rebuild.ideal, rebuild.samplers,
  rebuild.parts_world), seeded with np.random.default_rng(100) and torch.manual_seed(100), with
  N = 800 points, NS = 1000 samples and a 0.15 margin (step file 05, step 0).
- Each toy has its own A∩B region. The dial's is the deep square x1 < −0.15, x2 > 0.15 (3σ
  inside); at 0% no training point is there and it reads 0.00, while the whole top-left quadrant
  reads 0.20 ± 0.02, all of it the sliver around the corner, so the quadrant is reported only as a
  boundary-spill check and never alone as evidence (step file 05, steps 2 and 3). The answer key's
  is within 3 × 0.15 = 0.45 of each blob's centre, where a quadrant count reads the from-scratch
  origin blend as 0.25 (when-poe-reaches-a-and-b idea map, claim 5). The regions sit in different
  toys, so their raw counts are never compared as one metric.
- The bar's 0.30 and 0.17 are means over 3 data seeds with 2,000 samples (0.30 ± 0.10 exact,
  0.17 ± 0.04 tuned MALA, from phase5_tables). walk05.py runs one seed with 1,000 samples, and the
  step file's own run of it printed exact deep mass 0.0, 0.03, 0.276, 0.486, 0.453, 0.486, 0.709,
  0.76 at 0, 1, 2, 5, 10, 20, 50 and 100%, and at 5% tuned MALA 0.382 and UHA 0.343 against an
  exact 0.486. At low overlap a single seed swings because a handful of shared points decides it
  (step file 05, steps 4 and 5).
- Over 3 seeds tuned MALA and UHA agree within 0.02, reach the exact curve from 20% up (0.55
  against 0.59) and lag below it at 5% (0.17 and 0.18 against 0.30) and 10% (0.40 against 0.51);
  reverse diffusion gives 0.02 at 5% and overshoots at 100%, 0.91 against 0.76 (step file 05,
  step 5; phase5_tables). At 2% to 10% overlap the samplers recover a third to four fifths of the
  exact mass (overlap-dial idea map, claim 4).
- The whole-distribution check: TV = ½ Σ |p̂ − p| over 40 × 40 bins of [−1.1, 1.1]² (ideal.tv),
  set against the TV of 2,000 draws taken straight from the exact grid, the finite-sample floor.
  At 5% tuned MALA reads 0.24 and UHA 0.23 against a 0.15 floor; at 20%, 0.18 and 0.19 against
  0.18 (phase5_tables, first table; step file 05, step 5). A matching regional mass does not pass
  on its own.
- The parts-world dial: A∩B means one half's mean above 0.75 and the other's below −0.75. With an
  empty background (variance 0.01) the exact A∩B weight is 0.000 at 0%, a ghost with half-means
  (0.5, −0.5), and 1.000 from 0.1% up, because the real two-animal pair wins by 386.2 nats (400
  for the ghost's mismatch against −13.81 for the overlap weights). With a loose background
  (variance 1.0) it is 1.000 at every f (step file 05, step 6; p5.ghost_mismatch_nats,
  p5.overlap_weight_cost_nats, p5.matching_pair_advantage_nats).
- Step 7's copies come from copy.deepcopy(pre).requires_grad_(True): the moving-average copy has
  its gradients switched off, so fine-tuning it without that call trains nothing and raises no
  error (step file 05, step 7; step file 01, step 9's broken run).
- Step 7's short version (2,000 pretraining steps) printed deep mass 0.040, 0.007, 0.000, 0.008,
  0.005 at fine-tune steps 0, 10, 30, 100, 300; the full run (10,000 steps) starts at 0.159 (step
  file 05, step 7). At step 10,000 the grid counts a spurious blob near (0.75, 0.9) that the
  samplers never reach (REBUILD_WALK.md, brick "Fine-tune variant").
- The answer key is a second toy: one Gaussian blob of std 0.15 per quadrant at (±0.5, ±0.5), A
  is x1 > 0, B is x2 > 0, A∩B the top-right blob, shares counted within 3 std of a blob's centre
  on a 441 × 441 grid over [−1.1, 1.1]². Its A is the right half where the dial's is the left,
  so its A∩B is top-right where the dial's is top-left. Construction (1), p(x|A)·p(x|B)/p(x) with
  "contains" labels, holds 0.9886 at r = 0.001 (0.989 from r = 0.25 down); separately fine-tuned
  experts with leftover only next door hold 0.4999 at ε = 0.1, against the 0.5 that this
  construction's x → −x symmetry fixes (not a ceiling for every fine-tuned pair); own-part experts
  hold 0.99985 as a plain product (answer-key-quadrants README and answer_key.json, in
  artifacts/ideas/when-poe-reaches-a-and-b/runs/; ladder file 05, step 4).
- Train on a card with under 1 GB in use, else on the CPU (REBUILD_WALK.md, Pins). The full runs
  take about 10 minutes (phase5_dial) and 15 (phase5_finetune) on one card (step file 05, step 8).
Outputs:
- walk05.py: one script that runs end to end and prints each step's numbers, then the three
  answer-key numbers;
- the step file's steps 0 to 8 ticked, and the walk map that holds my place between sittings.
Target: the exact, seed-fixed numbers walk05.py prints equal the step file's printed run to the
digits shown (corner profile 0.25; 0% deep mass 0.0; the eight-value curve above; the parts
world's 0.0 then 1.0 with an empty background and 1.0 throughout with a loose one); at 5% tuned MALA
and UHA print below the exact value; the three answer-key numbers equal answer_key.json to 4
digits (0.9886, 0.4999, 0.9999). The 3-seed 0.30 and 0.17, and the TV against its floor (tuned
MALA 0.24 against 0.15 at 5%, 0.18 against 0.18 at 20%), are read from phase5_tables in step 5,
not reproduced. I judge, from the printed lines.
Progress: step 1 prints the count of A's points in A∩B at 5% (about 40), step 2 the corner profile
and the 0% masses, step 4 the eight-value curve, step 5 four sampler lines, step 6 two background
lines and the ghost's half-means, step 7 five fine-tune lines, then one line per answer-key number.
Proof: python walk05.py prints "0% overlap: ... at least 0.15 inside 0.0" and the eight-value curve
above, and the comparison in step 5 of this layer shows no differing exact number.
Limits: never edit rebuild/, the answer-key or scene folders, or the step file's code blocks; never
import answer_key.py, type its construction; I type, I do not paste; no commit.
Hand-off: the retrospective the walk closes on, and the /drip-retro command it prints.
```

added: where the reference and its numbers live (from step file 05) · the imports and seeds (from step file 05, step 0) · why shares are measured inside a supported region (from step file 05 and the when-poe-reaches-a-and-b idea map, claim 5) · that the bar's 0.30 and 0.17 are 3-seed means while `walk05.py` runs one seed, with the single-seed numbers it should print (from `phase5_tables` and step file 05, steps 4 and 5) · the sampler lag and reverse diffusion's overshoot (from step file 05 and the overlap-dial idea map, claim 4) · the parts-world dial and its 386.2 nats (from step file 05 and `walk-numbers.json`) · the moving-average gradient trap (from step files 05 and 01) · the fine-tune numbers and the spurious blob (from step file 05 and `REBUILD_WALK.md`) · the answer key's toy and its three numbers to four digits (from the answer-key README and `answer_key.json`) · the GPU guard and the full runs' times · outputs, target, progress, proof, limits and hand-off · from the Codex mathematics review: each toy's own A∩B region in place of "inside each blob and never by quadrant", since the dial toy has a deep square rather than blobs and its 0% deep square holds no training point, with the whole quadrant kept as a spill check · the TV check against its 2,000-draw floor on 40 × 40 bins of [−1.1, 1.1]² (the review wrote [−1.5, 1.5]²; `ideal.EDGES` spans [−1.1, 1.1]) · the guard that the answer key's A is the right half · the 0.5 as a property of the adjacent-leftover construction rather than a ceiling for every fine-tuned pair · the p5. key count, 24, split into walk05.py's 8 and the derivations' 16 (from the port of the review's checks into walk_numbers)
dropped: the bar's 0.30 and 0.17 as numbers `walk05.py` must print (why: they are means over 3 seeds and 2,000 samples, and `walk05.py` runs one seed with 1,000, which the step file's own run shows at 0.486 exact and 0.382 tuned MALA; they stay in the first line and are checked in step 5 against `phase5_tables`)

⏱ you 1 to 2 h (9 steps, plus the three answer-key numbers below) · AI ~25 min · stop: the prompt's bar and the three answer-key numbers

**Answer key for the four quadrants** (`artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/`). Rebuild these three numbers inside the same walk:

- construction (1) with "contains" labels keeps `0.989` in A∩B down to overlap r = 0.001;
- separately fine-tuned experts whose leftover sits only on the two quadrants next to their own put `0.5` in A∩B, fixed by that construction's symmetry;
- experts over separate parts reach `0.9999` with no A∩B data.

Layer 12's dial claim rests on these.

**Keep the two toys apart.** In the overlap dial, A is the left half and B the top half, so A∩B is the top-left region. In this answer key, A is the right half ($x_1>0$) and B the top half, so A∩B is the top-right blob.

**Why the three numbers differ.** Each one comes from a different construction, so they are not three scores of one thing.

With "contains" labels, dividing by the background leaves the A∩B blob at its original standard deviation σ. For a two-dimensional isotropic Gaussian the mass inside radius $3\sigma$ is

$$
\begin{aligned}
P(R\leq3\sigma)
&=1-\exp\left(-\frac{(3\sigma)^2}{2\sigma^2}\right) \\
&=1-e^{-9/2} \\
&\approx0.9889.
\end{aligned}
$$

The rule used is the radial CDF of a two-dimensional isotropic Gaussian (`p5.mass_within_3sd_2d`). The grid gives `0.9886` at r = 0.001.

With leftover next door, expert A keeps $1-2\varepsilon$ on bottom-right and ε on top-right and on bottom-left, and expert B keeps $1-2\varepsilon$ on top-left and ε on the same two. The point reflection $(x_1,x_2)\mapsto(-x_1,-x_2)$ turns A into B and B into A, so it leaves the product unchanged while swapping its top-right and bottom-left modes. With the blobs far apart those two modes carry almost all the weight, so

$$
\begin{aligned}
w_{\mathrm{TR}} &= w_{\mathrm{BL}}, \\
\frac{w_{\mathrm{TR}}}{w_{\mathrm{TR}}+w_{\mathrm{BL}}} &= \frac12.
\end{aligned}
$$

The grid gives `0.4999` at ε = 0.1. The `0.5` belongs to this symmetric construction; it is not a ceiling for every pair of separately fine-tuned experts. Put each expert's leftover on the top-right alone, A keeping $1-\varepsilon$ on bottom-right and B $1-\varepsilon$ on top-left, and the same grid gives the top-right `0.9997` at ε = 0.1 (`p5.leftover_top_right_only_eps0.1_share`).

With separate parts, the plain product multiplies the desired Gaussian along each axis by an equal Gaussian from the other expert's background. Squaring a Gaussian halves its variance, so the local variance becomes $\sigma^2/2$ and the same radius holds

$$
\begin{aligned}
P(R\leq3\sigma)
&=1-\exp\left(-\frac{(3\sigma)^2}{2(\sigma^2/2)}\right) \\
&=1-e^{-9} \\
&\approx0.9999.
\end{aligned}
$$

The closed form is `p5.mass_within_3sd_2d_half_variance`, and the grid gives `0.99985`.

### 5. You: compare your numbers with the reference

**Compare.** `python -m rebuild.walk_numbers` prints the `p5.` keys, and `rebuild/out/phase5/` holds the curves; your `walk05.py` prints the same numbers.

```bash
cd /home-mscluster/mmolefe/PhD/reduce-reuse-recycle
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m rebuild.walk_numbers | grep 'p5\.'
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m rebuild.phase5_tables
cd ~/Playground/PhD/simple-distributions-walk
export PYTHONPATH=/home-mscluster/mmolefe/PhD/reduce-reuse-recycle
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python walk05.py
```

⏱ you ~10 min · stop: these 8 of the 24 `p5.` lines read `corner_profile_peak_at_0: 0.25`, `corner_profile_half_mass_within: 0.0302`, `overlap_points_of_800_at_1pct: 8`, `overlap_points_of_800_at_5pct: 40`, `ghost_mismatch_nats: 400.0`, `overlap_weight_cost_nats: -13.81`, `matching_pair_advantage_nats: 386.2` and `interior_area_share_margin0.15: 0.7225`. `walk05.py` prints the 0.25 directly and a count near 40; the other six are the hand derivations of steps 2, 3 and 6, checked against your whiteboard. The first table of `phase5_tables` holds the bar's 3-seed numbers, `0.30 ± 0.10` exact and `0.17 ± 0.04` tuned MALA at 5%; your one-seed exact curve should equal the step file's printed run, and your sampler lines should sit below your exact line at 5%. An exact number that differs is either a lesson or a bug in one of the files; find which before the reading stop.

### 5b. You, optional: how Z moves with expert B's blob

**One more number to rebuild, optional.** Moving expert B's blob while A stays in the bottom-right corner changes Z, the product's total mass: `7.9 × 10⁻¹⁰` with B in the opposite corner, `5.3 × 10⁻⁵` next door, and `3.54` on top of A (`artifacts/scenes/three-ways-a-product-reaches-a-and-b/scene_key.json`, under `support.TL.Z`, `support.TR.Z` and `support.BR.Z`, each key named for where B's blob sits). Predict it before you compute it. "Nowhere: the product is zero" is a tempting wrong answer, because the raw product is almost zero everywhere, yet a sampler still returns samples at the origin.

**By hand first.** Two normalised two-dimensional Gaussians with common covariance $\sigma^2I$ multiply to a total mass of

$$
Z=\frac{1}{4\pi\sigma^2}\exp\left(-\frac{\lVert\mu_A-\mu_B\rVert^2}{4\sigma^2}\right).
$$

The rule used is [the integral of a product of two Gaussians](../../../../../../goal-setting/learning/mathematics/textbooks/linear-algebra/matrix-cookbook.pdf#page=42). With σ = 0.15 the prefactor is

$$
\frac{1}{4\pi(0.15)^2}=3.5368.
$$

That is `p5.Z_prefactor_sd0.15`, which `walk_numbers` prints to five decimals as `3.53678`.

B on top of A puts the centres at squared distance 0, next door at 1 and opposite at 2, so

$$
\begin{aligned}
Z_{\text{on top}} &= 3.5368, \\
Z_{\text{next door}} &= 3.5368\,e^{-1/0.09}\approx5.29\times10^{-5}, \\
Z_{\text{opposite}} &= 3.5368\,e^{-2/0.09}\approx7.90\times10^{-10}.
\end{aligned}
$$

The two smaller values are `p5.Z_next_door_by_hand` and `p5.Z_opposite_by_hand`. Then compute it with the answer key's grid (one blob of std 0.15 for each expert, a 441 × 441 grid over [−1.1, 1.1]², Z the sum of the product times the cell area), then compare:

```bash
cd /home-mscluster/mmolefe/PhD/reduce-reuse-recycle
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -c "import json; s = json.load(open('artifacts/scenes/three-ways-a-product-reaches-a-and-b/scene_key.json'))['support']; [print('B at', k, 'Z', f\"{s[k]['Z']:.3g}\", 'origin share', round(s[k]['at_origin'], 3)) for k in ('TL', 'TR', 'BR')]"
```

⏱ you 30 to 60 min · stop: it prints `B at TL Z 7.9e-10 origin share 0.989`, `B at TR Z 5.29e-05 origin share 0.032` and `B at BR Z 3.54 origin share 0.0`, and your three Z values match to the digits shown. The 0.989 at the origin with B opposite is where the normalised product's mass sits, so a sampler that reaches the product returns samples there.

### 6. AI, in the background: the scene

**Scene, already built:** [Three ways a product reaches A-and-B](https://claude.ai/artifact/9eFnG2rWEciPsceMwdem4u). It has 48 presses, one concept per press, and nothing appears before it is taught. Its numbers are checked against `scene_key.py` in the same folder (123 of 123 pass). For your own curves:

```text
/results-scene rebuild/out/phase5/

Starts from: rebuild/out/phase5/ in /home-mscluster/mmolefe/PhD/reduce-reuse-recycle:
overlap-dial.json (per seed and overlap, the exact whole-quadrant and deep masses, the
exact-sample TV floor, and each sampler's masses and TV), finetune.json, dial.log, finetune.log,
and the figures overlap-dial-curves.png, overlap-dial-quadrant-pictures.png, finetune-curves.png
and finetune-pictures.png; the layer 5 --math drip folder and its dispatch.md (step 2's
hand-off), for my notation.
Settled facts:
- python -m rebuild.phase5_tables prints the three tables every number on the scene comes from:
  the quadrant toy over 8 overlaps and 3 seeds, the parts world over 7 overlaps and two background
  variances, and the fine-tune variant over 8 checkpoints plus experts trained from scratch
  (rebuild/phase5_tables.py; step file 05, step 8).
- The quadrant toy: A is the left half, B the top half, A∩B the top-left quadrant; "deep" means
  x1 < −0.15 and x2 > 0.15; 2,000 samples per sampler, 3 data seeds, bars 1 sd over the seeds
  (rebuild/phase5_dial.py; step file 05, step 8).
- The existing figure colours black for exact, orange for tuned MALA, green for UHA and blue for
  reverse diffusion; keep them (step file 05, step 8).
- At 5% the exact deep mass is 0.30 ± 0.10 and tuned MALA 0.17 ± 0.04; at 0% the whole quadrant
  reads 0.20 ± 0.02 and the deep region 0.00; reverse diffusion overshoots at 100%, 0.91 against
  0.76 (phase5_tables).
- At fine-tune step 10,000 the grid counts a spurious blob near (0.75, 0.9) that the samplers
  never reach, so the grid's whole quadrant reads 0.021 while tuned MALA reads 0.291 and both
  samplers' TV is 0.90; label that point rather than read it as a result (REBUILD_WALK.md, brick
  "Fine-tune variant"; finetune.log).
- The scene already built, Three ways a product reaches A-and-B, is on the four-quadrant blob toy
  of the answer key, a different toy with A∩B top-right; keep its numbers off this scene
  (artifacts/scenes/three-ways-a-product-reaches-a-and-b/README.md).
- What the curves answer: the deep A∩B mass rounds to 0.00 at 0% on the quadrant toy, the parts
  world with an empty background gives a ghost instead, pretraining support does not survive 100
  fine-tune steps, and an expert loose where its concept is absent is the one way to A∩B with no
  overlap (step file 05, step 8).
Outputs: one published scene of the overlap-dial curves that I can scrub along the overlap share,
in my notation from dispatch.md.
Target: every number on the scene equals phase5_tables to the digits it prints; I judge.
Hand-off: write the scene's URL into step 6 of
artifacts/drips/simple-distributions-rebuild/ladder/05-the-overlap-dial.md, replacing "(URL lands
here)".
```

added: the folder's contents and the drip folder (from `rebuild/out/phase5/` and step 2) · the three tables, the toy, the seeds and samples, the colours (from `phase5_tables.py`, `phase5_dial.py` and step file 05) · the headline numbers (from `phase5_tables`) · the spurious blob at step 10,000 (from `REBUILD_WALK.md` and `finetune.log`) · the other scene's different toy (from its README) · what the curves answer (from step file 05, step 8) · outputs, target and the URL write-back · from the Codex mathematics review: the 0% deep mass as rounding to 0.00 (it is between 3.6e-7 and 1.0e-6 in `overlap-dial.json`, not zero)
dropped: nothing

Scene: (URL lands here)

⏱ you ~15 min · AI ~30 min, beside the reading stop · stop: the scene is published and its curves match `rebuild/out/phase5/`

**Outcome:** the overlap-dial curves as a scene you can scrub.

**Quiz, already built, for extra practice:** [the diffusion models quiz](https://claude.ai/artifact/MyTdAaVfZW3wiqhbLKg4wu), sections "When neither training set holds A and B" and "The four-quadrant answer key".

### 7. You: close the layer

Run the `/drip-retro` command the build printed at its close (⏱ about 25 minutes). Then paste this:

```text
In the background, dispatch xref-cadence over the files layer 5 of the composition ladder wrote: the
layer 5 --math drip folder under artifacts/drips/ and the retrospective the layer 5 walk closed on.

Settled facts:
- The reference sets to link against are:
  - the Study first line of
    artifacts/drips/simple-distributions-rebuild/ladder/05-the-overlap-dial.md: the vault notes
    When Neither Training Set Holds A And B, What An Expert Learns From Its Training Set and
    Composing Models With A Background under
    ~/Obsidian/Artificial Intelligence/Generative Modeling/Diffusion Models/Composition/; plan
    plans/05-composition-and-the-experiment/plans/03-mcmc-correction.md of the journey
    ~/goal-setting/learning/deep-learning/diffusion-models/06-diffusion-speciation-dynamics/;
    and plan 33-a-mixture-needs-heights.md of the journey
    ~/goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/;
  - the layer 5 rows of the Bricks table ("Overlap dial" and "Fine-tune variant") in
    artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md;
  - claims 2 and 4 of artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, and claims 2 (with 2a to 2c),
    3 and 5 and runs 1 and 2 of artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, which this
    layer rests on.
- ~/Playground/PhD/simple-distributions-walk/ is scratch space; link nothing into it (CLAUDE.md, the
  playground is disposable).
Outputs: links written both ways between those files and the reference sets.
Hand-off: the agent's report, listing every link it wrote, read before the reading stop starts.
```

added: the reference sets, by path (from this layer's Study first line, `REBUILD_WALK.md` and the two idea maps)
dropped: nothing

⏱ you 2 min to paste · AI ~10 min (estimate)

**Skip:** the fine-tune variant's code (keep its result).

---

⬅️ [Layer 4: the parts world](04-the-parts-world.md) · 🏠 [the guide](../PROMPTS.md) · [Reading stop: Bradley](05a-reading-stop-projective-composition.md) ➡️
