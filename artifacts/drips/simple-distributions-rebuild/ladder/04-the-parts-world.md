# ⭐ Layer 4: the parts world

⬅️ [Layer 3: ideal experts](03-ideal-experts-and-the-seven-kinds.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 5: the overlap dial](05-the-overlap-dial.md) ➡️

**Why it is critical.** It is the only place the cat-dog product is exact in more than two dimensions, and layer 8 computes the interaction term inside it.

**Depth.** Build: the product of diagonal Gaussians (the mosaic and the blend), the product of Gaussian mixtures (two animals), and the summed-score error. You 3.3 to 6.7 h in all.

**Study first.** The product of diagonal Gaussians: [Products And Quotients Of Diagonal Gaussians](../../../../../../Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Products%20And%20Quotients%20Of%20Diagonal%20Gaussians.md) and [Multivariate Normal Distribution](../../../../../../Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Multivariate%20Normal%20Distribution.md). The normaliser of a product: [the PoE factorisation and normaliser](../../../../../../goal-setting/learning/deep-learning/diffusion-models/05-poe-composition-diffusion/plans/07-poe-factorisation-and-normaliser.md) and [Why The Normaliser Has No Slope In X](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/Why%20The%20Normaliser%20Has%20No%20Slope%20In%20X.md). The mosaic, blend and two animals: [Kinds Of Composition, mosaic, blend or two animals](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/Kinds%20Of%20Composition.md#mosaic-blend-or-two-animals-from-one-product), the [parts-world write-up](../../../ideas/poe-overlap-dial/example-9-parts-world/example-9-parts-world.tex) and [the parts-world scene](https://claude.ai/artifact/6XzsUMYdEb2eopRyGkpYkh).

**Achieve.** `walk04.py` producing the mosaic, the blend and two animals from one product · on the whiteboard: two 1D Gaussians multiplied, their precisions adding and the mean landing precision-weighted · worked: `numbers.json` in example 9 · problems: the ones quiz 4.2 banks · quizzes 4.1 and 4.2 passed.

## The flow

```text
 AI, in the background                         │  You, in order
───────────────────────────────────────────────┼───────────────────────────────────────────────
 quizzes 4.1 and 4.2, prepared ──────────────▶ │  1. quiz 4.1: products of 1D Gaussians
   (paste during layer 3's build)              │       ▼
                                               │  2. /drip --math: products and mixtures
                                               │       ▼
                                               │  3. quiz 4.2: needed-now right, 80% of the rest
                                               │       ▼
                                               │  4. build: walk04.py, walk04b
 rebuild/ reference, built ──────────────────▶ │  5. compare: numbers.json, phase4.json
                                               │       ▼
 parts-world scene, already built ───────────▶ │  6. scene: operate it
                                               │       ▼
 xref-cadence links the layer ◀─────────────── │  7. /drip-retro, then the next layer
```

## The steps

### 0. AI, in the background: quizzes 4.1 and 4.2

**AI, one layer ahead.** Paste this when you start layer 3's build.

```text
In the background, prepare and build layer 4's two quizzes, bottom up, for the composition ladder
in artifacts/drips/simple-distributions-rebuild/. Goal: "multiply Gaussian and mixture experts
exactly in 16 dimensions". Quiz 4.1, --minutes 40: the product of two 1D Gaussians (precisions
add, the mean is precision-weighted), diagonal covariance, Gaussian mixtures and their weights.
Quiz 4.2, plan 04-the-parts-world.md, --minutes 60: the mosaic and the blend, the mixture-product
weight w_ij ∝ π_i ρ_j N(μ_i; ν_j, Σ_i + Λ_j), the summed-score error at ᾱ = 0.5.

Task: for each quiz, first dispatch prep-cadence (quiz-prep-pulse --build) with the goal, minutes
and topics above, then dispatch quiz-cadence (mode build, repo
/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain). Every path below is from that repo's root, except code paths, which are under toy/.
Starts from: the miss ledger (quiz-ledger) of every earlier quiz on this ladder. Quiz 4.1 takes its
material from artifacts/drips/simple-distributions-rebuild/ladder/04-the-parts-world.md and its
Study first line; quiz 4.2 takes it from
artifacts/drips/simple-distributions-rebuild/04-the-parts-world.md, steps 0 to 8.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- quiz-cadence stops when a quiz has no coverage file in
  artifacts/drips/simple-distributions-rebuild/.quiz-prep/, so prep-cadence runs first for each
  quiz (quiz-cadence, "The preparation gate").
- The step file links a textbook page beside each fundamental it uses (pp. 230, 242, 299 and 307 of
  introduction-to-probability-for-data-science.pdf, pp. 43 and 449 of
  pattern-recognition-and-machine-learning.pdf, p. 42 of matrix-cookbook.pdf, p. 110 of
  algebra-handbook.pdf) and carries three tables of timestamped video segments (the product of two
  Gaussians, the product of two mixtures, total variation distance); reuse them as nodes before
  searching again (step file 04, steps 2, 3, 5, 6 and 7).
- pdftotext is not installed on mscluster, so
  ~/.claude/skills/index-source/scripts/find-in-textbooks.py fails. pypdf does not import in any
  environment there either; open the PDFs with PyMuPDF (import pymupdf, version 1.28.2) from
  /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python (checked on mscluster109, 2026-10-18).
- The shelf index is ~/Obsidian/Mathematics/references/textbooks/Textbooks.md and the
  worked-examples index is ~/Obsidian/Mathematics/references/exercises/Worked Examples.md
  (quiz-prep-pulse, "The bank").
- The notation is the step file's: an image is x ∈ ℝ^16 drawn 4 × 4 row by row, pixels 0 to 7 the
  head and 8 to 15 the body, the cat +1 and the dog −1; an expert is N(m, diag v), tight at
  v = 0.01 and loose at v = 1 (step file 04, steps 1 and 2).
- The product per pixel, inputs N(m_1, diag v_1) and N(m_2, diag v_2): precisions add,
  1/v = 1/v_1 + 1/v_2; the mean is m = v(m_1/v_1 + m_2/v_2); the constant is
  Z = N(m_1; m_2, diag(v_1 + v_2)), derived by completing the square (step file 04, step 3).
- Every answer number traces to numbers.json in
  artifacts/ideas/poe-overlap-dial/example-9-parts-world/ or to rebuild/out/phase4/phase4.json:
  mosaic head mean 0.9802 and log Z −46.47; blend mean 0, sd 0.0707, 10 sds from each expert,
  log Z −1583.4; side-by-side pairs at unnormalised log weight −24.09, same-half pairs at
  −803.21, a gap Δ = 779.12; exactly, w_side = 1/(2(1 + e^−Δ)) is a hair below 0.5 and
  w_same = e^−Δ w_side, which float64 underflows to 0.0, so the printed weights are 0.5 and 0.0;
  at ᾱ = 0.5 the head mean is 0.2326 for the product of the noised experts against 0.6931 for
  the noised product (numbers.json; step file 04, steps 3 to 6).
- Sampler scores use 2,000 exact draws as a finite-sample baseline, not a lower bound: in the
  co-occurrence setting every sampler's weight error (0.009 to 0.014) is below the exact draws'
  0.026 (phase4.json; step file 04, step 7).
- Two published quizzes already hold parts-world questions: the diagonal Gaussian quiz,
  https://claude.ai/artifact/Vmd1FBg7oWwZq1onu9Jofa, section "Products of Gaussian mixtures", and
  the diffusion models quiz, https://claude.ai/artifact/MyTdAaVfZW3wiqhbLKg4wu, section "Mosaic,
  blend or two animals from one product" (overlap-dial idea map, run 7). Read them before writing,
  so the new pages extend them.
- The bar a learner meets: every question marked needed now right and at least 80% of the rest; a
  missed question returns in the next quiz (PROMPTS.md, "The bar for every quiz").
Outputs:
- two coverage files under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, each with a
  verdict per question: ready, blocked, or unprepared by choice;
- rows in the worked-examples index for every banked problem;
- two published quiz pages, each opening on a read-first list with minutes per item, a short path
  and a full path.
Target: no question on either page carries the verdict blocked; quiz 4.2 carries banked problems,
about 6 to 10 per concept for its three build-depth concepts (the diagonal product, the mixture
product, the summed-score error); every number in an answer traces to numbers.json, phase4.json,
the step file or a cited textbook page. I judge, from the pages.
Progress: prep-cadence reports, per quiz, its question count, node count and verdicts; quiz-cadence
reports each URL as it publishes.
Proof: the two coverage files exist under .quiz-prep/ and list no question as blocked; the two URLs
open.
Limits: change nothing in the step file's prose, in rebuild/ or in the example-9 folder; in the
ladder file, write only the two URLs.
Hand-off: write quiz 4.1's URL into step 1 and quiz 4.2's URL into step 3 of
artifacts/drips/simple-distributions-rebuild/ladder/04-the-parts-world.md, each replacing "(URL
lands here)".
```

added: the task's two dispatches and their order, the preparation gate (from the quiz-cadence agent definition) · the step file's textbook pages and video tables as ready nodes (from step file 04) · the missing `pdftotext`, and PyMuPDF in place of pypdf, which does not import on mscluster109 (checked 2026-10-18) · the shelf and worked-examples indexes (from the setup's quiz prompt) · the step file's notation and the product rule (from step file 04) · the reference numbers (from `numbers.json`) · the two quizzes already published (from the overlap-dial idea map, run 7) · the quiz bar (from `PROMPTS.md`) · outputs, target, progress, proof, limits and the URL write-back · the product rule's inputs renamed m_1, v_1 and m_2, v_2 to match step 3, the exact pair weights and their float64 underflow, and the exact draws as a baseline rather than a floor (from the Codex mathematics review)
dropped: "the way the setup section of artifacts/drips/simple-distributions-rebuild/PROMPTS.md does for layer 1" (why: `PROMPTS.md` has no setup section any more; the setup's quiz prompt is step 2 of [the setup](00-setup.md), and this prompt now carries everything it carried) · "weight 0.5 each" as an exact value, and "e^−779 relative" (from the Codex mathematics review: 0.5 and 0.0 are float64 roundings of the exact weights)

⏱ AI ~1.5 h (estimate), beside layer 3's build

### 1. You: quiz 4.1

Quiz 4.1: (URL lands here)

Built by step 0's background prompt. ⏱ you 20 to 40 min · stop: the bar, every *needed now* question right and at least 80% of the rest; below that, read the items your misses point at and retake.

### 2. You: the maths drip

```text
/drip --math the product of two diagonal Gaussians and the product of two Gaussian mixtures, every
pair and its weight, as steps 2 to 5 of
artifacts/drips/simple-distributions-rebuild/04-the-parts-world.md set them out

Starts from: quiz 4.1 passed; any question it marked missed becomes a first rung.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- An expert is N(m, diag v) over 16 pixels, each pixel independent, so its log-density is a sum
  over pixels; the cat expert of example 9(a) is tight on the head (v = 0.01) and loose on the
  body (v = 1), the dog the reverse (step file 04, step 2).
- Per pixel, for inputs N(m_1, diag v_1) and N(m_2, diag v_2), precisions add,
  1/v = 1/v_1 + 1/v_2, and the mean is precision-weighted, m = v(m_1/v_1 + m_2/v_2); the constant
  is Z = N(m_1; m_2, diag(v_1 + v_2)). Step 3 derives all three by adding the log-densities,
  collecting powers of x, completing the square and integrating over x (step file 04, step 3).
- A loose expert (v = 1) still pulls: its score −(x_d − m_d)/v_d is nonzero, which is why the
  mosaic head is 99/101 and not 1; only v_d → ∞ ignores a pixel (step file 04, steps 2 and 3).
- Worked numbers, mosaic: on a head pixel the cat has precision 100 and the dog 1, so the mean is
  (100 − 1)/101 = 0.9802; log Z = −46.47, 2.904 nats a pixel times 16 (step file 04, step 3;
  numbers.json, a).
- Worked numbers, blend: both experts at v = 0.01 give mean 0, variance 0.005 (sd 0.0707), 10 sds
  from each expert, log Z = −1583.4 (step file 04, step 4; numbers.json, b).
- The product of two mixtures is a constant times a mixture over every pair:
  (Σ_i w_i N_i)(Σ_j w'_j N'_j) = Σ_ij w̃_ij N_ij with w̃_ij = w_i w'_j Z_ij; the normaliser is
  Z_mix = Σ_ij w̃_ij and the weights are w*_ij = w̃_ij / Z_mix; the vault notes write the weight
  w_ij ∝ π_i ρ_j N(μ_i; ν_j, Σ_i + Λ_j) (step file 04, step 5; Products And Quotients Of Diagonal
  Gaussians).
- Worked numbers, two animals: cat-left × dog-right and cat-right × dog-left have unnormalised
  log weight −24.09 = log 0.25 + 16 pixels each disagreeing by 1 over variance 1.01; the
  same-half pairs have −803.21 = log 0.25 + 8 pixels disagreeing by 2 over variance 0.02 + 8 loose
  pixels agreeing at 0 over variance 2. The gap is 779.12, so exactly w_side = 1/(2(1 + e^−779.12))
  and w_same = e^−779.12 w_side; float64 underflows e^−779.12 to 0, so they print as 0.5 and 0.0
  (step file 04, step 5; numbers.json, c).
Outputs: the drip's thread folder under artifacts/drips/, with dispatch.md recording my notation,
the worked numbers I used and what clicked.
Target: I derive the precision sum, the precision-weighted mean and the pair weight without notes,
and reproduce 0.9802, −24.09 and −803.21 by hand; I judge.
Hand-off: the thread folder's path, which quiz 4.2, the build in step 4 and step 7's cross-links
read.
```

added: the expert's form, the product rule and the four sets of worked numbers (from step file 04 and `numbers.json`) · quiz 4.1's misses as first rungs · outputs, target and hand-off · the product rule's inputs renamed m_1, v_1 and m_2, v_2 with its completing-the-square derivation, the loose expert's nonzero pull, the mixture product's normaliser Z_mix, the pixelwise sums behind −24.09 and −803.21, and the exact pair weights with their float64 underflow (from the Codex mathematics review)
dropped: "weight 0.5 each" as an exact value, and "e^−779 relative" (from the Codex mathematics review: 0.5 and 0.0 are float64 roundings of the exact weights)

⏱ you 30 to 60 min · AI ~10 min · stop: you derive the Outcome below without notes

**Outcome:** you can derive that precisions add and that the mean is precision-weighted, plus the mixture-product weight w_ij ∝ π_i ρ_j N(μ_i; ν_j, Σ_i + Λ_j).

### 3. You: quiz 4.2

Quiz 4.2: (URL lands here)

⏱ you 30 to 60 min, the banked problems included · stop: the bar, then the build. Below the bar, the misses become the first rungs of a second `/drip --math`, and you retake before building.

### 4. You: the build: walk04.py

```text
/drip-execute-plan artifacts/drips/simple-distributions-rebuild/04-the-parts-world.md, I type every
block myself into ~/Playground/PhD/simple-distributions-walk/walk04.py; bar: the mosaic head 0.980,
the blend at 0, and the side-by-side weights printing 0.5 each in float64 with same-half pairs at
log weight -803.2

Task: walk the step file one step per message in its order, steps 0 to 8. I run walk04.py from
~/Playground/PhD/simple-distributions-walk/ with
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python and
PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy.
Starts from: the setup's folder and environment; quiz 4.2 passed; the layer 4 --math drip folder
under artifacts/drips/, whose dispatch.md holds my notation.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The reference is rebuild/parts_world.py and rebuild/phase4_parts_world.py; they match
  numbers.json to 1e-9 relative, largest relative difference 2.87e-16 (REBUILD_WALK.md, brick
  "Parts world"; rebuild/out/phase4/phase4.json).
- The reference numbers are artifacts/ideas/poe-overlap-dial/example-9-parts-world/numbers.json
  for the four example settings and rebuild/out/phase4/phase4.json for the samplers. python -m
  rebuild.walk_numbers holds 11 p4. keys for the step file's closed-form checks: the cat expert's
  1616-nat gap, Δ = 779.1241 (p4.delta_side_minus_same) and the exact same-half weight 2.14e-339
  (p4.w_same_half_exact) that float64 prints as 0.0. phase4.json's outside_projection_square counts
  0 projected points outside [−1.5, 1.5]² in every run (step file 04, "Where the numbers come
  from"; rebuild/out/walk-numbers.json).
- Step 0 imports cosine_beta_schedule from rebuild.model and pick_device from rebuild.device, with
  T = 100 and D = 16, so walk04.py fails with ModuleNotFoundError without the PYTHONPATH above
  (step file 04, step 0; setup, step 1).
- Pixels 0 to 7 are the head, 8 to 15 the body; LEFT is [0, 1, 4, 5, 8, 9, 12, 13]; the cat is +1
  and the dog −1 (step file 04, step 1).
- The four settings and their numbers: (a) mosaic, head mean 0.9802, body −0.9802, log Z −46.47;
  (b) blend, mean 0, sd 0.0707, log Z −1583.4; (c) two animals, side-by-side pairs at log weight
  −24.09, same-half pairs −803.21, a gap of 779.12, so exactly w_side = 1/(2(1 + e^−779.12)) and
  float64 prints it as 0.5 and the same-half weight as 0.0 after underflow; (d) at ᾱ = 0.5, head
  mean 0.2326 for the product of the noised experts against 0.6931 for the noised product
  (numbers.json; step file 04, steps 3 to 6).
- Step 7 scores each sampler against 100,000 exact draws by projected TV (two half-means, 40 × 40
  bins of [−1.5, 1.5]²; np.histogram2d drops points outside the square, and none fall outside in
  any run behind the step file) and by component-weight error. A separate 2,000 exact draws give
  the finite-sample baseline, not a lower bound: a sampler can score below it by chance (step
  file 04, step 7).
- With 1,000 samples on the mosaic the reference run printed UHA head mean 0.980, projected TV
  0.048; tuned MALA 0.977, 0.059; reverse diffusion 0.922, 0.746; 1,000 exact draws against the
  rest 0.027 (step file 04, step 7).
- With 2,000 samples, phase4.json holds mosaic projected TV 0.013 perfect, 0.732 reverse
  diffusion, 0.200 ULA, 0.191 MALA, 0.036 tuned MALA, 0.036 UHA; tuned MALA accepts 0.606 of its
  moves and plain MALA 0.992 (rebuild/out/phase4/phase4.json).
- Reverse diffusion with summed scores is step 7's broken run: it targets the product of the noised
  experts at every level, so the head mean ends at 0.923 where the product has 0.980 (step file 04,
  step 7; overlap-dial idea map, "What the words are", Du et al. 2023 Section 3).
- Train or sample on a card with under 1 GB in use, else on the CPU (REBUILD_WALK.md, Pins).
Outputs:
- walk04.py: one script that runs end to end and prints each step's worked number;
- step03-mosaic.png in ~/Playground/PhD/simple-distributions-walk/, the cat, the dog and the mosaic
  as tiles;
- the step file's steps 0 to 8 ticked, and the walk map that holds my place between sittings.
Target: step 8's seven lines agree with numbers.json to at least 9 significant figures, which
covers the bar's 0.980, 0 for the blend's mean, 0.5 and −803.2; step 7's tuned MALA and UHA land
near the 0.027 exact-draw baseline and reverse diffusion far above it. I judge, from the printed
lines.
Progress: step 1 prints the head and left pixel lists, step 2 the cat expert's log-density gap
beside scipy's, step 3 the mosaic line, step 4 the blend line, step 5 the four pair lines, step 6
the ᾱ = 0.5 line, step 7 one line per sampler, step 8 the seven comparisons.
Proof: python walk04.py prints, in step 8, seven "yours ... numbers.json ..." lines with equal
values to 9 significant figures, and the comparison in step 5 of this layer shows no differing
number.
Limits: never edit rebuild/, the example-9 folder or the step file's code blocks; I type, I do not
paste; no commit.
Hand-off: the retrospective the walk closes on, and the /drip-retro command it prints.
```

added: where the reference lives and how closely it matches (from `REBUILD_WALK.md` and `phase4.json`) · the missing `p4.` keys and the two reference files (from step file 04 and `walk-numbers.json`) · the imports and the PYTHONPATH (from step file 04 and the setup) · the pixel layout, the four settings' numbers, the sampler scoring and both sampler runs (from step file 04, `numbers.json` and `phase4.json`) · why reverse diffusion breaks (from step file 04 and the overlap-dial idea map) · the GPU guard (from `REBUILD_WALK.md`) · outputs, step 8's 9-figure check as the target, progress, proof, limits and hand-off · the exact pair weights and their float64 underflow, the histogram's dropped out-of-range points (none in any run), and the exact draws as a baseline rather than a floor (from the Codex mathematics review) · the 11 p4. keys in place of "has no p4. keys", and the outside-the-square count in phase4.json (from the port of the review's checks into walk_numbers and phase4.json)
dropped: "co-occurrence weights 0.5 and 0.5" and "weight 0.5" as exact values, and "the floor" for the exact draws (from the Codex mathematics review)

⏱ you 1 to 2 h (9 steps) · AI ~25 min · stop: the prompt's bar

**Outcome:** the mosaic, the blend and two animals side by side, all from one product.

### 4b. You: the second build: the summed-score error in walk04b.py

```text
/drip-execute-plan --request setting (d) of
artifacts/ideas/poe-overlap-dial/example-9-parts-world/parts_world.py by hand in
~/Playground/PhD/simple-distributions-walk/walk04b.py; bar: summed scores at ᾱ = 0.5 put the head at
0.233 where the noised product puts it at 0.693, against numbers.json

Task: cut the request into 4 to 6 pieces at the open and build one piece a round, each ending on a
test I run by hand. Settled at the invocation: where to build is walk04b.py above, run from
~/Playground/PhD/simple-distributions-walk/ with
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python; the bar is the one above; the time box is
2 hours. The plan file the open writes goes in artifacts/drips/simple-distributions-rebuild/ beside
04-the-parts-world.md.
Starts from: walk04.py from step 4, finished, and its retrospective; the layer 4 --math drip
folder.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- Setting (d) is lines 43 to 46 of parts_world.py: setting (a)'s two experts, each noised to
  ᾱ = 0.5, multiplied, and the head mean of that product set beside √0.5 times the clean
  product's head mean (parts_world.py).
- Noising a diagonal Gaussian to ᾱ gives N(√ᾱ m, ᾱ v + 1 − ᾱ); a mixture noises each component
  and keeps its weights (step file 04, step 6).
- numbers.json holds head_mean_product_of_noised 0.23257000444341597 and head_mean_noised_product
  0.6931046667076061 at abar 0.5 (numbers.json, d).
- Step 6 of the step file already prints both numbers inside walk04.py; this walk rebuilds them in
  a file of its own from parts_world.py, without walk04.py's code open (step file 04, step 6).
- Summing the noised experts' scores targets the product of the noised experts, whose tight
  variances were blurred toward 1 before the product could use them, so the summed score points
  the head at a third of where it should be (step file 04, step 6; Du et al. 2023, p. 4,
  eqs. 11 and 12, as the overlap-dial idea map's Sources cite).
- Reverse diffusion accumulates this error over 100 steps: head mean 0.923 against 0.980, projected
  TV 0.732 against the exact-draw baseline of 0.013 (step file 04, step 7; phase4.json).
Outputs:
- walk04b.py, printing the two head means beside numbers.json's values;
- the plan file the open writes, its pieces ticked;
- the walk map, and the retrospective the walk closes on.
Target: walk04b.py prints 0.2326 and 0.6931, each equal to numbers.json's d values to at least 9
significant figures. I judge, from the printed line.
Progress: each round ends on a test that prints one number I can check against the step file or
numbers.json.
Proof: python walk04b.py prints both head means beside numbers.json's, equal to 9 significant
figures.
Limits: never edit parts_world.py, numbers.json or rebuild/; I type, I do not paste; no commit.
Hand-off: the retrospective and the /drip-retro command it prints.
```

added: the decisions the open would ask, settled (where, the bar, the 2-hour box from the ⏱ line, where the plan goes) · what setting (d) computes, the noising rule and the two reference values to full precision (from `parts_world.py`, step file 04 and `numbers.json`) · that step 6 of the step file already prints both numbers · why the summed score misses and what it costs reverse diffusion (from step file 04, `phase4.json` and the overlap-dial idea map) · outputs, target, progress, proof, limits and hand-off
dropped: "a floor of 0.013", now the exact-draw baseline (from the Codex mathematics review)

⏱ you 1 to 2 h (one build concept) · AI ~35 min · stop: the prompt's bar

**Outcome:** Du et al.'s summed-score error, computed exactly in the parts world.

### 5. You: compare your numbers with the reference

**Compare.** `walk_numbers` has no phase 4 keys, so the reference numbers are `artifacts/ideas/poe-overlap-dial/example-9-parts-world/numbers.json` and `rebuild/out/phase4/phase4.json`; your `walk04.py` prints the same numbers.

```bash
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m json.tool artifacts/ideas/poe-overlap-dial/example-9-parts-world/numbers.json
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -c "import json; d = json.load(open('rebuild/out/phase4/phase4.json'))['settings']; [print(s, '|', c, '| TV', round(x['tv'], 3), '| weight error', round(x['weight_error'], 3), '| half-means', [round(m, 3) for m in x['half_means']]) for s in d for c, x in d[s]['samplers'].items()]"
cd ~/Playground/PhD/simple-distributions-walk
export PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python walk04.py
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python walk04b.py
```

⏱ you ~10 min · stop: every closed-form number your two scripts print equals `numbers.json` to 9 significant figures: `a.head_mean` 0.98019801980, `a.logZ` −46.465787495, `b.logZ` −1583.4068325, `b.product_sd` 0.070710678119, the side-by-side weight 0.5 (its float64 value: exactly it is 1/(2(1 + e^−779.12)), and the same-half weights underflow to 0.0), the same-half `logw_unnorm` −803.21380759, and `d` 0.23257000444 and 0.69310466671. The sampler lines are random draws, so read them for order rather than exact digits. In `phase4.json`, tuned MALA and UHA both have projected TV near 0.036, against the exact-draw baseline of 0.013. In the 1,000-sample run, tuned MALA is 0.059 and UHA is 0.048. Reverse diffusion is far above both at 0.732 in `phase4.json`. A closed-form number that differs is either a lesson or a bug in one of the files; find which before layer 5.

### 6. You: the scene, already built

**Scene, already built:** [the parts-world scene](https://claude.ai/artifact/6XzsUMYdEb2eopRyGkpYkh). **Quiz, already built, for extra practice:** [the diagonal Gaussian quiz](https://claude.ai/artifact/Vmd1FBg7oWwZq1onu9Jofa), section "Products of Gaussian mixtures".

⏱ you ~20 min to operate it (estimate, as for layer 1's scene) · stop: you can set a slider on each expert's variance and say before you look whether the head follows the cat, lands at the midpoint, or splits into two animals.

### 7. You: close the layer

Run the `/drip-retro` command each build printed at its close, walk04 first, then walk04b (⏱ about 25 minutes each). Then paste this:

```text
In the background, dispatch xref-cadence over the files layer 4 of the composition ladder wrote: the
layer 4 --math drip folder under artifacts/drips/, the plan file the walk04b --request walk wrote
in artifacts/drips/simple-distributions-rebuild/, and the retrospectives the two layer 4 walks
closed on.

Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The reference sets to link against are:
  - the Study first line of
    artifacts/drips/simple-distributions-rebuild/ladder/04-the-parts-world.md: the vault notes
    Products And Quotients Of Diagonal Gaussians and Multivariate Normal Distribution under
    ~/Obsidian/Mathematics/Statistics/Distributions/Multivariate Normal Distribution/, the vault
    notes Why The Normaliser Has No Slope In X and Kinds Of Composition under
    ~/Obsidian/Artificial Intelligence/Generative Modeling/Diffusion Models/Composition/, the
    journey plan 07-poe-factorisation-and-normaliser.md under
    ~/goal-setting/learning/deep-learning/diffusion-models/05-poe-composition-diffusion/plans/,
    and the write-up
    artifacts/ideas/poe-overlap-dial/example-9-parts-world/example-9-parts-world.tex;
  - the layer 4 row of the Bricks table ("Parts world") in
    artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md;
  - claims 7 and 9 and run 7 of artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, which this layer
    rests on.
- ~/Playground/PhD/simple-distributions-walk/ is scratch space; link nothing into it (CLAUDE.md, the
  playground is disposable).
Outputs: links written both ways between those files and the reference sets.
Hand-off: the agent's report, listing every link it wrote, read before layer 5 starts.
```

added: the reference sets, by path (from this layer's Study first line, `REBUILD_WALK.md` and the overlap-dial idea map)
dropped: nothing

⏱ you 2 min to paste · AI ~10 min (estimate)

---

⬅️ [Layer 3: ideal experts](03-ideal-experts-and-the-seven-kinds.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 5: the overlap dial](05-the-overlap-dial.md) ➡️
