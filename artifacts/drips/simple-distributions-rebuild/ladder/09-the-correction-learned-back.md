# ⭐ Layer 9: the correction, learned back

⬅️ [Correction prior art](08a-correction-prior-art.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 10: latent space](10-latent-space-and-vaes.md) ➡️

**Why it is critical.** This is the second half of the goal: a fix you can train. The fair test says whether composition plus a correction needs fewer joint examples than training on the joint directly.

**Depth.** Build: score matching as regression with ∇r_t as its target, and the fair test. You 2.3 to 4.7 h in all (rough), plus about 1 h of training beside you.

**Study first.** [Denoising score matching and its optimum](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/13-the-training-loss-two-ways.md), with [the objective derived three ways](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/20-score-matching-the-training-objective.md). [The conditional mean is the best guess](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/textbook/05-the-conditional-mean-is-the-best-guess.md) for quiz 9.1. Textbooks: Chen 6.3.3 (p126) and Goodfellow et al. 18.5 (p635). Video: Stanford CS236 lecture 13, [29:16 to 44:39](https://www.youtube.com/watch?v=8G-OsDs1RLI&t=1756s).

**Ideas to tie in.** From [the ideas that tie the ladder together](../../../ideas/ideas-that-tie-the-ladder-together.md): [the interaction score as the object a correction can learn](../../../ideas/ideas-that-tie-the-ladder-together.md#5-learn-the-identifiable-object-which-is-the-interaction-score) (needed, 2 to 3 h, no training); [an oracle correction cube for the fair test](../../../ideas/ideas-that-tie-the-ladder-together.md#7-turn-the-evaluation-into-an-oracle-correction-cube) (needed, 4 to 8 h). Each comes with an exact toy check.

**Achieve.** `walk09.py` printing the three arms' total variation at overlap 0.05 and 0.001, with exact and trained experts · on the whiteboard: the regression target ∇r_t at one point of the parts world · problems: the ones quiz 9.2 banks · quizzes 9.1 and 9.2 passed.

## The flow

```text
 AI, in the background                         │  You, in order
───────────────────────────────────────────────┼───────────────────────────────────────────────
 quizzes 9.1 and 9.2, prepared ──────────────▶ │  1. quiz 9.1: least squares, gradients
   (paste during the prior-art read)           │       ▼
                                               │  2. /drip --math: score matching as regression
                                               │       ▼
                                               │  3. quiz 9.2: needed-now right, 80% of the rest
                                               │       ▼
                                               │  4. build: walk09.py (--request)
 builds rebuild/phase9_correction.py ────────▶ │  5. compare: p9. numbers equal yours
                                               │       ▼
 results-scene, beside layer 12 ◀───────────── │  6. scene
                                               │       ▼
 xref-cadence links the layer ◀─────────────── │  7. /drip-retro, then the next layer
```

## The steps

### 0. AI, in the background: the layer 9 reference and quizzes 9.1 and 9.2, one layer ahead

**AI, one layer ahead.** Paste both when you start the prior-art read.

```text
In the background, build the layer 9 reference: rebuild/phase9_correction.py, a small correction
network trained on the exact gradient of r_t from rebuild/phase8_interaction.py, then the fair test
(joint condition, composition, composition plus correction) at overlap 0.05 and 0.001, sampled with
tuned MALA, with exact and with trained experts. First reproduce
artifacts/ideas/when-poe-reaches-a-and-b/runs/sampler-by-expert-interaction/ (MALA +0.001 above the
floor with exact experts and +0.267 with trained ones at r = 0.001). Add the numbers to
rebuild/walk_numbers.py as phase9() with keys p9.*, and run it once with the co3 python.

Task: in /home-mscluster/mmolefe/PhD/reduce-reuse-recycle, write rebuild/correction.py (the
four-blob toy, the trained experts, the correction and its two losses, the scoring) and
rebuild/phase9_correction.py (the run, python -m rebuild.phase9_correction --out
rebuild/out/phase9), the split the brick row names, and add phase9() to rebuild/walk_numbers.py.
Run from the repo root with /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python, on the cards
rebuild.device.pick_device picks.
Starts from: layer 8's reference: rebuild/interaction.py (r_t and ∇r_t in float64 for any four
parts_world.Mixture objects) and the p8. keys in rebuild/out/walk-numbers.json (step 0 of
artifacts/drips/simple-distributions-rebuild/ladder/08-the-interaction-term.md).
Settled facts:
- The brick row: the fair test from the parallel walk, a trained joint {A, B} condition against
  composition alone against composition plus a small correction fine-tuned on the same few A∩B
  samples, at overlap r = 0.05 and 0.001; the correction trained by regression on the exact ∇r_t
  and by denoising score matching with frozen experts; TV of tuned-MALA samples against the exact
  joint; files rebuild/correction.py and rebuild/phase9_correction.py; not built
  (artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, brick "The correction, learned
  back").
- The condition {A, B} trains on only the share r of the data that is A and B, while composition
  reuses all of it; the interaction term is exact in 2D, so a correction fine-tuned on the few A∩B
  samples can be scored against it and against the joint model trained on the same samples
  (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 9).
- The toy is run 7's, which is not layer 5's: four Gaussian blobs at (±0.5, ±0.5) with σ = 0.15,
  top-right weight r and (1 − r)/3 for each other blob; construction (1) with "contains" labels, A
  the two right blobs, B the two top blobs, composed target p(x|A)·p(x|B)/p(x); the exact joint
  p(x|A,B) is the top-right blob alone. Layer 5's quadrant toy in rebuild/phase5_dial.py (uniform
  squares, A left, B top) is a different toy (the run card, "The toy";
  artifacts/ideas/when-poe-reaches-a-and-b/runs/samplers-on-exact-experts/exact_toy.py).
- The four blobs are parts_world.Mixture objects in two dimensions, so rebuild/interaction.py
  applies unchanged (rebuild/parts_world.py; layer 8's reference).
- s_A + s_B − s_∅ + ∇r_t equals the noised joint's score at every level, so the exact correction
  turns the composed target into the joint (the definition of r in REBUILD_WALK.md, brick "The
  interaction term, exact").
- Run 7's cells, TV above the floor, mean ± sd over seeds 0, 1 and 2: at r = 0.05, exact experts
  with reverse diffusion +0.049 ± 0.003 and with annealed MALA −0.009 ± 0.007, trained experts
  +0.054 ± 0.007 and +0.051 ± 0.023; at r = 0.001, exact +0.556 ± 0.005 and +0.001 ± 0.006, trained
  +0.590 ± 0.030 and +0.267 ± 0.056; floors 0.133 and 0.124; answer-key top-right share 0.9889 and
  0.9886 (artifacts/ideas/when-poe-reaches-a-and-b/runs/sampler-by-expert-interaction/README.md,
  "Per cell").
- The errors interact: +0.054 ± 0.016 at r = 0.05 and +0.232 ± 0.046 at r = 0.001, all six seeds
  positive (the run card, "Effects on TV above the floor";
  artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 4 and run 7).
- The trained composed score's relative error, √mean‖s_trained − s_exact‖² / √mean‖s_exact‖² at
  2,000 noised exact composed samples, is 0.34 (r = 0.05) and 1.30 (r = 0.001) at level 0 and
  under 0.04 from level 30 up (the run card, "Score error").
- At r = 0.001 trained MALA reaches the top-right share 0.989 at the wrong place, centre (0.441,
  0.523) and TV 0.391, so the share alone hides the miss; report the share, the TV and the centre
  together (the run card; artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 5).
- Scoring: TV to the exact clean grid density on 60 × 60 cells over [−1.5, 1.5]², samples outside
  the box counted as error, the floor from 2,000 exact samples drawn by rejection, 2,000 samples
  per seed (the run card; exact_toy.total_variation). rebuild.ideal.tv bins 40 × 40 over
  [−1.1, 1.1]², so its numbers do not compare with run 7's.
- Trained experts: three per seed and r, for p(x), p(x|A) and p(x|B); the layer 1 network (4
  residual blocks, hidden 128, time embedding 32, zero-initialised output layers) in the energy
  wrapper, rebuild.model.make_model; the denoising loss with t uniform on 0 to 99; Adam 1e-3, batch
  1,000 drawn from one fixed set of 10,000 samples, 15,000 steps, EMA 0.999 and the EMA weights
  used; generator seed 100k + density index; final losses 0.26 to 0.27 for p(x) and 0.16 for the
  conditionals at r = 0.05, 0.25 and 0.12 at r = 0.001; about 540 s each, nine per card on two RTX
  A6000s (the run card, "Trained experts").
- rebuild.train.train(sample_fn, steps=15001, ...) draws a fresh batch from sample_fn each step, so
  pass a sample_fn that indexes the fixed 10,000-sample set, and steps=15000 (rebuild/train.py).
- The composed sampler reads log q_t = −E/√(1 − ᾱ_t), so the score is −∇E/√(1 − ᾱ_t):
  PortableDiffusionModel.p_energy and p_gradient in rebuild/model.py (the run card, "Trained
  experts").
- Run 7's MALA: 20 steps at each of the 100 levels and a clean step, the step tuned per level by a
  pilot run (seed + 1000) toward 0.6 acceptance and capped at 2v_t, one accept decision per sample
  (runs/samplers-on-exact-experts/mala_fixed.py through run_chain in run_samplers.py); acceptance
  landed at 0.599 to 0.601 in every cell. rebuild.samplers.tuned_mala adapts during the run with
  10 moves per level instead, so port run 7's rule into rebuild/correction.py for the
  reproduction (the run card, "The samplers"; rebuild/samplers.py).
- A trained model has no clean level, so its clean step uses the level-0 networks (ᾱ_0 = 0.99937),
  which moves TV by at most 0.002 (the run card, "The clean step for trained experts").
- Top-right samples per 10,000-sample training set, seeds 0, 1, 2: p(x) 506, 513, 522 at r = 0.05
  and 11, 11, 11 at r = 0.001; p(x|A) 1,408, 1,373, 1,403 and 34, 23, 27; p(x|B) 1,363, 1,380,
  1,381 and 30, 24, 30 (the run card, "Top-right samples in the training sets").
- A correction can make learned scores worse: Feynman-Kac correction improves the fit with exact
  scores (0.64 to 0.08 sliced Wasserstein-2) and worsens it with scores learned from 10,000 samples
  (0.39 with 1 particle to 2.82 with 256) (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md,
  claim 4.1, from Soiffer et al. 2026, Table 1).
- The layer 1 energy wrapper outputs 2x at initialisation, so a correction built on it does not
  start at zero (REBUILD_WALK.md, brick "ResNet score network").
- rebuild/walk_numbers.py holds worked numbers, and run results are read from rebuild/out/phase*/
  (its docstring), so phase9() reads the fair-test TVs from phase9.json rather than retraining.
Outputs:
- rebuild/correction.py: the four-blob toy as Mixture objects, run 7's trained experts and MALA
  ported, the correction network (output zero at initialisation), its two losses, and the scoring;
- rebuild/phase9_correction.py: the reproduction, then the fair test. The A∩B samples every arm
  sees are the top-right samples of the p(x) set, the share r of the data. The arms are the joint
  (the layer 1 network trained on those samples alone); composition, with exact and with trained
  experts; and composition plus a correction, with exact and with trained experts, the correction
  trained once by regression on the exact ∇r_t at noised copies of those samples and once by
  denoising score matching on them with the experts frozen. Every arm is sampled with run 7's MALA
  and scored by TV against the exact joint, with its own 2,000-sample floor, at both r and 3 seeds;
- rebuild/out/phase9/phase9.json: one row per arm, expert kind, correction loss, r and seed, with
  the TV, the TV above the floor, the top-right share, the origin share, the sample centre, the
  MALA acceptance, and every setting (correction size, steps, learning rate, number of A∩B samples)
  beside the numbers;
- rebuild/out/phase9/fair-test.png: TV above the floor (y) for each arm (x), exact and trained
  experts as two lines, one panel per r;
- phase9() in rebuild/walk_numbers.py, added to its __main__ loop: ∇r_t at one fixed point of the
  four-blob toy computed there, and the fair-test TVs read from phase9.json as p9. keys;
- rebuild/out/walk-numbers.json rewritten with the p9. keys added.
Target: each of run 7's eight cells reproduces within 2 sd of the card's mean (exact MALA at
r = 0.001 inside +0.001 ± 0.012, trained MALA inside +0.267 ± 0.112); the identity check s_A + s_B
− s_∅ + ∇r_t − s_joint stays under 1e-10 at every level; the fair-test table is complete at both r,
3 seeds per cell, mean ± sd. The arms' values are the result and carry no target. Every p1. to p8.
value is unchanged. I judge, from the printed table.
Progress: the reproduction prints first, eight cells beside the card's; then the identity check;
then each correction's training loss; then one fair-test line per arm, expert kind, r and seed.
Proof: python -m rebuild.phase9_correction --out rebuild/out/phase9 prints "reproduction: 8 of 8
cells within 2 sd of run 7" and the fair-test table; python -m rebuild.walk_numbers | grep 'p9\.'
prints the keys.
Limits: write nothing under artifacts/; read run 7's code and port it, never import from the runs
folder; change no existing function in rebuild/ and no p1. to p8. value; no commit.
Hand-off: rebuild/out/phase9/phase9.json and the p9. keys, which step 4's walk, step 5's comparison
and step 6's scene read; at the top of phase9_correction.py's docstring, one line per r with the
fair test's answer.
```

added: the function and run split and the run command (from the brick row and the Pins) · claim 9's statement of the fair test (from the `when-poe-reaches-a-and-b` idea map) · which toy, and that it is not layer 5's (from the run card and `exact_toy.py`) · layer 8's `interaction.py` as the source of ∇r_t, in place of `phase8_interaction.py`, which is the parts-world run and not the functions (from layer 8's step 0) · the identity the exact correction meets (from the brick row's definition of r) · run 7's eight cells, the interaction, the score error and the hidden miss (from the run card and claims 4 and 5) · the scoring grid and why `rebuild.ideal.tv` does not compare (from the run card and `ideal.py`) · the training recipe, the fixed-set `sample_fn`, the energy scaling and the clean step (from the run card, `train.py` and `model.py`) · run 7's MALA and why `tuned_mala` differs (from the run card and `samplers.py`) · the training-set counts (from the run card) · the Feynman-Kac warning (from claim 4.1) · the 2x initialisation (from the brick row) · the `walk_numbers` split (from its docstring) · design choices a fresh session would otherwise have to ask about, mine to change: the A∩B samples are the p(x) set's top-right samples, the joint arm is the layer 1 network on them, both corrections run on both expert kinds, and the reproduction tolerance is 2 sd
dropped: nothing

```text
In the background, prepare and build layer 9's two quizzes, bottom up, for the composition ladder
in artifacts/drips/simple-distributions-rebuild/: quiz 9.1 and quiz 9.2.

Task: for each quiz, first dispatch prep-cadence (quiz-prep-pulse --build) with the goal and minutes
below, then dispatch quiz-cadence (mode build, repo
/home-mscluster/mmolefe/PhD/reduce-reuse-recycle). The goal for both is "train a correction and
read the fair test".
- Quiz 9.1, --minutes 40: the college maths under layer 9. Least-squares regression recovers a
  conditional expectation, the gradient of a scalar function.
- Quiz 9.2, --minutes 60: layer 9 itself. Score matching as regression, why the score residual is
  ∇r_t, the three arms of the fair test, why trained experts cost more under a better sampler.
Starts from: layer 8's two quizzes and the layer 8 --math drip folder, which hold the notation for
r(x) and r_t. Nothing from layer 9 has run, and layer 9 has no step file yet. Quiz 9.1 takes its
material from the Study first line of
artifacts/drips/simple-distributions-rebuild/ladder/09-the-correction-learned-back.md; quiz 9.2
takes it from that file's steps 2 and 4, the brick row "The correction, learned back" in
artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, claims 4 and 9 and run 7 of
artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, and run 7's card,
artifacts/ideas/when-poe-reaches-a-and-b/runs/sampler-by-expert-interaction/README.md.
Settled facts:
- quiz-cadence stops when a quiz has no coverage file in .quiz-prep/, so prep-cadence runs first for
  each quiz (quiz-cadence, "The preparation gate").
- The textbooks are Chen 6.3.3 (p126) and Goodfellow et al. 18.5 (p635), with the conditional-mean
  chapter and the training-loss chapter linked on the Study first line, and Stanford CS236 lecture
  13, 29:16 to 44:39, as the timed video (this layer's Study first line).
- pdftotext is not installed on mscluster, so
  ~/.claude/skills/index-source/scripts/find-in-textbooks.py fails; open the PDFs with PyMuPDF from
  the co3 environment (checked on mscluster109: import pymupdf works in co3, version 1.28.2, and
  import pypdf fails everywhere).
- The shelf index is ~/Obsidian/Mathematics/references/textbooks/Textbooks.md and the
  worked-examples index is ~/Obsidian/Mathematics/references/exercises/Worked Examples.md
  (quiz-prep-pulse, "The bank").
- The notation: q(x_t | x_0) = N(√ᾱ_t x_0, (1 − ᾱ_t) I) over T = 100 steps of a cosine schedule;
  ε = −√(1 − ᾱ_t)·∇log q_t; r_t = log q_t(x|A,B) − [log q_t(x|A) + log q_t(x|B) − log q_t(x)],
  whose gradient is what the composed score misses of the joint's score
  (ladder/01-one-diffusion-expert.md, step 2; REBUILD_WALK.md, brick "The interaction term,
  exact").
- The fair test: a joint {A, B} condition trained on only the share r of the data that is A and B,
  against composition, against composition plus a small correction fine-tuned on the same few A∩B
  samples, at r = 0.05 and 0.001 (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 9;
  REBUILD_WALK.md, brick "The correction, learned back").
- A better sampler makes expert error cost more: trained experts add 0.005 TV under reverse
  diffusion and 0.059 under annealed MALA at r = 0.05, and 0.035 and 0.266 at r = 0.001, because
  the trained composed score is worst at low noise (relative error 0.34 and 1.30 at level 0, under
  0.04 from level 30 up), where MALA spends its last steps (the run card, "The answer" and "Score
  error").
- Every number an answer uses comes from run 7's card, the idea map, the p8. keys in
  rebuild/out/walk-numbers.json or a cited textbook page; the p9. keys do not exist yet, because
  the layer 9 reference is built beside these quizzes.
Outputs:
- two coverage files under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, each with a
  verdict per question: ready, blocked, or unprepared by choice;
- rows in the worked-examples index for every banked problem;
- two published quiz pages, each opening on a read-first list with minutes per item, a short path
  and a full path.
Target: no question on either page carries the verdict blocked; quiz 9.2 carries banked problems,
about 6 to 10 per concept at the build depth (score matching as regression with ∇r_t as its
target, the fair test); every number in an answer traces to a source named above. I judge, from
the pages.
Progress: prep-cadence reports, per quiz, its question count, node count and verdicts; quiz-cadence
reports each URL as it publishes.
Proof: the two coverage files exist under .quiz-prep/ and list no question as blocked; the two
URLs open.
Limits: change nothing in the step files' prose, the idea maps, the run card or rebuild/; in the
ladder files, write only the two URLs.
Hand-off: write quiz 9.1's URL into step 1 and quiz 9.2's URL into step 3 of
artifacts/drips/simple-distributions-rebuild/ladder/09-the-correction-learned-back.md, each
replacing "(URL lands here)".
```

added: every field the setup's quiz prompt carries, so this prompt stands alone instead of pointing at "the setup section of PROMPTS.md", which the guide no longer holds (the setup is `ladder/00-setup.md`) · quiz 9.2's material named by path in place of "plan PROMPTS.md (layer 9) with REBUILD_WALK.md", since layer 9's prompts live in this file · the preparation gate (from the quiz-cadence agent definition) · the textbooks and the video (from this layer's Study first line) · the missing `pdftotext` and the PyMuPDF route (from the textbook pass) · the notation (from layer 1 and the brick row) · the fair test and the sampler-by-expert numbers (from claim 9 and the run card) · that the p9. keys do not exist yet · the URL write-back
dropped: nothing

⏱ AI ~2.5 h for both, the training included (estimate), beside the prior-art read

### 1. You: quiz 9.1

Quiz 9.1: (URL lands here)

⏱ you 20 to 40 min · stop: the bar, every *needed now* question right and at least 80% of the rest; below that, read the items your misses point at and retake.

### 2. You: the maths drip

```text
/drip --math score matching as regression, and why the score residual is the gradient of r_t

Starts from: quiz 9.1 passed; any question it marked missed becomes a first rung. The layer 8
--math drip folder under artifacts/drips/ holds my notation for r(x), r_t and their ε-space form.
Settled facts:
- A squared loss is minimised by the conditional mean, so the denoising loss E‖ε_θ(x_t, t) − ε‖²
  is minimised by E[ε | x_t] = −√(1 − ᾱ_t)·∇log q_t(x_t): score matching is a regression (the
  conditional-mean and training-loss chapters on this layer's Study first line; layer 1's
  identity, ladder/01-one-diffusion-expert.md, step 2).
- s_A + s_B − s_∅ + ∇r_t is the noised joint's score at every level, so a network regressed on
  ∇r_t learns exactly what composition misses (REBUILD_WALK.md, brick "The interaction term,
  exact"); in ε-space the target is ε_joint − ε_A − ε_B + ε_∅.
- With the experts frozen and the loss taken on the few A∩B samples, the correction's optimum is
  the gap between the noised density of those samples and the composition, which is ∇r_t only as
  far as the samples cover the joint (the same conditional-mean argument; the brick row's two
  losses, regression on the exact ∇r_t and denoising score matching with frozen experts).
- The worked point: ∇r_t at the fixed image of the parts world is the p8. key the layer 8
  reference printed (rebuild/out/walk-numbers.json).
- The trained composed score's relative error is 1.30 at level 0 for r = 0.001 and under 0.04 from
  level 30 up, so a correction is needed most where the noise is lowest
  (artifacts/ideas/when-poe-reaches-a-and-b/runs/sampler-by-expert-interaction/README.md, "Score
  error").
Outputs: the drip's thread folder under artifacts/drips/, with dispatch.md recording my notation,
the worked numbers I used and what clicked.
Target: I write the regression and its target ∇r_t without notes, and compute ∇r_t at the parts
world's fixed image by hand to the digits the p8. key shows. I judge.
Hand-off: the thread folder's path, which quiz 9.2, step 6's scene and step 7's cross-links read.
```

added: the conditional-mean argument (from this layer's Study first line and layer 1) · the identity and the ε-space target (from the brick row) · what the frozen-expert loss converges to (worked from the same argument and the brick row's two losses) · the worked point from the p8. keys, for the whiteboard item in Achieve · the low-noise score error (from the run card) · layer 8's drip folder as the notation source
dropped: nothing

⏱ rough: you 30 to 60 min · AI ~10 min · stop: you write the regression and its target without notes

**Outcome:** you can write the regression a correction network solves, and say why its target is ∇r_t.

### 3. You: quiz 9.2

Quiz 9.2: (URL lands here)

⏱ you 30 to 60 min, the banked problems included · stop: the bar, then the build. Below the bar, the misses become the first rungs of a second `/drip --math`, and you retake before building.

### 4. You: the build, walk09.py

```text
/drip-execute-plan --request a small correction network trained on the exact gradient of r_t, then
the fair test (joint condition, composition, composition plus correction) at overlap 0.05 and 0.001,
sampled with tuned MALA, with exact and with trained experts, as layer 9 of
artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md; I type every block myself into
~/Playground/PhD/simple-distributions-walk/walk09.py; bar: first reproduce
artifacts/ideas/when-poe-reaches-a-and-b/runs/sampler-by-expert-interaction/ (MALA +0.001 above the
floor with exact experts and +0.267 with trained ones at r = 0.001), then the three arms' total
variation at each overlap

Task: cut the layer into 4 to 6 pieces, write the step file at the open, and build, teach and test
one piece per round. I type every block into walk09.py and run it with
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python from
~/Playground/PhD/simple-distributions-walk/, with
PYTHONPATH=/home-mscluster/mmolefe/PhD/reduce-reuse-recycle. Training runs in the background
beside me.
Starts from: the layer 9 reference from step 0, rebuild/correction.py,
rebuild/phase9_correction.py, rebuild/out/phase9/phase9.json and the p9. keys, whose settings
(correction size, steps, learning rate, the A∩B samples) I use; rebuild/interaction.py from layer
8; quiz 9.2 passed; the layer 9 --math drip folder under artifacts/drips/, whose dispatch.md holds
my notation.
Settled facts:
- The toy is run 7's: four blobs at (±0.5, ±0.5) with σ = 0.15, top-right weight r and (1 − r)/3
  for each other; construction (1), A the right blobs, B the top blobs; the exact joint is the
  top-right blob (the run card, "The toy"). It is not layer 5's quadrant toy.
- Run 7 at r = 0.001, TV above the floor: exact experts with annealed MALA +0.001 ± 0.006, trained
  +0.267 ± 0.056; with reverse diffusion +0.556 ± 0.005 and +0.590 ± 0.030. At r = 0.05: MALA
  −0.009 ± 0.007 and +0.051 ± 0.023, reverse diffusion +0.049 ± 0.003 and +0.054 ± 0.007; floors
  0.133 (r = 0.05) and 0.124 (r = 0.001)
  (artifacts/ideas/when-poe-reaches-a-and-b/runs/sampler-by-expert-interaction/README.md, "Per
  cell").
- Scoring: TV on 60 × 60 cells over [−1.5, 1.5]², samples outside the box counted as error, the
  floor from 2,000 exact samples, 2,000 samples per seed, seeds 0, 1, 2 (the run card).
- Training sets of 10,000 at r = 0.001 hold 11 top-right samples for p(x) and 23 to 34 for each
  conditional; trained MALA there reaches the share 0.989 at the wrong place, centre (0.441, 0.523)
  (the run card).
- The trained experts are the layer 1 network in the energy wrapper, Adam 1e-3, batch 1,000 from a
  fixed 10,000-sample set, 15,000 steps, EMA 0.999, about 540 s each; the composed score is
  −∇E/√(1 − ᾱ_t), and the clean step uses the level-0 networks (the run card).
- MALA is run 7's: 20 steps per level over 100 levels plus a clean step, tuned per level toward 0.6
  acceptance by a pilot run and capped at 2v_t, as ported into rebuild/correction.py (the run card;
  step 0's reference).
- s_A + s_B − s_∅ + ∇r_t is the noised joint's score, so the exact correction makes composition
  sample the joint (REBUILD_WALK.md, brick "The interaction term, exact").
- A correction can amplify learned-score error: Feynman-Kac correction worsens learned scores from
  0.39 to 2.82 sliced Wasserstein-2 as particles go from 1 to 256
  (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 4.1).
- Report the share, the TV and the centre together, never the share alone
  (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 5).
Outputs:
- walk09.py: one script that reproduces run 7's eight cells, trains the correction both ways, and
  prints the fair-test table: TV, TV above the floor, top-right share and centre for every arm,
  expert kind and r, mean ± sd over 3 seeds;
- the step file the walk writes at its open, ticked piece by piece.
Target: the reproduction lands within 2 sd of run 7's card in all eight cells; every fair-test
cell is within 2 sd of the reference's p9. value, using the reference's 3-seed sd. I judge, from
the printed table.
Progress: piece 1 prints the toy's answer-key share, 0.9889 at r = 0.05; piece 2 the reproduction;
piece 3 the exact ∇r_t and the identity check; piece 4 each correction's training loss; pieces 5
and 6 the fair-test table.
Proof: python walk09.py prints "reproduction: 8 of 8 cells within 2 sd of run 7" and the fair-test
table, and step 5's comparison shows every cell inside the reference's spread.
Limits: never edit rebuild/, the idea maps or the run folder; I type, I do not paste; no commit.
Hand-off: the retrospective the walk closes on, and the /drip-retro command it prints.
```

added: the reference's outputs and settings as the starting point (from step 0) · the toy and that it is not layer 5's (from the run card) · run 7's eight cells, the scoring, the training-set counts, the hidden miss, the training recipe and MALA (from the run card) · the identity (from the brick row) · the Feynman-Kac warning and the three-numbers rule (from claims 4.1 and 5) · the tolerance of 2 sd, which is mine
dropped: nothing

⏱ rough: you 1 to 2 h (about 6 pieces) · AI ~35 min, plus about 1 h of training beside you (estimate) · stop: the prompt's bar

**Outcome:** the fair test's answer. It also shows why the correction has to be accurate at low noise: a better sampler makes expert error cost more, because the trained scores are worst at low noise (relative error `1.30`), exactly where MALA spends its last steps.

### 5. You: compare your numbers with the reference

**Compare.** `python -m rebuild.walk_numbers` prints the `p9.` keys; your `walk09.py` prints the same total variations within the sampling noise the reference reports.

```bash
cd /home-mscluster/mmolefe/PhD/reduce-reuse-recycle
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m rebuild.walk_numbers | grep 'p9\.'
grep -A 9 '^\*\*r = ' artifacts/ideas/when-poe-reaches-a-and-b/runs/sampler-by-expert-interaction/README.md
cd ~/Playground/PhD/simple-distributions-walk
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python walk09.py
```

⏱ you ~15 min · stop: your eight reproduction cells sit within 2 sd of the run card's table (exact MALA at r = 0.001 inside `+0.001 ± 0.012`, trained MALA inside `+0.267 ± 0.112`), and each fair-test cell sits within 2 sd of its `p9.` value. A cell outside is either a lesson or a bug in one of the two files; find which before layer 12.

### 6. AI, in the background: the scene

```text
/results-scene the layer 9 fair-test outputs the walk wrote under
~/Playground/PhD/simple-distributions-walk/

Starts from: the fair-test table walk09.py printed (step 4) and the reference's
rebuild/out/phase9/phase9.json and fair-test.png (step 0); the layer 9 --math drip folder and its
dispatch.md for my notation (step 2).
Settled facts:
- The fair test: a joint {A, B} condition trained on only the share r of the data that is A and B,
  against composition, against composition plus a small correction fine-tuned on the same few A∩B
  samples, at r = 0.05 and 0.001, with exact and with trained experts, scored by TV against the
  exact joint (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 9; REBUILD_WALK.md,
  brick "The correction, learned back").
- The toy is four blobs at (±0.5, ±0.5) with σ = 0.15, and the exact joint is the top-right blob;
  at r = 0.001 each training set of 10,000 holds 11 top-right samples for p(x) and 23 to 34 for
  each conditional (run 7's card,
  artifacts/ideas/when-poe-reaches-a-and-b/runs/sampler-by-expert-interaction/README.md).
- Run 7 is the baseline the scene sits beside: trained MALA +0.267 ± 0.056 above the floor at
  r = 0.001 against +0.001 ± 0.006 with exact experts, the interaction +0.232 ± 0.046 (the run
  card).
- TV, the top-right share and the sample centre appear together, because the share alone hid
  trained MALA's miss at r = 0.001 (0.989 share, TV 0.391) (claim 5; the run card).
- ~/Playground/PhD/simple-distributions-walk/ is scratch space, so the scene keeps its own copy of
  every number it shows; every number comes from walk09.py's table, phase9.json or the run card,
  never estimated.
Outputs: one published scene, in my notation from dispatch.md: TV above the floor for each arm at
r = 0.05 and 0.001, exact and trained experts, with the samples behind each cell.
Hand-off: write the scene's URL into step 6 of
artifacts/drips/simple-distributions-rebuild/ladder/09-the-correction-learned-back.md, replacing
"(URL lands here)".
```

added: the reference's outputs beside the walk's, and the drip folder for notation (from steps 0 and 2) · the fair test, the toy and run 7's baseline (from claim 9, the brick row and the run card) · the three-numbers rule (from claim 5) · that the scene keeps its own numbers because the playground is disposable (from CLAUDE.md) · the URL write-back
dropped: nothing

Scene: (URL lands here)

⏱ you ~15 min · AI ~30 min, beside layer 12's gate · stop: the scene is published and its numbers match your walk's

**Outcome:** the fair test as a scene: total variation for each of the three arms at overlap 0.05 and 0.001, with exact and with trained experts.

### 7. You: close the layer

Run the `/drip-retro` command the build printed at its close (⏱ about 25 minutes). Then paste this:

```text
In the background, dispatch xref-cadence over the files layer 9 of the composition ladder wrote: the
layer 9 --math drip folder under artifacts/drips/, the step file the --request walk of step 4
wrote at its open, and the retrospective the layer 9 walk closed on.

Settled facts:
- The reference sets to link against are the textbook chapters, the journey plan and the video on
  the Study first line of
  artifacts/drips/simple-distributions-rebuild/ladder/09-the-correction-learned-back.md; the layer 9
  row of the Bricks table in artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md ("The
  correction, learned back") and the journey plan it links, the score function; claims 4 and 9 and
  run 7 of artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md; and run 7's card,
  artifacts/ideas/when-poe-reaches-a-and-b/runs/sampler-by-expert-interaction/README.md.
- ~/Playground/PhD/simple-distributions-walk/ is scratch space; link nothing into it (CLAUDE.md, the
  playground is disposable).
Outputs: links written both ways between those files and the reference sets.
Hand-off: the agent's report, listing every link it wrote, read before layer 12 starts.
```

added: the reference sets, by path (from this layer's Study first line, `REBUILD_WALK.md`, the `when-poe-reaches-a-and-b` idea map and its run card)
dropped: nothing

⏱ you 2 min to paste · AI ~10 min (estimate)

---

⬅️ [Correction prior art](08a-correction-prior-art.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 10: latent space](10-latent-space-and-vaes.md) ➡️
