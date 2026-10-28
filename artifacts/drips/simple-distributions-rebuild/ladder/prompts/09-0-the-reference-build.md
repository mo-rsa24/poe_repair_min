# Layer 9, step 0: the reference build

⬅️ [Layer 9, step 0](../09-the-correction-learned-back.md#0-ai-in-the-background-the-layer-9-reference-and-quizzes-91-and-92-one-layer-ahead)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
when you start the prior-art read, beside the quiz prompt for this step.

```text
In the background, build the layer 9 reference: rebuild/phase9_correction.py, a small correction
network trained on the exact gradient of r_t, then the fair test (joint condition, composition,
composition plus correction) at overlap 0.05 and 0.001, sampled with tuned MALA, with exact and
with trained experts.

Background
The composition ladder in artifacts/drips/simple-distributions-rebuild/ walks from one 2D diffusion
expert to a correction for product-of-experts composition, and toy/rebuild/ in this repo is
Claude's reference code for every layer. Layer 9 is the second half of the goal: a fix that can be
trained, and a fair test of whether composition plus a correction needs fewer joint examples than
training on the joint directly. This reference is built one layer ahead, during the prior-art read,
so my walk in step 4 has numbers to match. Its brick row in REBUILD_WALK.md says "not built".

Objectives
1. Reproduce artifacts/ideas/when-poe-reaches-a-and-b/runs/sampler-by-expert-interaction/ (run 7)
   first, whose MALA cells sit +0.001 above the floor with exact experts and +0.267 with trained
   ones at r = 0.001.
2. In /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy, write
   rebuild/correction.py: the four-blob toy as Mixture objects, run 7's trained experts and MALA
   ported, the correction network with output zero at initialisation, its two losses, and the
   scoring.
3. Write rebuild/phase9_correction.py, run as python -m rebuild.phase9_correction --out
   rebuild/out/phase9, which runs the reproduction and then the fair test.
4. In the fair test, give every arm the same A∩B samples: the top-right samples of the p(x) set,
   the share r of the data.
5. Run four arms: the joint (the layer 1 network trained on those samples alone); composition with
   exact and with trained experts; and composition plus a correction with exact and with trained
   experts.
6. Train the correction twice: once by regression on the exact ∇r_t at noised copies of those
   samples, and once by denoising score matching on them with the experts frozen.
7. Sample every arm with run 7's MALA and score it by total variation against the exact joint, with
   its own 2,000-sample floor, at both r and 3 seeds.
8. Add phase9() to rebuild/walk_numbers.py and its __main__ loop: ∇r_t at one fixed point of the
   four-blob toy computed there, and the fair-test total variations read from phase9.json as p9.
   keys.
9. Run python -m rebuild.walk_numbers once with the co3 python, so rebuild/out/walk-numbers.json is
   rewritten with the p9. keys added.

What to read
- toy/rebuild/interaction.py: r_t and ∇r_t in float64 for any four parts_world.Mixture objects,
  layer 8's reference.
- toy/rebuild/out/walk-numbers.json, the p8. keys: layer 8's numbers.
- artifacts/drips/simple-distributions-rebuild/ladder/08-the-interaction-term.md, step 0: how
  layer 8's reference was built.
- artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, brick "The correction, learned
  back": the brick row this reference fills.
- artifacts/ideas/when-poe-reaches-a-and-b/runs/sampler-by-expert-interaction/README.md: run 7's
  card, with every number to reproduce.
- artifacts/ideas/when-poe-reaches-a-and-b/runs/samplers-on-exact-experts/exact_toy.py,
  mala_fixed.py and run_samplers.py: run 7's toy and MALA, to port.
- artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claims 4, 4.1, 5 and 9.
- toy/rebuild/train.py, toy/rebuild/model.py, toy/rebuild/samplers.py, toy/rebuild/parts_world.py
  and toy/rebuild/device.py: the functions to call.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- Run from toy/ with /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python, on the cards
  rebuild.device.pick_device picks (layer 9, step 0).
- The brick row is the fair test from the parallel walk: a trained joint {A, B} condition against
  composition alone against composition plus a small correction fine-tuned on the same few A∩B
  samples, at overlap r = 0.05 and 0.001 (REBUILD_WALK.md, brick "The correction, learned back").
- The brick row trains the correction by regression on the exact ∇r_t and by denoising score
  matching with frozen experts, scores TV of tuned-MALA samples against the exact joint, names the
  files rebuild/correction.py and rebuild/phase9_correction.py, and is marked not built
  (REBUILD_WALK.md, brick "The correction, learned back").
- The condition {A, B} trains on only the share r of the data that is A and B, while composition
  reuses all of it (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 9).
- The interaction term is exact in 2D, so a correction fine-tuned on the few A∩B samples can be
  scored against it and against the joint model trained on the same samples
  (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 9).
- The toy is run 7's, which is not layer 5's: four Gaussian blobs at (±0.5, ±0.5) with σ = 0.15,
  top-right weight r and (1 − r)/3 for each other blob (the run card, "The toy"; exact_toy.py).
- It uses construction (1) with "contains" labels, A the two right blobs, B the two top blobs,
  composed target p(x|A)·p(x|B)/p(x), and the exact joint p(x|A,B) is the top-right blob alone (the
  run card, "The toy"; exact_toy.py).
- Layer 5's quadrant toy in rebuild/phase5_dial.py (uniform squares, A left, B top) is a different
  toy (the run card, "The toy").
- The four blobs are parts_world.Mixture objects in two dimensions, so rebuild/interaction.py
  applies unchanged (rebuild/parts_world.py; layer 8's reference).
- s_A + s_B − s_∅ + ∇r_t equals the noised joint's score at every level, so the exact correction
  turns the composed target into the joint (the definition of r in REBUILD_WALK.md, brick "The
  interaction term, exact").
- Run 7's cells give TV above the floor as mean ± sd over seeds 0, 1 and 2 (the run card, "Per
  cell").
- At r = 0.05, exact experts give +0.049 ± 0.003 with reverse diffusion and −0.009 ± 0.007 with
  annealed MALA, and trained experts +0.054 ± 0.007 and +0.051 ± 0.023 (the run card, "Per cell").
- At r = 0.001, exact experts give +0.556 ± 0.005 and +0.001 ± 0.006, and trained experts +0.590 ±
  0.030 and +0.267 ± 0.056 (the run card, "Per cell").
- The floors are 0.133 and 0.124, and the answer-key top-right shares 0.9889 and 0.9886, at r = 0.05
  and 0.001 (the run card, "Per cell").
- The errors interact: +0.054 ± 0.016 at r = 0.05 and +0.232 ± 0.046 at r = 0.001, all six seeds
  positive (the run card, "Effects on TV above the floor"; IDEA_MAP.md, claim 4 and run 7).
- The trained composed score's relative error, √mean‖s_trained − s_exact‖² / √mean‖s_exact‖² at
  2,000 noised exact composed samples, is 0.34 (r = 0.05) and 1.30 (r = 0.001) at level 0, and
  under 0.04 from level 30 up (the run card, "Score error").
- At r = 0.001 trained MALA reaches the top-right share 0.989 at the wrong place, centre (0.441,
  0.523) and TV 0.391, so the share alone hides the miss (the run card; IDEA_MAP.md, claim 5).
- Report the share, the TV and the centre together (IDEA_MAP.md, claim 5).
- Scoring is TV to the exact clean grid density on 60 × 60 cells over [−1.5, 1.5]², with samples
  outside the box counted as error (the run card; exact_toy.total_variation).
- The floor comes from 2,000 exact samples drawn by rejection, with 2,000 samples per seed (the run
  card; exact_toy.total_variation).
- rebuild.ideal.tv bins 40 × 40 over [−1.1, 1.1]², so its numbers do not compare with run 7's (the
  run card; ideal.py).
- Run 7 trains three experts per seed and r, for p(x), p(x|A) and p(x|B), each the layer 1 network
  (4 residual blocks, hidden 128, time embedding 32, zero-initialised output layers) in the energy
  wrapper, rebuild.model.make_model (the run card, "Trained experts").
- Each uses the denoising loss with t uniform on 0 to 99, Adam 1e-3, batch 1,000 drawn from one
  fixed set of 10,000 samples, 15,000 steps, EMA 0.999 with the EMA weights used, and generator
  seed 100k + density index (the run card, "Trained experts").
- Final losses are 0.26 to 0.27 for p(x) and 0.16 for the conditionals at r = 0.05, and 0.25 and
  0.12 at r = 0.001 (the run card, "Trained experts").
- Each expert takes about 540 s, nine per card on two RTX A6000s (the run card, "Trained experts").
- rebuild.train.train(sample_fn, steps=15001, ...) draws a fresh batch from sample_fn each step, so
  pass a sample_fn that indexes the fixed 10,000-sample set, and steps=15000 (rebuild/train.py).
- The composed sampler reads log q_t = −E/√(1 − ᾱ_t), so the score is −∇E/√(1 − ᾱ_t), through
  PortableDiffusionModel.p_energy and p_gradient in rebuild/model.py (the run card, "Trained
  experts").
- Run 7's MALA takes 20 steps at each of the 100 levels and a clean step, the step tuned per level
  by a pilot run (seed + 1000) toward 0.6 acceptance and capped at 2v_t, with one accept decision
  per sample (runs/samplers-on-exact-experts/mala_fixed.py through run_chain in run_samplers.py).
- Acceptance landed at 0.599 to 0.601 in every cell (the run card, "The samplers").
- rebuild.samplers.tuned_mala adapts during the run with 10 moves per level instead, so port run 7's
  rule into rebuild/correction.py for the reproduction (the run card, "The samplers";
  rebuild/samplers.py).
- A trained model has no clean level, so its clean step uses the level-0 networks (ᾱ_0 = 0.99937),
  which moves TV by at most 0.002 (the run card, "The clean step for trained experts").
- Top-right samples per 10,000-sample training set, seeds 0, 1, 2, are 506, 513, 522 for p(x) at
  r = 0.05 and 11, 11, 11 at r = 0.001 (the run card, "Top-right samples in the training sets").
- For p(x|A) they are 1,408, 1,373, 1,403 at r = 0.05 and 34, 23, 27 at r = 0.001 (the run card).
- For p(x|B) they are 1,363, 1,380, 1,381 at r = 0.05 and 30, 24, 30 at r = 0.001 (the run card).
- A correction can make learned scores worse: Feynman-Kac correction improves the fit with exact
  scores (0.64 to 0.08 sliced Wasserstein-2) and worsens it with scores learned from 10,000 samples
  (0.39 with 1 particle to 2.82 with 256) (IDEA_MAP.md, claim 4.1, from Soiffer et al. 2026,
  Table 1).
- The layer 1 energy wrapper outputs 2x at initialisation, so a correction built on it does not
  start at zero (REBUILD_WALK.md, brick "ResNet score network").
- rebuild/walk_numbers.py holds worked numbers, and run results are read from rebuild/out/phase*/,
  so phase9() reads the fair-test TVs from phase9.json rather than retraining (its docstring).
- The A∩B samples, the joint arm's network, both corrections on both expert kinds, and the 2 sd
  reproduction tolerance are design choices of mine, open to change (layer 9, step 0).

Rules
- Write nothing under artifacts/.
- Read run 7's code and port it; never import from the runs folder.
- Change no existing function in rebuild/ and no p1. to p8. value.
- No commit.
- Print the reproduction first, eight cells beside the card's; then the identity check; then each
  correction's training loss; then one fair-test line per arm, expert kind, r and seed.

What to hand back
- rebuild/correction.py and rebuild/phase9_correction.py.
- rebuild/out/phase9/phase9.json: one row per arm, expert kind, correction loss, r and seed, with
  the TV, the TV above the floor, the top-right share, the origin share, the sample centre, the MALA
  acceptance, and every setting (correction size, steps, learning rate, number of A∩B samples)
  beside the numbers.
- rebuild/out/phase9/fair-test.png: TV above the floor (y) for each arm (x), exact and trained
  experts as two lines, one panel per r.
- phase9() in rebuild/walk_numbers.py, and rebuild/out/walk-numbers.json rewritten with the p9. keys.
- At the top of phase9_correction.py's docstring, one line per r with the fair test's answer.
- phase9.json and the p9. keys are what step 4's walk, step 5's comparison and step 6's scene read.
- The proof: python -m rebuild.phase9_correction --out rebuild/out/phase9 prints "reproduction: 8 of
  8 cells within 2 sd of run 7" and the fair-test table, and python -m rebuild.walk_numbers | grep
  'p9\.' prints the keys.

What you will see
Along the way: the reproduction's eight cells beside the card's, then the identity check, then each
correction's training loss, then the fair-test lines.
At the end, a closing report: what was asked, in one sentence; what was done, with each file's path;
PASS or MISS against each of run 7's eight cells within 2 sd of the card's mean (exact MALA at
r = 0.001 inside +0.001 ± 0.012, trained MALA inside +0.267 ± 0.112), the identity check s_A + s_B −
s_∅ + ∇r_t − s_joint under 1e-10 at every level, the fair-test table complete at both r with 3
seeds per cell as mean ± sd, and every p1. to p8. value unchanged (the arms' values are the result
and carry no target); what is still open; at most one question for me.

I judge, from the printed table: done when 8 of 8 cells reproduce within 2 sd, the identity check stays under 1e-10, the fair-test table is complete and no p1. to p8. value moved.
```

added: the seven sections, from the field-labelled version · the ladder's purpose and why this runs one layer ahead, as Background (from the layer's Why line and step 0's heading) · the function and run split and the run command (from the brick row and the Pins) · layer 8's `interaction.py` as the source of ∇r_t, in place of `phase8_interaction.py`, which is the parts-world run and not the functions (from layer 8's step 0) · design choices a fresh session would otherwise ask about, mine to change: the A∩B samples are the p(x) set's top-right samples, the joint arm is the layer 1 network on them, both corrections run on both expert kinds, and the reproduction tolerance is 2 sd
dropped: nothing
