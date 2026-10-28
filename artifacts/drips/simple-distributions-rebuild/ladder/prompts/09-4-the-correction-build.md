# Layer 9, step 4: the correction build

⬅️ [Layer 9, step 4](../09-the-correction-learned-back.md#4-you-the-build-walk09py)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
after quiz 9.2 is passed.

```text
/drip-execute-plan --request a small correction network trained on the exact gradient of r_t, then
the fair test (joint condition, composition, composition plus correction) at overlap 0.05 and 0.001,
sampled with tuned MALA, with exact and with trained experts, as layer 9 of
artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md; I type every block myself into
~/Playground/PhD/simple-distributions-walk/walk09.py; bar: first reproduce
artifacts/ideas/when-poe-reaches-a-and-b/runs/sampler-by-expert-interaction/ (MALA +0.001 above the
floor with exact experts and +0.267 with trained ones at r = 0.001), then the three arms' total
variation at each overlap

Background
This is layer 9's build in the composition ladder (artifacts/drips/simple-distributions-rebuild/):
I type a correction network and the fair test from an empty file, and the fair test says whether
composition plus a correction needs fewer joint examples than training on the joint directly.
Claude's reference is already built in toy/rebuild/ (step 0 of this layer), and my file and the
reference print the same numbers within sampling noise; a difference is a lesson or a bug. I have
passed quiz 9.2, and the layer 9 maths drip folder under artifacts/drips/ holds my notation in its
dispatch.md. Training runs in the background beside me.

Objectives
1. Cut the layer into 4 to 6 pieces, write the step file at the open, and build, teach and test one
   piece per round.
2. I type every block into walk09.py and run it from ~/Playground/PhD/simple-distributions-walk/
   with /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python and
   PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy.
3. Reproduce run 7's eight cells first.
4. Train the correction both ways: by regression on the exact ∇r_t, and by denoising score matching
   with the experts frozen.
5. Print the fair-test table: TV, TV above the floor, top-right share and centre for every arm,
   expert kind and r, mean ± sd over 3 seeds.
6. Tick each piece in the step file as it lands.

What to read
- toy/rebuild/correction.py, toy/rebuild/phase9_correction.py, toy/rebuild/out/phase9/phase9.json
  and the p9. keys in toy/rebuild/out/walk-numbers.json: the layer 9 reference, whose settings
  (correction size, steps, learning rate, the A∩B samples) I use.
- toy/rebuild/interaction.py: r_t and ∇r_t from layer 8.
- artifacts/ideas/when-poe-reaches-a-and-b/runs/sampler-by-expert-interaction/README.md: run 7's
  card.
- artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, layer 9's brick row.
- The layer 9 --math drip folder under artifacts/drips/ and its dispatch.md: my notation.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The toy is run 7's: four blobs at (±0.5, ±0.5) with σ = 0.15, top-right weight r and (1 − r)/3
  for each other (the run card, "The toy").
- It uses construction (1), A the right blobs, B the top blobs, and the exact joint is the
  top-right blob; it is not layer 5's quadrant toy (the run card, "The toy").
- Run 7 at r = 0.001, TV above the floor: exact experts with annealed MALA +0.001 ± 0.006, trained
  +0.267 ± 0.056; with reverse diffusion +0.556 ± 0.005 and +0.590 ± 0.030 (the run card, "Per
  cell").
- Run 7 at r = 0.05: MALA −0.009 ± 0.007 and +0.051 ± 0.023, reverse diffusion +0.049 ± 0.003 and
  +0.054 ± 0.007 (the run card, "Per cell").
- The floors are 0.133 at r = 0.05 and 0.124 at r = 0.001 (the run card, "Per cell").
- Scoring is TV on 60 × 60 cells over [−1.5, 1.5]², with samples outside the box counted as error,
  the floor from 2,000 exact samples, 2,000 samples per seed, seeds 0, 1, 2 (the run card).
- Training sets of 10,000 at r = 0.001 hold 11 top-right samples for p(x) and 23 to 34 for each
  conditional (the run card).
- Trained MALA at r = 0.001 reaches the share 0.989 at the wrong place, centre (0.441, 0.523) (the
  run card).
- The trained experts are the layer 1 network in the energy wrapper, Adam 1e-3, batch 1,000 from a
  fixed 10,000-sample set, 15,000 steps, EMA 0.999, about 540 s each (the run card).
- The composed score is −∇E/√(1 − ᾱ_t), and the clean step uses the level-0 networks (the run card).
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
- The tolerance of 2 sd is my choice (layer 9, step 4).

Rules
- I type; nothing is pasted into walk09.py.
- Never edit rebuild/, the idea maps or the run folder.
- No commit.
- Piece 1 prints the toy's answer-key share, 0.9889 at r = 0.05; piece 2 the reproduction; piece 3
  the exact ∇r_t and the identity check; piece 4 each correction's training loss; pieces 5 and 6
  the fair-test table. Show each before moving on.

What to hand back
- walk09.py, one script that reproduces run 7's eight cells, trains the correction both ways, and
  prints the fair-test table.
- The step file the walk writes at its open, ticked piece by piece.
- The retrospective the walk closes on, and the /drip-retro command it prints.
- The proof: python walk09.py prints "reproduction: 8 of 8 cells within 2 sd of run 7" and the
  fair-test table, and step 5's comparison shows every cell inside the reference's spread.

What you will see
Along the way: one piece per round, each ending on the number it printed and a yes to move on.
At the end, a closing report: what was asked, in one sentence; what was done, with walk09.py's path
and the step file's; PASS or MISS against the reproduction landing within 2 sd of run 7's card in
all eight cells and every fair-test cell within 2 sd of the reference's p9. value, using the
reference's 3-seed sd; what is still open; at most one question.

I judge, from the printed table: done when 8 of 8 reproduction cells sit within 2 sd of run 7 and every fair-test cell sits within 2 sd of its p9. value.
```

added: the seven sections, from the field-labelled version · the layer's purpose and the reference as Background (from the layer's Why line and step 0) · run 7's card and the brick row under What to read (from the facts that cite them)
dropped: nothing
