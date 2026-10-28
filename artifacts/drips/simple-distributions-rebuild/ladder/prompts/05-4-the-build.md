# Layer 5, step 4: the build, walk05.py

⬅️ [Layer 5, step 4](../05-the-overlap-dial.md#4-you-the-build-walk05py)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`.

```text
/drip-execute-plan artifacts/drips/simple-distributions-rebuild/05-the-overlap-dial.md, I type every
block myself into ~/Playground/PhD/simple-distributions-walk/walk05.py, measuring the dial's A∩B in
its deep square and the answer key's within 3 std of each blob, with the whole quadrant only as a
spill check; bar: deep A∩B mass 0.00 at 0% and 0.30 at 5% overlap (exact), and tuned MALA's 0.17
at 5%

Background
This is layer 5's build in the composition ladder (artifacts/drips/simple-distributions-rebuild/):
I type the overlap dial from the step file, one step per message, and then rebuild the three
answer-key numbers that layer 12's dial claim rests on. Claude's reference code is under toy/ in
this repo; my file and the reference print the same numbers, and a difference is a lesson or a bug.
I start from the setup's folder and environment and walk04.py from layer 4, with quiz 5.2 passed,
and the layer 5 maths drip folder under artifacts/drips/ holds my notation in its dispatch.md.

Objectives
1. Walk the step file's steps 0 to 8, one per message, in its order.
2. Rebuild, inside the same walk, the three answer-key numbers listed under step 4 of
   artifacts/drips/simple-distributions-rebuild/ladder/05-the-overlap-dial.md, typing the answer
   key's construction rather than importing answer_key.py.
3. I type every block into walk05.py and run it from ~/Playground/PhD/simple-distributions-walk/
   with /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python and
   PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy.
4. Tick each step in the step file as it lands, and keep my place in the walk map between sittings.

What to read
- artifacts/drips/simple-distributions-rebuild/05-the-overlap-dial.md: the step file, including
  "Where the numbers come from".
- artifacts/drips/simple-distributions-rebuild/ladder/05-the-overlap-dial.md, step 4: the three
  answer-key numbers and why they differ.
- toy/rebuild/phase5_dial.py and toy/rebuild/phase5_finetune.py: the reference.
- toy/rebuild/out/phase5/overlap-dial.json and finetune.json, printed as tables by
  python -m rebuild.phase5_tables.
- artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/, its README and
  answer_key.json: the answer key.
- The layer 5 maths drip's dispatch.md: my notation.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The reference is rebuild/phase5_dial.py and rebuild/phase5_finetune.py, whose results are
  toy/rebuild/out/phase5/overlap-dial.json and finetune.json, printed as tables by
  python -m rebuild.phase5_tables (step file 05, "Where the numbers come from").
- The worked numbers are 24 p5. keys of python -m rebuild.walk_numbers, 8 for walk05.py's own toys
  and 16 for the step and ladder files' derivations (step file 05, "Where the numbers come from").
- From step 0 on, phases 1 to 4 are imported from the reference (rebuild.ideal, rebuild.samplers,
  rebuild.parts_world), seeded with np.random.default_rng(100) and torch.manual_seed(100), with
  N = 800 points, NS = 1000 samples and a 0.15 margin (step file 05, step 0).
- Each toy has its own A∩B region, and since the regions sit in different toys their raw counts are
  never compared as one metric (step file 05, steps 2 and 3; when-poe-reaches-a-and-b idea map,
  claim 5).
- The dial's A∩B region is the deep square x1 < −0.15, x2 > 0.15 (3σ inside); at 0% no training
  point is there and it reads 0.00 (step file 05, steps 2 and 3).
- At 0% the whole top-left quadrant reads 0.20 ± 0.02, all of it the sliver around the corner, so
  the quadrant is reported only as a boundary-spill check and never alone as evidence (step file
  05, steps 2 and 3).
- The answer key's A∩B region is within 3 × 0.15 = 0.45 of each blob's centre, where a quadrant
  count reads the from-scratch origin blend as 0.25 (when-poe-reaches-a-and-b idea map, claim 5).
- The bar's 0.30 and 0.17 are means over 3 data seeds with 2,000 samples: 0.30 ± 0.10 exact and
  0.17 ± 0.04 tuned MALA (phase5_tables).
- walk05.py runs one seed with 1,000 samples, and the step file's own run of it printed exact deep
  mass 0.0, 0.03, 0.276, 0.486, 0.453, 0.486, 0.709 and 0.76 at 0, 1, 2, 5, 10, 20, 50 and 100%
  (step file 05, step 4).
- In that run, at 5%, tuned MALA printed 0.382 and UHA 0.343 against an exact 0.486 (step file 05,
  step 5).
- At low overlap a single seed swings because a handful of shared points decides it (step file 05,
  steps 4 and 5).
- Over 3 seeds tuned MALA and UHA agree within 0.02, reach the exact curve from 20% up (0.55 against
  0.59) and lag below it at 5% (0.17 and 0.18 against 0.30) and 10% (0.40 against 0.51) (step file
  05, step 5; phase5_tables).
- Reverse diffusion gives 0.02 at 5% and overshoots at 100%, 0.91 against 0.76 (step file 05, step
  5; phase5_tables).
- At 2% to 10% overlap the samplers recover a third to four fifths of the exact mass (overlap-dial
  idea map, claim 4).
- The whole-distribution check is TV = ½ Σ |p̂ − p| over 40 × 40 bins of [−1.1, 1.1]² (ideal.tv),
  set against the TV of 2,000 draws taken straight from the exact grid, the finite-sample floor
  (phase5_tables, first table; step file 05, step 5).
- At 5% tuned MALA reads 0.24 and UHA 0.23 against a 0.15 floor; at 20%, 0.18 and 0.19 against 0.18;
  a matching regional mass does not pass on its own (phase5_tables, first table; step file 05, step
  5).
- The parts-world dial: A∩B means one half's mean above 0.75 and the other's below −0.75 (step file
  05, step 6).
- With an empty background (variance 0.01) the exact A∩B weight is 0.000 at 0%, a ghost with
  half-means (0.5, −0.5), and 1.000 from 0.1% up, because the real two-animal pair wins by 386.2
  nats (400 for the ghost's mismatch against −13.81 for the overlap weights) (step file 05, step 6;
  p5.ghost_mismatch_nats, p5.overlap_weight_cost_nats, p5.matching_pair_advantage_nats).
- With a loose background (variance 1.0) the parts-world A∩B weight is 1.000 at every f (step file
  05, step 6).
- Step 7's copies come from copy.deepcopy(pre).requires_grad_(True): the moving-average copy has its
  gradients switched off, so fine-tuning it without that call trains nothing and raises no error
  (step file 05, step 7; step file 01, step 9's broken run).
- Step 7's short version (2,000 pretraining steps) printed deep mass 0.040, 0.007, 0.000, 0.008 and
  0.005 at fine-tune steps 0, 10, 30, 100 and 300, and the full run (10,000 steps) starts at 0.159
  (step file 05, step 7).
- At step 10,000 the grid counts a spurious blob near (0.75, 0.9) that the samplers never reach
  (REBUILD_WALK.md, brick "Fine-tune variant").
- The answer key is a second toy: one Gaussian blob of std 0.15 per quadrant at (±0.5, ±0.5), A is
  x1 > 0, B is x2 > 0, A∩B the top-right blob, shares counted within 3 std of a blob's centre on a
  441 × 441 grid over [−1.1, 1.1]² (answer-key-quadrants README, in
  artifacts/ideas/when-poe-reaches-a-and-b/runs/).
- The answer key's A is the right half where the dial's is the left, so its A∩B is top-right where
  the dial's is top-left (ladder file 05, step 4).
- Construction (1), p(x|A)·p(x|B)/p(x) with "contains" labels, holds 0.9886 at r = 0.001 (0.989
  from r = 0.25 down) (answer_key.json; ladder file 05, step 4).
- Separately fine-tuned experts with leftover only next door hold 0.4999 at ε = 0.1, against the
  0.5 that this construction's x → −x symmetry fixes, which is not a ceiling for every fine-tuned
  pair (answer_key.json; ladder file 05, step 4).
- Own-part experts hold 0.99985 as a plain product (answer_key.json; ladder file 05, step 4).
- Train on a card with under 1 GB in use, else on the CPU (REBUILD_WALK.md, Pins).
- The full runs take about 10 minutes (phase5_dial) and 15 (phase5_finetune) on one card (step file
  05, step 8).

Rules
- I type; nothing is pasted into walk05.py.
- Never edit toy/rebuild/, the answer-key or scene folders, or the step file's code blocks.
- Never import answer_key.py; type its construction.
- No commit.
- Show each printed line before moving on: step 1 prints the count of A's points in A∩B at 5% (about
  40), step 2 the corner profile and the 0% masses, step 4 the eight-value curve, step 5 four
  sampler lines, step 6 two background lines and the ghost's half-means, step 7 five fine-tune
  lines, then one line per answer-key number.

What to hand back
- walk05.py, one script that runs end to end and prints each step's numbers, then the three
  answer-key numbers.
- The step file's steps 0 to 8 ticked, and the walk map that holds my place between sittings.
- The retrospective the walk closes on, and the /drip-retro command it prints.
- The proof: python walk05.py prints "0% overlap: ... at least 0.15 inside 0.0" and the eight-value
  curve above, and the comparison in step 5 of layer 5 shows no differing exact number.

What you will see
Along the way: one step per message, each ending on the line it printed and a yes to move on.
At the end, a closing report: what was asked, in one sentence; what was done, with walk05.py's
path; PASS or MISS against the exact, seed-fixed numbers equal to the step file's printed run to
the digits shown (corner profile 0.25; 0% deep mass 0.0; the eight-value curve above; the parts
world's 0.0 then 1.0 with an empty background and 1.0 throughout with a loose one), tuned MALA and
UHA printing below the exact value at 5%, and the three answer-key numbers equal to answer_key.json
to 4 digits (0.9886, 0.4999, 0.9999); the 3-seed 0.30 and 0.17 and the TV against its floor (tuned
MALA 0.24 against 0.15 at 5%, 0.18 against 0.18 at 20%) are read from phase5_tables in step 5, not
reproduced; what is still open; at most one question.

I judge, from the printed lines: done when the seed-fixed exact numbers match the step file's run, the samplers sit below exact at 5%, and the answer-key numbers read 0.9886, 0.4999 and 0.9999.
```

added: the seven sections, from the field-labelled version · the answer-key rebuild and "type the construction, never import answer_key.py" as objectives (from the old Task and Limits lines)
dropped: nothing (the earlier version had already dropped the bar's 0.30 and 0.17 as numbers walk05.py must print, since they are 3-seed means over 2,000 samples and walk05.py runs one seed with 1,000; they stay in the first line and are read from `phase5_tables` in step 5)
