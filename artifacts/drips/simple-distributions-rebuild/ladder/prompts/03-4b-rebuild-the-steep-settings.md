# Layer 3, step 4b: the idea-walk rebuild, walk03b.py

⬅️ [Layer 3, step 4b](../03-ideal-experts-and-the-seven-kinds.md#4b-you-the-idea-walk-rebuild-walk03bpy)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`.

```text
/drip-execute-plan --request rebuild run 2 of
artifacts/ideas/poe-overlap-dial/kinds-of-composition-on-the-papers-toys.ipynb by hand in
~/Playground/PhD/simple-distributions-walk/walk03b.py; bar: with 5% of each set rotated, the exact
composition puts 0.75 of its mass on co-occurrence while reverse diffusion draws a chimera column
with 0.21, and at 2% overlap reverse diffusion, ULA and MALA add a dry-dog cluster that tuned MALA
and UHA do not

Background
This is the idea-walk rebuild in layer 3 of the composition ladder
(artifacts/drips/simple-distributions-rebuild/). Run 2 of the kinds notebook showed that near the
edge of what the data supports, a weak sampler draws the picture of a data failure, which is why
every later result is graded against an exact target. I rebuild it by hand, reusing walk03.py from
step 4a (its exact grid, perfect-sample baseline, kinds and dials) and walk02c.py from layer 2 (its
tuned MALA); the notebook's printed run 2 table is the answer key.

Objectives
1. Cut run 2 into 4 to 6 pieces, and write the step file at the open.
2. Build, teach and test one piece per round.
3. I type every block into walk03b.py and run it with
   /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python and
   PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy.
4. End with walk03b.py printing one block per steep setting (the exact measure, then measure and TV
   per sampler) and a figure of the three settings with one panel per sampler, in the typing
   folder.

What to read
- artifacts/ideas/poe-overlap-dial/kinds-of-composition-on-the-papers-toys.ipynb, run 2 (cells 17
  to 19): the code, and its cell 18 table as the answer key.
- artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 4 and run 6: why only the closest samplers are
  read on image results.
- toy/rebuild/phase3_kinds.py, Run 2, and toy/rebuild/out/phase3/phase3.json, run2: the reference.
- ~/Playground/PhD/simple-distributions-walk/walk03.py and walk02c.py.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- Run 2 runs three kinds at the steep points of Figure 2, where a sampler's error and the data's
  limit could be mistaken for each other: co-occurrence with A's top edge at 0.4, contention with 5%
  of each set rotated, and interaction at 2% overlap (notebook, cell 17).
- The samplers are perfect draws, reverse diffusion with the composed score, ULA and MALA at
  0.05·v_t, tuned MALA adapted to 0.6 acceptance, and UHA at 0.3·√v_t with damping 0.5 and 3
  leapfrog steps, 2,000 samples each (notebook, cells 2 and 18).
- The notebook's run 2 table, measure / TV per sampler, for co-occurrence at top edge 0.4, exact
  0.44: perfect 0.43 / 0.11, reverse diffusion 0.38 / 0.22, ULA 0.40 / 0.19, MALA 0.42 / 0.18,
  tuned MALA 0.42 / 0.13, UHA 0.43 / 0.13 (notebook, cell 18).
- For contention at 5% rotated, exact 0.75: perfect 0.73 / 0.10, reverse diffusion 0.21 / 0.69, ULA
  0.26 / 0.57, MALA 0.24 / 0.57, tuned MALA 0.53 / 0.30, UHA 0.53 / 0.27 (notebook, cell 18).
- For interaction at 2% overlap, exact −0.38: perfect −0.38 / 0.05, reverse diffusion −0.35 / 0.59,
  ULA −0.37 / 0.26, MALA −0.37 / 0.27, tuned MALA −0.38 / 0.08, UHA −0.38 / 0.08 (notebook, cell
  18).
- At contention 5% rotated, reverse diffusion's samples form one column at x1 ≈ 0 between the bars,
  a chimera the exact composition does not contain (notebook, cell 19).
- At interaction 2%, reverse diffusion, ULA and MALA put a second cluster at the dry position at the
  band's bottom edge (notebook, cell 19).
- A weak sampler near the edge of what the data supports draws the kind's failure picture, which
  without the exact composition beside it would be blamed on missing data, so only the samplers
  that come closest to the perfect-sample baseline in 2D (tuned MALA, UHA) are read on image
  results (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 4 and run 6).
- rebuild/phase3_kinds.py reproduces run 2 within 0.05 of the notebook on every row, its own
  tolerance for these sampler numbers (toy/rebuild/phase3_kinds.py, Run 2;
  toy/rebuild/out/phase3/phase3.json, run2).

Rules
- Never edit the notebook or toy/rebuild/.
- I type; I do not paste.
- No commit.
- Show each piece's output before moving on: piece 1 prints the three exact measures, each sampler
  piece prints its row for all three settings, and the figure piece saves the panels.

What to hand back
- walk03b.py, printing one block per steep setting: the exact measure, then measure and TV per
  sampler.
- The figure of the three settings with one panel per sampler, in the typing folder.
- The step file the walk writes at its open, ticked piece by piece.
- The retrospective the walk closes on, and the /drip-retro command it prints.
- The proof: python walk03b.py prints the three blocks, and step 5's comparison with python -m
  rebuild.phase3_kinds (Run 2) shows each row within 0.05.

What you will see
Along the way: one piece per round, each ending on what it printed or saved and a yes to move on.
At the end, a closing report: what was asked, in one sentence; what was done, with walk03b.py's
path, the figure's and the step file's; PASS or MISS against exact measures of 0.44, 0.75 and
−0.38, every sampler row's measure and TV within 0.05 of the notebook's run 2 table (the tolerance
phase3_kinds holds the reference to), and the figure showing reverse diffusion's chimera column at
contention 5% and the dry-dog cluster at interaction 2%; what is still open; at most one question.

I judge, from the printed table and the figure: done when the exact measures read 0.44, 0.75 and −0.38, every row sits within 0.05 of the notebook, and the figure shows the chimera column and the dry-dog cluster.
```

added: the seven sections, from the field-labelled version · what run 2 showed, as the Background (from the step's Outcome)
dropped: nothing (the earlier version had already replaced "samplers that reach the floor" with "the samplers that come closest to the perfect-sample baseline", from the Codex mathematics review, since one perfect draw is random and not a floor)
