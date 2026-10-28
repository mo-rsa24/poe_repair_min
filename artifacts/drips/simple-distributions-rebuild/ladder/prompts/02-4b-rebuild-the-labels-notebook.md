# Layer 2, step 4b: the first idea-walk rebuild, walk02b.py

⬅️ [Layer 2, step 4b](../02-composing-two-experts.md#4b-you-the-first-idea-walk-rebuild-walk02bpy)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`.

```text
/drip-execute-plan --request rebuild sections 1 to 3 of
artifacts/ideas/poe-overlap-dial/what-an-expert-learns-and-kinds-of-composition.ipynb by hand in
~/Playground/PhD/simple-distributions-walk/walk02b.py; bar: the gradient difference is exactly 0.0
when only the labels change, construction (1) puts 1.00 of its mass in A∩B under "contains A"
against 0.08 under "exactly A", and a classifier trained on set membership guides sideways in A∩B,
(-0.38, -0.04) against (-0.14, +0.99) with true labels

Background
This is the first of two idea-walk rebuilds in layer 2 of the composition ladder
(artifacts/drips/simple-distributions-rebuild/). The notebook above found that a density expert
never reads its labels, that the meaning of a label decides construction (1), and that a classifier
trained on the labels the data carries learns a shortcut. I rebuild its sections 1 to 3 by hand,
starting from walk02.py from step 4a; the notebook's printed outputs are the answer key.

Objectives
1. Cut sections 1 to 3 of the notebook into 4 to 6 pieces, and write the step file at the open.
2. Build, teach and test one piece per round.
3. I type every block into walk02b.py and run it with
   /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python from
   ~/Playground/PhD/simple-distributions-walk/.
4. End with walk02b.py printing the gradient difference, the two top-left masses and the two
   guidance directions.

What to read
- artifacts/ideas/poe-overlap-dial/what-an-expert-learns-and-kinds-of-composition.ipynb, sections 1
  to 3: the code, and its printed outputs as the answer key.
- artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 1: the repaired claim.
- ~/Playground/PhD/simple-distributions-walk/walk02.py from step 4a.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- No sampler appears: every density is an ideal expert, its training points smoothed by N(0, σ²I),
  computed exactly on a 241 × 241 grid over [−1.1, 1.1]², and A∩B is the top-left quadrant
  (notebook, cells 0 and 1).
- Section 1: A's set is (−0.5, −0.5), (−0.6, −0.4), (−0.5, 0.5) under two labellings (notebook,
  cell 3).
- In section 1 a tiny denoiser (Linear 3 to 32, SiLU, Linear 32 to 2, seed 1) takes one gradient
  step on each labelling; the labels never enter the loss, so the largest gradient difference is
  exactly 0.0 (notebook, cells 3 and 4).
- At σ = 0.15 a third of A's mass sits in the top-left quadrant (notebook, cells 3 and 4).
- Section 2: the pool adds B's points (0.5, 0.5), (0.6, 0.4), (−0.4, 0.6), one of them secretly A
  (notebook, cell 5).
- Under "contains A" every point whose label set includes A trains p(x|A); under "exactly A" a point
  labelled {A, B} drops out of p(x|A) (notebook, cells 5 and 6).
- Construction (1), p(x|A)p(x|B)/p(x) at σ = 0.15, puts 1.00 of its mass in A∩B under "contains"
  and 0.08 under "exactly", where its leftover mass sits near the origin (notebook, cells 5 and 6;
  artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 1).
- Section 3: two sets of 400 points with 20% hidden in the other concept's half (notebook, cell 7).
- A classifier (2 to 64 to 64 to 1, SiLU, Adam 3e-3, weight decay 1e-3, 1,500 steps, seed 0)
  trained on set membership is unsure in A∩B, about 0.5, and its mean guidance direction there is
  (−0.38, −0.04) (notebook, cells 7 and 8).
- Trained on the true labels x2 > 0, the same classifier's direction is (−0.14, +0.99) (notebook,
  cells 7 and 8).
- The notebook draws its data from np.random.default_rng(0) with torch.manual_seed(0) in cell 1, so
  section 3's directions repeat only when the draws happen in the notebook's order (notebook, cell
  1).
- The repaired claim: a density expert learns its training set's density, and labels matter only in
  the parts that read them, the conditioning scheme of construction (1) and any classifier
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 1).

Rules
- Never edit the notebook or toy/rebuild/.
- I type; I do not paste.
- No commit.
- Show each piece's printed number before moving on: piece 1 prints the gradient difference, piece
  2 the masses under each scheme, piece 3 the classifier's mean p(B|x) in A∩B and its two
  directions.

What to hand back
- walk02b.py, printing the gradient difference, the two top-left masses and the two guidance
  directions.
- The step file the walk writes at its open, ticked piece by piece.
- The retrospective the walk closes on, and the /drip-retro command it prints.
- The proof: python walk02b.py prints "largest difference between the two versions' gradients: 0.0"
  and the four numbers above, and step 5's comparison against the notebook shows no differing
  number.

What you will see
Along the way: one piece per round, each ending on the number it printed and a yes to move on.
At the end, a closing report: what was asked, in one sentence; what was done, with walk02b.py's
path and the step file's; PASS or MISS against a gradient difference of exactly 0.0, top-left masses
of 1.00 and 0.08, and guidance directions of (−0.38, −0.04) and (−0.14, +0.99) to two decimals with
the notebook's seeds and order; what is still open; at most one question.

I judge, from the printed lines: done when the gradient difference is exactly 0.0, the masses read 1.00 and 0.08, and the two directions match the notebook to two decimals.
```

added: the seven sections, from the field-labelled version · the three things the notebook found, as the Background (from the step's Outcome)
dropped: nothing
