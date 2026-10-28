# Layer 8, step 4b: the label term, walk08b.py

⬅️ [Layer 8, step 4b](../08-the-interaction-term.md#4b-you-the-label-term-walk08bpy)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
after step 4a's walk has closed.

```text
/drip-execute-plan --request the label term with report-style labels on the quadrant toy: A is read
with probability 0.9 when true, and inside A∩B, B is read only when A was read; a 4-way classifier
over the label pairs against two separate classifiers gives log p(A,B|x) - log p(A|x) - log p(B|x);
I type every block myself into ~/Playground/PhD/simple-distributions-walk/walk08b.py

Background
This is layer 8's second build in the composition ladder (artifacts/drips/simple-distributions-rebuild/).
Step 4a showed the interaction term in the pixels, the wet dog. This walk shows it in the labels:
zero for clean labels, real for report-mined ones like NIH's, and hardest on the negation cells. The
layer 8 reference from step 0 already holds the exact values as the p8.label_term_* keys, so my
classifier estimate has an answer key.

Objectives
1. At the walk's open, cut the label-term toy into 4 to 6 pieces and write the step file, then
   build, teach and test one piece per round.
2. I type every block into walk08b.py and run it with
   /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python from
   ~/Playground/PhD/simple-distributions-walk/.
3. Piece by piece: the label counts per cell; the exact term per cell; the 4-way classifier's
   accuracy; the two binary classifiers' accuracy; the estimated term beside the exact one.
4. Train the 4-way classifier and the two binary classifiers on the same points, and print the label
   term's mean per label cell, inside A∩B and outside it, beside the exact p8.label_term_* values.
5. Tick each piece in the step file as it lands.

What to read
- walk08.py from step 4a.
- The p8.label_term_* keys in rebuild/out/walk-numbers.json (step 0's reference): the exact values.
- artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claims 6, 7.3 and 7.4.
- rebuild/phase5_dial.py: the A and B definitions and MARGIN.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The toy is points uniform on [−1, 1]², A true when x1 < 0 and B true when x2 > 0 (the A and B of
  rebuild/phase5_dial.py), so A∩B is the top-left quadrant
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 7.4; rebuild/phase5_dial.py docstring).
- Claim 7.4's expected label term is +0.105 nats for (A,B) inside A∩B, very negative for (¬A,B)
  there, and zero elsewhere (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 7.4).
- Inside A∩B, P(A,B|x) / [P(A|x)·P(B|x)] = 0.9q / (0.9 · 0.9q) = 1/0.9 for any probability q of
  reading B once A was read, so +0.105 = −ln 0.9; (¬A,B) has probability 0 there, so its term is −∞
  and a classifier reports it as strongly negative; and outside A∩B at most one label is uncertain,
  so the term is 0 (the arithmetic of claim 7.4's labelling).
- The term is zero for clean labels and nonzero for report-mined ones, and it is a different thing
  from the wet dog of step 4a (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 7.3).
- On chest X-rays it matters most for the negation cells, pneumothorax without drain and drain
  without pneumothorax, and 20% to 54% of NIH pneumothoraces have no drain
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 6 and "The idea, as it stands").
- A margin of 0.15 inside a quadrant keeps boundary points out of a region count
  (rebuild/phase5_dial.py, MARGIN).
- The tolerances 0.05 nats and −2 below are estimates, to be changed here if a classifier on a few
  thousand points cannot reach them (this prompt's earlier added line).

Rules
- I type; I do not paste.
- Never edit rebuild/ or the idea maps.
- No commit.
- Piece 1 prints the label counts per cell; piece 2 the exact term per cell; pieces 3 and 4 each
  classifier's accuracy; piece 5 the estimated term beside the exact one; show each before moving
  on.

What to hand back
- walk08b.py: one script that trains the 4-way classifier and the two binary classifiers on the same
  points and prints the label term's mean per label cell, inside A∩B and outside it, beside the
  exact p8.label_term_* values.
- The step file the walk writes at its open, ticked piece by piece.
- The retrospective the walk closes on, and the /drip-retro command it prints.
- The proof: python walk08b.py prints "(A,B) inside A∩B: <x> nats, exact 0.1054" with x within 0.05
  of 0.1054.

What you will see
Along the way: one piece per round, each ending on the number it printed and a yes to move on.
At the end, a closing report: what was asked, in one sentence; what was done, with walk08b.py's path
and the step file's; PASS or MISS against the classifier's (A,B) term inside A∩B (points at least
0.15 inside) within 0.05 nats of +0.105, the term outside A∩B (points at least 0.15 from both
boundaries) within 0.05 of 0, and (¬A,B) inside A∩B below −2; what is still open; at most one
question.

I judge, from the printed lines: done when the (A,B) term inside A∩B is within 0.05 nats of +0.105, the term outside A∩B is within 0.05 of 0, and (¬A,B) inside A∩B is below −2.
```

added: the seven sections, from the field-labelled version · the tolerances' status as estimates, moved from the old added line into the prompt as a fact, since it still governs the bar
dropped: nothing
