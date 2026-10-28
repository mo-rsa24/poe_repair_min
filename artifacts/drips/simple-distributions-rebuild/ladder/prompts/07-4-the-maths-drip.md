# Layer 7, step 4: the maths drip

⬅️ [Layer 7, step 4](../07-a-without-b.md#4-you-the-maths-drip)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
after quiz 7.1.

```text
/drip --math quotients of Gaussians, why dividing by an expert whose density falls off like e^(-d²)
pushes mass away without limit, and why a positive label ("no drain", a factor p(no drain|x) ≤ 1)
stays bounded where dividing by the drain expert does not

Background
This is layer 7 of the composition ladder in artifacts/drips/simple-distributions-rebuild/: A
without B. The medical cells are A without B (pneumothorax without drain), and at the paper's own
weight negation sends mass off every mode when the experts have sharp tails. The maths comes before
the build and teaches only what the build needs: when a weighted difference of log densities stays
a density, and why a positive label is the safer choice wherever one exists. I have passed quiz
7.1; any question it marked missed becomes a first rung.

Objectives
1. Teach the quotient of two Gaussians, and why a negative power subtracts precision.
2. Show why dividing by an expert whose density falls off like e^(−d²) pushes mass away without
   limit.
3. Work N(0, 1) divided by N(0, 0.5)^w at two values of w by hand, tails included.
4. Show why a positive label, a factor p(no drain|x) ≤ 1, stays bounded where dividing by the drain
   expert does not.

What to read
- Quiz 7.1's results: the misses that become first rungs.
- artifacts/drips/simple-distributions-rebuild/ladder/07-a-without-b.md, its Study first line: the
  vault notes and textbook for this layer.
- artifacts/ideas/poe-overlap-dial/IDEA_MAP.md and artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md:
  the dial's numbers and the medical counts.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- For Gaussians, q_a/q_b^α is a density only while 1/v_a − α/v_b > 0, because a negative power
  subtracts precision (vault notes Product, Mixture And Negation and Powers Of Gaussians).
- Where p_B → 0, p_B^(−w) → ∞, so negation pushes mass toward wherever B is least likely, not just
  out of B (step file 02, step 4).
- On the ring and bar (8 ring modes of std 0.03 at radius 0.5, a bar of half-width 0.2, ideal
  experts smoothed by 0.05), the exact mass on the 2 in-bar modes is 0.146, 0.121, 0.085, 0.037,
  0.0, 0.0 at w = 0, 0.02, 0.05, 0.1, 0.3, 1.0, and the mass away from every ring mode is 0.387,
  0.384, 0.391, 0.439, 0.863, 1.0 (kinds notebook, Figure 2; rebuild/kinds.py).
- Negation works only near w ≈ 0.05, and the notebook's own weight is w = 0.3, the 1.3 and −0.3 of
  NegationEBMDiffusionModel (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, run 3 and "What the words
  are").
- p(no drain|x) is at most 1, so p(x|PTX)·p(no drain|x)^w never exceeds p(x|PTX) (Bayes' rule).
- Negation's exponents sum to 1, so it is unchanged by an invertible warp (total variation 7e-17)
  (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 10 and run 6).
- 1,906 of 3,543 NEATX pneumothorax films have no drain, and what is missing is a drain label on
  negatives, so a positive "no drain" label needs one (when-poe-reaches-a-and-b idea map, claims 6
  and 8).
- Report-mined labels add a label term that matters most for these negation cells
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, "The idea, as it stands").

Rules
- One rung per message, in LaTeX, with worked numbers.

What to hand back
- The drip's thread folder under artifacts/drips/, with dispatch.md recording my notation, the
  worked numbers I used and what clicked.
- The folder's path, which quiz 7.2, step 8's scene and step 9's cross-links read.

What you will see
Along the way: one rung per message, each ending on the drip's control line.
At the end, a closing report: what was asked, in one sentence; what was done, with the thread
folder's path; PASS or MISS against the target below; what is still unclear to me; at most one
question.

I judge, without notes: done when I say when (1 + w) log p_A − w log p_B stays a proper density, work N(0, 1) divided by N(0, 0.5)^w at two values of w by hand, and say why the positive label stays bounded.
```

added: the seven sections, from the field-labelled version · the whiteboard example as an objective (from this layer's Achieve line) · "one rung per message, in LaTeX" as a rule (from the layer 1 maths drip prompt's shape)
dropped: nothing
