# Layer 8, step 2: the maths drip

⬅️ [Layer 8, step 2](../08-the-interaction-term.md#2-you-the-maths-drip)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
after quiz 8.1.

```text
/drip --math Bayes' rule for two labels and conditional independence, ending at r(x) = log p(x|A,B)
- [log p(x|A) + log p(x|B) - log p(x)] and its ε-space form ε_joint - ε_A - ε_B + ε_∅

Background
This is layer 8 of the composition ladder in artifacts/drips/simple-distributions-rebuild/: the
interaction term, the first half of the research goal, defined exactly here before any learned
version of it is trusted. The maths comes before the build and teaches only the derivation the
build needs. I have passed quiz 8.1; any question it marked missed becomes a first rung. Layer 1's
maths drip folder under artifacts/drips/ holds my notation for q(x_t | x_0) and the score.

Objectives
1. Teach Bayes' rule for two labels and conditional independence given x.
2. Derive r(x) = log p(x|A,B) − [log p(x|A) + log p(x|B) − log p(x)] and rewrite it by Bayes as the
   label term plus a constant.
3. Show why r is constant when the labels are fixed functions of the image.
4. Derive its ε-space form ε_joint − ε_A − ε_B + ε_∅ from layer 1's ε = −√(1 − ᾱ_t)·∇log q_t.
5. Reproduce +0.105 = −ln 0.9 by hand on claim 7.4's labels.

What to read
- Quiz 8.1's results: the misses that become first rungs.
- Layer 1's maths drip folder under artifacts/drips/ and its dispatch.md: my notation.
- artifacts/drips/simple-distributions-rebuild/ladder/08-the-interaction-term.md, its Study first
  line: the journey plans, textbook chapters and vault notes for this layer, among them section 8.2,
  the route through p(c1, c2 | x).
- artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 7, and
  artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claims 3 and 9.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- By Bayes, log p(x|A) = log p(A|x) + log p(x) − log p(A), so r(x) = log p(A,B|x) − log p(A|x) −
  log p(B|x) + log[p(A)p(B)/p(A,B)], which makes r the label term plus a constant (the route through
  p(c1, c2 | x), section 8.2 of the textbook on this layer's Study first line;
  artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 9).
- Construction (1), score(x|A) + score(x|B) − score(x), targets p(x|A)p(x|B)/p(x), which is
  p(x)·p(A|x)·p(B|x) up to a constant, and assumes A and B independent given x
  (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 3 and "What the words are").
- When each label is a fixed function of the image the term is exactly zero, so the wet dog is a
  data condition rather than a third operator, and the shift was found at 20% overlap and missed at
  0% (mean x1 −0.79 against −0.48) (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 7.1 and run
  1).
- With report-mined labels the term survives, +0.105 nats for (A,B) inside A∩B on claim 7.4's
  labels, which is −ln 0.9 (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claims 7.3 and 7.4).
- ε = −√(1 − ᾱ_t)·∇log q_t, so ∇r_t = −(ε_joint − ε_A − ε_B + ε_∅)/√(1 − ᾱ_t), the gap between the
  joint's score and the composed score (ladder/01-one-diffusion-expert.md, step 2; REBUILD_WALK.md,
  brick "The interaction term, exact").
- poe_repair_min caches that residual for 25 pairs at all 50 steps, at guidance 7.5
  (artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md, claim 1 and Sources).

Rules
- One rung per message, in LaTeX, with worked numbers.

What to hand back
- The drip's thread folder under artifacts/drips/, with dispatch.md recording my notation, the
  worked numbers I used and what clicked.
- The folder's path, which quiz 8.2, step 6's scene and step 7's cross-links read.

What you will see
Along the way: one rung per message, each ending on the drip's control line.
At the end, a closing report: what was asked, in one sentence; what was done, with the thread
folder's path; PASS or MISS against the target below; what is still unclear to me; at most one
question.

I judge, without notes: done when I derive r and its ε-space form, say why r is constant when the labels are fixed functions of the image, and reproduce +0.105 = −ln 0.9 by hand.
```

added: the seven sections, from the field-labelled version · "one rung per message, in LaTeX" as a rule (from the layer 1 maths drip prompt's shape)
dropped: nothing
