# Layer 5a, step 4: the scene

⬅️ [Layer 5a, step 4](../05a-reading-stop-projective-composition.md#4-ai-in-the-background-the-scene)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
after the combined read of step 2 has closed.

```text
/concept-scene projective composition: the background-divided product and the
factorized-conditionals condition on the quadrant toy, with the background as a choice

Background
The reading stop of the composition ladder (artifacts/drips/simple-distributions-rebuild/) read two
Bradley papers on when summed scores are exact. This scene puts the background in my hands: on the
four-quadrant toy I change the background and watch each of Definition 5.2's three conditions hold
or break. It uses the four-quadrant answer key because its background-divided numbers exist and
layer 5's square toy has none.

Objectives
1. Build one scene on the four-quadrant answer-key toy where the background is a control with three
   settings: empty, the four-quadrant marginal, and the unconditional.
2. Show each of Definition 5.2's three conditions as holding or breaking under each setting.
3. Use my words and the differences the combined read named.
4. Publish it and write its URL into the reading stop's step 4.

What to read
- COMBINE_MAP.md under
  ~/goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/drips/ in its
  combine-<idea-slug>/ folder (step 2's hand-off): my words and the differences the read named.
- artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/README.md and answer_key.json:
  the toy and its numbers.
- artifacts/scenes/when-a-product-reaches-a-and-b/checks/scene_key.py: the earlier scene's key
  script.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The scene's toy is the four-quadrant answer key: one Gaussian blob per quadrant at (±0.5, ±0.5),
  std 0.15, A is x1 > 0, B is x2 > 0, with products on a 441 × 441 grid over [−1.1, 1.1]²
  (artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/README.md).
- Experts that model only their own coordinate and take the other from the background marginal of
  all four quadrants satisfy Definition 5.2, and with no A∩B data they put 0.9999 (plain product)
  and 0.989 (background-divided, p_A·p_B/p_bg) in the top-right blob (same card, rescue 2b).
- Experts trained from scratch on one quadrant each put 0.989 at the origin and 0.006 in the
  top-right blob, normaliser 7.9e-10 (same card, construction (2) at ε = 0).
- One conditional model with "contains" labels, composed as p(x|A)·p(x|B)/p(x), keeps 0.989 in the
  top-right blob for every A∩B share r from 0.25 down to 0.001, normaliser equal to r (same card,
  construction (1)).
- 0.989 recurs because one blob holds 1 − e^(−9/2) of its mass within 3 std (when-poe IDEA_MAP.md,
  run 4).
- Definition 5.2's three conditions are: x on M_i independent of the rest under p_i; all blocks
  independent under p_b; p_i equal to p_b off M_i (read-ahead 05 of 2502.04549, chunk 45).
- The same experts compose with an empty background and fail with the unconditional as background
  (2502.04549 Figure 3, A and B; read-ahead 05, chunk 53).
- In 2D a uniform background changes the plain product only by a constant
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 2).
- Three scenes already exist and are not to be redrawn differently: "Three ways a product reaches
  A-and-B" https://claude.ai/artifact/9eFnG2rWEciPsceMwdem4u and "Where the Product Lands"
  https://claude.ai/artifact/2A7YDnfKe8EPsS7yFDVcNT on this toy, and the 2502.04549 chapter "Divide
  Out the Wall" https://claude.ai/artifact/KS2mvPznS1VKNdWz7B6f85 (ladder/05-the-overlap-dial.md;
  when-poe IDEA_MAP.md, run 4; READ_MAP.md).

Rules
- Every number comes from answer_key.json, scene_key.py in
  artifacts/scenes/when-a-product-reaches-a-and-b/checks/, or a key script written before the build,
  never estimated.
- Link the three existing scenes rather than redraw them.
- List the scene's frames before building, and wait for a yes.

What to hand back
- One published scene where the background is a control and each of Definition 5.2's three
  conditions shows as holding or breaking.
- Its URL in step 4 of
  artifacts/drips/simple-distributions-rebuild/ladder/05a-reading-stop-projective-composition.md,
  replacing "(URL lands here)".
- The proof: the key script's printed numbers beside the numbers the scene shows.

What you will see
Along the way: the scene's frames listed before it is built, with a yes asked.
At the end, a closing report: what was asked, in one sentence; what was done, with the URL; PASS or
MISS against every number on the scene equal to the answer key or the scene's own key script; what
is still open; at most one question.

I judge, by operating it: done when changing the background makes each of the three conditions hold or break and every number equals the answer key or the scene's key script.
```

added: the seven sections, from the field-labelled version · a frame list with a yes before the build (from the layer 1 scene prompt's shape) · why the four-quadrant toy stands in for layer 5's square toy, moved into Background (from the earlier added line)
dropped: nothing
