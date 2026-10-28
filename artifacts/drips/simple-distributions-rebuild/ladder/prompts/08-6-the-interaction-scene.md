# Layer 8, step 6: the scene

⬅️ [Layer 8, step 6](../08-the-interaction-term.md#6-ai-in-the-background-the-scene)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
after layer 8's maths drip and reference have both run.

```text
/math-scene the folder the layer 8 --math drip wrote under artifacts/drips/: the interaction term in
the parts world, with the wet rule as a toggle and the noise level as a slider

Background
Layer 8 of the composition ladder (artifacts/drips/simple-distributions-rebuild/) defined the
interaction term exactly in the parts world, the first half of the research goal. This scene puts
the wet rule (the cat half shifts when a dog is present) on a toggle and the noise level on a
slider, in my own notation, so I can see where r moves and where it does not.

Objectives
1. Build one scene of the parts world's interaction term, with the wet rule as a toggle and the
   noise level as a slider over t = 0, 25, 50, 75 and 99.
2. Show what the reference found when the toggle flips, not the brick row's expectation.
3. Write it in the notation the maths drip's dispatch.md records.
4. Publish it and write its URL into layer 8's step 6.

What to read
- The layer 8 maths drip folder under artifacts/drips/ and its dispatch.md (step 2's hand-off): my
  notation.
- rebuild/out/phase8/phase8.json and the p8. keys of python -m rebuild.walk_numbers (step 0): every
  number per setting and level.
- The docstring of rebuild/phase8_interaction.py: where r moved with the wet rule and where it did
  not.
- artifacts/results/joint-against-poe-by-kind-of-composition/five-interactions.md: the wet dog at
  scale.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The parts world is 16 pixels drawn as a 4 × 4 image, a cat +1 and a dog −1 on their pixels with
  variance 0.01, each expert a mixture of diagonal Gaussians, so every noised density is exact (step
  file 04-the-parts-world.md, steps 1, 5 and 6).
- The noise levels are t = 0, 25, 50, 75 and 99, where ᾱ_t is 0.9994, 0.8356, 0.4783, 0.1335 and
  0.0 (the p1.abar_* keys in rebuild/out/walk-numbers.json).
- With clean labels r is constant at t = 0 whether the wet rule is on or off, so the toggle changes r
  only where phase8.json shows it does, and the scene shows that, not the brick row's expectation
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 7.1; the docstring of
  rebuild/phase8_interaction.py).
- On Stable Diffusion v1-4 the product paints rain streaks over a dry dog where the joint prompt
  gives wet fur, in 4 of 4 seeds
  (artifacts/results/joint-against-poe-by-kind-of-composition/five-interactions.md).

Rules
- Every number on the scene comes from phase8.json, the p8. keys or the drip's dispatch.md, never
  estimated.
- List the scene's frames before building, and wait for a yes.

What to hand back
- One published scene, in my notation from dispatch.md.
- Its URL in step 6 of artifacts/drips/simple-distributions-rebuild/ladder/08-the-interaction-term.md,
  replacing "(URL lands here)".

What you will see
Along the way: the scene's frames listed before it is built, with a yes asked.
At the end, a closing report: what was asked, in one sentence; what was done, with the URL; whether
every number on it traced to phase8.json, the p8. keys or dispatch.md; where the toggle moved r and
where it did not; what is still open; at most one question.

I judge, by operating it: done when flipping the wet rule and moving the noise level change r only where phase8.json says they do, and every number on it is traceable.
```

added: the seven sections, from the field-labelled version · a frame list with a yes before the build (from the layer 1 scene prompt's shape) · the judge line, since the field-labelled version carried no target (from the reference-only number rule and claim 7.1)
dropped: nothing
