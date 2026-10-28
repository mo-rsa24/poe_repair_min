# Layer 9, step 2: the maths drip

⬅️ [Layer 9, step 2](../09-the-correction-learned-back.md#2-you-the-maths-drip)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`.

```text
/drip --math score matching as regression, and why the score residual is the gradient of r_t

Background
This is layer 9 of the composition ladder in artifacts/drips/simple-distributions-rebuild/: a
correction network trained to add back what composition misses. The maths comes before the build,
and teaches only the derivation the build needs. I have passed quiz 9.1; any question it marked
missed becomes a first rung.

Objectives
1. Show that a squared loss is minimised by the conditional mean, so the denoising loss is a
   regression on the score.
2. Show that a network regressed on ∇r_t learns exactly what composition misses, and write its
   ε-space target.
3. Work out what the correction converges to when the experts are frozen and the loss is taken on
   the few A∩B samples.
4. Compute ∇r_t at the parts world's fixed image by hand, against the p8. key.

What to read
- The conditional-mean and training-loss chapters on the Study first line of
  artifacts/drips/simple-distributions-rebuild/ladder/09-the-correction-learned-back.md: the
  regression argument.
- artifacts/drips/simple-distributions-rebuild/ladder/01-one-diffusion-expert.md, step 2: layer 1's
  identity.
- artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, bricks "The interaction term,
  exact" and "The correction, learned back": the identity and the two losses.
- toy/rebuild/out/walk-numbers.json, the p8. keys: the worked point.
- The layer 8 --math drip folder under artifacts/drips/ and its dispatch.md: my notation for r(x),
  r_t and their ε-space form.
- Quiz 9.1's results: the misses that become first rungs.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- A squared loss is minimised by the conditional mean, so the denoising loss E‖ε_θ(x_t, t) − ε‖² is
  minimised by E[ε | x_t] = −√(1 − ᾱ_t)·∇log q_t(x_t), and score matching is a regression (the
  conditional-mean and training-loss chapters on this layer's Study first line; layer 1's identity,
  ladder/01-one-diffusion-expert.md, step 2).
- s_A + s_B − s_∅ + ∇r_t is the noised joint's score at every level, so a network regressed on
  ∇r_t learns exactly what composition misses (REBUILD_WALK.md, brick "The interaction term,
  exact").
- In ε-space the target is ε_joint − ε_A − ε_B + ε_∅ (REBUILD_WALK.md, brick "The interaction term,
  exact").
- With the experts frozen and the loss taken on the few A∩B samples, the correction's optimum is
  the gap between the noised density of those samples and the composition, which is ∇r_t only as
  far as the samples cover the joint (the same conditional-mean argument; the brick row's two
  losses, regression on the exact ∇r_t and denoising score matching with frozen experts).
- The worked point is ∇r_t at the fixed image of the parts world, the p8. key the layer 8 reference
  printed (rebuild/out/walk-numbers.json).
- The trained composed score's relative error is 1.30 at level 0 for r = 0.001 and under 0.04 from
  level 30 up, so a correction is needed most where the noise is lowest
  (artifacts/ideas/when-poe-reaches-a-and-b/runs/sampler-by-expert-interaction/README.md, "Score
  error").

Rules
- One rung per message, in LaTeX, with worked numbers from walk_numbers.
- Keep the exact-target regression and the frozen-expert loss apart at every rung where both
  appear.

What to hand back
- The drip's thread folder under artifacts/drips/, with dispatch.md recording my notation, the
  worked numbers I used and what clicked.
- The folder's path, which quiz 9.2, step 6's scene and step 7's cross-links read.

What you will see
Along the way: one rung per message, each ending on the drip's control line.
At the end, a closing report: what was asked, in one sentence; what was done, with the thread
folder's path; PASS or MISS against the target below; what is still unclear to me; at most one
question.

I judge, without notes: done when I write the regression and its target ∇r_t, and compute ∇r_t at the parts world's fixed image by hand to the digits the p8. key shows.
```

added: the seven sections, from the field-labelled version · the four derivations as separate objectives, in the order the facts set them out · "keep the two losses apart" as a rule (from the brick row's two losses, which converge to different things) · the rung format rule (from layer 1's maths drip)
dropped: nothing
