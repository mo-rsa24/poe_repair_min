# Layer 9, step 6: the fair-test scene

⬅️ [Layer 9, step 6](../09-the-correction-learned-back.md#6-ai-in-the-background-the-scene)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
beside layer 12's gate.

```text
/results-scene the layer 9 fair-test outputs the walk wrote under
~/Playground/PhD/simple-distributions-walk/

Background
Layer 9 of the composition ladder (artifacts/drips/simple-distributions-rebuild/) trained a
correction and ran the fair test: a joint condition, composition, and composition plus a
correction, at two overlaps, with exact and with trained experts. This scene shows that result in
my own notation, beside run 7 as the baseline, so I can operate it rather than read a table.

Objectives
1. Build one scene of TV above the floor for each arm at r = 0.05 and 0.001, exact and trained
   experts, with the samples behind each cell.
2. Show TV, the top-right share and the sample centre together for every cell.
3. Write it in the notation the layer 9 maths drip's dispatch.md records.
4. Publish it and write its URL into layer 9's step 6.

What to read
- The fair-test table walk09.py printed (step 4).
- toy/rebuild/out/phase9/phase9.json and fair-test.png: the reference's outputs (step 0).
- The layer 9 --math drip folder under artifacts/drips/ and its dispatch.md: my notation (step 2).
- artifacts/ideas/when-poe-reaches-a-and-b/runs/sampler-by-expert-interaction/README.md: run 7's
  card, the baseline.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The fair test is a joint {A, B} condition trained on only the share r of the data that is A and
  B, against composition, against composition plus a small correction fine-tuned on the same few
  A∩B samples, at r = 0.05 and 0.001, with exact and with trained experts, scored by TV against the
  exact joint (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 9; REBUILD_WALK.md,
  brick "The correction, learned back").
- The toy is four blobs at (±0.5, ±0.5) with σ = 0.15, and the exact joint is the top-right blob
  (run 7's card).
- At r = 0.001 each training set of 10,000 holds 11 top-right samples for p(x) and 23 to 34 for
  each conditional (run 7's card).
- Run 7 is the baseline the scene sits beside: trained MALA +0.267 ± 0.056 above the floor at
  r = 0.001 against +0.001 ± 0.006 with exact experts, the interaction +0.232 ± 0.046 (the run
  card).
- TV, the top-right share and the sample centre appear together, because the share alone hid
  trained MALA's miss at r = 0.001 (0.989 share, TV 0.391) (claim 5; the run card).
- ~/Playground/PhD/simple-distributions-walk/ is scratch space, so the scene keeps its own copy of
  every number it shows (~/.claude/CLAUDE.md, the playground is disposable).

Rules
- Every number comes from walk09.py's table, phase9.json or the run card, never estimated.
- List the scene's frames before building it, and ask for a yes.

What to hand back
- One published scene, in my notation from dispatch.md.
- Its URL in step 6 of
  artifacts/drips/simple-distributions-rebuild/ladder/09-the-correction-learned-back.md, replacing
  "(URL lands here)".

What you will see
Along the way: the scene's frames listed before it is built, with a yes asked.
At the end, a closing report: what was asked, in one sentence; what was done, with the URL; whether
every number on it traced to walk09.py's table, phase9.json or the run card, and whether they match
my walk's; what is still open; at most one question.

I judge, by operating it: done when the scene is published, shows TV, share and centre for every arm at both overlaps, and its numbers match my walk's.
```

added: the seven sections, from the field-labelled version · the frames-before-build stop (from layer 1's scene step) · the run card under What to read (from the facts that cite it) · the target, from this step's stop line on the page ("the scene is published and its numbers match your walk's")
dropped: nothing
