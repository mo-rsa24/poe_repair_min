# Layer 4, step 4b: the summed-score error, walk04b.py

⬅️ [Layer 4, step 4b](../04-the-parts-world.md#4b-you-the-second-build-the-summed-score-error-in-walk04bpy)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`.

```text
/drip-execute-plan --request setting (d) of
artifacts/ideas/poe-overlap-dial/example-9-parts-world/parts_world.py by hand in
~/Playground/PhD/simple-distributions-walk/walk04b.py; bar: summed scores at ᾱ = 0.5 put the head at
0.233 where the noised product puts it at 0.693, against numbers.json

Background
This is the second build in layer 4 of the composition ladder
(artifacts/drips/simple-distributions-rebuild/): Du et al.'s summed-score error, computed exactly
in the parts world. Step 6 of the step file already prints both numbers inside walk04.py; this walk
rebuilds them in a file of its own from parts_world.py, without walk04.py's code open. I start from
walk04.py from step 4, finished, its retrospective, and the layer 4 maths drip folder.

Objectives
1. Cut the request into 4 to 6 pieces at the open, and write the plan file in
   artifacts/drips/simple-distributions-rebuild/ beside 04-the-parts-world.md.
2. Build one piece a round, each ending on a test I run by hand.
3. I type every block into walk04b.py and run it from ~/Playground/PhD/simple-distributions-walk/
   with /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python.
4. End with walk04b.py printing the two head means beside numbers.json's values.

What to read
- artifacts/ideas/poe-overlap-dial/example-9-parts-world/parts_world.py, lines 43 to 46: setting
  (d).
- artifacts/ideas/poe-overlap-dial/example-9-parts-world/numbers.json, setting d: the reference
  values.
- artifacts/drips/simple-distributions-rebuild/04-the-parts-world.md, steps 6 and 7: the noising
  rule and what the error costs reverse diffusion.
- The layer 4 maths drip folder and its dispatch.md: my notation.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The decisions the open would ask are settled at the invocation: build in walk04b.py, the bar is
  the one above, the time box is 2 hours, and the plan file goes in
  artifacts/drips/simple-distributions-rebuild/ beside 04-the-parts-world.md (layer 4, step 4b).
- Setting (d) is lines 43 to 46 of parts_world.py: setting (a)'s two experts, each noised to
  ᾱ = 0.5, multiplied, and the head mean of that product set beside √0.5 times the clean product's
  head mean (parts_world.py).
- Noising a diagonal Gaussian to ᾱ gives N(√ᾱ m, ᾱ v + 1 − ᾱ), and a mixture noises each component
  and keeps its weights (step file 04, step 6).
- numbers.json holds head_mean_product_of_noised 0.23257000444341597 and head_mean_noised_product
  0.6931046667076061 at abar 0.5 (numbers.json, d).
- Step 6 of the step file already prints both numbers inside walk04.py (step file 04, step 6).
- Summing the noised experts' scores targets the product of the noised experts, whose tight
  variances were blurred toward 1 before the product could use them, so the summed score points
  the head at a third of where it should be (step file 04, step 6; Du et al. 2023, p. 4, eqs. 11
  and 12, as the overlap-dial idea map's Sources cite).
- Reverse diffusion accumulates this error over 100 steps: head mean 0.923 against 0.980, projected
  TV 0.732 against the exact-draw baseline of 0.013 (step file 04, step 7; phase4.json).

Rules
- Never edit parts_world.py, numbers.json or toy/rebuild/.
- I type; I do not paste.
- No commit.
- Stop at the 2-hour time box.
- Each round ends on a test that prints one number I can check against the step file or
  numbers.json, shown before moving on.

What to hand back
- walk04b.py, printing the two head means beside numbers.json's values.
- The plan file the open writes, its pieces ticked.
- The walk map, and the retrospective the walk closes on with the /drip-retro command it prints.
- The proof: python walk04b.py prints both head means beside numbers.json's, equal to 9 significant
  figures.

What you will see
Along the way: one piece per round, each ending on the number its test printed and a yes to move
on.
At the end, a closing report: what was asked, in one sentence; what was done, with walk04b.py's
path and the plan file's; PASS or MISS against walk04b.py printing 0.2326 and 0.6931, each equal to
numbers.json's d values to at least 9 significant figures; what is still open; at most one
question.

I judge, from the printed line: done when walk04b.py prints 0.2326 and 0.6931, each equal to numbers.json's d values to 9 significant figures.
```

added: the seven sections, from the field-labelled version · the 2-hour time box as a stop in Rules (from the settled invocation decisions)
dropped: nothing (the earlier version had already replaced "a floor of 0.013" with the exact-draw baseline, from the Codex mathematics review)
