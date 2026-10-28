# Layer 5, step 6: the overlap-dial curves scene

⬅️ [Layer 5, step 6](../05-the-overlap-dial.md#6-ai-in-the-background-the-scene)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
in the background beside the reading stop.

```text
/results-scene rebuild/out/phase5/

Background
Layer 5 of the composition ladder (artifacts/drips/simple-distributions-rebuild/) measured how much
A-and-B data a product needs. A scene of the answer key's toy already exists, Three ways a product
reaches A-and-B (https://claude.ai/artifact/9eFnG2rWEciPsceMwdem4u). This scene is of my own
overlap-dial curves from rebuild/out/phase5/ in
/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy, in my notation, so I can
scrub along the overlap share.

Objectives
1. Build one scene of the overlap-dial curves that I can scrub along the overlap share.
2. Write it in the notation the layer 5 maths drip's dispatch.md records.
3. Publish it and write its URL into layer 5's step 6.

What to read
- toy/rebuild/out/phase5/overlap-dial.json: per seed and overlap, the exact whole-quadrant and deep
  masses, the exact-sample TV floor, and each sampler's masses and TV.
- toy/rebuild/out/phase5/finetune.json, dial.log and finetune.log.
- The figures in toy/rebuild/out/phase5/: overlap-dial-curves.png,
  overlap-dial-quadrant-pictures.png, finetune-curves.png and finetune-pictures.png.
- The layer 5 maths drip folder and its dispatch.md (step 2's hand-off): my notation.
- artifacts/scenes/three-ways-a-product-reaches-a-and-b/README.md: the other scene's toy, to keep
  apart.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- python -m rebuild.phase5_tables prints the three tables every number on the scene comes from: the
  quadrant toy over 8 overlaps and 3 seeds, the parts world over 7 overlaps and two background
  variances, and the fine-tune variant over 8 checkpoints plus experts trained from scratch
  (toy/rebuild/phase5_tables.py; step file 05, step 8).
- The quadrant toy: A is the left half, B the top half, A∩B the top-left quadrant, and "deep" means
  x1 < −0.15 and x2 > 0.15 (toy/rebuild/phase5_dial.py; step file 05, step 8).
- The quadrant toy runs 2,000 samples per sampler and 3 data seeds, with bars of 1 sd over the seeds
  (toy/rebuild/phase5_dial.py; step file 05, step 8).
- The existing figure colours black for exact, orange for tuned MALA, green for UHA and blue for
  reverse diffusion (step file 05, step 8).
- At 5% the exact deep mass is 0.30 ± 0.10 and tuned MALA 0.17 ± 0.04 (phase5_tables).
- At 0% the whole quadrant reads 0.20 ± 0.02 and the deep region 0.00 (phase5_tables).
- Reverse diffusion overshoots at 100%, 0.91 against 0.76 (phase5_tables).
- At fine-tune step 10,000 the grid counts a spurious blob near (0.75, 0.9) that the samplers never
  reach, so the grid's whole quadrant reads 0.021 while tuned MALA reads 0.291 and both samplers'
  TV is 0.90 (REBUILD_WALK.md, brick "Fine-tune variant"; finetune.log).
- The scene already built, Three ways a product reaches A-and-B, is on the four-quadrant blob toy of
  the answer key, a different toy with A∩B top-right
  (artifacts/scenes/three-ways-a-product-reaches-a-and-b/README.md).
- What the curves answer: the deep A∩B mass rounds to 0.00 at 0% on the quadrant toy, the parts
  world with an empty background gives a ghost instead, pretraining support does not survive 100
  fine-tune steps, and an expert loose where its concept is absent is the one way to A∩B with no
  overlap (step file 05, step 8).

Rules
- Every number on the scene comes from phase5_tables, never estimated.
- Keep the existing colours: black exact, orange tuned MALA, green UHA, blue reverse diffusion.
- Label the step-10,000 fine-tune point as the spurious blob rather than read it as a result.
- Keep the answer-key scene's numbers off this scene.
- List the scene's frames before building it, and wait for a yes.

What to hand back
- One published scene of the overlap-dial curves, scrubbable along the overlap share, in my notation
  from dispatch.md.
- Its URL in step 6 of artifacts/drips/simple-distributions-rebuild/ladder/05-the-overlap-dial.md,
  replacing "(URL lands here)".

What you will see
Along the way: the scene's frames listed before it is built, with a yes asked.
At the end, a closing report: what was asked, in one sentence; what was done, with the URL; PASS or
MISS against every number on the scene equal to phase5_tables to the digits it prints; what is
still open; at most one question.

I judge, by scrubbing it: done when every number on the scene equals phase5_tables to the digits it prints.
```

added: the seven sections, from the field-labelled version · the existing scene's URL in the Background (from the step's "Scene, already built" line) · the colours, the spurious-blob label and the other scene's toy turned into rules (from the settled facts that said "keep them", "label that point" and "keep its numbers off") · the frames listed before the build, as a stop (from the layer 1 scene prompt's shape)
dropped: nothing
