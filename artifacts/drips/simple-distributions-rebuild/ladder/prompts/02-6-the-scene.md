# Layer 2, step 6: the scene

⬅️ [Layer 2, step 6](../02-composing-two-experts.md#6-ai-in-the-background-the-scene)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
in the background beside layer 3's maths.

```text
/math-scene the folder the layer 2 --math drip wrote under artifacts/drips/: reverse diffusion with
summed scores leaving a faint ring between the two true modes of ring × bar, against annealed MALA
landing on them

Background
Layer 2 of the composition ladder (artifacts/drips/simple-distributions-rebuild/) taught the two
constructions, why summed scores miss the product, and the accept step. This scene shows reverse
diffusion with summed scores leaving a faint ring between ring × bar's two true modes, while
annealed MALA lands on them, in my own notation, so I can operate it rather than read about it.

Objectives
1. Build one scene of ring × bar sampled by reverse diffusion with summed scores and by annealed
   (tuned) MALA.
2. Write it in the notation the layer 2 maths drip's dispatch.md records.
3. Publish it and write its URL into layer 2's step 6.

What to read
- The layer 2 maths drip folder under artifacts/drips/ and its dispatch.md (step 2's hand-off): my
  notation.
- artifacts/drips/simple-distributions-rebuild/02-composing-and-sampling.md, steps 9 and 10: ring ×
  bar and its shares.
- artifacts/ideas/poe-overlap-dial/kinds-of-composition-on-the-papers-toys.ipynb, Figure 1 and run
  1: the faint ring's numbers.
- toy/rebuild/out/phase2/step9.json and toy/rebuild/out/phase2/phase2.json: the reference shares.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- Ring × bar is step file 02's step 9 target: the ring of eight Gaussian modes on a circle of radius
  0.5 (sd 0.03) and the bar uniform on |x1| < 0.2, |x2| < 1, 400 points each, smoothed by σ = 0.05
  (step file 02, step 9).
- The centres (0, 0.5) and (0, −0.5) are the only ring-mode centres inside the bar (step file 02,
  step 9).
- Because both experts are Gaussian smoothed, the product keeps tails everywhere: 0.933 of its mass
  lies within 0.2 of those two centres and 0.065 within 0.2 of the other six ring modes (step file
  02, step 9).
- On that target the exact share within 0.1 of the two in-bar modes is 0.713, and reverse diffusion
  gives 0.468, ULA 0.356, MALA at 0.05·ν_t 0.370 and UHA 0.563 (step file 02, step 9).
- The annealed samplers end at π_0, whose exact share is 0.636; after a clean final level of 10
  moves they give ULA 0.407, MALA 0.411 and UHA 0.620 against 0.713
  (toy/rebuild/out/phase2/step9.json).
- "Annealed MALA" in the scene is tuned MALA, adapted to 0.6 acceptance, because at the fixed step
  0.05·ν_t MALA accepts nearly every move and leaves the same stragglers as ULA (step file 02, step
  9).
- The faint ring is the kinds notebook's intersection row (ring × a bar of half-width 0.05): reverse
  diffusion reaches TV 0.31 against a floor of 0.11, and tuned MALA reaches 0.11 against 0.11
  (artifacts/ideas/poe-overlap-dial/kinds-of-composition-on-the-papers-toys.ipynb, Figure 1 and run
  1).
- With the trained phase 1 experts, the in-bar share is 0.928 for the trained composition, 0.580 for
  reverse diffusion and 0.850 for tuned MALA (toy/rebuild/out/phase2/phase2.json; step file 02,
  step 10).

Rules
- Every number on the scene comes from step file 02, the kinds notebook, phase2.json or the drip's
  dispatch.md, never estimated.
- List the scene's frames before building it, and wait for a yes.

What to hand back
- One published scene, in my notation from dispatch.md.
- Its URL in step 6 of artifacts/drips/simple-distributions-rebuild/ladder/02-composing-two-experts.md,
  replacing "(URL lands here)".

What you will see
Along the way: the scene's frames listed before it is built, with a yes asked.
At the end, a closing report: what was asked, in one sentence; what was done, with the URL; whether
every number on it traced to step file 02, the kinds notebook, phase2.json or dispatch.md; what is
still open; at most one question.

I judge, by operating it: done when reverse diffusion leaves the faint ring, tuned MALA lands on the two in-bar modes, and every number on it matches my drip's and is traceable.
```

added: the seven sections, from the field-labelled version · the frames listed before the build, as a stop (from the layer 1 scene prompt's shape) · the target, from the step's stop line ("the scene is published and its numbers match your drip's") and its Outcome
dropped: nothing (the earlier version had already dropped "the product keeps only the two ring modes", from the Codex mathematics review, since the smoothed product keeps mass elsewhere)
