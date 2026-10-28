# Layer 4, step 4: the build, walk04.py

⬅️ [Layer 4, step 4](../04-the-parts-world.md#4-you-the-build-walk04py)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`.

```text
/drip-execute-plan artifacts/drips/simple-distributions-rebuild/04-the-parts-world.md, I type every
block myself into ~/Playground/PhD/simple-distributions-walk/walk04.py; bar: the mosaic head 0.980,
the blend at 0, and the side-by-side weights printing 0.5 each in float64 with same-half pairs at
log weight -803.2

Background
This is layer 4's build in the composition ladder (artifacts/drips/simple-distributions-rebuild/):
I type the parts world from the step file, one step per message, and produce the mosaic, the blend
and two animals side by side, all from one product. Claude's reference code is under toy/ in this
repo; my file and the reference print the same numbers, and a difference is a lesson or a bug. I
start from the setup's folder and environment, with quiz 4.2 passed, and the layer 4 maths drip
folder under artifacts/drips/ holds my notation in its dispatch.md.

Objectives
1. Walk the step file's steps 0 to 8, one per message, in its order.
2. I type every block into walk04.py and run it from ~/Playground/PhD/simple-distributions-walk/
   with /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python and
   PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy.
3. Save step03-mosaic.png, the cat, the dog and the mosaic as tiles, in
   ~/Playground/PhD/simple-distributions-walk/.
4. Tick each step in the step file as it lands, and keep my place in the walk map between sittings.

What to read
- artifacts/drips/simple-distributions-rebuild/04-the-parts-world.md: the step file, including
  "Where the numbers come from".
- toy/rebuild/parts_world.py and toy/rebuild/phase4_parts_world.py: the reference.
- artifacts/ideas/poe-overlap-dial/example-9-parts-world/numbers.json: the four example settings.
- toy/rebuild/out/phase4/phase4.json: the samplers.
- The layer 4 maths drip's dispatch.md: my notation.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The reference is rebuild/parts_world.py and rebuild/phase4_parts_world.py, which match
  numbers.json to 1e-9 relative, largest relative difference 2.87e-16 (REBUILD_WALK.md, brick
  "Parts world"; toy/rebuild/out/phase4/phase4.json).
- The reference numbers are numbers.json for the four example settings and phase4.json for the
  samplers (step file 04, "Where the numbers come from").
- python -m rebuild.walk_numbers holds 11 p4. keys for the step file's closed-form checks, among
  them the cat expert's 1616-nat gap, Δ = 779.1241 (p4.delta_side_minus_same) and the exact
  same-half weight 2.14e-339 (p4.w_same_half_exact) that float64 prints as 0.0
  (toy/rebuild/out/walk-numbers.json).
- phase4.json's outside_projection_square counts 0 projected points outside [−1.5, 1.5]² in every
  run (step file 04, "Where the numbers come from").
- Step 0 imports cosine_beta_schedule from rebuild.model and pick_device from rebuild.device, with
  T = 100 and D = 16, so walk04.py fails with ModuleNotFoundError without the PYTHONPATH above
  (step file 04, step 0; setup, step 1).
- Pixels 0 to 7 are the head and 8 to 15 the body; LEFT is [0, 1, 4, 5, 8, 9, 12, 13]; the cat is +1
  and the dog −1 (step file 04, step 1).
- Setting (a), the mosaic: head mean 0.9802, body −0.9802, log Z −46.47 (numbers.json; step file
  04, step 3).
- Setting (b), the blend: mean 0, sd 0.0707, log Z −1583.4 (numbers.json; step file 04, step 4).
- Setting (c), two animals: side-by-side pairs at log weight −24.09, same-half pairs −803.21, a gap
  of 779.12, so exactly w_side = 1/(2(1 + e^−779.12)), and float64 prints it as 0.5 and the
  same-half weight as 0.0 after underflow (numbers.json; step file 04, step 5).
- Setting (d): at ᾱ = 0.5, head mean 0.2326 for the product of the noised experts against 0.6931 for
  the noised product (numbers.json; step file 04, step 6).
- Step 7 scores each sampler against 100,000 exact draws by projected TV (two half-means, 40 × 40
  bins of [−1.5, 1.5]²) and by component-weight error (step file 04, step 7).
- np.histogram2d drops points outside the square, and none fall outside in any run behind the step
  file (step file 04, step 7).
- A separate 2,000 exact draws give the finite-sample baseline, not a lower bound: a sampler can
  score below it by chance (step file 04, step 7).
- With 1,000 samples on the mosaic the reference run printed UHA head mean 0.980, projected TV
  0.048; tuned MALA 0.977, 0.059; reverse diffusion 0.922, 0.746; 1,000 exact draws against the
  rest 0.027 (step file 04, step 7).
- With 2,000 samples, phase4.json holds mosaic projected TV 0.013 perfect, 0.732 reverse diffusion,
  0.200 ULA, 0.191 MALA, 0.036 tuned MALA and 0.036 UHA (toy/rebuild/out/phase4/phase4.json).
- In that run tuned MALA accepts 0.606 of its moves and plain MALA 0.992
  (toy/rebuild/out/phase4/phase4.json).
- Reverse diffusion with summed scores is step 7's broken run: it targets the product of the noised
  experts at every level, so the head mean ends at 0.923 where the product has 0.980 (step file 04,
  step 7; overlap-dial idea map, "What the words are", Du et al. 2023 Section 3).
- Train or sample on a card with under 1 GB in use, else on the CPU (REBUILD_WALK.md, Pins).

Rules
- I type; nothing is pasted into walk04.py.
- Never edit toy/rebuild/, the example-9 folder or the step file's code blocks.
- No commit.
- Show each printed line before moving on: step 1 prints the head and left pixel lists, step 2 the
  cat expert's log-density gap beside scipy's, step 3 the mosaic line, step 4 the blend line, step
  5 the four pair lines, step 6 the ᾱ = 0.5 line, step 7 one line per sampler, step 8 the seven
  comparisons.

What to hand back
- walk04.py, one script that runs end to end and prints each step's worked number.
- step03-mosaic.png in ~/Playground/PhD/simple-distributions-walk/.
- The step file's steps 0 to 8 ticked, and the walk map that holds my place between sittings.
- The retrospective the walk closes on, and the /drip-retro command it prints.
- The proof: python walk04.py prints, in step 8, seven "yours ... numbers.json ..." lines with equal
  values to 9 significant figures, and the comparison in step 5 of layer 4 shows no differing
  number.

What you will see
Along the way: one step per message, each ending on the line it printed and a yes to move on.
At the end, a closing report: what was asked, in one sentence; what was done, with walk04.py's path
and the figure's; PASS or MISS against step 8's seven lines agreeing with numbers.json to at least 9
significant figures (which covers the bar's 0.980, 0 for the blend's mean, 0.5 and −803.2), and
step 7's tuned MALA and UHA landing near the 0.027 exact-draw baseline with reverse diffusion far
above it; what is still open; at most one question.

I judge, from the printed lines: done when step 8's seven lines agree with numbers.json to 9 significant figures and step 7 puts tuned MALA and UHA near 0.027 with reverse diffusion far above.
```

added: the seven sections, from the field-labelled version · the figure as an objective
dropped: nothing (the earlier version had already dropped "co-occurrence weights 0.5 and 0.5" as exact values and "the floor" for the exact draws, from the Codex mathematics review)
