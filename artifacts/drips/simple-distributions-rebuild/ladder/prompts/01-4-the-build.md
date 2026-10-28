# Layer 1, step 4: the build

⬅️ [Layer 1, step 4](../01-one-diffusion-expert.md#4-you-the-build)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`.

```text
/drip-execute-plan artifacts/drips/simple-distributions-rebuild/01-one-expert-from-scratch.md

Background
This is layer 1's build in the composition ladder (artifacts/drips/simple-distributions-rebuild/):
I type one 2D diffusion expert from an empty file, following the step file one step per message.
Claude's reference code is toy/rebuild/ in this repo; my file and the reference print the same
numbers, and a difference is a lesson or a bug. I have passed quiz 1.2, and the layer 1 maths drip
folder under artifacts/drips/ holds my notation in its dispatch.md.

Objectives
1. Walk the step file's steps 0 to 11, one per message, in its order.
2. I type every block into ~/Playground/PhD/simple-distributions-walk/walk01.py and run it from that
   folder with /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python and
   PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy.
3. Tick each step in the step file as it lands, and keep my place in the walk map between sittings.
4. End with a samples figure from my trained expert beside toy/datasets-1000-samples-each.png.

What to read
- artifacts/drips/simple-distributions-rebuild/01-one-expert-from-scratch.md: the step file.
- artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, Pins and Bricks: what each brick
  was checked against.
- toy/rebuild/datasets.py, toy/rebuild/model.py, toy/rebuild/train.py: the reference.
- The layer 1 maths drip's dispatch.md: my notation.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- Every worked number prints from python -m rebuild.walk_numbers, run from toy/, as 41 p1. keys
  (REBUILD_WALK.md, Pins; toy/rebuild/out/walk-numbers.json).
- The ResNet has 565,506 parameters; the notebook prints 565,606 because it counts out_logvar, which
  the beta_forward variance never reads (REBUILD_WALK.md, brick "ResNet score network").
- With zero output layers, the noise prediction equals 2x at initialisation (same brick).
- The noise prediction is the gradient of the energy E = |net(x, t) − x|² (brick "Energy
  parameterisation").
- The schedule is computed in float64 (brick "Cosine schedule and DDPM").
- The full run is 15,001 steps of Adam at 1e-3, batch 1000, with a moving average of 0.999, about 5
  minutes on one card; step 9 builds a short version of 2,000 steps on the bar dataset (brick
  "Training"; step file, step 9).
- A trained expert passes when the total variation between 5,000 model and 5,000 data samples, on
  40 × 40 bins of [−1.5, 1.5]², is within 0.10 of the total variation between two data draws
  (brick "Training").
- 9 of 10 datasets pass; the ring passes only without the last step's noise, 0.179 with it and
  0.075 without, against a floor of 0.049 (step file, step 11).
- After 2,000 steps the bar dataset reads 0.239 against a floor of 0.098, a miss by design; the full
  run reads 0.115 against 0.095 (step file, step 11).
- A deepcopy of the moving-average model has its gradients switched off, so fine-tuning it trains
  nothing and raises no error (step file, step 9's broken run).
- Train on a card with under 1 GB in use, else on the CPU (step file, step 0; the GPU guard in
  environment/hpc/execution-protocol.md).

Rules
- I type; nothing is pasted into walk01.py.
- Never edit toy/rebuild/ or the step file's code blocks.
- No commit.
- Step 1 prints the normal density values, step 4 the ᾱ_t values, step 6 the parameter count, step 8
  the initial loss on the bar dataset, and step 11 the total-variation line; show each before moving
  on.

What to hand back
- walk01.py, one script that runs end to end and prints each step's worked number.
- The step file's steps 0 to 11 ticked, and the walk map.
- The samples figure beside toy/datasets-1000-samples-each.png.
- The retrospective the walk closes on, and the /drip-retro command it prints.
- The proof: python walk01.py prints "bar: TV model <x>  floor <y>" with x − y ≤ 0.10 after the full
  run, and layer 1's compare step shows no differing number.

What you will see
Along the way: one step per message, each ending on the number it printed and a yes to move on.
At the end, a closing report: what was asked, in one sentence; what was done, with walk01.py's path
and the figure's; PASS or MISS against every p1. number equal to walk_numbers to the digits shown,
565,506 parameters, and the bar dataset within 0.10 of its floor after the full run; what is still
open; at most one question.

I judge, from the printed lines: done when every p1. number matches, the count is 565,506, and the bar dataset's total variation sits within 0.10 of its floor.
```

added: the seven sections, from the field-labelled version · the step file's own place-keeping as an objective
dropped: nothing (the earlier version had already dropped "samples match the figure" as a pass line, since it has no tolerance; the figure stays as an output to look at)
