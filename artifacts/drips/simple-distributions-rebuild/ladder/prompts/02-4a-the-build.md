# Layer 2, step 4a: the build, walk02.py

⬅️ [Layer 2, step 4a](../02-composing-two-experts.md#4a-you-the-build-walk02py)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`.

```text
/drip-execute-plan artifacts/drips/simple-distributions-rebuild/02-composing-and-sampling.md, I type
every block myself into ~/Playground/PhD/simple-distributions-walk/walk02.py

Background
This is layer 2's build in the composition ladder (artifacts/drips/simple-distributions-rebuild/):
I type my own product of two experts, annealed ULA and per-sample MALA from the step file, one step
per message. Claude's reference code is under toy/ in this repo; my file and the reference print
the same numbers, and a difference is a lesson or a bug. I start from the setup's folder and
environment, with quiz 2.2 passed, and the layer 2 maths drip folder under artifacts/drips/ holds my
notation in its dispatch.md.

Objectives
1. Walk the step file's steps 0 to 10, one per message, in its order.
2. I type every block into ~/Playground/PhD/simple-distributions-walk/walk02.py and run it from that
   folder with /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python and
   PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy.
3. Save step07-acceptance.png (one sample's acceptance over 10 moves) and step09-ring-times-bar.png
   (the four samplers side by side) in the typing folder.
4. Tick each step in the step file as it lands, and keep my place in the walk map between sittings.

What to read
- artifacts/drips/simple-distributions-rebuild/02-composing-and-sampling.md: the step file.
- toy/rebuild/compose.py, toy/rebuild/samplers.py and toy/anneal_samplers.py: the reference.
- toy/rebuild/out/walk-numbers.json and toy/rebuild/out/phase2/step9.json: the worked numbers.
- The layer 2 maths drip's dispatch.md: my notation.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The reference is rebuild/compose.py, rebuild/samplers.py and the fixed anneal_samplers.py in toy/
  (step file 02, opening).
- Every worked number prints from python -m rebuild.walk_numbers, run from toy/, as 53 p2. keys
  (step file 02, opening; toy/rebuild/out/walk-numbers.json).
- The 14 p2.step9_ keys among them are read from toy/rebuild/out/phase2/step9.json, produced by
  python -m rebuild.phase2_step9 (step file 02, opening; toy/rebuild/out/walk-numbers.json).
- Step 0 imports the cosine schedule and the device guard from rebuild/ and the samplers from
  anneal_samplers.py, so walk02.py runs without walk01.py (step file 02, opening).
- Train or sample on a card with under 1 GB in use, else on the CPU (toy/rebuild/device.py; the GPU
  guard in ~/Playground/PhD/poe_repair_min/environment/hpc/execution-protocol.md).
- anneal_samplers.py divides the MALA proposal's log-density by 4·ss and accepts each sample on its
  own (REBUILD_WALK.md, brick "MALA and CHA fix").
- tests/test_anneal_samplers.py holds 6 tests and all pass (REBUILD_WALK.md, brick "MALA and CHA
  fix").
- The notebook's mixture is sharpened by a factor 3.5: E_mix = −logsumexp(−3.5E_A, −3.5E_B), which
  is 3.4991 at E_A = 1, E_B = 3 (step file 02, step 3).
- The notebook's negation uses w = 0.3, and p_A^1.3·p_B^−0.3 = 0.3299 at p_A = 0.5, p_B = 2 (step
  file 02, step 4).
- ULA at ss = 0.9 on N(0, 1) has stationary variance 2/(2 − ss) = 1.8182, and 200,000 chains for
  300 moves give 1.806 (step file 02, step 5; p2.ula_stationary_var_ss0.9,
  p2.ula_simulated_var_ss0.9).
- With the fixed divisor the MALA variance comes out near 1 and with the old one about four times
  that: the step file's run printed 0.989 and 3.745, the test file's seed 0.989 and 4.026 (step file
  02, step 6's broken run).
- Per-sample acceptance on N(0, 1) at ss = 1.5, 4,000 chains and 10 moves has mean 0.656 and sd
  0.153 across samples (step file 02, step 7).
- One accept decision for the whole batch would make all 4,000 entries equal, while one uniform
  shared but compared with each sample's own α would not (step file 02, step 7).
- leapfrog_step_c returns the tensor it was given, so the old CHA's accept test read the change in
  log p alone, and the fixed class passes copies (step file 02, step 8;
  p2.cha_alias_same_tensor True).
- The leapfrog toy gives x' 1.105, v' 0.18425 and H 0.625 to 0.62749 (step file 02, step 8).
- Step 9's target is ring × bar as exact ideal experts, 400 points each, smoothed by σ = 0.05 (step
  file 02, step 9).
- The centres (0, 0.5) and (0, −0.5) are the only ring-mode centres inside the bar; the composition
  puts 0.933 of its mass within 0.2 of them and, because both experts are Gaussian smoothed, 0.065
  within 0.2 of the other six ring modes (step file 02, step 9).
- Step 9's step sizes are 0.05·ν_t for ULA and MALA and 0.3·√ν_t for UHA, with
  ν_t = 1 − ᾱ_t + ᾱ_t·σ², 10 moves per level over 100 levels (step file 02, step 9).
- The last annealing level has ᾱ_0 = 0.9994, so the anneal ends at π_0, the composition smoothed
  further (variance 0.00313 per component against the clean 0.0025) (step file 02, step 9).
- π_0's exact share within 0.1 of the two in-bar modes is 0.636 (p2.step9_exact_share_pi0); the
  clean composition's is 0.713 (p2.step9_exact_share_clean); walk02.py's ULA, MALA and UHA shares
  are read against 0.636 (step file 02, step 9).
- The step file's run printed shares on the two in-bar modes of exact 0.713 (the clean
  composition), 1000 perfect draws 0.718, ULA 0.356, MALA 0.370, UHA 0.563 and reverse diffusion
  0.468 (step file 02, step 9).
- MALA at 0.05·ν_t accepts nearly every move and so behaves like ULA (step file 02, step 9).
- python -m rebuild.phase2_step9 reruns step 9 with the same seeds and adds a clean final level, 10
  moves on the clean composition after the anneal (ss = 0.05·σ² for ULA and MALA, h = 0.3·σ for
  UHA), giving ULA 0.407, MALA 0.411 and UHA 0.620, read against 0.713
  (toy/rebuild/out/phase2/step9.json; p2.step9_share_*_plus_clean_level).
- Step 10 runs python -m rebuild.phase2_compose on the phase 1 checkpoints in
  toy/rebuild/out/phase1/ckpt/ (step file 02, step 10).
- The trained t = 0 energy splits the in-bar mass 0.10 and 0.90 where the data's product splits it
  0.54 and 0.46, so 2D ground truth comes from ideal experts, never from a trained energy at t = 0
  (step file 02, step 10; toy/rebuild/out/phase2/phase2.json).

Rules
- I type; nothing is pasted into walk02.py.
- Never edit toy/rebuild/, toy/anneal_samplers.py or the step file's code blocks.
- No commit.
- Show each printed number before moving on: step 1 prints mean 1.0, variance 0.5 and log Z
  -2.2655; step 2 prints 0.9802, 0.6931 and 0.2326; step 5 the ULA variance against 1.8182; step 6
  the two stationary variances; step 7 the acceptance mean and sd; step 8 "leapfrog returned the
  tensor it was given: True"; step 9 one share per sampler.

What to hand back
- walk02.py, one script that runs end to end and prints each step's worked number.
- step07-acceptance.png and step09-ring-times-bar.png in the typing folder.
- The step file's steps 0 to 10 ticked, and the walk map that holds my place between sittings.
- The retrospective the walk closes on, and the /drip-retro command it prints.
- The proof: python walk02.py prints "acceptance per sample: mean <m>  sd <s>" with s above 0, and
  the comparison in step 5 of layer 2 shows no differing p2. number.

What you will see
Along the way: one step per message, each ending on the number it printed and a yes to move on.
At the end, a closing report: what was asked, in one sentence; what was done, with walk02.py's path
and the two figures' paths; PASS or MISS against every p2. number equal to walk_numbers to the
digits shown, a per-sample acceptance sd above 0 at step 7, and at step 9 the exact share 0.713
with UHA's share above ULA's and MALA's, all three read against π_0's 0.636, and
step09-ring-times-bar.png beside toy/rebuild/out/phase2/reference/product-0.png showing samples on
the two in-bar modes; what is still open; at most one question.

I judge, from the printed lines and the picture: done when every p2. number matches walk_numbers, step 7's acceptance sd is above 0, and step 9 prints 0.713 with UHA above ULA and MALA against 0.636.
```

added: the seven sections, from the field-labelled version · the two figures as an objective
dropped: nothing (the earlier version had already dropped "my ring × bar matches the paper's figure" as a pass line, since it has no tolerance; the figure stays as a picture to read beside `product-0.png`, and the per-sample acceptance is checked by its sd above 0)
