# Layer 6, step 1: the layer 6 reference

⬅️ [Layer 6, step 1](../06-guidance-as-a-dial.md#1-ai-in-the-background-the-layer-6-reference)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
when you start the reading stop, so the reference is ready for layer 6's compare step.

```text
In the background, build the layer 6 reference: rebuild/phase6_guidance.py, the guidance dial for
w = 1 to 7.5 on the quadrant toy and the parts world.

Background
The composition ladder in artifacts/drips/simple-distributions-rebuild/ ran every layer so far at
guidance weight 1, but Stable Diffusion's cat-dog hybrid happens at weight 7.5, and the correction
in layer 9 has to work at the weight the real model uses. Layer 6 turns guidance into a dial. Claude
writes the reference code in toy/rebuild/ one layer ahead, so that my own walk06.py has numbers to
match; the reference for layers 1 to 5 is already in rebuild/, with outputs in rebuild/out/phase1
to phase5. Work in /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy, on the
current branch.

Objectives
1. Write rebuild/guidance.py, the guided target as a function of w for both toys, starting from
   rebuild/ideal.py and rebuild/parts_world.py.
2. Write the guided target p(x)·p(A|x)^w·p(B|x)^w in code as
   (1 − 2w)·log p(x) + w·log p(x, A) + w·log p(x, B).
3. On the quadrant toy, compute it in closed form for the four-quadrant answer-key toy on ideal.py's
   241 × 241 grid (ideal.grid, ideal.CELL, ideal.grid_density), at A∩B shares r = 0.05 and 0.001.
4. In the parts world, build p(x), p(x|A) and p(x|B) as parts_world.Mixture objects over the
   placements phase5_dial.pw_experts builds, with p(x) the mixture of every placement either set
   holds, at both backgrounds; estimate Z and the A∩B share at w ≠ 1 by self-normalised importance
   sampling from exact draws of p(x) (Mixture.sample), whose weights p(A|x)^w·p(B|x)^w lie in
   [0, 1], and print each with its standard error.
5. On layer 5's toy, take p(x) as make_sets' two sets pooled and compute the guided target's A∩B
   share per w at f = 0.01, 0.05 and 0.2, expected near 1.0 at every w, as a check of the guided
   target's code.
6. Per toy and w, run tuned MALA, UHA and reverse diffusion on the noised guided target
   (samplers.Target), 2,000 samples each, with the settings phase5_dial.sample_2d uses, beside 2,000
   draws from the exact target as the floor (ideal.perfect_samples on the grid; exact draws in the
   parts world), scored by ideal.tv and parts_world.tv_projected.
7. Use a w grid from 1 to 7.5 that includes both ends, and find the w where the parts-world blend
   appears, or record that none up to 7.5 does.
8. Write rebuild/phase6_guidance.py, the dial, run from toy/ as python -m rebuild.phase6_guidance
   with /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python, printing a w = 1 check line first,
   then A∩B share, total variation and Z per toy and w.
9. Add those numbers to rebuild/walk_numbers.py as phase6() with keys p6.*, add it to the __main__
   tuple, and run it once with the co3 python.

What to read
- rebuild/ideal.py: the grid, the ideal experts, logq, perfect_samples and tv.
- rebuild/parts_world.py: Mixture, noised, sample, mixture_product and tv_projected.
- rebuild/phase5_dial.py: make_sets, pw_experts and sample_2d.
- rebuild/walk_numbers.py: how worked numbers are written.
- artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, brick "Guidance as a tilted
  density", and its Pins.
- artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/README.md and answer_key.py:
  the four-quadrant toy and its w = 1 numbers.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The brick is the target p(x)·p(A|x)^w·p(B|x)^w computed exactly on the quadrant toy and the parts
  world for w = 1 to 7.5, with A∩B share, total variation and Z per w, in rebuild/guidance.py and
  rebuild/phase6_guidance.py (REBUILD_WALK.md, brick "Guidance as a tilted density").
- SDXL composes "a cat" and "a dog" with one model and two prompts at w = 7.5 over 50 DDIM steps, as
  ε_∅ + w(ε_A − ε_∅) + w(ε_B − ε_∅); the tilted target is inferred from the cache's residual formula
  gs*(eps_j − eps_a − eps_b + eps_uncond), and the 2D work measured construction (1) only at w = 1
  (artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md, claim 1 and Sources).
- At w = 1 the target is construction (1), p(x|A)·p(x|B)/p(x), which equals construction (2)'s
  p_A·p_B only where p(x) is flat (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 3).
- The four-quadrant toy has one Gaussian blob per quadrant at (±0.5, ±0.5), std 0.15, A is x1 > 0
  and B is x2 > 0, and with "contains" labels and a share r in the top-right, construction (1) puts
  0.989 within 3 std of (0.5, 0.5) for every r from 0.25 to 0.001, with normaliser Z = r, and at
  r = 0 it puts 0.970 at the origin
  (artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/README.md, answer_key.py).
- 0.989 is 1 − e^(−9/2), the mass one blob holds within 3 std (when-poe-reaches-a-and-b idea map,
  run 4).
- Layer 5's quadrant sets (phase5_dial.make_sets) have an unconditional p(x), the two sets pooled,
  which is what one model trained on both with the labels dropped learns; with overlap share f, A is
  uniform on the left half with f of its mass top-left and B uniform on the top half with f
  top-left, so p(x) is (1 − f)/2 bottom-left, f top-left, (1 − f)/2 top-right and 0 bottom-right,
  and the guided target p(x)^(1 − 2w)·p_A^w·p_B^w is then f top-left and 0 elsewhere, so its A∩B
  share is 1 at every w for any f > 0, up to the ideal experts' smoothing at the square edges
  (rebuild/phase5_dial.py, make_sets; Bayes' rule).
- ideal.logq weights every data point equally, so blobs weighted r and (1 − r)/3 need a weighted
  form, and since a Gaussian noised to step t is N(√ᾱ_t μ, (ᾱ_t v + 1 − ᾱ_t) I), each blob's noised
  variance is ᾱ_t·0.15² + 1 − ᾱ_t (rebuild/ideal.py; parts_world.Mixture.noised).
- p(A|x) and p(B|x) are at most 1, so the guided target never exceeds p(x) and stays a density for
  every w ≥ 0 when all three densities come from one labelled set (Bayes' rule).
- Raising a Gaussian to the power w multiplies its precision by w (vault note Powers Of Gaussians).
- A power of a mixture is not a mixture, and parts_world.mixture_product's closed form covers the
  plain product only (rebuild/parts_world.py, module docstring).
- Success is three numbers reported together, the share in A∩B, the total variation to the exact
  composed density, and Z, because reverse diffusion's share 0.983 against 0.989 hid a total
  variation 0.049 above the floor (when-poe-reaches-a-and-b idea map, claim 5 and run 5).
- With exact experts at r = 0.05, annealed ULA, fixed MALA and UHA sit at the exact-sample floor
  (total variation 0.124 to 0.132 against 0.133 ± 0.006) and reverse diffusion with summed scores at
  0.182 (when-poe-reaches-a-and-b idea map, run 5).
- In layer 5's parts world, with an empty background (variance 0.01) the product at 0% overlap is a
  ghost with half-means (0.5, −0.5) and A∩B weight 0.000, and 1.000 from 0.1% up, and with a loose
  background (variance 1.0) it is 1.000 at every overlap (step file 05, step 6).
- The parts-world blend is example 9(b): both experts tight on every pixel, product mean 0, 10 sds
  from each expert, log Z −1583.4 (step file 04, step 4); no source shows that guidance produces it
  at any w.
- On the mosaic, reverse diffusion leaves the head mean at 0.923 where the product has 0.980, a
  projected total variation of 0.732 against a floor of 0.013 (step file 04, step 7).
- walk_numbers.py holds worked-example numbers, each written by put(key, value), while run results
  (masses and total variations per sampler) are read from rebuild/out/phaseN/ instead; its __main__
  runs (phase1, phase2, phase3, phase5) and writes rebuild/out/walk-numbers.json by a relative path,
  so it runs from toy/ (rebuild/walk_numbers.py).
- The tolerances are an exact number within 0.005 and a sampler number within 0.03
  (rebuild/phase3_kinds.py).
- rebuild.device.pick_device takes the first card with under 1 GB in use, else the CPU
  (rebuild/device.py; REBUILD_WALK.md, Pins).

Rules
- Add the two files and phase6() only.
- Leave ideal.py, parts_world.py, kinds.py and the phase 1 to 5 scripts as they are.
- Write nothing in the ladder, the step files or REBUILD_WALK.md.
- No commit.
- Never assume the parts-world blend appears: record None when no w up to 7.5 gives one.
- The w = 1 check prints first; then one row per w for the quadrant toy and then for the parts
  world, each with the share and Z, and per sampler the share and total variation.

What to hand back
- rebuild/guidance.py and rebuild/phase6_guidance.py; the phase script prints a w = 1 check line,
  marked matches or differs against the answer key, then one row per toy and w.
- rebuild/out/phase6/phase6.json with every row, and a figure of the guided target per w on both
  toys.
- phase6() in rebuild/walk_numbers.py, added to the __main__ tuple: the w grid as p6.w_grid, the
  exact share and Z per toy and w as p6.<toy>_<measure>_w<w>, and the blend's w as p6.blend_w (None
  when no w up to 7.5 gives one).
- rebuild/out/walk-numbers.json rewritten, its 84 earlier keys unchanged.
- A report naming the w grid, every p6. key with its value, and the brick row's new mark; steps 6
  and 7 of artifacts/drips/simple-distributions-rebuild/ladder/06-guidance-as-a-dial.md read the
  keys and phase6.json.
- The proof: python -m rebuild.phase6_guidance prints "w = 1 check: matches"; then
  python -m rebuild.walk_numbers | grep 'p6\.' lists every p6. key; the evidence is
  rebuild/out/phase6/phase6.json.

What you will see
Along the way: the w = 1 check line first, then the quadrant toy's rows, then the parts world's,
each estimated number with its standard error.
At the end, a closing report: what was asked, in one sentence; what was done, with each file's
path; PASS or MISS against the quadrant toy's top-right share within 0.005 of 0.989 and Z within
0.005·r of r at w = 1 for both values of r, every sampler row carrying its total variation beside
the floor, and every estimated parts-world number marked as an estimate with its standard error;
the blend's w or None; what is still open; at most one question.

I judge, from the printed rows: done when the w = 1 check matches at both values of r, every sampler row carries its floor, and every estimate carries its standard error.
```

added: the seven sections, from the field-labelled version · the layer's reason as Background (from the ladder file's "Why it is critical" line) · "never assume the blend appears" as a rule, from the earlier dropped line
dropped: the earlier added line's history of where each fact came from (why: every fact now carries its source in the prompt itself); the assumption that a parts-world blend appears at some w up to 7.5 stays dropped (why: no source shows one does, so the reference records None when none does)
