# Layer 6, step 6: the build

⬅️ [Layer 6, step 6](../06-guidance-as-a-dial.md#6-you-the-build)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
after quiz 6.2.

```text
/drip-execute-plan --request the guidance dial for w = 1 to 7.5 on the quadrant toy and the parts
world, starting from rebuild/ideal.py and rebuild/parts_world.py, reporting A∩B share, total
variation and Z per w, as layer 6 of artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md; I
type every block myself into ~/Playground/PhD/simple-distributions-walk/walk06.py

Background
This is layer 6's build in the composition ladder (artifacts/drips/simple-distributions-rebuild/): I
type the guidance dial from an empty file, one piece per round. Stable Diffusion's cat-dog hybrid
happens at guidance weight 7.5, and every earlier layer ran at 1, so this layer measures what the
weight does on both toys. Claude's reference is rebuild/guidance.py and rebuild/phase6_guidance.py,
built in step 1; my file and the reference print the same numbers, and a difference is a lesson or
a bug. I have passed quiz 6.2, and the layer 6 maths drip folder under artifacts/drips/ holds my
notation in its dispatch.md.

Objectives
1. At the walk's open, cut the layer into 4 to 6 pieces and write them as the step file
   artifacts/drips/simple-distributions-rebuild/06-guidance-as-a-dial.md, in the shape of the step
   files for layers 1 to 5.
2. Build, teach and test one piece per round, in this order: the w = 1 check, the quadrant toy's
   share and Z per w, the samplers' total variation, then the parts world.
3. I type every block into walk06.py and run it from ~/Playground/PhD/simple-distributions-walk/
   with /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python and
   PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy.
4. Use the reference's w grid in walk06.py.
5. Show that w = 1 reproduces construction (1) exactly.
6. Find the w where the parts-world blend appears, or show that none up to 7.5 does.
7. Tick each piece in the step file as it lands, and keep my place in the walk map between sittings.

What to read
- The layer 6 maths drip folder under artifacts/drips/ and its dispatch.md (step 4): my notation.
- rebuild/guidance.py, rebuild/phase6_guidance.py, the p6. keys and rebuild/out/phase6/phase6.json
  (step 1): the reference.
- artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, brick "Guidance as a tilted
  density".
- artifacts/drips/simple-distributions-rebuild/05-the-overlap-dial.md: the step-file shape and "How
  to walk it".

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The w grid, the p6. keys and phase6.json are step 1's, and walk06.py uses the same grid (from
  step 1).
- The brick is the target p(x)·p(A|x)^w·p(B|x)^w computed exactly on the quadrant toy and the parts
  world for w = 1 to 7.5, with A∩B share, total variation and Z per w (REBUILD_WALK.md, brick
  "Guidance as a tilted density").
- SDXL's hybrid comes from one model and two prompts at w = 7.5 over 50 DDIM steps, the tilted
  target is inferred from the cache's residual formula, and the 2D work measured only w = 1
  (artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md, claim 1 and Sources).
- At w = 1 the target is construction (1), p(x|A)·p(x|B)/p(x), which equals p_A·p_B only where p(x)
  is flat (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 3).
- The quadrant toy is the four-quadrant answer-key toy, blobs at (±0.5, ±0.5), std 0.15, A is x1 > 0
  and B is x2 > 0, and at w = 1 with "contains" labels it puts 0.989 within 3 std of (0.5, 0.5) for
  every A∩B share r from 0.25 to 0.001, with Z = r
  (artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/README.md, answer_key.py).
- Layer 5's quadrant sets (phase5_dial.make_sets) have an unconditional p(x), the two sets pooled,
  and on them the guided target's A∩B share is near 1 at every w for any f > 0 (step 1's
  reference).
- p(A|x) and p(B|x) are at most 1, so the guided target never exceeds p(x) and stays a density for
  every w ≥ 0 (Bayes' rule).
- A power of a mixture is not a mixture, so in the parts world Z and the A∩B share at w ≠ 1 are
  estimates with a standard error (rebuild/parts_world.py; from step 1).
- Success is three numbers together, the share in A∩B, the total variation to the exact density,
  and Z, because reverse diffusion's share 0.983 against 0.989 hid a total variation 0.049 above the
  floor (when-poe-reaches-a-and-b idea map, claim 5 and run 5).
- With exact experts at r = 0.05, annealed ULA, fixed MALA and UHA sit at the floor (0.124 to 0.132
  against 0.133 ± 0.006) and reverse diffusion at 0.182 (when-poe-reaches-a-and-b idea map, run 5).
- In layer 5's parts world, an empty background (variance 0.01) gives a ghost at 0% overlap and A∩B
  weight 1.000 from 0.1% up, and a loose background (variance 1.0) gives 1.000 at every overlap
  (step file 05, step 6).
- The parts-world blend is example 9(b), product mean 0, 10 sds from each expert, log Z −1583.4
  (step file 04, step 4), and no source shows that guidance produces it at any w.
- The tolerances are an exact number within 0.005 and a sampler number within 0.03
  (rebuild/phase3_kinds.py).
- Phases 1 to 5 are imported from the reference, as step file 05 does (step file 05, "How to walk
  it").

Rules
- I type; I do not paste.
- Never edit rebuild/ or the ladder files.
- No commit.
- Each piece prints its number as it lands: the w = 1 check first, then the quadrant toy's share and
  Z per w, then the samplers' total variation, then the parts world; show each before moving on.

What to hand back
- The step file artifacts/drips/simple-distributions-rebuild/06-guidance-as-a-dial.md, its pieces
  ticked, and the walk map that holds my place between sittings.
- walk06.py: one script that prints, per toy and w, the A∩B share and Z, and per sampler the share
  and total variation beside the floor.
- On the whiteboard, a two-mode 1D density raised to w = 1, 2 and 7.5 (this layer's Achieve line).
- The step file's path, the retrospective the walk closes on, and the /drip-retro command it
  prints.
- The proof: python walk06.py prints the w = 1 check within the tolerances below, and step 7's
  comparison shows no differing number.

What you will see
Along the way: one piece per round, each ending on the number it printed and a yes to move on.
At the end, a closing report: what was asked, in one sentence; what was done, with walk06.py's path
and the step file's; PASS or MISS against the quadrant toy's top-right share within 0.005 of 0.989
and Z within 0.005·r of r at w = 1, every exact number equal to its p6. key to the digits shown, and
every sampler number within 0.03 of phase6.json; the w where the parts-world blend appears, or that
none up to 7.5 does; what is still open; at most one question.

I judge, from the printed lines: done when the w = 1 check holds, every exact number matches its p6. key, every sampler number is within 0.03 of phase6.json, and the walk names the blend's w or that none up to 7.5 does.
```

added: the seven sections, from the field-labelled version · the step file's place-keeping as an objective (from the layer 1 build prompt's shape)
dropped: the bar's half that the walk "finds the w where the parts-world blend appears" as a certain outcome stays dropped (why: no source shows guidance produces the blend; the bar accepts the answer that none up to 7.5 does)
