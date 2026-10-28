# Layer 5, step 2: the maths drip

⬅️ [Layer 5, step 2](../05-the-overlap-dial.md#2-you-the-maths-drip)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`.

```text
/drip --math steps 4 and 7 of artifacts/drips/simple-distributions-rebuild/05-the-overlap-dial.md:
why the product weights shared places most, and what fine-tuning does to inherited support

Background
This is layer 5 of the composition ladder in artifacts/drips/simple-distributions-rebuild/: the
overlap dial, which answers how much A-and-B data a product needs, and whether pretraining supplies
it. The maths comes before the build and teaches only what the build needs. I have passed quiz 5.1;
any question it marked missed becomes a first rung.

Objectives
1. Teach the quadrant toy and why a few points shared by both sets carry a large share of the
   product.
2. Derive the averaged deep-mass formula and check it at 5% and 100%.
3. Work the deep region's uniform-area benchmark 0.7225 and the 0% corner profile's peak 0.25 by
   hand.
4. Read the fine-tune numbers and say whether pretraining rescues A∩B or is erased.

What to read
- artifacts/drips/simple-distributions-rebuild/05-the-overlap-dial.md, steps 1 to 4 and 7: the
  derivation in this walk's notation.
- toy/rebuild/out/walk-numbers.json: the p5. keys.
- python -m rebuild.phase5_tables and toy/rebuild/out/phase5/finetune.json: the 3-seed curve and
  the fine-tune numbers.
- artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 2: the repaired claim.
- Quiz 5.1's results: the misses that become first rungs.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The quadrant toy: A is the left half, B the top half, A∩B the top-left quadrant; each expert holds
  800 points, a share f of them in A∩B, smoothed by σ = 0.05 (step file 05, step 1).
- Each point lands in A∩B independently with probability f, so A has 8 points there on average at
  1% and 40 at 5%, and one data seed's count differs (walk_numbers,
  p5.overlap_points_of_800_at_1pct and _5pct, which are 800f).
- The product's mass at a point is the product of both experts' densities there (step file 05,
  step 4).
- Outside A∩B at least one expert is near zero almost everywhere, so a few points shared by both
  sets carry a large share of the product (step file 05, step 4).
- Averaged over the data, the deep mass is
  M_D(f) = f²C_D / (f²C + f(1−f)C_1 + (1−f)²C_0), with corner overlap C_0 = 0.0008, edge overlaps
  C_1 = 0.053, the square C = 0.890 and its deep part C_D = 0.675 (step file 05, step 4;
  p5.deep_mass_C0_corner, p5.deep_mass_C1_edges, p5.deep_mass_C_square,
  p5.deep_mass_CD_deep_square and p5.deep_mass_formula_MD).
- That formula gives 0.31 at 5% and 0.76 at 100% (step file 05, step 4; p5.deep_mass_formula_MD).
- Deep A∩B means x1 < −0.15 and x2 > 0.15, 3σ inside (step file 05, step 3).
- At 100% overlap a uniform density would put 0.85 × 0.85 = 0.7225 in the deep region, a benchmark
  the product exceeds, since its exact mass there is 0.76 (step file 05, step 3;
  p5.interior_area_share_margin0.15).
- At 0% the two squares touch only at the origin (step file 05, step 2).
- Along each axis the 0% product is Φ(−x/σ)Φ(x/σ) = u(1 − u) with u = Φ(x/σ), largest at 0 where it
  equals 0.25, with half its mass within 0.0302 of the boundary (step file 05, step 2;
  p5.corner_profile_peak_at_0, p5.corner_profile_half_mass_within).
- At 0% the whole quadrant reads 0.20 ± 0.02 and the deep region rounds to 0.00 (step file 05,
  steps 2 and 3).
- The exact deep mass over 3 data seeds is 0.00 at 0%, 0.05 ± 0.05 at 1%, 0.09 at 2%, 0.30 ± 0.10
  at 5%, 0.51 at 10%, 0.59 at 20%, 0.71 at 50% and 0.76 at 100% (step file 05, step 4;
  python -m rebuild.phase5_tables).
- Fine-tuning: one model pretrained on all four quadrants, two copies fine-tuned at 0% overlap with
  Adam at 1e-4 (step file 05, step 7).
- Grid / tuned MALA / UHA deep mass reads 0.159 / 0.158 / 0.163 at step 0, 0.089 / 0.104 / 0.122 at
  step 10, 0.001 / 0.056 / 0.068 at step 30, 0.005 / 0.004 / 0.0055 at step 100, and at most 0.001
  from step 300 (step file 05, step 7; finetune.json).
- Experts trained from scratch give 0.000 (step file 05, step 7; finetune.json).
- Rescue would mean the pretrained support survives long enough for fine-tune length to be a dial,
  and erasure at once means pretraining cannot stand in for overlap data (step file 05, step 7).
- Claim 2 of the overlap-dial idea map, repaired: the product puts samples in A∩B only where some
  expert carries density there, from overlap in the training data or from an expert loose outside
  its concept's region, and support inherited from pretraining does not survive fine-tuning
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 2).
- The two reasons the layer asks for are why 5% overlap puts 0.30 of the product's mass deep in
  A∩B, more than 5%, and why pretraining support is gone within 30 to 100 fine-tune steps (layer
  5, step 2's Outcome).

Rules
- One rung per message, in LaTeX, with worked numbers from walk_numbers and phase5_tables.
- Keep the 8 and 40 shared points as averages of a random count at every rung that uses them.

What to hand back
- The drip's thread folder under artifacts/drips/, with dispatch.md recording my notation, the
  worked numbers I used and what clicked.
- The folder's path, which quiz 5.2, the build in step 4, the scene in step 6 and step 7's
  cross-links read.

What you will see
Along the way: one rung per message, each ending on the drip's control line.
At the end, a closing report: what was asked, in one sentence; what was done, with the thread
folder's path; PASS or MISS against the target below; what is still unclear to me; at most one
question.

I judge, without notes: done when I give both reasons in the layer's Outcome and work 0.25 and 0.7225 by hand.
```

added: the seven sections, from the field-labelled version · the two reasons, copied from the step's Outcome so the prompt stands alone · "keep the shared-point counts as averages" as a rule (from the Codex review's correction that 8 and 40 are averages of a random count)
dropped: nothing
