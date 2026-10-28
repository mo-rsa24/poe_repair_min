# Layer 4, step 2: the maths drip

⬅️ [Layer 4, step 2](../04-the-parts-world.md#2-you-the-maths-drip)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`.

```text
/drip --math the product of two diagonal Gaussians and the product of two Gaussian mixtures, every
pair and its weight, as steps 2 to 5 of
artifacts/drips/simple-distributions-rebuild/04-the-parts-world.md set them out

Background
This is layer 4 of the composition ladder in artifacts/drips/simple-distributions-rebuild/: the
parts world, where a cat expert and a dog expert over 16 pixels multiply exactly into a mosaic, a
blend or two animals. The maths comes before the build and teaches only what the build needs. I
have passed quiz 4.1; any question it marked missed becomes a first rung.

Objectives
1. Teach the diagonal Gaussian expert over 16 independent pixels.
2. Derive the per-pixel product by completing the square: precisions add, the mean is
   precision-weighted, and the constant Z.
3. Work the mosaic and the blend with their numbers.
4. Derive the product of two mixtures as a mixture over every pair, with its weights and normaliser.
5. Work the two-animals pair weights by hand, including why float64 prints them as 0.5 and 0.0.

What to read
- artifacts/drips/simple-distributions-rebuild/04-the-parts-world.md, steps 2 to 5: the derivation
  in this walk's notation.
- artifacts/ideas/poe-overlap-dial/example-9-parts-world/numbers.json: the worked numbers, as
  settings a, b and c.
- The vault note Products And Quotients Of Diagonal Gaussians, linked on layer 4's Study first line.
- Quiz 4.1's results: the misses that become first rungs.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- An expert is N(m, diag v) over 16 pixels, each pixel independent, so its log-density is a sum over
  pixels (step file 04, step 2).
- The cat expert of example 9(a) is tight on the head (v = 0.01) and loose on the body (v = 1), and
  the dog the reverse (step file 04, step 2).
- Per pixel, for inputs N(m_1, diag v_1) and N(m_2, diag v_2), precisions add, 1/v = 1/v_1 + 1/v_2,
  and the mean is precision-weighted, m = v(m_1/v_1 + m_2/v_2) (step file 04, step 3).
- The constant is Z = N(m_1; m_2, diag(v_1 + v_2)) (step file 04, step 3).
- Step 3 derives all three by adding the log-densities, collecting powers of x, completing the
  square and integrating over x (step file 04, step 3).
- A loose expert (v = 1) still pulls: its score −(x_d − m_d)/v_d is nonzero, which is why the mosaic
  head is 99/101 and not 1; only v_d → ∞ ignores a pixel (step file 04, steps 2 and 3).
- Mosaic: on a head pixel the cat has precision 100 and the dog 1, so the mean is
  (100 − 1)/101 = 0.9802 (step file 04, step 3; numbers.json, a).
- Mosaic: log Z = −46.47, 2.904 nats a pixel times 16 (step file 04, step 3; numbers.json, a).
- Blend: both experts at v = 0.01 give mean 0, variance 0.005 (sd 0.0707), 10 sds from each expert,
  and log Z = −1583.4 (step file 04, step 4; numbers.json, b).
- The product of two mixtures is a constant times a mixture over every pair:
  (Σ_i w_i N_i)(Σ_j w'_j N'_j) = Σ_ij w̃_ij N_ij with w̃_ij = w_i w'_j Z_ij (step file 04, step 5).
- Its normaliser is Z_mix = Σ_ij w̃_ij and its weights are w*_ij = w̃_ij / Z_mix (step file 04,
  step 5).
- The vault notes write the weight as w_ij ∝ π_i ρ_j N(μ_i; ν_j, Σ_i + Λ_j) (Products And Quotients
  Of Diagonal Gaussians).
- Two animals: cat-left × dog-right and cat-right × dog-left have unnormalised log weight
  −24.09 = log 0.25 + 16 pixels each disagreeing by 1 over variance 1.01 (step file 04, step 5;
  numbers.json, c).
- The same-half pairs have −803.21 = log 0.25 + 8 pixels disagreeing by 2 over variance 0.02 + 8
  loose pixels agreeing at 0 over variance 2 (step file 04, step 5; numbers.json, c).
- The gap is 779.12, so exactly w_side = 1/(2(1 + e^−779.12)) and w_same = e^−779.12 w_side (step
  file 04, step 5; numbers.json, c).
- float64 underflows e^−779.12 to 0, so the two weights print as 0.5 and 0.0 (step file 04, step 5;
  numbers.json, c).

Rules
- One rung per message, in LaTeX, with worked numbers from numbers.json.
- Keep each exact weight apart from its float64 printout at every rung where both appear.

What to hand back
- The drip's thread folder under artifacts/drips/, with dispatch.md recording my notation, the
  worked numbers I used and what clicked.
- The folder's path, which quiz 4.2, the build in step 4 and step 7's cross-links read.

What you will see
Along the way: one rung per message, each ending on the drip's control line.
At the end, a closing report: what was asked, in one sentence; what was done, with the thread
folder's path; PASS or MISS against the target below; what is still unclear to me; at most one
question.

I judge, without notes: done when I derive the precision sum, the precision-weighted mean and the pair weight, and reproduce 0.9802, −24.09 and −803.21 by hand.
```

added: the seven sections, from the field-labelled version · "keep the exact weight apart from its float64 printout" as a rule (from the Codex review's correction that 0.5 and 0.0 are roundings)
dropped: nothing (the earlier version had already dropped "weight 0.5 each" and "e^−779 relative" as exact values, from the Codex mathematics review)
