# Layer 2, step 2: the maths drip

⬅️ [Layer 2, step 2](../02-composing-two-experts.md#2-you-the-maths-drip)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`.

```text
/drip --math the two constructions p(x|A)p(x|B)/p(x) and p_A p_B, total variation distance, why
summed scores under reverse diffusion miss the product (Du et al. 2023, eqs. 11 and 12), and why
MALA's proposal divisor is 4·ss, as steps 1, 2 and 6 of
artifacts/drips/simple-distributions-rebuild/02-composing-and-sampling.md set them out

Background
This is layer 2 of the composition ladder in artifacts/drips/simple-distributions-rebuild/: two
experts composed, and the samplers that reach the composition. The maths comes before the build and
teaches only what the build needs. I have passed quiz 2.1; any question it marked missed becomes a
first rung. Layer 1's maths drip folder under artifacts/drips/ holds my notation for q(x_t | x_0)
and the score.

Objectives
1. Teach the two constructions and what each one targets, with the factor that separates them.
2. Teach total variation distance and why a sampler's value is read against the exact-sample floor.
3. Derive why summed noised scores target the product of the noised experts, not the noised
   product, on the worked instance below.
4. Derive why MALA's proposal log-density divides by 4·ss, and check the divisor toy by hand.
5. Close on the three numbers every composition reports.

What to read
- artifacts/drips/simple-distributions-rebuild/02-composing-and-sampling.md, steps 1, 2 and 6: the
  derivation in this walk's notation.
- toy/rebuild/out/walk-numbers.json: every worked number, as the p2. keys.
- Layer 1's maths drip folder under artifacts/drips/ and its dispatch.md: my notation.
- Quiz 2.1's results: the misses that become first rungs.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- Write a(x) = p(x|A), b(x) = p(x|B) and r(x) for the background density of all the data (step file
  02, step 1).
- Construction (1), score(x|A) + score(x|B) − score(x), targets a·b/r, which is
  r(x)·p(A|x)·p(B|x) up to a constant and equals p(x|A,B) when A and B are conditionally
  independent given x (step file 02, step 1).
- Construction (2) targets a·b (step file 02, step 1).
- The two differ by the factor 1/r, so after normalising they agree only where r is constant on the
  region where a·b > 0 (step file 02, step 1; artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md,
  claim 3; artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 3).
- The idea maps write (1) as p(x|A)p(x|B)/p(x) with p(x) the background, while the step file's p(x)
  is the composed product (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 3;
  artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 3).
- The product of two Gaussians: N(0, 1)·N(2, 1) is N(1, 0.5) with log Z = log N(0; 2, 2) = -2.2655,
  the p2.prod_mean, p2.prod_var and p2.prod_logZ keys (step file 02, step 1;
  toy/rebuild/out/walk-numbers.json).
- Summed noised scores target the product of the noised experts, not the noised product; with
  A = N(1, 0.1²) and B = N(−1, 1) at ᾱ = 0.5 the clean product mean is 0.9802, the noised product
  mean 0.6931 and the product of the noised experts 0.2326 (step file 02, step 2; the
  p2.clean_product_mean, p2.noised_product_mean and p2.product_of_noised_mean keys; Du et al. 2023,
  p.4, eqs. 11 and 12).
- That error does not shrink with more steps or better networks: on the 16-pixel mosaic it leaves
  the head mean at 0.923 where the exact product has 0.980 (step file 02, step 2's broken run).
- Total variation is half the summed absolute difference between two binned distributions, and a
  sampler's value is read against the floor that exact samples of the same size reach, never
  against 0 (step file 03, step 4; the Wootters table under step file 02, step 2).
- The estimator behind the sampler TVs here uses 40 × 40 bins on the canvas [−1.1, 1.1]², the exact
  density's mass per bin against each bin's sample count divided by the full sample count, never by
  the count that landed inside the canvas, so the result is a windowed histogram diagnostic and not
  the full-space TV (toy/rebuild/ideal.py, tv; the kinds notebook, cell 1).
- Step 5's answer key uses 60 × 60 bins on [−1.5, 1.5]² and counts samples outside that box as
  error (artifacts/ideas/when-poe-reaches-a-and-b/runs/samplers-on-exact-experts/exact_toy.py,
  total_variation).
- The MALA proposal has variance 2·ss, so its log-density divides by 4·ss; the repository's code
  divided by 2·ss² (step file 02, step 6).
- One move on N(0, 1) from x = 1 with ss = 0.5 and z = 0.3 gives x' = 0.8 and a log-density change
  of 0.18 (step file 02, step 6; the p2.mala_* keys).
- With 4·ss that move gives log forward -0.045, log backward -0.18, log α 0.045 and accept
  probability 1.0 (step file 02, step 6; the p2.mala_* keys).
- With 2·ss² it gives -0.18, -0.72, log α -0.36 and accept probability 0.6977 (step file 02, step 6;
  the p2.mala_* keys).
- Every composition reports three numbers together: the A∩B share, the total variation to the exact
  target and the normaliser Z (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 5).
- Reverse diffusion's share of 0.983 against the exact 0.989 hid a total variation 0.049 above the
  floor (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 5 and run 5).
- The four outcome sentences are: construction (1) is construction (2) divided by a background
  expert; summed scores miss the product; for a fixed MALA step the accept step makes the target a
  stationary law of the chain, which removes the proposal's step-size bias, while a finite run can
  still miss through poor mixing or too few moves, and the tuned MALA here adapts while it samples,
  so it is judged against the exact-sample floor; every composition reports the A∩B share, the
  total variation to the exact target and the normaliser Z together (layer 2, step 2's Outcome).

Rules
- One rung per message, in LaTeX, with worked numbers from walk_numbers.
- Keep construction (1)'s background p(x) and the step file's composed p(x) apart at every rung
  where both appear.

What to hand back
- The drip's thread folder under artifacts/drips/, with dispatch.md recording my notation, the
  worked numbers I used and what clicked.
- The folder's path, which step 6's scene and step 7's cross-links read.

What you will see
Along the way: one rung per message, each ending on the drip's control line.
At the end, a closing report: what was asked, in one sentence; what was done, with the thread
folder's path; PASS or MISS against the target below; what is still unclear to me; at most one
question.

I judge, without notes: done when I state the four outcome sentences and reproduce the divisor toy by hand, log α 0.045 with 4·ss against -0.36 with 2·ss².
```

added: the seven sections, from the field-labelled version · the four outcome sentences, copied from the step's Outcome so the prompt stands alone · "keep the two p(x) apart" as a rule (from the Codex review's note on the clash between the idea maps' p(x) and the step file's)
dropped: nothing
