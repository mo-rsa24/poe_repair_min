# Layer 3, step 2: the maths drip

⬅️ [Layer 3, step 2](../03-ideal-experts-and-the-seven-kinds.md#2-you-the-maths-drip)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`.

```text
/drip --math why smoothing the training points by N(0, σ²I) is the best a diffusion model can learn
at that noise level, and under which assumptions, as steps 1 and 2 of
artifacts/drips/simple-distributions-rebuild/03-ideal-experts-and-the-seven-kinds.md set them out

Background
This is layer 3 of the composition ladder in artifacts/drips/simple-distributions-rebuild/: the
ideal expert, which makes every later composition target exact, so a failure can be blamed on the
sampler or the data rather than on a network's approximation. The maths comes before the build and
teaches only what the build needs. I have passed quiz 3.1; any question it marked missed becomes a
first rung. Layer 2's maths drip folder under artifacts/drips/ holds my notation for the two
constructions and the summed scores.

Objectives
1. Teach the ideal expert as a Gaussian kernel density estimate of the training points.
2. Derive why its score is the optimal learned score, and name the assumptions that make it so.
3. Derive the noised density q_t and its variance v_t, and check v_t at t = 50 by hand.
4. Derive the score of the noised mixture with softmax responsibilities.
5. Say which sampler the exactness helps and which it does not.

What to read
- artifacts/drips/simple-distributions-rebuild/03-ideal-experts-and-the-seven-kinds.md, steps 1
  and 2: the derivation in this walk's notation.
- toy/rebuild/out/walk-numbers.json: every worked number, as the p3. and p1. keys.
- The plan "the exact score of a Gaussian mixture", linked on layer 3's Study first line.
- Layer 2's maths drip folder under artifacts/drips/ and its dispatch.md: my notation.
- Quiz 3.1's results: the misses that become first rungs.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The ideal expert of x_1, …, x_N at smoothing σ is p_σ(x) = (1/N)Σ N(x; x_i, σ²I), a Gaussian
  kernel density estimate with bandwidth σ (step file 03, step 1).
- Write the training set as p̂_0(x) = (1/N)Σ δ(x − x_i); Gaussian corruption of variance σ² gives
  p_σ(x) = (1/N)Σ N(x; x_i, σ²I) (step file 03, step 1).
- With unlimited capacity and the denoising objective taken over the whole distribution and
  minimised globally, the optimal learned score is ∇_x log p_σ(x) (step file 03, step 1).
- The rebuild chooses this kernel density estimate as its clean expert before applying the forward
  process (step file 03, step 1).
- A network can do worse by underfitting or differently by extrapolating (step file 03, step 1).
- σ = 0.05 is a chosen smoothing, not the schedule's smallest noise: √(1 − ᾱ_0) = 0.0251
  (p1.sqrt_1m_abar_0).
- A model trained on the bare points would have component variance 1 − ᾱ_t at step t; the extra
  ᾱ_t σ² in v_t comes from noising p_σ (step file 03, steps 1 and 2).
- In run 2 of the idea map, trained 2D networks matched the ideal experts within 0.04 nats, so in
  that run a trained expert added nothing measurable past its smoothed data
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, run 2 and claim 7).
- The worked instance: three points (−0.5, −0.5), (−0.6, −0.4), (−0.5, 0.5) at σ = 0.15 give a
  log-density at (−0.5, 0) of -2.6288 by the formula and by scikit-learn's KernelDensity
  (p3.ideal_logdensity_formula, p3.ideal_logdensity_sklearn).
- Noising scales each kernel by √ᾱ_t and convolves it with N(0, (1 − ᾱ_t)I), and convolving two
  Gaussians adds their variances, so q_t(x) = (1/N)Σ N(x; √ᾱ_t x_i, v_t I) with
  v_t = 1 − ᾱ_t + ᾱ_t σ² (step file 03, step 2).
- With σ = 0.05, v_t is 1.0, 0.5229 and 0.00313 at t = 99, 50 and 0 (the p3.noised_ideal_var_*
  keys).
- At t = 50, ᾱ_t = 0.4783, and 1 − 0.4783 + 0.4783 × 0.0025 = 0.5229 (the p3.noised_ideal_var_*
  and p1.abar_50 keys).
- The score of that mixture, derived: put μ_i = √ᾱ_t x_i and φ_i(x) = N(x; μ_i, v_t I), so
  q_t = (1/N)Σ φ_i (step file 03, step 2).
- Differentiating the log gives ∇ log q_t = Σ_i ∇φ_i / Σ_j φ_j (step file 03, step 2).
- Writing ∇φ_i = φ_i ∇ log φ_i turns that into Σ_i r_i(x) ∇ log φ_i, with r_i = φ_i / Σ_j φ_j the
  softmax responsibilities over the training points (step file 03, step 2; the plan "the exact
  score of a Gaussian mixture").
- The score of an isotropic Gaussian is (μ_i − x)/v_t, so
  ∇ log q_t(x) = Σ_i r_i(x)(√ᾱ_t x_i − x)/v_t (step file 03, step 2; the plan "the exact score of
  a Gaussian mixture").
- Because each q_t is exact, every annealing target π_t = compose(log q^A_t, log q^B_t) and every
  score a sampler uses is exact (step file 03, step 2).
- Summed noised scores under reverse diffusion still target the product of the noised experts, not
  the noised product, so exact experts do not make reverse diffusion exact (step file 02, step 2).

Rules
- One rung per message, in LaTeX, with worked numbers from walk_numbers.
- Keep the chosen smoothing σ apart from the schedule's noise at every rung where both appear.

What to hand back
- The drip's thread folder under artifacts/drips/, with dispatch.md recording my notation, the
  worked numbers I used and what clicked.
- The folder's path, which step 7's cross-links read.

What you will see
Along the way: one rung per message, each ending on the drip's control line.
At the end, a closing report: what was asked, in one sentence; what was done, with the thread
folder's path; PASS or MISS against the target below; what is still unclear to me; at most one
question.

I judge, without notes: done when I derive q_t and v_t from the kernel density estimate, reproduce v_t = 0.5229 at t = 50 and the -2.6288 instance by hand, and say which sampler the exactness helps and which it does not.
```

added: the seven sections, from the field-labelled version · "keep σ apart from the schedule's noise" as a rule (from the Codex review's note that σ = 0.05 is a chosen smoothing, separate from the schedule's 0.0251)
dropped: nothing (the earlier version had already narrowed "in 2D a trained expert adds nothing past its smoothed data" to the one run that measured it, from the Codex mathematics review)
