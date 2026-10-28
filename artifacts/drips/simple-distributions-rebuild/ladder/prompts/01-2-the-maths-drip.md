# Layer 1, step 2: the maths drip

⬅️ [Layer 1, step 2](../01-one-diffusion-expert.md#2-you-the-maths-drip)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`.

```text
/drip --math the forward process, the cosine schedule, and why the noise prediction is the score
rescaled, as steps 4, 5 and 8 of
artifacts/drips/simple-distributions-rebuild/01-one-expert-from-scratch.md set them out

Background
This is layer 1 of the composition ladder in artifacts/drips/simple-distributions-rebuild/: one 2D
diffusion expert, built and understood before any composing. The maths comes before the build, and
teaches only the derivation the build needs. I have passed quiz 1.1; any question it marked missed
becomes a first rung.

Objectives
1. Teach the forward process q(x_t | x_0) = N(√ᾱ_t x_0, (1 − ᾱ_t) I) on the step file's schedule.
2. Derive the sampled-noise identity ε = −√(1 − ᾱ_t)·∇log q(x_t | x_0) for one clean point.
3. Derive by posterior averaging the marginal identity E[ε | x_t] = −√(1 − ᾱ_t)·∇log q_t(x_t),
   which is what the trained network learns.
4. Check the conditional identity on the worked instance below, by hand.

What to read
- artifacts/drips/simple-distributions-rebuild/01-one-expert-from-scratch.md, steps 4, 5 and 8: the
  derivation in this walk's notation.
- toy/rebuild/out/walk-numbers.json: every worked number, as the p1. keys.
- Quiz 1.1's results: the misses that become first rungs.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The forward process runs T = 100 steps, with ᾱ_t = ∏(1 − β_j) from the cosine schedule with
  offset 0.008 (step file 01, step 4).
- Code index t = 0 is the first noisy state, so ᾱ_t equals the schedule's boundary value C_{t+1},
  and ᾱ_0 = 0.9994 (step file 01, step 4).
- For one clean point x_0 and one sampled ε, ε = −√(1 − ᾱ_t)·∇log q(x_t | x_0) is the conditional
  score (step file 01, step 5).
- Averaged over the clean points given x_t, the squared-error optimum is
  E[ε | x_t] = −√(1 − ᾱ_t)·∇log q_t(x_t), the marginal score (step file 01, step 8; the Codex
  mathematics review of 2026-10-18).
- python -m rebuild.walk_numbers prints ᾱ_t and √(1 − ᾱ_t) at t = 0, 25, 50, 75 and 99 as the
  p1.abar_* and p1.sqrt_1m_abar_* keys (toy/rebuild/out/walk-numbers.json).
- The worked instance: clean point (0, 0.5), ε = (0.3, −1.2), t = 50, √(1 − ᾱ_50) = 0.7223,
  x_50 = (0.2167, −0.521); p1.example_score and p1.example_minus_eps_over_sigma are both
  (−0.4153, 1.6613) (toy/rebuild/out/walk-numbers.json).
- The cosine schedule's derivation is out of scope; the schedule is given (layer 1's Skip line).

Rules
- One rung per message, in LaTeX, with worked numbers from walk_numbers.
- Keep the conditional and the marginal identity apart at every rung where both appear.

What to hand back
- The drip's thread folder under artifacts/drips/, with dispatch.md recording my notation, the
  worked numbers I used and what clicked.
- The folder's path, which quiz 1.2, layer 1's scene and its closing links read.

What you will see
Along the way: one rung per message, each ending on the drip's control line.
At the end, a closing report: what was asked, in one sentence; what was done, with the thread
folder's path; PASS or MISS against the target below; what is still unclear to me; at most one
question.

I judge, without notes: done when I derive the sampled-noise identity from q(x_t | x_0), derive the marginal identity by posterior averaging, and reproduce (−0.4153, 1.6613) by hand.
```

added: the seven sections, from the field-labelled version · the two identities as separate objectives · "keep the two identities apart" as a rule (from the Codex review's reason for splitting them)
dropped: nothing
