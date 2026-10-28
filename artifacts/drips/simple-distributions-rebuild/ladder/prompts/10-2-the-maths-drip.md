# Layer 10, step 2: the maths drip

⬅️ [Layer 10, step 2](../10-latent-space-and-vaes.md#2-you-the-maths-drip)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`.

```text
/drip --math change of variables and the Jacobian, and why a composition whose exponents sum to 1 is
unchanged by an invertible warp

Background
This is layer 10 of the composition ladder in artifacts/drips/simple-distributions-rebuild/. The
layer adds one variable, the latent, and asks whether a composition survives an invertible change of
space. The maths comes before the build and teaches only what the build needs. I passed quiz R.1 at
the reading stop; any question it marked missed on change of variables becomes a first rung.

Objectives
1. Teach how a density moved through an invertible map x = g(z) picks up |det J|⁻¹.
2. Show that ∏ p_i^(w_i) pushed through g picks up |det J|^(−Σw_i), so construction (1) and
   negation are unchanged and the plain product is not.
3. Work the run's warp by hand as the instance, down to det J = 49 at z = 0 for a = 3.
4. Name the VAE decoder's volume factor without deriving it.

What to read
- artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 10: the factor argument.
- artifacts/ideas/when-poe-reaches-a-and-b/runs/compose-before-or-after-a-warp/warp_check.py and
  warp_check.json: the worked instance.
- artifacts/drips/simple-distributions-rebuild/ladder/10-latent-space-and-vaes.md, Study first line:
  Theorem 6.1 and its proof.
- Quiz R.1's results: the misses that become first rungs.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- A density moved through an invertible map x = g(z) picks up |det J|⁻¹ (claim 10).
- Each conditional and the unconditional pick up the same factor, so p(x|A)·p(x|B)/p(x) picks it up
  once, exactly as the latent composition pushed through g does (claim 10).
- Construction (1) has exponents 1 + 1 − 1 = 1 and negation (1 + w) − w = 1 (claim 10).
- The plain product has exponents summing to 2, so it picks up |det J|⁻¹ twice in pixel space and
  once from the latent (claim 10).
- The worked instance is the run's warp, x1 = z1 + a·tanh(2 z1), x2 = z2 + a·tanh(2 z2) + 0.4 z1
  (runs/compose-before-or-after-a-warp/warp_check.py).
- The warp is lower triangular, so det J is the product of the two diagonal terms, (1 + 2a)² at
  z = 0, which gives the largest area stretches 2.56, 9 and 49 at a = 0.3, 1 and 3 (warp_check.py
  and warp_check.json).
- The general statement is Theorem 6.1 of Bradley et al. 2025a (arXiv 2502.04549), proved in
  appendix I (the layer 10 page, Study first line).
- A VAE decoder is not invertible; √det(JᵀJ) as its volume factor is "likely, verify", so the drip
  names it and does not derive it (claim 10).

Rules
- One rung per message, in LaTeX, with the worked numbers from warp_check.json.
- Name the VAE volume factor as unverified wherever it appears.

What to hand back
- The drip's thread folder under artifacts/drips/, with dispatch.md recording my notation, the
  worked numbers I used and what clicked.
- The folder's path, which step 3's build and step 6's idea walk read.

What you will see
Along the way: one rung per message, each ending on the drip's control line.
At the end, a closing report: what was asked, in one sentence; what was done, with the thread
folder's path; PASS or MISS against the target below; what is still unclear to me; at most one
question.

I judge, without notes: done when I show that ∏ p_i^(w_i) pushed through g picks up |det J|^(−Σw_i) and compute det J = 49 at z = 0 for a = 3 by hand.
```

added: the seven sections, from the field-labelled version · "name the decoder volume factor as unverified" as a rule (from claim 10's "likely, verify" mark)
dropped: nothing
