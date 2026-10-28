# Layer 10, step 6: the VAE idea walk

⬅️ [Layer 10, step 6](../10-latent-space-and-vaes.md#6-you-the-idea-walk)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`.

```text
/drip-idea resume artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md at claim 11: whether a
beta-VAE, beta-TCVAE or FactorVAE on the 2D toy shows what happens inside Stable Diffusion's latent,
read against Theorem 6.1 of arXiv 2502.04549, and what the smallest check is

Background
Layer 10 of the composition ladder (artifacts/drips/simple-distributions-rebuild/) showed in code
that construction (1) survives an invertible warp and the plain product does not. A VAE decoder is
not invertible, so the open question is whether a small VAE on the 2D toy can say anything about
Stable Diffusion's latent. Claim 11 of the idea map holds that question. walk10.py and the step 4
comparison reproduce run 6 (claim 10's warp check), and the layer 10 maths drip folder under
artifacts/drips/ holds my notation (step 2).

Objectives
1. Walk claim 11 from its place under Held claims, in repo
   /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain.
2. Stop when the smallest VAE check is chosen or the claim is marked dead.
3. Update claim 11's row in the Position table and its Held claims row with the mark.
4. If a check is chosen, add a Checks outstanding row naming the check and what each outcome means.

What to read
- artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md: claims 10 and 11, the Held claims row, the
  Runs table (run 6), the Routes table (route 7) and Checks outstanding (claim 2a).
- The route 7 prompt in that map's routes/ folder: Theorem 6.1 and Lemma 6.2.
- walk10.py in ~/Playground/PhD/simple-distributions-walk/ and the layer 10 maths drip's
  dispatch.md: what I built and my notation.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- Claim 11 is held (the map's Held claims row).
- Unsupervised disentanglement is not identifiable without supervision or inductive bias (Locatello
  et al. 2019, confident), so a β-VAE on 2D data may not put A and B on separate axes (the map's
  Held claims row).
- A 2D latent from 2D data speaks to coordinates, whether A and B fall on separate axes (the
  orthogonal-mean-difference condition of Bradley et al. 2025a, Lemma 8.1), not to a
  high-dimensional latent (the same row).
- The map's own route to settle it is claim 10's warp check first, then a 2D VAE whose latent is
  deliberately rotated 45° against A and B, composed in the latent, against one aligned with them
  (the same row).
- Claim 10's warp check is done as run 6: construction (1) and negation agree to 4e-17 and 7e-17
  total variation across the warp (Runs table, run 6).
- In run 6 the plain product differs by 0.035, 0.080 and 0.126 at largest area stretches 2.6, 9 and
  49 (Runs table, run 6).
- A VAE decoder is not invertible; its analogue of |det J| is √det(JᵀJ) of the pullback metric,
  marked "likely, verify" (claim 10).
- Route 7, the pressure-test on whether exponent-sum invariance is known, is emitted with no return
  file (Routes table, route 7; no routes/07-exponent-sum-invariance.return.md exists).
- Theorem 6.1 and Lemma 6.2 of arXiv 2502.04549 say projective composition in pixel space is
  projective composition in any feature space an invertible map reaches (route 7's prompt).
- The quadrant toy: A is x1 > 0, B is x2 > 0, one blob per quadrant at (±0.5, ±0.5), std 0.15
  (Checks outstanding, claim 2a).

Rules
- Each round closes one part of the claim with a mark: holds, needs a check, wrong as stated with
  its repair, or dead.
- Write only in that idea map.
- Launch no run.
- No commit.

What to hand back
- Claim 11's rows in the Position table and Held claims, with the mark.
- If a check is chosen, its Checks outstanding row with what each outcome means.
- Claim 11's row is what step 7's cross-links read.
- The proof: grep -n "^| 11\." artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md prints the
  claim's new mark.

What you will see
Along the way: one part of the claim per round, each ending on its mark and the control line.
At the end, a closing report: what was asked, in one sentence; what was done, with the rows changed
in the idea map; PASS or MISS against the smallest VAE check written with its outcomes, or claim 11
marked dead with its reason; what is still open; at most one question.

I judge, from the idea map: done when the smallest VAE check is written with its outcomes, or claim 11 is marked dead with the reason.
```

added: the seven sections, from the field-labelled version · why the question is open, in Background (from the layer 10 page's Why it is critical line)
dropped: nothing
