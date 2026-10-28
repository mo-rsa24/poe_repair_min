/pressure-test Is it known in the diffusion-composition literature that a composition whose exponents sum to 1 gives the same answer in a latent space and in pixel space, while the plain product of experts does not, and does that matter for composing latent diffusion models such as Stable Diffusion?

## The claim, in my words

Take experts as densities and compose them as a weighted product ∏ p_i(x)^{w_i}. If x = g(z) is an invertible map (a stand-in for a decoder), every density picks up the same factor |det J|⁻¹, so the composed density picks up |det J|^{−Σw_i}, while composing in z and decoding picks it up once. They agree exactly when Σw_i = 1. Construction (1), p(x|A)·p(x|B)/p(x) (Composable Diffusion AND, Liu et al. 2022; classifier-free composition), has weights 1 + 1 − 1 = 1, and negation p_A^{1+w}/p_B^{w} has (1 + w) − w = 1, so both are unchanged by where they are computed. The plain product p_A·p_B (two separately trained experts, the product of experts in Du et al. 2023, Reduce, Reuse, Recycle) has weights summing to 2, so composing in a latent space reweights it by how much the decoder stretches each region. For a VAE decoder, which is not invertible, I expect the analogue of |det J| to be the volume factor √det(JᵀJ) of the pullback metric (unverified).

## What has been checked

A 2D check in /home-mscluster/mmolefe/PhD/reduce-reuse-recycle/artifacts/ideas/when-poe-reaches-a-and-b/runs/compose-before-or-after-a-warp/ (README.md, warp_check.py, run.log): the four-quadrant toy (Gaussian blobs at (±0.5, ±0.5), std 0.15) in a latent space, decoded by a fixed invertible warp at four strengths. Construction (1) and negation agree across the warp to 4e-17 total variation (rounding). The plain product differs by 0.035, 0.080 and 0.126 at warps whose largest area stretch is 2.6, 9 and 49 times. The idea walk this belongs to is /home-mscluster/mmolefe/PhD/reduce-reuse-recycle/artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 10.

Related work I already have: Bradley et al. 2025a, Mechanisms of Projective Composition of Diffusion Models (arXiv 2502.04549), Theorem 6.1 and Lemma 6.2: projective composition in pixel space is projective composition in any feature space reached by an invertible map, with no need to know the map.

## The ask

1. Is the exponent-sum condition (Σw_i = 1 makes a weighted product invariant to reparameterisation) stated anywhere for diffusion or energy-based composition, or in the product-of-experts or geometric-mixture literature (log-linear pooling, "externally Bayesian" pooling)? Name where.
2. Does any paper discuss composing in a latent diffusion model's latent against pixel space, the decoder's volume change, or the pullback metric of a VAE decoder in this context?
3. Verdict: known, a small new observation worth a paragraph, or wrong? And what would make it matter in practice for Stable Diffusion, whose decoder is not invertible?

Return. Close with a fenced return package, and also write it to /home-mscluster/mmolefe/PhD/reduce-reuse-recycle/artifacts/ideas/when-poe-reaches-a-and-b/routes/07-exponent-sum-invariance.return.md:
  origin: artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md · claim 10 · route 7
  ask: is exponent-sum invariance under reparameterisation known, and does it matter for latent diffusion
  deliverable: the verdict, where it is stated if anywhere, and confidence flags
  changes for the walk: one line on how claim 10 should be worded
Then I paste it into the idea walk and say `integrate`.
