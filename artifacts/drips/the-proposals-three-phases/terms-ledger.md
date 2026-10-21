# Term ledger: Composability Gap PhD proposal (v2)

Every term the proposal defines, with its own words and where they sit, then the terms it uses
without defining. A walk through the proposal explains a term from this file first.

Source: `/mnt/c/Users/molef/Downloads/PhD/Proposal/Composability Gap - PhD Proposal (v2).pdf`.
Section numbers are the proposal's own.

## Defined in the proposal

| Term | The proposal's words | Where |
|---|---|---|
| logical composition | "the mathematical operator used to combine conditions, which we term as 'logical composition'" | Abstract; Chapter 1 |
| semantic composition | "the conventional case in which a single natural-language prompt specifies all desired factors and the model attempts to realize them jointly" | §2.2.4 |
| composability gap | "This gap between logical and semantic composition"; measured as "the latent distance between semantic composition and logical composition (PoE) at each denoising step" | Chapter 1; §5.1.4 |
| simple product | "p̂(x) ∝ p1(x) p2(x)" | §2.2.4 |
| Bayes composition | "p̂(x) := p(x \| c1, c2)", which under conditional independence is "∝ p(x \| c1) p(x \| c2) / p(x)" | §2.2.4, Eq. 2.28 |
| AND operator | the noise-prediction form of Bayes composition, unconditional prediction plus two weighted conditional differences | §2.2.4, Eq. 2.30 |
| projective composition | "the composed distribution is judged by the part each source is meant to contribute" | §2.2.4 |
| disentangled | "each latent dimension encodes at most one ground-truth factor of variation" | §2.1.1 |
| correct-heads | "predictors attached to the intended subset" of the latent | §2.1.2 |
| wrong-heads | "predictors attached to other subsets", trained through a gradient reversal layer so the encoder removes the factor there | §2.1.2 |
| common and salient latents | SepVAE's split: `c` for variation shared by background and target data, `s` for variation only in the target data | §2.1.2 |
| Groups 1 to 4 | co-occurring concepts, feature-space disentanglement, feature overlap, adversarial collision | §5.1.1 |
| `d_t`, `d_T` | mean squared difference between the single-prompt latent and the PoE latent at step `t`; `d_T` is the same at the final denoising step | §5.1.4 |
| Regimes A to D | baseline VAE on standard dSprites; the same VAE on correlated dSprites; stronger disentanglement or weak supervision on correlated dSprites; a partitioned latent on correlated dSprites | §5.2.2 |
| withheld combination | "an orientation–scale pair that is deliberately removed from the training set" | §5.2.3 |
| `z = (z_c, z_d1, z_d2)` | shared context, then one block per factor: orientation and scale in Phase 2, cardiomegaly and pleural thickening in Phase 3 | §5.2.2; §5.3.2, Eq. 5.2 |
| held-out co-morbid presentation | "a joint cardiomegaly–pleural-thickening case that is absent or strongly reduced during generative training" | §5.3.3 |

## Used but never defined

| Term | First use | Most likely reading (from outside the document) |
|---|---|---|
| PoE | §5.1.2 | product of experts; §3.1 spells the phrase out once but never the abbreviation, and never says whether PoE means the simple product or the AND operator of Eq. 2.30 |
| monolithic prompt | §5.1.2 | the single prompt of semantic composition |
| solo `c1`, solo `c2` | §5.1.2 | one-concept runs from the same noise; they are generated but no Phase 1 metric uses them |
| semantic manifold | §5.1.3 | the set of images a text prompt can reach |
| manifold-supported co-occurrence | §5.1 | concept pairs the training data already shows together |
| `u`, "the generative variable" | §5.3.2 | the latent the Phase 3 diffusion model denoises; its relation to `(z_c, z_d1, z_d2)` is not stated |
| shared downstream evaluator | §5.2, §6.2 | the one conditional latent diffusion model reused across Regimes A to D; its conditioning is not described |

## Where this repo already owns a term

`d_T` and `d_t` carry the same meaning in this repo's
[data dictionary](../../../context/data/02-dictionary.md): the distance between the single-prompt
and PoE predictions per step and at the final step. PoE composition as this project runs it on SDXL
is described in the [PoE composition context file](../../../context/world/poe-composition.md).
The proposal's "composability gap" and "logical composition" have no entry in `context/`.
