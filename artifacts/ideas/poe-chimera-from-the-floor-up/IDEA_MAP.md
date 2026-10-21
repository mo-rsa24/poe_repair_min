# 💡 The PoE chimera, rebuilt from the floor up

## Position in the idea

| Claim | Mark | Settled by |
|---|---|---|
| **1 (current)**: a small setting reproduces the chimera and the joint-prompt success side by side | **needs a check** | **rung 0, the closed-form slot toy: does the exact product's mode land on a fused point while a joint condition lands on both slots** |
| 2. the chimera is in the target distribution, not in how the sampler reaches it | needs a check | rung 1 (exact against learned score, DDIM against a clean-product MCMC chain) and the free bound on the model's share, step 24 of the running order |
| 3. what "a cat" was trained to mean decides what the product asks for | open | |
| 4. guidance and the null branch push the product toward its most extreme point | open | |
| 5. the network's inductive bias shapes the chimera beyond what the exact score would give | open | |
| 6. each rung's result decides whether the next is worth building | open | |

Load-bearing: claim 1. If no setting below SDXL shows the chimera beside a working joint condition,
the ladder has no control to hold fixed and nothing to climb.

## Table of contents

- [Position in the idea](#position-in-the-idea)
- [Quick context: where you are](#quick-context-where-you-are)
- [The idea, as it stands](#the-idea-as-it-stands)
- [Facts pinned before round 1](#facts-pinned-before-round-1)
- [The eleven threads, placed](#the-eleven-threads-placed)
- [The claims](#the-claims)
- [What the words are](#what-the-words-are)
- [Held claims](#held-claims)
- [Dead ends](#dead-ends)
- [Checks outstanding](#checks-outstanding)
- [Runs](#runs)
- [Routes](#routes)
- [Sources](#sources)
- [Next step](#next-step)

## Quick context: where you are

Navigation: ⬅️ [Position](#position-in-the-idea) | 📋 [TOC](#table-of-contents) | [Next](#the-idea-as-it-stands) ➡️

**What the idea is**

A ladder of small controlled settings, each bringing back one ingredient of SDXL with the others
held fixed, built to find which ingredient makes product-of-experts composition render one fused
animal where the joint prompt renders two.

**Where the walk is**

The opening round is delivered. Claim 1 is next and has not had a round yet.

**What compile would produce today**

Destination A, thin: the thread placement and a proposed rung order, with every claim still open.

## The idea, as it stands

Navigation: ⬅️ [Quick context](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#facts-pinned-before-round-1) ➡️

On SDXL, product-of-experts composition of "a cat" and "a dog" renders one fused animal, and the
joint prompt "a cat and a dog" renders two. The paper measures the gap as the interaction term
`r_t = ε_joint − ε_PoE` per step, and has shown it is causal, acts in an early window, and can be
learned by an adapter. SDXL carries too many candidate causes at once to say which one produces the
gap. The idea is to rebuild the failure in settings small enough that the exact product of the two
experts can be computed or sampled, then add back one ingredient per rung (a learned network, a
convolutional or attention architecture, a latent space, a text pathway) and watch for the rung at
which the chimera first appears or first changes.

The working intuition to test is that composition asks a pretrained model for a distribution it
never saw. The field has a sharper version of that: under the Bayes reading of classifier-free
guidance, each expert's push is the gradient of an implicit classifier `log p(c | x)`, so the
product asks for images that each classifier scores highly at once. What those classifiers learned
"a cat" means, from how the training images were captioned, decides whether "both at once" is a
cat beside a dog or a single animal that half-satisfies each.

## Facts pinned before round 1

Navigation: ⬅️ [The idea](#the-idea-as-it-stands) | 📋 [TOC](#table-of-contents) | [Next](#the-eleven-threads-placed) ➡️

| Fact | Value | Where it comes from |
|---|---|---|
| Prediction target | epsilon | the role's pin list, stated in the brief |
| Process | variance-preserving | the brief |
| Space the diffusion runs in | SDXL VAE latent, `4 × 128 × 128`, 48× compression from `1024 × 1024 × 3` | the learning journey's substrate file |
| Sampler | DDIM, 50 steps for per-epoch renders, 20 for dose renders | the brief; the mismatch is in memory `dose-vs-per-epoch-sampler-mismatch` |
| Guidance scale | 7.5 per expert, so the null prediction enters PoE at weight `1 − 2·7.5 = −14` | the brief; the weight follows from the composition rule |
| Training-set size for the animal concepts | **unknown** | nothing in the repo or the SDXL report pins it |
| Intrinsic dimension of the animal concepts | **unknown** | no estimate exists for this data |

The last two rows mean the role's layer 4 (generalised or memorised) cannot be answered on SDXL.
Every toy rung fixes both numbers by construction, which is one of the things the ladder buys.

## The eleven threads, placed

Navigation: ⬅️ [Pinned facts](#facts-pinned-before-round-1) | 📋 [TOC](#table-of-contents) | [Next](#the-claims) ➡️

| Thread | Bears on the failure? | The field's term for the connection | Claim | Rung that tests it |
|---|---|---|---|---|
| Learning a distribution | yes, as the frame | the product is the distribution `p(x)·Π p(c_i \| x)`; composition asks for its high-density region, which training never sampled | 3 | 0 |
| Neural networks as the learner | yes, possibly | inductive bias; locality and equivariance of convolutional denoisers produce patch-wise recombination of training images | 5 | 2 |
| Diffusion as the learner (denoising, score, SDE, OT views) | no, as a discriminator | the views describe one object; none predicts a different product, so none predicts the chimera where another does not | 2 | none needed |
| Sampling (MCMC, ULA, HMC) | yes, as the control | the target-or-route question; a correct MCMC chain on the clean product tells you what the target is | 2 | 1 |
| Stochastic calculus, Fokker-Planck | yes, same thread as sampling | non-commutation of noising and multiplying (Du et al.): the sum of noised scores is not the score of the noised product. Your suspicion has this name | 2 | 1 |
| Conditioning, CFG, null branch | yes, strongly | the implicit classifier; guidance as an exponent on it; CFG as a predictor-corrector rather than the tilted distribution's score | 3 and 4 | every rung, as an axis |
| Cross-attention and text | partly; same thread as training data | whether a count can be represented in a single-concept conditional | 3 and 5 | 3 |
| The latent space, pullback metric | not as a cause; useful as a readout | pullback metric (Park et al., arXiv 2307.12868); a readout of where trajectories part, not a knob on the failure | 5 | 3, as an instrument |
| The training data | yes, load-bearing for claim 3 | caption exhaustivity: "a cat" learned as "only a cat"; frame-filling crops as a spatial prior | 3 | 0 and 1 |
| Other model families | mostly outside the paper's object | energy-based models and flows can express a product (energies add, log-densities add); a VAE or GAN has no density to multiply | 6 | parked |
| Train it or correct it at sampling time | already answered on SDXL | correctors alone are null there; the ladder can say *why*, by showing whether the product assigns "both" any mass at all | 2 and 6 | 1 |

**Threads the list is missing**

The implicit classifier and what captions teach it (claim 3). Locality and equivariance of the
denoiser (claim 5). Where each object goes: the product of two frame-filling experts has no
spatial arrangement to prefer, so something must break the left-right symmetry, which is the
disjoint-support design and the speciation window (claims 1 and 3). The guidance exponent as a
push toward the product's argmax (claim 4).

## The claims

Navigation: ⬅️ [Threads](#the-eleven-threads-placed) | 📋 [TOC](#table-of-contents) | [Next](#what-the-words-are) ➡️

### 1. 🔍 A small setting reproduces the chimera and the joint-prompt success side by side  ← current

Mark: needs a check. Load-bearing. Two plain 2D Gaussians cannot carry it: their product is a
Gaussian at the precision-weighted midpoint, which is the chimera, and no point in that plane means
"both". The smallest setting that can is one with slots, where a scene holding two objects is a
point of its own. The disjoint-support toy's four numbers are that setting.

- [x] 1.1 a space with one object per point cannot tell "both" from "fused": holds, by the Gaussian product rule; drawn in [why the product blends](why-the-product-blends.html) (published at https://claude.ai/artifact/CsiVYYB1j7arRMv7iGZzMa)   ← current
- [ ] 1.2 the slot toy gives "both" and "fused" separate points, and asks which the product prefers
- [ ] 1.3 a joint condition beside the two experts, so the success side exists in the same toy

### 2. 🔍 The chimera is in the target distribution, not in how the sampler reaches it

Mark: needs a check. SDXL already leans this way (a Langevin corrector alone is null, SuperDiff's
missing piece is the same residual, Soiffer et al. say no inference-time fix reaches an
out-of-distribution target). Nothing yet separates Du's sampler error from the model error.

### 3. The single-concept conditionals learned "a cat" as "only a cat", and that decides what the product asks for

Mark: open.

### 4. Guidance and the null branch push the product toward its most extreme point

Mark: open.

### 5. The network's inductive bias shapes the chimera beyond what the exact score gives

Mark: open.

### 6. Each rung's result decides whether the next is worth building

Mark: open.

## What the words are

Navigation: ⬅️ [The claims](#the-claims) | 📋 [TOC](#table-of-contents) | [Next](#held-claims) ➡️

| My phrase | The field's name | What it means | Confidence |
|---|---|---|---|
| composing two separately prompted scores | product of experts; conjunction (Composable Diffusion) | sum the guided pushes of each prompt, add the unconditional prediction once | confident |
| a distribution the model never saw | the product's support lying outside the training distribution; compositional generalisation | the high-density region of `Π p(c_i \| x)` holds images no training caption described | confident on the framing |
| the step between the noisy product and the clean product | non-commutation of noising and multiplying | the product of two noised densities is not the noised product, so summed scores at `t > 0` follow no single distribution's forward process | confident (Du et al.) |
| what each prompt's push really is | the implicit classifier | `ε(x_t, c) − ε(x_t, ∅) ∝ −∇ log p(c \| x_t)`, the gradient of a classifier the model defines by Bayes' rule | confident (Ho and Salimans) |
| "a cat" meaning "just a cat" | caption exhaustivity; reporting bias | captions name what is salient, so an image holding a cat and a dog is rarely captioned "a cat" and the solo classifier scores it low | likely; the naming is borrowed from captioning work, verify |
| when adding scores works | projective composition; factorized conditionals | the conditions under which the linear score sum samples the intended composition | likely, verify (Bradley et al.) |
| the denoiser stitching parts locally | locality and equivariance bias; patch-mosaic creativity | a convolutional denoiser behaves like a local patch-wise denoiser and recombines patches across training images | likely, verify (Kamb and Ganguli) |
| a plane where "both" and "fused" are the same point | the binding problem; the superposition catastrophe | summing the feature vectors of two objects loses which feature belongs to which object (von der Malsburg) | confident on the terms |
| giving each object its own place in the vector | role-filler binding; tensor product representation | a vector built as slot ⊗ feature, so "cat in slot 1, dog in slot 2" differs from a blend (Smolensky 1990) | confident |
| a model that keeps objects apart | object-centric representation; slot attention | a fixed set of slot vectors that compete for parts of the input (Locatello et al., arXiv 2006.15055) | confident |
| the joint prompt mixing up which animal gets which feature | attribute binding failure; catastrophic neglect | the joint-prompt failure modes in text-to-image models (Chefer et al., Attend-and-Excite, arXiv 2301.13826); a different failure from PoE's, same underlying binding problem | confident on the paper, the link to PoE is this walk's |
| the moment a sample picks its layout | speciation time; symmetry breaking | when two copies of a noisy state stop landing in different classes | confident (Biroli et al., role source 34) |

## Held claims

Navigation: ⬅️ [What the words are](#what-the-words-are) | 📋 [TOC](#table-of-contents) | [Next](#dead-ends) ➡️

| Claim | What is unresolved | What would settle it |
|---|---|---|

## Dead ends

Navigation: ⬅️ [Held claims](#held-claims) | 📋 [TOC](#table-of-contents) | [Next](#checks-outstanding) ➡️

| Claim | The workaround | Why it failed |
|---|---|---|
| 1 | two plain 2D Gaussians as the bottom rung | their product is always a single Gaussian at the midpoint; the space has no point meaning "both", so the success side of the comparison cannot exist |

## Checks outstanding

Navigation: ⬅️ [Dead ends](#dead-ends) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

| Claim | The check | What each outcome means |
|---|---|---|
| 1 | rung 0: the closed-form slot toy from [the disjoint-support idea](../disjoint-marginal-support-and-poe-chimeras/IDEA_MAP.md), with a joint condition added beside the two experts | the product's mode sits on a mixed point while the joint condition sits on both slots: the ladder has its control, climb. Both sit on both slots: the chimera needs something the toy lacks, and claim 5 moves up |
| 2 | the free bound on the model's share, step 24 of the running order, already designed in scope 06 | a large late-step `‖r_t‖` says most of the gap is the model's, and rung 1's MCMC arm becomes confirmation rather than discovery |

## Runs

Navigation: ⬅️ [Checks outstanding](#checks-outstanding) | 📋 [TOC](#table-of-contents) | [Next](#routes) ➡️

| # | Anchor | What it executed | State | Finding |
|---|---|---|---|---|

## Routes

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#sources) ➡️

| # | Anchor | The ask | Return line | State |
|---|---|---|---|---|

## Sources

Navigation: ⬅️ [Routes](#routes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Source | What it gives the idea | Confidence |
|---|---|---|
| [the disjoint-support idea](../disjoint-marginal-support-and-poe-chimeras/IDEA_MAP.md) | rung 0 and rung 1 already designed: four numbers, closed-form products, then small score networks with an additive sampler | read this session |
| [which variable explains what PoE is missing](../which-variable-explains-what-poe-is-missing/IDEA_MAP.md) | the error A (sampler) and error B (model) split, and that readouts are not knobs | read this session |
| [the probe battery](../three-product-comparison/IDEA_MAP.md) | "a cat" × "a cat" as the product with no class conflict, predicted to give one saturated cat | read this session |
| [the same-concept finding](../../../report/is-the-adapter-just-copying-the-joint-prompt/does-the-adapter-return-two-when-both-prompts-name-the-same-concept.md) | the plain product draws one animal when both prompts name the same concept; the failure is plurality itself, with no fusion of two kinds needed | read this session |
| [the report index](../../../report/00-INDEX.md), rungs 1 to 5 and 7 | what is shown; correctors, SuperDiff and search are all null or inconclusive without the correction | read this session |
| `~/goal-setting/learning/deep-learning/diffusion-models/latent-diffusion-architecture-rebuilt/artifacts/substrate.md` | a toy latent diffusion model with a concept pair never trained jointly; its concepts are regions (a quadrant, a border), so they cannot collide on the same pixels | read this session |
| `~/goal-setting/learning/deep-learning/diffusion-models/07-speciation-before-divergence/MASTER_PLAN.md` | a two-mode 2D toy with a branch probe for speciation time | master plan read this session |
| `~/goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/MASTER_PLAN.md` | the floor-up sampler theory; the timing puzzle Du's argument does not obviously answer | mission and question read this session |
| Du et al., *Reduce, Reuse, Recycle*, arXiv 2302.11552 | non-commutation; MCMC-corrected samplers | confident |
| Soiffer et al., *Catastrophic Compositional Generation*, arXiv 2606.23920 | no inference-time technique alone reaches an out-of-distribution composed target | in the reading register |
| Ho and Salimans, *Classifier-Free Diffusion Guidance*, arXiv 2207.12598 | the implicit-classifier reading | confident |
| Liu et al., *Composable Diffusion*, arXiv 2206.01714 | the conjunction rule | confident |
| Bradley et al., *Mechanisms of Projective Composition of Diffusion Models*, arXiv 2502.04549 | when the linear score sum composes, on synthetic data | likely, verify the id and the claim |
| Bradley and Nakkiran, *Classifier-Free Guidance is a Predictor-Corrector*, arXiv 2408.09000 | CFG does not sample the tilted distribution `p(x)·p(c \| x)^w` | likely, verify |
| Kamb and Ganguli, *An analytic theory of creativity in convolutional diffusion models*, arXiv 2412.20292 | locality and equivariance give patch mosaics | likely, verify |
| Okawa et al., *Compositional Abilities Emerge Multiplicatively*, arXiv 2310.09336 | compositional generalisation in diffusion on a synthetic task | likely, verify |
| [why the product blends](why-the-product-blends.html) | four steps on a 2D plane: one Gaussian pair, three-by-three modes with their nine weights, a spread slider, and the feature-axes view where "both" and "fused" coincide | built this session; samples from the exact product, seeded |
| Biroli et al., *Dynamical regimes of diffusion models*, arXiv 2402.18491 | speciation time and the cloning probe | confident |
| Park et al., *Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry*, arXiv 2307.12868 | the pullback metric as a readout | confident |

## Next step

Navigation: ⬅️ [Sources](#sources) | 📋 [TOC](#table-of-contents)

Walk claim 1 with `next`. Its round decides what rung 0 has to contain for the success side of the
comparison to exist at all.
