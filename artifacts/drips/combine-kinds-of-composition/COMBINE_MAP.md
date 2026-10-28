# 📖 Combined reading: kinds of composition in diffusion models

## Position

| Idea | State |
|---|---|
| (a) what each source means by "composition", and its kinds | compared |
| (b) co-occurrence and contention: region products, leakage, chimeras | compared |
| (c) intersection: PoE, logical AND, why summed scores miss the product | compared |
| (d) interaction and attribute binding: does it need more than a product | compared **← current** |

All four ideas are compared. Nothing was tested with numbers, so no idea is marked `tested`.

## Table of contents

- [Position](#position)
- [Why this comparison](#why-this-comparison)
- [Sources](#sources)
- [Idea (a): what "composition" means](#idea-a-what-composition-means)
- [Idea (b): co-occurrence and contention](#idea-b-co-occurrence-and-contention)
- [Idea (c): intersection](#idea-c-intersection)
- [Idea (d): interaction and attribute binding](#idea-d-interaction-and-attribute-binding)
- [Symbols](#symbols)
- [Differences](#differences)

## Why this comparison

Navigation: ⬅️ [Position](#position) | 📋 [TOC](#table-of-contents) | [Next](#sources) ➡️

The [overlap-dial idea walk](../../ideas/poe-overlap-dial/IDEA_MAP.md) proposes four kinds of composition (co-occurrence, intersection, interaction, contention) in [the composition terms](../../ideas/poe-overlap-dial/composition-terms.md), and its claim 7 says interaction is not a third operator. This reading checks how recent papers name and split the same cases, and whether any of them treats interaction as needing a term beyond the product.

Every plain reading and comparison line below is this walk's own reading of the quoted passage. Page numbers are PDF pages of the arXiv version. The walk ran as a route with the sources chosen by the route prompt, so no source was kept or dropped by the reader in a round. It is tied to no learning journey, so this map is the record and no `combined-reading.md` was written to Obsidian.

## Sources

Navigation: ⬅️ [Why](#why-this-comparison) | 📋 [TOC](#table-of-contents) | [Next](#idea-a-what-composition-means) ➡️

| Short name | Paper | arXiv | Used in |
|---|---|---|---|
| Bradley25a | Bradley, Nakkiran, Berthelot, Thornton, Susskind, *Mechanisms of Projective Composition of Diffusion Models* | 2502.04549 | a, c, d |
| Bradley25b | Bradley, *Local Mechanisms of Compositional Generalization in Conditional Diffusion* | 2509.16447 | b |
| Liu22 | Liu, Li, Du, Torralba, Tenenbaum, *Compositional Visual Generation with Composable Diffusion Models* | 2206.01714 | a, d |
| SuperDiff | Skreta, Atanackovic, Bose, Tong, Neklyudov, *The Superposition of Diffusion Models Using the Itô Density Estimator* | 2412.17762 | a |
| CompBench++ | Huang, Duan, Sun, Xie, Li, Liu, *T2I-CompBench++* | 2307.06350 | a |
| MultiDiffusion | Bar-Tal, Yariv, Lipman, Dekel, *MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation* | 2302.08113 | b |
| Bounded | Dahary, Patashnik, Aberman, Cohen-Or, *Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation* | 2403.16990 | b |
| CO3 | Dutta, Chen, Rajagopalan, Wei, Roy Choudhury, *Steer Away From Mode Collisions* | 2509.25940 | b, d |
| RRR | Du et al., *Reduce, Reuse, Recycle* | 2302.11552 | c |
| FKC | Skreta et al., *Feynman-Kac Correctors in Diffusion* | 2503.02819 | c |
| Catastrophic | Soiffer, Squires, Guan, Hartford, Ravikumar, *Catastrophic Compositional Generation* | 2606.23920 | c, d |

## Idea (a): what "composition" means

Navigation: ⬅️ [Sources](#sources) | 📋 [TOC](#table-of-contents) | [Next](#idea-b-co-occurrence-and-contention) ➡️

`combined > what "composition" means > idea 1 of 4 · 4 sources`

**Bradley25a p.5, Definition 4.1**
> "Given a collection of distributions {pᵢ} along with associated 'projection' functions {Πᵢ : ℝⁿ → ℝᵏ}, we call a distribution p̂ a projective composition if ∀i: Πᵢ♯p̂ = Πᵢ♯pᵢ."

In plain words: a composition works when the result looks exactly like expert i once you look only at the features expert i is about. Πᵢ picks those features: a set of pixels, or a feature such as style. The paper sets this against two older definitions (p.4). The "simple product" p₁p₂ cannot leave either expert's support. The "Bayes composition" p(x|c₁)p(x|c₂)/p(x) cannot either, because it is a conditional of a distribution that never contained both.

**Liu22 p.7, eqs. 9 and 10**
> "Concept Conjunction (AND). … p(x|c₁,…,cₙ) ∝ p(x) ∏ p(cᵢ|x). Here we assume the concepts are conditionally independent given x."

In plain words: composition has two operators, AND and NOT. AND conditions one model on several concepts at once, assuming the concepts are independent once the image is known. Each p(cᵢ|x) is an implicit classifier built from a conditional and an unconditional model.

**SuperDiff p.2**
> "(i) generating samples from a mixture of densities (an OR operation over models) or (ii) generating samples that are equally likely under all densities (an AND operation over models)."

In plain words: composition is "superposition", and it comes in two forms. OR samples the mixture. AND samples points the two models score equally, which is a different object from the product. On p.5 the same two labels appear swapped (mixture called AND, equal densities called OR). That reads as a slip, since p.2 and Figure 2 agree with each other.

**CompBench++ p.1, Fig. 1 caption**
> "…attribute binding (including color, shape, and texture), generative numeracy, object relationships (including 2D/3D-spatial relationship and non-spatial relationship), and complex compositions."

In plain words: for text-to-image benchmarks, composition means following a prompt with several objects, attributes and relations. The kinds are kinds of prompt, not kinds of density operator.

**How they compare**

Same: every source treats composition as combining separately specified concepts into one sample.

Differs (*different scope*): Bradley25a, Liu22 and SuperDiff define composition as an operation on densities. CompBench++ defines it by prompt category, with no operator.

Differs (*they disagree*): "AND" means the Bayes product p(x)∏p(x|cᵢ)/p(x) in Liu22, the equal-density set in SuperDiff, and Bradley25a treats it as one of several definitions that fail out of distribution.

Adds: Bradley25a adds that a composition is defined only once you say *which features* of each expert must be preserved. Composition over features (style + content) is the same kind of object as composition over pixel regions.

Skips: no source names co-occurrence, intersection and interaction as three kinds. Only CompBench++ uses the word "interaction", and it means a relation verb between two objects.

## Idea (b): co-occurrence and contention

Navigation: ⬅️ [Idea (a)](#idea-a-what-composition-means) | 📋 [TOC](#table-of-contents) | [Next](#idea-c-intersection) ➡️

`combined > co-occurrence and contention > idea 2 of 4 · 4 sources`

**Bradley25b p.4, Figure 3 caption**
> "A CPC is a conditional distribution over a set of conditions c_J that factorizes independently into the marginals over x_{M_j} conditioned on c_j, where M_j are disjoint subsets. An LCS is a conditional score … such that the score at each pixel i depends only on a subset N_i of other pixels … as well as a subset L_i ⊂ J of conditions."

In plain words: when each concept owns its own pixels (a conditional projective composition, CPC), the score at a pixel depends only on nearby pixels and nearby conditions (a local conditional score, LCS). The paper proves the two are equivalent. CLEVR models that generalize to more objects have local conditional scores, and models that fail do not (p.1).

**MultiDiffusion p.1**
> "…the reference diffusion model is applied to different regions in the generated image, predicting a denoising sampling step for each. In turn, the MultiDiffusion takes a global denoising sampling step reconciling all these different steps via least squares optimal solution."

In plain words: each region is denoised by its own call to the model, and where regions overlap, the shared pixels take the least-squares compromise. This is a region-restricted composition built at the sampler, with a layout given by hand.

**Bounded p.1**
> "…the primary issue stems from inadvertent semantic leakage between subjects in the denoising process. This leakage is attributed to the diffusion model's attention layers, which tend to blend the visual features of different subjects."

In plain words: when one image holds two similar subjects (a kitten and a puppy), features of one appear on the other. The paper traces this to attention mixing features across subjects, and fixes it by blocking attention between subject regions.

**CO3 p.2, eq. 1**
> "Our corrector distribution p̃(x|C) ∝ p(x|C) / ∏ᵢ p(x|cᵢ) assigns low probability to regions where p(x|C) overlaps with individual p(x|cᵢ); we deem them as degenerate modes dominated by a single concept."

In plain words: in "a cat and a dog", the joint-prompt model sometimes lands in a mode that looks like only one concept (one is missing, faint, or "colliding"). CO3 calls this a mode collision. It divides the joint by each single-concept model to push samples away from those modes.

**How they compare**

Same: all four locate the failure where two concepts compete for the same part of the image or the same internal features.

Differs (*different scope*): Bradley25b and MultiDiffusion separate the concepts by pixel region. Bounded separates them by attention. CO3 does not separate them at all and corrects the joint model's modes.

Differs (*different assumption*): Bradley25b gets the regions from the score's own locality, with no layout given. MultiDiffusion needs the layout as input.

Adds: CO3 adds that the failure includes *dominance* (one concept wins), not only fusion. Bounded adds that the failure is worst for *semantically similar* subjects.

Skips: no source uses the word "chimera". CO3 and Bounded call the fused outcome "colliding" and "blend" respectively.

## Idea (c): intersection

Navigation: ⬅️ [Idea (b)](#idea-b-co-occurrence-and-contention) | 📋 [TOC](#table-of-contents) | [Next](#idea-d-interaction-and-attribute-binding) ➡️

`combined > intersection > idea 3 of 4 · 4 sources`

**RRR p.4, eqs. 11 and 12**
> "∇ₓ log q̃ₜᵖʳᵒᵈ(xₜ) = ∇ₓ log ∫dx₀ q¹(x₀)q²(x₀)q(xₜ|x₀). For t > 0, this quantity is not equal to the sum of the scores of the two models…"

In plain words: to sample a product by reverse diffusion you need the score of the *noised product*. Adding the two models' scores gives the score of the *product of the noised* densities instead. These differ whenever there is noise, so reverse diffusion with summed scores samples something else. MCMC with an energy function fixes this.

**FKC p.3 to 4, eq. 15 and the sentence after eq. 16**
> "Product: pₜᵖʳᵒᵈ(x) = (1/Zₜ) q¹ₜ(x)q²ₜ(x) … this heuristic does not sample from the prescribed marginals (including the final distribution), except in special cases."

In plain words: FKC takes the product of the noised densities as the path to follow. It derives the extra weight each sample must carry so that the path is followed exactly, then resamples by those weights (sequential Monte Carlo).

**Bradley25a p.6, Theorem 5.3, and p.7**
> "The sampling guarantee follows from the commutativity of composition with the diffusion noising process, i.e. C[p⃗ₜ] = Nₜ[C[p⃗]]."

In plain words: when the experts are factorized conditionals (each changes only its own set of coordinates and copies a shared background elsewhere), composing then noising equals noising then composing. Plain reverse diffusion with summed scores is then exact. So the RRR problem belongs to experts that overlap on the same coordinates.

**Catastrophic p.10, Table 1**
> "…FKC strictly worsens performance when applied to learned models. It is only when the algorithm is supplied with the exact analytical scores that it achieves the expected theoretical improvements."

In plain words: two errors stack. The sampler's error is fixed by FKC. The score error, where the target lies outside every expert's training data, is not fixed by any sampler, and with learned scores more FKC particles make the fit worse.

**How they compare**

Same: the product of experts is the agreed meaning of intersection, and reverse diffusion with summed scores does not sample it in general.

Differs (*different order of steps*): RRR corrects with MCMC at each noise level. FKC reweights and resamples particles along the path. Bradley25a shows that no correction is needed for factorized experts.

Differs (*different scope*): RRR, FKC and Bradley25a analyse the sampler with exact or well-trained scores. Catastrophic shows that out of distribution the score error dominates.

Adds: Catastrophic adds the two-by-two split the project's claim 4 asks for (sampler error against score error, in against out of distribution), with numbers. Bradley25a p.4 adds that the simple product has zero mass wherever either expert has zero mass.

Skips: none of the four separate "square inside circle" (nested shapes) from "a shape that is both".

## Idea (d): interaction and attribute binding

Navigation: ⬅️ [Idea (c)](#idea-c-intersection) | 📋 [TOC](#table-of-contents) | [Next](#symbols) ➡️

`combined > interaction and attribute binding > idea 4 of 4 · 4 sources`

**Bradley25a p.4**
> "Because the two events c₁ and c₂ are conditionally independent given x (since they are deterministic functions of x), we can compute p̂ in terms of the individual conditionals: p̂(x) := p(x|c₁,c₂) ∝ p(x|c₁)p(x|c₂)/p(x)."

In plain words: this is the project's claim 7 argument, stated in a published paper. Labels that are fixed functions of the image make the interaction factor exactly 1. The paper then shows the result still cannot place mass where the training distribution had none. On p.17 it notes that cluttered training data lets "the models learn how multiple objects should interact and even overlap correctly".

**Catastrophic p.2, and Appendix B p.21**
> "…if two biological perturbations a and a′ are causally separable, it can be shown that the double perturbation distribution P(x|do(a),do(a′)) can be expressed as a geometric combination of the control distribution P(x) and the single perturbation distributions…"

In plain words: when two causes act on different hidden mechanisms (non-overlapping intervention targets), the joint effect is P_a·P_a′/P_0, the same Bayes form as above. When they act on the same mechanism, this form is wrong. Appendix B says the separable case generalizes Bradley25a's factorized conditionals. The paper cites Xu et al. 2024, *Automated discovery of pairwise interactions from unstructured data*, for detecting the non-separable pairs.

**Liu22 p.13, failure cases**
> "The second type of failure is because the diffusion models confuse the objects' attributes. In (c), the generated image contains 'a red bear' while…"

In plain words: binding an attribute to the wrong object is reported as a failure of the AND operator. It is not treated as a missing term.

**CO3 p.2**
> "…while p(x|C) may assign high probability to weakly conforming images due to training noise or limited multi-concept data, dividing by the marginals removes this bias…"

In plain words: CO3 keeps a model of the joint prompt, p(x|C), as the main term and divides out the single-concept models. Whatever the joint carries beyond the product comes from that joint model, learned from data with both concepts.

**How they compare**

Same: no source adds an interaction factor to a product of single-concept experts at sampling time. A joint effect comes from data or from a model trained on the joint.

Differs (*different assumption*): Bradley25a makes the interaction factor vanish by treating labels as functions of the image. Catastrophic makes the product form exact by assuming the causes act on separate mechanisms, and names the opposite case as a non-separable interaction.

Adds: Catastrophic adds a name and a test for interaction: overlapping intervention targets, detectable from data (Xu et al. 2024, recalled through the citation, not opened). CO3 adds a sampling rule that uses a joint model alongside the single-concept models.

Skips: Liu22 and CompBench++ treat binding as accuracy on a prompt and give no account of when a product could or could not bind.

## Symbols

Navigation: ⬅️ [Idea (d)](#idea-d-interaction-and-attribute-binding) | 📋 [TOC](#table-of-contents) | [Next](#differences) ➡️

| Object | Project | Bradley25a | Liu22 | RRR / FKC | Catastrophic | CO3 |
|---|---|---|---|---|---|---|
| expert density for A | p_A(x) | pᵢ(x) | p(x\|cᵢ) | q¹(x) | P_a(x) | p(x\|cᵢ) |
| shared background | p(x) | p_b(x) (any reference, often empty) | p(x) (unconditional) | none | P_0(x) (control) | none |
| joint-prompt model | p(x\|A,B) | not used | not used | not used | P(x\|do(a),do(a′)) | p(x\|C) |
| region or feature owned by A | S | Mᵢ, Πᵢ | none | none | intervention targets T(I) | none |
| noised density at time t | q_A,t | pₜᵢ = Nₜ[pᵢ] | none | qₜ | Pₜ | pₜ |

Bradley25a's background p_b can be an empty scene. Liu22's p(x) is always the unconditional model. Both are written "background", and they behave very differently (Bradley25a Figure 3, settings A and B).

## Differences

Navigation: ⬅️ [Symbols](#symbols) | 📋 [TOC](#table-of-contents)

| Idea | Kind | Source A says | Source B says | Resolved |
|---|---|---|---|---|
| a | they disagree | Liu22 p.7: AND is p(x)∏p(x\|cᵢ)/p(x) | SuperDiff p.2: AND is the set where all models give equal density | open: two different objects share one name |
| a | slip in the printed paper | SuperDiff p.2 and Fig. 2: mixture is OR, equal density is AND | SuperDiff p.5: mixture called AND, equal density called OR | read: p.2 and Fig. 2 agree, so p.5 is the slip |
| a | different scope | Bradley25a p.5: composition is defined by the features each expert must keep | CompBench++ p.2: composition is defined by prompt category | open: the two do not map one to one |
| a | different symbol | CompBench++ p.5: "Non-spatial relationships usually describe the interactions between two objects" (hold, wear) | the project: interaction is concepts changing each other's appearance | open: same word, different meaning |
| b | different assumption | Bradley25b p.4: regions come from the score's locality, with no layout | MultiDiffusion p.1: regions are given as input | read |
| b | different scope | CO3 p.2: the failure includes one concept dominating | the project: contention is a fused chimera | open: the literature has dominance, leakage and fusion as separate symptoms |
| c | stated or derived | RRR p.4: summed scores miss the noised product | Bradley25a p.6: summed scores are exact for factorized conditionals | resolved by reading: both hold, for different expert pairs |
| c | different scope | FKC p.4: reweighting fixes the sampler | Catastrophic p.10: with learned scores out of distribution, reweighting makes the fit worse | resolved by reading: different error sources |
| d | different assumption | Bradley25a p.4: interaction factor is 1 when labels are functions of x | Catastrophic p.2: the product form holds when causes act on separate mechanisms | open: whether medical labels are better modelled as functions of the image or as causes |
