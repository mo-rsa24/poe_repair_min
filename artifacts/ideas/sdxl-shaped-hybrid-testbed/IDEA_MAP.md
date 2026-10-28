# 💡 An SDXL-shaped testbed for the cat-dog hybrid

## Position in the idea

| Claim | Mark | Settled by |
|---|---|---|
| **1. A hybrid in this testbed has the same cause as SDXL's cat-dog hybrid (current)** | **wrong as stated** | **SDXL's cause is not established, so the testbed cannot inherit it; repair proposed: switch candidate causes one at a time and match each by a signature that the existing SDXL cache also carries** |
| 2. The testbed covers everything the 2D toy covered | wrong as stated | the 2D measures that need the exact composed density do not exist in pixel space; repair proposed: score in the space of drawing parameters, where the generator's density is known |
| 3. A parser of the drawing scores each output as both present, one missing, or fused | needs a check | outcome categories fixed before training, then the parser run on held-out training draws and on hand-built hybrids |
| 4. A tiny text encoder gives the testbed SDXL's word similarity | wrong as stated | a jointly trained encoder puts two words close only if their pictures look alike; repair proposed: a frozen embedding table with the cosine set by hand |
| 5. Reading each word's cross-attention map shows where the hybrid comes from | wrong as stated | a map is an observation; repair proposed: every probe paired with an intervention |
| 6. Each experiment fits in minutes to an hour on the cluster | needs a check | one timed training run of the smallest model on the `co3` environment |

Load-bearing: claim 1. If no hybrid in the testbed can be matched to SDXL's, the testbed explains its own hybrids and nothing about the one the project studies.

## Table of contents

- [Position in the idea](#position-in-the-idea)
- [Quick context: where you are](#quick-context-where-you-are)
- [The idea, as it stands](#the-idea-as-it-stands)
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

A small text-to-image diffusion model with each of SDXL's structural parts, trained on 32 × 32 procedurally drawn shapes, in which each candidate cause of a hybrid can be switched on alone, so the cause behind SDXL's cat-dog hybrid can be narrowed by matching signatures.

**Where the walk is**

Claim 1, the load-bearing one, is current on its first round. It is marked wrong as stated, with a repair proposed.

**What compile would produce today**

Neither destination. Claim 1's repair has not been walked, and the probe-depth table (which cause each probe can isolate) does not exist yet.

**The bar and the time box**

Bar: at compile, claim 1 is settled with at least two candidate causes, each carrying a signature computable both in the testbed and from the SDXL training cache, and a table places every probe as isolating or not isolating each cause; every row names its check. Judged by the reader against the claim ledger. Time box: two sittings of about 90 minutes.

## The idea, as it stands

Navigation: ⬅️ [Quick context](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claims) ➡️

SDXL composes "a cat" and "a dog" with one model and two prompts, at guidance `7.5`, and often draws one fused animal. A testbed with SDXL's parts at small scale (a text encoder over a small vocabulary, a UNet with cross-attention at two resolutions, a pooled text embedding added to the time embedding) is trained on procedurally drawn shapes, so every output can be parsed back into what was drawn. Each candidate cause of a hybrid (what single-concept captions describe, the guidance weight, how close the two words' embeddings sit, whether the two words' attention shares pixels, the sampler) is switched on alone. Each leaves a signature, such as the size of the interaction residual over denoising steps, and the same signature can be read from SDXL's existing cache of noise predictions. A cause is a candidate for SDXL's hybrid when its signature matches there.

## The claims

Navigation: ⬅️ [The idea](#the-idea-as-it-stands) | 📋 [TOC](#table-of-contents) | [Next](#what-the-words-are) ➡️

### 1. 🔧 A hybrid in this testbed has the same cause as SDXL's cat-dog hybrid  ← current

Mark: wrong as stated, repair proposed. Load-bearing. SDXL's cause is not yet known. The main project's lower bound on the interaction, computed from its training cache for 25 pairs, did not track which pairs blend (`poe_repair_min/artifacts/results/does-interaction-strength-predict-which-pairs-blend/figure-explainer.md`). The SDXL hybrid comes from one model with two prompts, composed as ε_∅ + w(ε_A − ε_∅) + w(ε_B − ε_∅) at w = `7.5` (inferred from the cache's residual formula `gs*(eps_j − eps_a − eps_b + eps_uncond)`). That is construction (1) tilted by w, targeting p(x)·p(A|x)^w·p(B|x)^w, and the 2D walk measured construction (1) only at w = 1.

Repaired wording: the testbed produces hybrids by candidate causes switched one at a time, and a cause becomes a candidate for SDXL's hybrid when the signature it leaves in the testbed also appears in SDXL's cache.

### 2. 🔧 The testbed covers everything the 2D toy covered

Mark: wrong as stated, repair proposed. The support rule and the two constructions carry over as rates. Total variation to the exact product above an exact-sample floor (the `0.049` sampler result) needs the composed density, which pixel space does not give. The generator's density over drawing parameters is known exactly, so a parser that maps each output back to its parameters makes those measures computable in parameter space.

### 3. 🔍 A parser of the drawing scores each output as both present, one missing, or fused

Mark: needs a check. The contention forms (fusion, dominance, leakage) have to be fixed as parser categories before any model is trained.

### 4. 🔧 A tiny text encoder gives the testbed SDXL's word similarity

Mark: wrong as stated, repair proposed. CLIP puts "cat" near "dog" through contrastive training on images where cats and dogs look alike. A small encoder trained with the UNet has no such pressure. A frozen embedding table with the two words' cosine set by hand makes similarity a dial, and it separates word similarity from visual similarity, which SDXL's encoders tangle.

### 5. 🔧 Reading each word's cross-attention map shows where the hybrid comes from

Mark: wrong as stated, repair proposed. Two words attending to the same pixels can be the cause of a hybrid or its consequence. Each probe needs an intervention beside it (mask one word's attention to its region, swap the two maps, drop cross-attention at one resolution).

### 6. 🔍 Each experiment fits in minutes to an hour on the cluster

Mark: needs a check. Every data dial (overlap, caption style) needs its own training run.

## What the words are

Navigation: ⬅️ [The claims](#the-claims) | 📋 [TOC](#table-of-contents) | [Next](#held-claims) ➡️

| My phrase | The field's name | What it means | Confidence |
|---|---|---|---|
| composing two prompts at guidance 7.5 | Composable Diffusion AND with classifier-free guidance | ε_∅ + Σ w(ε_c − ε_∅), whose target is p(x)·∏ p(c\|x)^w | confident on the operator, the tilt is inferred |
| cat and dog sitting close in the encoder | cohyponym entanglement | in Stable Diffusion, prompting two cohyponyms lowers generation quality by about 9% (DAAM) | confident |
| one object's features leaking onto the other | semantic leakage | attention mixes the two subjects' features, worst for similar subjects (Bounded Attention) | confident |
| one concept vanishing | catastrophic neglect | Attend-and-Excite's name for a subject missing from the image | recalled, not opened |
| new images stitched from training pieces | patch mosaic | locality and equivariance of a convolutional denoiser combine local training patches into novel images (Kamb and Ganguli) | confident |
| the shapes' attributes as axes | concept space | each image is a point on axes such as shape, colour, size; compositional ability is read per axis (Okawa et al., Park et al.) | confident |

## Held claims

Navigation: ⬅️ [What the words are](#what-the-words-are) | 📋 [TOC](#table-of-contents) | [Next](#dead-ends) ➡️

| Claim | What is unresolved | What would settle it |
|---|---|---|

## Dead ends

Navigation: ⬅️ [Held claims](#held-claims) | 📋 [TOC](#table-of-contents) | [Next](#checks-outstanding) ➡️

| Claim | The workaround | Why it failed |
|---|---|---|

## Checks outstanding

Navigation: ⬅️ [Dead ends](#dead-ends) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

| Claim | The check | What each outcome means |
|---|---|---|
| 3 | Fix the outcome categories, then run the parser on 1,000 held-out training draws and on 200 hand-built hybrids (a red dot cluster, a blue circle, a circle made of dots) | above 99% correct on both: the parser is the scorer. Below it on hybrids: the categories are not separable from pixels and need redefining |
| 6 | Train the smallest model on single-concept captions in `co3` on one GPU node until the parser calls 95% of single-concept samples correct | under 30 minutes: dials can be swept as parallel jobs. Over an hour: shrink the UNet or the vocabulary first |

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
| [the parent walk](../when-poe-reaches-a-and-b/IDEA_MAP.md), claim 12 | the findings this testbed must reproduce: the support rule, the two constructions, the sampler result, the kinds of composition | read in this session |
| [the composition terms](../poe-overlap-dial/composition-terms.md) | contention's three forms (fusion, dominance, leakage); the four questions that place any composition | read in this session |
| `poe_repair_min/context/world/poe-composition.md` and `chimera.md` | the SDXL hybrid: one model, two prompts, guidance `7.5`, 50 DDIM steps; scored by an instance counter that cannot tell two dogs from a cat and a dog | read in this session |
| `poe_repair_min/artifacts/results/does-interaction-strength-predict-which-pairs-blend/figure-explainer.md` | the cache of ε_A, ε_B, ε_joint, ε_∅ at all 50 steps for 25 pairs under `/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache/`; the interaction bound did not track fail rate | read in this session |
| `poe_repair_min/artifacts/ideas/disjoint-marginal-support-and-poe-chimeras/IDEA_MAP.md` | the frame-filling-crops explanation of the chimera, parked with a two-stage toy design | read in this session |
| Tang et al. 2023, *What the DAAM* ([arXiv 2210.04885](https://arxiv.org/abs/2210.04885)) | cross-attention attribution maps for Stable Diffusion; cohyponyms lower generation quality by about 9% | confident on the paper, the 9% from a search summary |
| Okawa et al. 2023, *Compositional Abilities Emerge Multiplicatively* (NeurIPS 2023) | a synthetic-shape diffusion testbed with concepts as axes | confident |
| Park et al. 2024, *Emergence of Hidden Capabilities* ([arXiv 2406.19370](https://arxiv.org/abs/2406.19370)) | learning dynamics in concept space on synthetic data; authors Park, Okawa, Lee, Tanaka, Lubana | confident it exists, contents not opened |
| Kamb and Ganguli 2025, *An analytic theory of creativity in convolutional diffusion models* (ICML 2025, [arXiv 2412.20292](https://arxiv.org/abs/2412.20292)) | locality and equivariance build novel images as patch mosaics, an inductive bias that could build factorised experts without masks | confident |
| Dahary et al. 2024, *Bounded Attention* ([arXiv 2403.16990](https://arxiv.org/abs/2403.16990)) | semantic leakage through attention, worst for similar subjects | confident, quoted in the composition terms |

## Next step

Navigation: ⬅️ [Sources](#sources) | 📋 [TOC](#table-of-contents)

Claim 1 is current. `fix` lists the candidate causes with the signature each leaves; `how` names the smallest check on the SDXL cache.
