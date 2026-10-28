# When a product of experts reaches A-and-B: the compiled idea

**This page is closed.** Its findings, with the later results that correct some of its numbers, are walked from [the prompt guide](../../drips/simple-distributions-rebuild/PROMPTS.md). Open the guide, not this page.

The one-page state of [the overlap-dial idea map](IDEA_MAP.md), for the session walking [the unbuild route prompt](routes/03-unbuild-walk.prompt.md) and for anyone picking the idea up cold. Every line traces to a run, a source or a file named in the map.

## The idea, said plainly

Two diffusion models, each trained on one concept, are composed at sampling time by adding their scores, which targets the product of their densities. A product can only put mass where both experts already have mass, so samples that are both A and B come from three possible sources: A-and-B examples inside each training set (the overlap dial), support inherited from a pretrained model the experts were fine-tuned from, and the network's own extrapolation. In 2D the third source is absent: trained networks match the smoothed-data ideal within `0.04` nats.

"Composition" covers two operators and two data conditions:

- **A product over separate parts** gives co-occurrence: two objects, each in its own region.
- **A product over the same variables** gives intersection: one thing with both properties.
- **Whether each expert saw its object in more than one place** decides between co-occurrence and a chimera.
- **Whether any expert saw A as it looks under B** decides whether an interaction (a wet dog) appears.

Noisy or diagnosis-level labels add one label term.

The sampler is a separate axis. Reverse diffusion with summed scores misses the product, except for mixtures, while annealed UHA and tuned MALA reach the exact target on most kinds. Near the edge of what the data supports, a weak sampler draws the same picture as a data failure. So the two failures can be told apart only where the exact composition is known, which is why the work runs on exact toys first: the paper's 2D datasets and a 16-pixel Gaussian-mixture parts world, where the cat-dog mosaic, blend and side-by-side outcomes are all computed in closed form.

The medical cells wanted, pneumothorax without a drain and drain without pneumothorax, are a negation, and they are rare rather than absent.

## The claim ledger

| # | Claim, as it stands | Mark | What settled it |
|---|---|---|---|
| 1 | A density expert learns its training set's density; labels matter only where something reads them | ✅ settled, repaired | [worked-examples notebook](what-an-expert-learns-and-kinds-of-composition.ipynb) sections 1 to 3; run 2 |
| 2 ⚓ | The product puts samples in A∩B with no A∩B training data | 🔍 needs a check | exact 2D answer is no (`0.12` in the crossing at 0% overlap); the open part is whether fine-tuning from a pretrained model supplies the support: route 3, phase 5 |
| 3 | One conditional model, score(x\|A)+score(x\|B)−score(x), and two separate experts compose the same thing | 🔧 wrong as stated, not walked | (1) targets p(x\|A)p(x\|B)/p(x), (2) targets p_A·p_B; equal only when p(x) is flat; every chest X-ray generator found is one conditional model (route 2) |
| 4 | Sampler failures and data failures can be told apart only where the exact composition is known | 🔧 repaired, not settled | runs 4 to 6: reverse diffusion draws a chimera where the data supports co-occurrence (`0.21` against exact `0.75`); UHA and tuned MALA reach the floor; dominance defeats every sampler |
| 5 | "Fraction inside A∩B" needs a coverage number beside it | 🔧 repaired in practice | total variation distance to the exact composition, used in every run since run 4 |
| 6 | The toy answers the pneumothorax question | 🔧 wrong as stated | the cells are negation and rare, not absent: 80% (radiologist) and 46.2% (counted) of NIH pneumothorax images carry a drain; "without a drain" must also mean untreated (route 2) |
| 7 | Composition is two operators, two data conditions and one label term | ✅ settled, repaired | notebook section 4; runs 3 and 4; route 1 (no paper adds an interaction term to a product); route 2 (diagnosis labels carry the label term) |
| 8 | The paper's 2D toys, one dial per kind, host every kind but leakage | 🔍 needs a check | run 3 passed six of seven kinds; dominance blurs into empty intersection |
| 9 | The parts world replicates the cat-dog hybrid exactly | ✅ settled, repaired | run 7: [the parts-world write-up](example-9-parts-world/example-9-parts-world.tex), [the parts-world scene](https://claude.ai/artifact/6XzsUMYdEb2eopRyGkpYkh) |

## What it takes to build

Each step is written as markdown walked with `/drip-code` or `/drip --code`: definition, derivation, worked example with real numbers, then code in small steps with NumPy, matplotlib, scikit-learn, seaborn and PyTorch. Steps 1 to 5 are route 3's phases.

1. **The notebook in PyTorch.** Datasets, the ResNet score model, the energy parameterisation, the cosine schedule, training. Needs nothing. Check: samples match `datasets-1000-samples-each.png`.
2. **Wrappers and samplers.** Product, mixture and negation; annealed ULA, MALA, UHA from `anneal_samplers.py`, with MALA's two bugs fixed without breaking `inf_sample.py`. Needs step 1. Check: the paper's ring-times-bar and left-plus-right figures, with per-sample MALA acceptance.
3. **Ideal experts and the 2D suite.** Seven kinds, one dial each, exact targets. Needs step 2. Check: [the kinds notebook](kinds-of-composition-on-the-papers-toys.ipynb)'s Figure 1 and Figure 2 numbers reproduced.
4. **The parts world.** Gaussian-mixture experts on 16 pixels, exact products, then every sampler against the exact product. Needs step 3. Check: `example-9-parts-world/numbers.json` reproduced, with each sampler's component-weight error and TV.
5. **The overlap dial with fine-tuning.** A∩B mass against overlap from 0% to 100%, and a pretrained-then-fine-tuned variant at 0% overlap. Needs steps 3 and 4. Check: the curves for the exact composition, tuned MALA and UHA. This answers claim 2.
6. **Negation on the toys.** A∧¬B and ¬A∧B on the quadrant setup and the parts world, with the exact target, before any chest X-ray. Needs step 5. Check: mass in the A-without-B cell against the exact composition, per sampler.

## Still open

| What | What settles it |
|---|---|
| claim 2, the fine-tune support | route 3, phase 5 |
| claim 3, construction (1) against (2) | one walk round, then a 2D run of both constructions on the same data |
| claim 4, dominance (unequal mode weights) | a reweighting sampler such as Feynman-Kac correctors (arXiv 2503.02819) on the dominance row |
| claim 6, negation | build step 6 |
| claim 8, dominance against intersection | a dominance pair whose broad expert sits off the sharp one's modes |
| `anneal_samplers.py` MALA and CHA bugs | route 3, phase 2 |
| three stuck pictures (Diffusion 33, Multivariate Normal 9 and 10) | a word from the user on each stuck line |
| the Codex bridge (`--no-daemon` rejected by codex-cli 0.147.0-alpha.6.5) | remove the flag from `codex_bridge.py` |
| the label-meaning route prompt `03-what-a-label-teaches-and-the-toy-axes.prompt.md` from another session | append as claim 10 if brought into this walk |

## The route out

Route 3, `/unbuild --walk` over `notebooks/simple_distributions.ipynb`, is running as session `4bdfbc90` and carries steps 1 to 5. Its return lands in `routes/03-unbuild-walk.return.md`, and `integrate` in this walk folds it in.
