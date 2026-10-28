# 💡 When product-of-experts composition reaches A-and-B: a 2D overlap dial

**This map is closed.** Its findings, with the later results that correct some of its numbers, are walked from [the prompt guide](../../drips/simple-distributions-rebuild/PROMPTS.md). Open the guide, not this map.

## Position in the idea

| Claim | Mark | Settled by |
|---|---|---|
| 1. A density expert learns its training set's density; labels matter only where something reads them (repaired from: an expert trained on "A" data learns the concept A) | wrong as stated, settled on the repair | notebook sections 1 to 3 (gradient difference `0.0`; composed A∩B mass `1.00` "contains" against `0.08` "exactly"); run 2 (trained within `0.04` nats of ideal) |
| 2. The product puts samples in A∩B only where some expert carries density there: from overlap in the training data, or from an expert that is loose outside its concept's region; support inherited from pretraining does not survive fine-tuning (repaired from: the product puts samples in A∩B even when no training sample is A∩B) | wrong as stated, settled on the repair | route 3: quadrant toy deep A∩B mass `0.00` at 0% overlap, `0.30` at 5%, `0.59` at 20% (exact); parts world `0.000` at 0% (a ghost blend), `1.000` from 0.1% overlap, and `1.000` at 0% when each expert is loose on the other half (background variance 1.0); pretrained support falls from `0.16` to under `0.01` within 30 fine-tune steps (exact) and 100 (samplers). Medical: 20% to 54% of NIH pneumothoraces have no drain, far above the 0.1% the parts world needs |
| 3. Construction (1), score(x\|A)+score(x\|B)−score(x), targets p(x\|A)p(x\|B)/p(x), which is construction (2)'s p_A·p_B divided by a background expert; the two agree only when the background is flat (repaired from: they compose the same thing) | wrong as stated, settled on the repair | settled in the parallel walk `when-poe-reaches-a-and-b`, claim 3; with "contains" labels construction (1) keeps `0.989` in A∩B down to overlap 10⁻⁴ |
| 4. Sampler failures and missing-data failures can be told apart only where the exact composition is known (repaired after `break 4`, from: can be told apart) | wrong as stated | run both samplers on analytic experts, whose diffused scores are exact, before any trained model; published precedent with numbers in Soiffer et al. 2026, Table 1; route 3: at 2% to 10% overlap tuned MALA and UHA recover a third to four fifths of the exact A∩B mass (5%: `0.17` against `0.30`), reverse diffusion far less (`0.02`) and overshoots at 100% (`0.91` against `0.76`) |
| 5. "Fraction of samples inside A∩B" is the number for "it worked" | wrong as stated | needs a second number for coverage, distance to the exact product |
| 6. The toy answers the pneumothorax question | wrong as stated | the medical cells wanted are A∧¬B and ¬A∧B, which is negation, and they are rare rather than absent: 80% of NIH pneumothorax test images carry a drain (Oakden-Rayner et al. 2019, radiologist) and 46.2% of 3,543 others do (counted from Cheplygina et al. 2025's non-expert labels), so 20% to 54% are pneumothorax without a drain; a treated pneumothorax can lose its pleural line, so "pneumothorax without drain" must also mean untreated; no chest X-ray generator found composes separately trained models (route 2) |
| 7. Composition is two operators and two data conditions: a product over a partition (co-occurrence) or over the same variables (intersection), with interaction a data condition and noisy labels adding one label term (repaired from: composition is one operation) | wrong as stated, settled on the repair | notebook sections 4a to 4c; runs 3 and 4; route 1 (no paper adds an interaction term to a product; the clean-label argument is Bradley et al. 2025a's Bayes composition, p.4); route 2: a diagnosis is read from its findings with uncertainty ("The CXR is not a diagnostic test for HF", Pan et al. 2021), so diagnosis labels carry the label term while finding labels given the diagnosis do not |
| 8. The paper's own 2D toy datasets, in settings with one dial per kind, host every kind but leakage and isolate the conditions each needs (settled on alternative (a), from: a shapes-image testbed) | needs a check | run 3: the suite on ideal experts, each kind's verdict at its base setting and its failure at the dial's far end |
| 9. The parts world (16 pixels, Gaussian-mixture experts) replicates the cat-dog hybrid exactly: mosaic, blend or two animals, decided by where each expert is consistent (repaired from: the 2D toys replicate it) | wrong as stated, settled on the repair | run 7: [the parts-world write-up](example-9-parts-world/example-9-parts-world.tex), [the parts-world scene](https://claude.ai/artifact/6XzsUMYdEb2eopRyGkpYkh) |

Load-bearing: claim 2. If a product cannot place mass where neither expert's training data was, composition cannot fill the missing cells and the medical hope rests on something else.

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

Measure, on 2D data with known ground truth, how often a product of two diffusion experts lands in the A-and-B region as the share of A-and-B samples in each expert's training set is turned from 0% to 100%, and separate the failures caused by the sampler from those caused by missing data.

**Where the walk is**

Claims 1, 2, 3, 7 and 9 are settled on their repairs. Claims 4, 5, 6 and 8 hand over to the layers of the rebuild walk file (`artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md`). Claim 8 (the shapes testbed) was appended at the settle. Claim 7 has had `words`, `break` (twice) and `how`, and its repair (two operators, two data conditions, one label term) waits on `settle 7`. Runs 1 and 2 are done. Two runs are offered and not launched: the label-term toy (7.4) and the fine-tune support check on claim 2. Route 1 (the literature taxonomy) is integrated; its 11 proposed changes to the terms file wait on a yes each. Route 2 (chest X-ray findings and diagnoses) is emitted.

**What compile would produce today**

Destination A, compiled 2026-10-17 into [the compiled idea](COMPILED.md): claim 2's checks have run (route 3): the product reaches A∩B only where some expert carries density there, and the medical cells carry it.

## The idea, as it stands

Navigation: ⬅️ [Quick context](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claims) ➡️

Two diffusion models are each trained on one concept's data and composed at sampling time by adding their scores, which targets the product of their densities (a product of experts). A diffusion model is a density model: it learns the density of its training set, and labels reach it only through a conditioning scheme or a classifier. A product can only put mass where every factor already has mass, so the A-and-B samples it produces come from three possible sources: A-and-B samples inside each training set (the overlap dial), support inherited from a pretrained model that each expert was fine-tuned from, and extrapolation by the network itself. In 2D, trained MLP experts match the ideal smoothed-data experts within `0.04` nats, so the third source is absent there and the first two carry the experiment.

"Composition" names more than one thing. Co-occurrence (two objects in separate regions) and intersection (one thing satisfying both on the same pixels) are two different products: over a partition of the image, and over the same variables. A chimera is the same-variable product applied where a partition was meant, and in 2D it appears only when both experts saw their object in the same place. Appearance interaction (a wet dog) needs no extra operator, only training data showing A as it looks under B. Report-mined labels add a real term, log p(A,B|x) − log p(A|x) − log p(B|x), which matters most for the negation cells the chest X-ray work wants (pneumothorax without drain, drain without pneumothorax). Reverse diffusion with summed scores samples something other than the product even with perfect experts (Du et al. 2023), so the sampler is a separate axis, checked on analytic experts first.

## The claims

Navigation: ⬅️ [The idea](#the-idea-as-it-stands) | 📋 [TOC](#table-of-contents) | [Next](#what-the-words-are) ➡️

### 1. ✅ A density expert learns its training set's density (repaired)

Mark: wrong as stated. A diffusion model fits the density of the samples it is shown. For construction (2) the label never enters the loss, so "hidden B samples inside A's set" and "samples labelled both A and B in A's set" are the same training set to expert A, and the overlap dial is a dial on p_A's mass inside the A∩B region.

Labels do matter in two places. In construction (1), it depends on whether the condition "A" means "contains A" (every sample with A in its label set trains p(x|A)) or "exactly A" (samples labelled {A, B} train a separate condition and are absent from p(x|A)). A classifier p(B|x) trained on these labels learns that hidden-B samples are not B, so classifier guidance toward B pushes away from A∩B. That is the toy's version of a shortcut.

- [x] 1.1 example: one three-point training set under two labellings gives expert A identical gradients (difference exactly `0.0`), shown in [the worked-examples notebook](what-an-expert-learns-and-kinds-of-composition.ipynb)
- [x] 1.2 break: labels matter for the conditioning scheme in (1) and for any classifier; in the notebook, "contains A" puts `1.00` of the composed mass in A∩B and "exactly A" puts `0.08`, and a set-membership classifier's guidance in A∩B points `(−0.38, −0.04)` against `(−0.14, +0.99)` with true labels

### 2. ✅ The product puts samples in A∩B only where some expert carries density there (repaired)

Mark: needs a check. Load-bearing.

Three sources can give the product mass in A∩B: overlap data in each training set, support inherited from a pretrained model, and the network's own extrapolation (absent in 2D, run 2). Route 1 asked whether the quadrant toy's plain product tests co-occurrence, since the field writes co-occurrence as a background-divided product p_A·p_B/p_bg (Bradley et al. 2025a). In 2D the background is uniform, so the two differ by a constant and the quadrant toy does test co-occurrence. On images the background is not uniform, and the testbed of claim 8 has to compare both operators.

### 3. 🔧 The two expert constructions compose the same thing

Mark: wrong as stated. Walked in its own round.

### 4. 🔧 Sampler failures and missing-data failures can be told apart only where the exact composition is known

Mark: wrong as stated, repair proposed after `break 4`. Near the edge of what the data supports, a weak sampler produces the same picture as a data failure: a chimera column at 5% rotation where the exact composition is `0.75` co-occurrence, a dry-dog cluster at 2% overlap where the exact composition is wet (run 6). On trained or image experts the exact composition is not available, so the two cannot be separated there directly. Repair: separate them in 2D with ideal experts, where the exact composition is computable, and carry each sampler's error signature per kind to the image experiments; use only samplers that reach the floor in 2D (tuned MALA, UHA) when reading image results. Dominance stays unresolved for every sampler (mode weights). Not settled.

### 5. 🔧 "Fraction inside A∩B" is the success number

Mark: wrong as stated. Walked in its own round.

### 6. 🔧 The toy answers the pneumothorax question

Mark: wrong as stated. Walked in its own round.

### 9. ✅ The parts world replicates the cat-dog hybrid exactly (repaired)

Mark: wrong as stated for the 2D toys, repair proposed. Appended at the `break` after claim 4. A 2D slot holds one number, so the only chimera it can show is a midpoint value that is neither cat nor dog. Image chimeras are mostly mosaics: some parts follow the cat, others the dog. Repair: each sample is 16 coordinates rendered as a 4×4 image, and each expert is a Gaussian mixture over those coordinates. A noised Gaussian mixture is a Gaussian mixture, and the product of two Gaussian mixtures is a Gaussian mixture with every pair of components, means, covariances and weights in closed form, so the target stays exact in 16 dimensions with no grid. For one pair of components the product's precision is the sum of precisions and its mean is the precision-weighted mean, so per coordinate the tighter expert wins: cat tight on the head coordinates and dog tight on the body coordinates gives a cat head on a dog body (the mosaic), and both tight everywhere gives the midpoint blend. Not settled.

### 8. 🔍 The paper's 2D toys, one dial per kind, host every kind but leakage

Mark: needs a check. Appended at claim 1's settle. Small RGB images of procedurally drawn shapes, scored by a deterministic parser rather than a learned classifier, so each kind of composition has an exact count. The 2D toys stay for the sampler question, because only they give exact product densities.

- [x] 8.1 fix, alternative (a): extend the paper's own 2D figures (ring × bar, left + right mixture) with one dial per kind, keeping exact densities. It hosts intersection, disjunction, co-occurrence, contention, dominance, interaction and negation. It cannot host leakage, which needs two attributes per object. Alternative (b) is the shapes-image testbed, kept for leakage and realism.
- [x] 8.2 how (a), run 3 (passes for six kinds; dominance blurs into intersection; negation breaks at the paper's weight): build each kind's pair from ideal experts, compute the exact product on the grid, and check that the base setting gives its kind's verdict and the dial's far end gives its failure. No training, no sampler, a few minutes.

### 7. ✅ Composition is two operators and two data conditions (repaired)

Mark: wrong as stated. Appended in round 2 of claim 1. Co-occurrence is a product over disjoint regions, intersection is a product over the same variables, and interaction is a product plus an interaction term. A chimera is the product computed exactly, over the wrong variables. Definitions are drafted in [the composition terms](composition-terms.md). The overlap dial's quadrant setup is a co-occurrence toy, and the notebook's ring-times-bar is an intersection toy.

- [x] 7.1 break: when each label is a fixed function of the image, the interaction term is exactly zero, so interaction is not a third operator; it is a same-variable product whose A∩B appearance must be present in the training data (notebook section 4c: shift found at 20% overlap, missed at 0%). Proposed repair: two operators (partition, same variables) and one data condition (does either expert's data contain A as it looks under B). Not settled.
- [x] 7.2 how: trained energy-parameterised MLP experts match the ideal ones on 4b and 4c (run 2), so in 2D a trained expert adds nothing past its smoothed data
- [ ] 7.4 how 7.3: report-style labels on the quadrant (B read in A∩B only when A was read), a 4-way classifier over label pairs against two separate classifiers; expected label term `+0.105` nats for (A,B) and very negative for (¬A,B) inside A∩B, zero elsewhere
- [x] 7.3 break again: with noisy, report-mined labels, p(A,B|x) is not p(A|x)p(B|x), so a label term survives. It is a different thing from the wet-dog case. "Interaction" splits into appearance interaction (a data condition, no term) and label dependence given the image (a real term, zero for clean labels, nonzero for NIH-style labels). Repair: two operators, two data conditions, one label term. Not settled.

## What the words are

Navigation: ⬅️ [The claims](#the-claims) | 📋 [TOC](#table-of-contents) | [Next](#held-claims) ➡️

| My phrase | The field's name | What it means | Confidence |
|---|---|---|---|
| composing A and B, "both A and B" | product of experts (PoE), conjunction | sampling from a density proportional to p_A(x)·p_B(x) | confident |
| score(x\|A)+score(x\|B)−score(x) | classifier-free guidance composition, "Composable Diffusion" AND (Liu et al. 2022) | sums of conditional scores minus the unconditional, which targets p(x)·p(A\|x)·p(B\|x) under conditional independence of the labels | confident on the formula |
| B samples hidden in A's set | label noise / co-occurrence; in the medical case, a spurious correlation or shortcut | a feature that travels with the label in the training data without defining it | confident |
| A-and-B absent from training | compositional generalisation, out-of-support generation | producing combinations of attributes never seen together | confident |
| the drain the classifier leans on | shortcut learning (Geirhos et al. 2020), the NIH chest-drain confounder (Oakden-Rayner et al.) | the classifier predicts the label from a correlated artefact | confident on the names, verify the drain percentage against the source |
| the sampler being wrong | the score of a product of diffused densities is not the score of the diffused product | summing per-expert scores at noise level t gives ∇log(q_A,t·q_B,t), which is not ∇log of the noised p_A·p_B | confident, Du et al. 2023 Section 3 |
| a background-divided product | projective composition (Bradley et al. 2025a) | p_b·∏ p_i/p_b, exact under reverse diffusion when concepts sit on disjoint coordinates | confident |
| one concept swallowing the other | mode collision (CO3), dominance | the product sits on one expert's mode | confident |
| colour or shape on the wrong object | semantic leakage, attribute leakage | an attribute crosses from one subject to another, worst for similar subjects | confident |
| "not B" | negation composition | p_A(x)^(1+w) / p_B(x)^w, as in the notebook's `NegationEBMDiffusionModel` (1.3 and −0.3) | confident |

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
| 2 | Quadrant setup, analytic Gaussian-mixture experts, overlap dial at 0, 5, 20, 50, 100%, annealed MALA; fraction of samples in the top-left quadrant | near 0 at 0% overlap: the product only intersects support, and extrapolation is the only rescue. Clearly above 0: something in the sampler or the smoothing is leaking mass, which needs explaining before it is trusted |
| 4 | Same analytic experts, reverse diffusion with summed scores against annealed MALA, at 100% overlap | a gap here is sampler error with zero model error |
| 2 (support from pretraining) | Pretrain one 2D expert on all four quadrants, fine-tune copies on A-only and B-only data (0% overlap) for 0 to N steps; product mass in A∩B against fine-tune steps | mass stays well above the from-scratch value (near 0) for short fine-tunes: the pretrained model supplies the missing overlap, which is the mechanism the chest X-ray plan can lean on. Mass falls to near 0 as fine-tuning lengthens: fine-tuning erases the support, and its length becomes a dial |
| sampler code | `anneal_samplers.py` MALA: proposal log-density divides by `2*ss**2` where the proposal variance is `2*ss`; MALA and CHA accept or reject the whole batch with one `torch.rand(1)` | fix both before any MALA or CHA number counts, or a "MALA fails" result is a code failure |

## Runs

Navigation: ⬅️ [Checks outstanding](#checks-outstanding) | 📋 [TOC](#table-of-contents) | [Next](#routes) ➡️

| # | Anchor | What it executed | State | Finding |
|---|---|---|---|---|
| 1 | claims 1 and 7 | [the worked-examples notebook](what-an-expert-learns-and-kinds-of-composition.ipynb), ideal experts (Gaussian-smoothed training points), exact products on a 241×241 grid, `co3` env | done | label-invariance holds for (2); conditioning scheme and classifier break it; chimera appears only when both experts saw their object in the same slot (chimera mass `0.66` centred, `0.21` position-varied); the interaction shift appears at 20% overlap (mean x1 `−0.79`) and not at 0% (`−0.48`) |
| 2 | claim 7 | how-7: trained energy MLPs against ideal experts on 4b (contention) and 4c (interaction), 3 seeds, outputs in `runs/how-7/`, ran on mscluster88 in 178 s | done | trained ≈ ideal everywhere: 4b centred chimera `0.652 ± 0.005` against `0.663`, co-occurrence `0.003` both; 4c mean x1 at f=0 `−0.499` against `−0.485`, at f=0.2 `−0.788` against `−0.789`; every expert within `0.04` nats of ideal. Small 2D MLPs add no extrapolation; composition reaches only what the data covers |
| 3 | claim 8 | how (a): seven kinds on the paper's toy datasets (toy_gmm, bars, toy_gauss, toy_box, left and right sets), ideal experts, exact products on the grid, one dial per kind; the background run died with its session, so it was rebuilt in-session as [the kinds notebook](kinds-of-composition-on-the-papers-toys.ipynb), 9 s on mscluster109 GPU 1 | done | six of seven kinds show success at base and failure at the far end: intersection `0.71 → 0.24`, disjunction B share `0.50 → 0.004`, co-occurrence `0.83 → 0.10`, contention `0.68 → 0.001` (chimera `0.988`), interaction `−0.41 → −0.21`; dominance moves continuously and blurs into empty intersection; negation fails at the paper's w = 0.3 (`0.891` of mass off every mode), working only near w ≈ 0.05 |
| 4 | claims 4 and 8 | [the kinds notebook](kinds-of-composition-on-the-papers-toys.ipynb) rebuilt as Figure 1 (p1, p2, exact, perfect sampler, reverse diffusion, annealed ULA, MALA, UHA per kind, exact noised scores of ideal experts) and Figure 2 (exact dial curves); ULA and UHA from `anneal_samplers.py` unchanged, MALA rewritten per-sample in the notebook; 48 s on mscluster109 GPU 0 | done | UHA within `0.02` TV of the perfect-sampler floor on five kinds; reverse diffusion fails on every kind except disjunction (intersection TV `0.31` against floor `0.11`, negation doubles in-bar mass `0.20` against `0.10`); MALA accepted `0.99` to `1.00`, so at `0.05·v_t` it equals ULA and is not yet tested; dominance defeats every sampler (TV `0.29` to `0.58`, floor `0.11`) because none moves mass between separated modes to match unequal weights |
| 5 | claim 4 | run 1 in [the kinds notebook](kinds-of-composition-on-the-papers-toys.ipynb): MALA with per-level step adaptation to 0.6 acceptance, ULA at the same steps | done | tuned MALA used steps `18` to `82`× larger and reached the floor on four kinds (intersection TV `0.11`, co-occurrence `0.11`, interaction `0.08`, disjunction `0.22`); dominance stays at `0.30` (floor `0.11`); ULA at the same steps diverges (keeps `1` of 2,000 on disjunction). The accept step is what makes large steps safe |
| 6 | claim 4 | run 2 in the same notebook: all samplers at Figure 2's steep points | done | contention at 5% rotated: exact co-occurrence `0.75`, reverse diffusion `0.21` with a chimera column the exact composition does not contain; interaction at 2%: reverse diffusion, ULA, MALA add a dry-dog cluster (TV `0.26` to `0.59`) while tuned MALA and UHA match (TV `0.08`). A weak sampler near the data's edge draws the kind's failure picture |
| 7 | claim 9 | example 9: the 16-pixel parts world, exact Gaussian and mixture products, written up in [the parts-world write-up](example-9-parts-world/example-9-parts-world.tex) (PDF beside it), numbers from `parts_world.py` | done | mosaic head `+0.980` / body `−0.980` (log Z `−46.5`); blend mean `0`, 10 sds from each expert (log Z `−1583.4`); co-occurrence weights `0.5`, `0.5` against same-half pairs at `e^−779` relative; summed scores at ᾱ = 0.5 put the head at `0.233` where the noised product has `0.693`; scene: [the parts-world scene](https://claude.ai/artifact/6XzsUMYdEb2eopRyGkpYkh), quiz: [diffusion models quiz](https://claude.ai/artifact/MyTdAaVfZW3wiqhbLKg4wu) and [diagonal Gaussian quiz](https://claude.ai/artifact/Vmd1FBg7oWwZq1onu9Jofa) |

## Routes

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#sources) ➡️

| # | Anchor | The ask | Return line | State |
|---|---|---|---|---|
| 1 | claim 7 | `/paper-scout` seeded by arXiv 2509.25940, 2307.14012, 2607.11758, 2605.07078, 2506.08894, then `/drip-read --combine --lean` over four ideas, writing citations into [the composition terms](composition-terms.md); prompt in [the taxonomy route prompt](routes/01-composition-taxonomy-literature.prompt.md) | `routes/01-composition-taxonomy-literature.return.md`: shortlist, per-kind verdict, whether interaction needs a term, proposed changes to [yours] entries | returned and integrated: [the taxonomy return](routes/01-composition-taxonomy-literature.return.md) |
| 2 | claim 6 | `/paper-scout` on chest X-ray findings against diagnoses, label co-occurrence, the drain confounder, then `/drip-read --combine --lean`; prompt in [the findings route prompt](routes/02-findings-and-diagnoses-literature.prompt.md), written in another session | `routes/02-findings-and-diagnoses-literature.return.md`: shortlist, per-term verdict, NIH drain rate with source, whether any chest X-ray generator composes separate models | returned and integrated 2026-10-17 into claims 6 and 7, proposals 12 to 17 applied: [the findings and diagnoses return](routes/02-findings-and-diagnoses-literature.return.md) |
| 3 | all claims, final handoff | `/unbuild --walk notebooks/simple_distributions.ipynb` in PyTorch, five phases (notebook rebuild, wrappers and samplers with the MALA fix, the 2D suite, the parts world, the overlap dial with fine-tuning); prompt in [the unbuild route prompt](routes/03-unbuild-walk.prompt.md) | `routes/03-unbuild-walk.return.md`: code location, each phase's check, the MALA fix and test, phase 5's curves | returned and integrated 2026-10-17: the return and code are on branch `unbuild-walk` (commits `1481288`, `17d4d35`, worktree `.claude/worktrees/unbuild-walk/`), not merged; the return is `artifacts/ideas/poe-overlap-dial/routes/03-unbuild-walk.return.md` on that branch |

## Sources

Navigation: ⬅️ [Routes](#routes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Source | What it gives the idea | Confidence |
|---|---|---|
| Du et al. 2023, *Reduce, Reuse, Recycle* (`~/goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/sampling-composed-models/compositional-generation-with-energy-based-diffusion-models-and-mcmc.pdf`) | why summed scores under reverse diffusion miss the product (p.4, eqs. 11 and 12); annealed ULA, MALA, HMC with energy parameterisation | confident, page checked by route 1 |
| Bradley et al. 2025a, *Mechanisms of Projective Composition of Diffusion Models* (arXiv 2502.04549) | projective composition, the background-divided product, when summed scores are exact, Bayes composition (p.4) | confident, read by route 1 |
| Soiffer et al. 2026, *Catastrophic Compositional Generation* (arXiv 2606.23920) | sampler error separated from score error with numbers (Table 1); interaction as non-separable causes | confident, read by route 1 |
| Dutta et al. 2025, CO3 (arXiv 2509.25940); Dahary et al. 2024, *Bounded Attention* (arXiv 2403.16990) | mode collision (one concept dominates); semantic leakage | confident, read by route 1 |
| the full shortlist of 17 | see [the taxonomy return](routes/01-composition-taxonomy-literature.return.md) | as flagged there |
| Oakden-Rayner et al. 2019, *Hidden Stratification* (arXiv 1909.12475); Cheplygina et al. 2025 (arXiv 2309.02244); Pan et al. 2021; Ehrlich et al. 2024; Bustos et al. 2020, *PadChest* | the drain rates, the finding against diagnosis line, the three compound effects | confident, read by route 2; the 46.2% is counted, method in [the findings and diagnoses return](routes/02-findings-and-diagnoses-literature.return.md) |
| Liu et al. 2022, *Compositional Visual Generation with Composable Diffusion Models* | the AND and NOT operators built from conditional and unconditional scores | confident |
| Hinton 2002, *Training Products of Experts by Minimizing Contrastive Divergence* | the product-of-experts definition | confident |
| `notebooks/simple_distributions.ipynb` | the JAX reference: product, mixture and negation wrappers, annealed ULA, MALA, UHA, the 2D datasets | read in this session |
| `anneal_samplers.py` | the PyTorch annealed samplers to reuse | read in this session, two bugs noted under Checks outstanding |

## Next step

Navigation: ⬅️ [Sources](#sources) | 📋 [TOC](#table-of-contents)

Either `settle` claim 1 and open claim 2 (which now carries the fine-tune support check), or `settle 7`. Optional: `run` the label-term toy, `run` the fine-tune check, `launch` route 1.
