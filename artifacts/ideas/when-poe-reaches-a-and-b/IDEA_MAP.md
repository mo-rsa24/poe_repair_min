# 💡 When a product of two diffusion experts reaches A-and-B

## Position in the idea

| Claim | Mark | Settled by |
|---|---|---|
| 1. An expert trained on A-labelled data learns the concept A | wrong as stated | repair proposed: it learns its training set's density, and in construction (2) the label never reaches the loss |
| 2. A product reaches A∩B only as far as each expert's density does, and generates a missing combination only when that density is borrowed or built (repaired from: a product can put samples in A∩B that neither training set contains) | wrong as stated, settled on the repair | borrowed: construction (1) with "contains" labels holds `0.989` in A∩B for any A∩B data share down to `0.001`, while separately fine-tuned experts cap at `0.5` (run 1, run 2); built: experts over separate parts reach `0.9999` with no A∩B data (run 1), and the treated lung without a drain they risk is the counterfactual a stress test wants and a risk to training only where the air rim vanishes (route 5); graded: the toy separates sampler error from data error (run 5); with trained experts at r = `0.001` the share still reaches `0.989` under annealed MALA but at the wrong place (total variation `0.391`), and reverse diffusion reaches only half (run 7) |
| 3. Construction (1) and construction (2) compose different things, and (2) divided by a background expert becomes (1) (repaired from: they compose the same thing) | wrong as stated, settled on the repair | (1) targets p(x\|A)·p(x\|B)/p(x), (2) targets p_A·p_B; measured differences: rarity (`0.989` against a `0.5` cap, run 1), space (unchanged against `0.035` to `0.126` total variation, run 6), labels (`1.00` under "contains" against `0.08` under "exactly", earlier walk); p_A·p_B/p_b is Bradley et al.'s background-divided product |
| 4. Sampler failures and missing-data failures can be told apart where the exact density is known, and with trained experts the measure includes their interaction (repaired from: they can be told apart) | wrong as stated, settled on the repair | run 5: with exact experts, annealed MCMC sits at the exact-sample floor and reverse diffusion `0.049` above it; break 4.1: with learned scores a correction can amplify score error off the data (Soiffer et al. 2026); run 7 confirms the interaction: `+0.054` total variation at r = `0.05`, `+0.232` at r = `0.001` |
| 5. Success is three numbers reported together: the share in A∩B, the total variation to the exact composed density, and the normaliser Z; on films, a classifier rate plus coverage against real held-out films (repaired from: the fraction inside A∩B is the number) | wrong as stated, settled on the repair | run 5: reverse diffusion's share `0.983` against `0.989` hid a total variation `0.049` above the floor; a quadrant count misreads the from-scratch origin blend as `0.25`; a collapsed sampler would score `1.0` |
| **6. The toy answers the pneumothorax question (current) | **wrong as stated** | **the films wanted are pneumothorax without drain and drain without pneumothorax, which is a negation (A and not B); on NIH the first is common (1,906 of 3,543 NEATX films) and the second exists unlabelled, so what is missing is a drain label on negatives and the paired drain-removed film, not a film (route 5)** |
| 7. A more sophisticated model (region-aware, bounding boxes, GLIDE) changes whether composition reaches A∩B | wrong as stated | a better network only gets closer to its target density, and the toy's ideal experts already are that target; region and box conditioning change the target itself (rescue 2b), and GLIDE composed as in Liu et al. 2022 is construction (1) on a large pretrained base (rescue 2a's strong form). What sophistication adds beyond that is inductive bias across image parts, which 2D cannot show |
| 8. Composition is needed to fix the chest-drain shortcut, because nothing else generates pneumothorax films without a drain | wrong as stated | 1,906 of 3,543 NEATX pneumothorax films have no drain, and masked edits (RadEdit; FastDiME, Weng et al. 2024) make the paired drain-removed film; repaired: on the drain, composition is one arm against upweighting real films and editing, run after B1's drain gap shows the shortcut exists, and composition's own job is a shortcut that cannot be masked (AP against PA view, scanner) (route 5) |
| 9. The 2D toy reconstructs the failures seen at scale (hybrids, dry dogs) and judges joint models against composition fairly | needs a check | hybrids and dry dogs are reproduced in 2D by the earlier walk (chimera mass `0.66` when both experts saw their object in the same place; the wet-dog shift found at 20% overlap and missed at 0%); failures that come from text encoders and cross-attention (leakage, neglect) have no 2D counterpart. The fair test: a trained joint condition {A, B} against trained composition and against composition plus a small correction fine-tuned on the same few A∩B samples, at r = `0.05` and `0.001` |
| 10. Composing in pixel space or in a latent space gives the same answer | wrong as stated | by change of variables, a composition whose exponents sum to 1 (construction (1): 1 + 1 − 1; negation: (1 + w) − w) is unchanged by an invertible map between the spaces, while the plain product (exponents sum to 2) picks up a factor of the map's volume change, \|det J\|; run 6 confirms it on the quadrant toy: construction (1) and negation agree across an invertible warp to `4e-17` total variation, the plain product differs by `0.035` to `0.126` as the warp's largest area stretch grows from `2.6` to `49` |
| 11. β-VAE, β-TCVAE or FactorVAE on the 2D toy show what happens inside Stable Diffusion's latent | held | see Held claims |
| 12. A small text-conditioned image diffusion model (a tiny text encoder, cross-attention, procedurally drawn shapes) reproduces Stable Diffusion's hybrids and shows where they come from | needs a check | train on single-concept captions, prompt the joint caption and compose with the product; score hybrids exactly with a parser of the drawing; dial token-embedding similarity, data overlap and fine-tuning set size, and read each token's cross-attention map |

Load-bearing: claim 2. If a product cannot place samples where neither expert's data put any, composition cannot generate a missing combination on its own, and the medical hope has to rest on something the experts bring beyond their data. The chest-drain case does not supply a missing combination: pneumothorax without a drain is 54% of NEATX films (claim 8), so the medical case for composition needs a combination that is actually absent from the data.

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

On 2D data with known ground truth, measure how often a product of two diffusion experts lands in the A-and-B region as the share of A-and-B samples in each expert's training set is turned from 0% to 100%, and separate the failures the sampler causes from the failures missing data causes, before trusting composition to generate the missing chest X-ray films.

**Where the walk is**

Claim 2, the load-bearing one, is current and has had one round. Its repair names two rescues to test (support kept from pretraining, an expert that factorises over parts) and one fallback (use the 2D dial only to calibrate samplers). Claim 1 has its opening mark and no round of its own. Route 1 (pictures of hidden labels and the two conditioning rules) is out.

**What compile would produce today**

Neither destination is earned. Claim 2 is load-bearing and marked wrong as stated, with a repair that has not been walked.

## The idea, as it stands

Navigation: ⬅️ [Quick context](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claims) ➡️

Two diffusion experts are composed at sampling time by adding their scores, which targets a weighted product of their densities. A product reaches the region that is both A and B only as far as each expert's density does, so composition generates a combination missing from training only when that density is borrowed or built. It is borrowed from support kept in pretraining, and most strongly by one conditional model with "contains" labels composed as p(x|A)·p(x|B)/p(x), which holds `0.989` of the composed density in A∩B while A∩B is rare (down to a share of `0.001`) and is unchanged by the space it is computed in. It is built by experts that each model only their own part of the image, which reach A∩B with no A∩B data when the parts are independent. The 2D toy is the answer key for both: with exact experts, annealed MCMC samplers reach the composed density and reverse diffusion with summed scores misses it by a measurable amount. For chest films, the drain case asks for no missing film. NIH holds 1,906 pneumothorax films without a drain, and the paired film with only the drain removed is made by masked editing, which is a composition anchored to the real film. Whether a classifier trained on NIH uses the drain at all is unshown in this project until B1's drain gap is measured, and composition's own medical job is a shortcut that cannot be masked.

## The claims

Navigation: ⬅️ [The idea](#the-idea-as-it-stands) | 📋 [TOC](#table-of-contents) | [Next](#what-the-words-are) ➡️

### 1. 🔧 An expert trained on A-labelled data learns the concept A

Mark: wrong as stated, repair proposed. A diffusion model fits the density of the samples it is shown. In construction (2) the label never enters the loss, so "B samples hidden in A's set, unlabelled" and "samples labelled A and B in A's set" are the same training set to expert A. The overlap dial is a dial on how much of p_A's mass sits in A∩B.

### 2. ✅ A product reaches A∩B only as far as each expert's density does (settled on the repair)

Mark: wrong as stated, repair proposed. Load-bearing. A product p_A·p_B is zero wherever either factor is zero, so it reaches A∩B only as far as each expert's density does. The notebook's own conjunction (toy_gmm ring × vertical bar) lands on the two ring modes at `(0, ±0.5)`, the 2 of 8 modes (25% of the ring's data) that already sit inside the bar. With those two modes removed from the ring and the bar cut at `0.4 < |x2| < 0.6`, the exact product's normaliser falls from `Z = 0.313` (`0.996` of it near `(0, ±0.5)`) to `1.6e-07`, none of it near `(0, ±0.5)`. Smoothing the experts by a width `h = 0.1` brings back 31% of the full pair's Z, and only `0.005` of that lands near `(0, ±0.5)`. The rest sits at the bar's edge beside the 45-degree modes, near `(0.24, −0.315)`. Smoothing makes the product non-empty without putting it in A∩B (sandbox in the vault note's section "When neither training set holds A and B").

Repaired wording: a product reaches A∩B only as far as each expert's density does, and it generates combinations missing from training only when something gives each expert density there.

- [x] 2a support kept from pretraining (needs a check: route 3's notebook): pretrain one 2D expert on all four cells, fine-tune copies on A-only and B-only data (0% overlap), product mass in A∩B against fine-tune steps. Separately trained experts fine-tuned on "A, not B" and "B, not A" also learn "not B" and "not A", so the top-right share starts at `0.25` and is at most about `0.5`. Construction (1) keeps the pretraining support in its base density p(x) and does not learn "not B" under "contains" labels, so it is the stronger form of this rescue whenever A∩B is rare in the data rather than absent
- [x] 2b settled: built-in factorisation works when the parts are independent, and the open risk is the treated-lung chimera. An expert that models only its own part (projective composition, Bradley et al. 2025a) reaches the top-right blob with no A∩B data (`0.9999` plain product, `0.989` background-divided, run 1). Trained networks do not learn the factorisation unprompted, so it is built in with masks, region-restricted labels, or box conditioning (claim 7). On chest films a drain changes the lung it treats, so a factorised pneumothorax expert pairs a treated-looking lung with a clean chest wall. That film is close to real ones (the check film after a drain is pulled, a small pneumothorax watched without one, high confidence on the BTS guidance). For a stress test it is the counterfactual wanted, made by an edit anchored to the real film; for training it is a risk only where the air rim vanishes and a positive label sits on a film that shows nothing (route 5)
- [x] 2c settled: the toy is the sampler answer key that grades rescues 2a and 2b. With exact experts, annealed ULA, fixed MALA and UHA sit at the exact-sample floor (total variation `0.124` to `0.132` against `0.133`), and reverse diffusion with summed scores sits `0.049` above it (run 5)

### 3. ✅ The two constructions compose different things (settled on the repair)

Mark: wrong as stated. Construction (1) targets p(x|A)·p(x|B)/p(x), which is p(x)·p(A|x)·p(B|x) up to a constant. Construction (2) targets p_A·p_B. The two agree only where p(x) is flat across the support.

### 4. ✅ Sampler and data failures can be told apart where the exact density is known (settled on the repair)

Mark: needs a check. They separate only where the exact composed density is known. The check is every sampler run on ideal experts, whose noised scores are exact, compared with the exact product on a grid.


- [x] 4.1 break: run 5 used exact experts, where score error is zero. With learned scores the two errors interact rather than add: Soiffer et al. 2026 (Table 1, p.10) report that, with the target outside the training data, Feynman-Kac correction improves the fit with exact scores (`0.64` to `0.08` sliced Wasserstein-2, non-factorised case) and worsens it with scores learned from 10,000 samples (`0.39` with 1 particle to `2.82` with 256, factorised case), as quoted in the composition terms. Repair: a 2 × 2 design (exact or trained experts, reverse diffusion or annealed MCMC) that reports the interaction beside the two main effects
### 5. ✅ Success is three numbers reported together (settled on the repair)

Mark: wrong as stated. A sampler that collapses onto one point inside A∩B scores `1.0`. A coverage number has to sit beside it.

### 6. 🔧 The toy answers the pneumothorax question  ← current

Mark: wrong as stated. The films wanted are A and not B, which is negation. On NIH neither is absent. 1,906 of the 3,543 NEATX pneumothorax films have no drain. B1's follow-up count puts about 1,100 drain films without a pneumothorax label (range 900 to 1,400, estimated from how close each film sits to a labelled drain film; none has been looked at). What is missing is a drain label on negatives and the paired film with only the drain removed (route 5).

### 7. 🔧 A more sophisticated model changes whether composition reaches A∩B

Mark: wrong as stated. Appended in claim 2's round on 2b, from the learner's question. With the same target density, a better network only approximates that density more closely, and where the target is zero a perfect model is zero too. Region-aware and bounding-box conditioning change which density each expert targets, so they are rescue 2b built in. GLIDE composed as score(x|c1)+score(x|c2)−score(x) (Liu et al. 2022) is construction (1) on a base pretrained on far more combinations, so it is rescue 2a's strong form. The one thing a sophisticated image model can add that the toy cannot show is inductive bias across image parts (convolutional locality, attention), which needs a small image testbed.

### 8. 🔍 Composition is needed to fix the chest-drain shortcut

Mark: wrong as stated. Appended from the learner's question after the search on 2b. Nothing else needs to generate pneumothorax films without a drain on NIH, because 1,906 of the 3,543 NEATX films are exactly that. The paired film with only the drain removed comes from masked editing, and RadEdit is itself a composition: the edit region is guided by a prompt and the keep region is pinned to the inverted real film, which is rescue 2b with one expert replaced by the real film. RadEdit as published removes the pneumothorax and keeps the drain, and a classifier trained on CANDID-PTX falls from 93.3 to 17.9 accuracy on 629 such films (Table 2). FastDiME (Weng et al. 2024) removes and adds drains on the NEATX films.

Whether the shortcut exists on NIH is unshown in this project. Oakden-Rayner et al. report AUC `0.940` with a drain and `0.770` without on the NIH test set, Thian et al. 2021 find 0.95 against 0.94 at six outside hospitals, and TorchXRayVision's DenseNet moves a median of `+0.001` between drain and drain-free films of the same patient over 237 pairs (on a score scale where most pneumothorax films sit between 0.50 and 0.53). That model shows a split between groups (514 of 1,906 drain-free films below its threshold against 180 of 1,637 drain films), which may come from which patients get a drain rather than from the tube.

Repaired wording: on the drain shortcut, composition is one arm against upweighting the real drain-free films and against masked editing, run only after B1's drain gap shows the shortcut exists. Composition's own job is a shortcut that cannot be masked, such as AP against PA view or scanner (route 5).

### 9. 🔍 The 2D toy reconstructs the failures seen at scale and judges joint models fairly

Mark: needs a check. Appended from the learner's question after 2c. The toy holds the support rule, hybrids, dry dogs and sampler bias with exact answers. It cannot show failures that come from how a text encoder and cross-attention bind words to regions. A joint model has its own shortcoming that the toy can show: the condition {A, B} trains on only the share r of data that is A and B, while composition reuses all of it. Curating and fine-tuning enter here: the interaction term, log p(A, B | x) − log p(A | x) − log p(B | x), is exact in 2D, so a small correction fine-tuned on the few A∩B samples can be scored against it and against the joint model trained on the same samples.

### 10. 🔧 Composing in pixel space or in a latent space gives the same answer

Mark: wrong as stated. Appended from the learner's question about VAEs and latent diffusion. A density moved through an invertible map x = g(z) picks up a factor \|det J\|⁻¹. Each conditional and the unconditional density pick up the same factor, so p(x\|A)·p(x\|B)/p(x) picks it up once, exactly as the latent composition pushed through g does: construction (1) and negation (exponents summing to 1) give the same answer in either space. The plain product picks it up twice in pixel space and once from the latent, so the two differ by \|det J\|⁻¹, a reweighting by how much the decoder stretches each region. A VAE decoder is not invertible, and its analogue of \|det J\| is the volume factor √det(JᵀJ) of the pullback metric (likely, verify). Check: a fixed invertible warp of the quadrant toy, composing before and after it.

### 12. 🔍 A small text-conditioned image model reproduces Stable Diffusion's hybrids and shows where they come from

Mark: needs a check. Appended from the learner's question after claim 10. 32 × 32 images of procedurally drawn shapes (the red clean circle, the blue scattered dots, and a few more), a tiny text encoder over a few words, and a small UNet whose cross-attention reads it. Train on single-concept captions only, then ask for two concepts three ways: the joint caption, construction (1), and the plain product. A parser of the drawing scores each output exactly (both shapes present, one missing, or fused into a hybrid). The dials that 2D cannot hold: how similar the two words' embeddings are (cat and dog sit close in CLIP; Bradley et al. 2025a, Figure 8), whether each word's cross-attention map lands on its own region, and how few joint examples a fine-tune needs. Prior synthetic-shape studies of compositional generalisation in diffusion are parts to borrow: Okawa et al. 2023, *Compositional Abilities Emerge Multiplicatively* (confident); Park et al. 2024, *Emergence of Hidden Capabilities* (likely, verify).

## What the words are

Navigation: ⬅️ [The claims](#the-claims) | 📋 [TOC](#table-of-contents) | [Next](#held-claims) ➡️

| My phrase | The field's name | What it means | Confidence |
|---|---|---|---|
| composing A and B, "both A and B" | product of experts (PoE), conjunction | sampling from a density proportional to p_A(x)·p_B(x) | confident |
| score(x\|A)+score(x\|B)−score(x) | Composable Diffusion AND (Liu et al. 2022), classifier-free guidance composition | targets p(x)·p(A\|x)·p(B\|x), which assumes A and B are independent given x | confident |
| B samples hidden in A's set | label noise; in the medical case a spurious correlation or shortcut | a feature that travels with the label in the data without defining it | confident |
| A-and-B absent from training | compositional generalisation, out-of-support generation | producing combinations never seen together | confident |
| the sampler being wrong | the score of a product of noised densities is not the score of the noised product | summing per-expert scores at noise level t gives ∇log(q_A,t·q_B,t), not ∇log of the noised p_A·p_B | confident, Du et al. 2023 Section 3 |
| "not B" | negation composition | p_A(x)^(1+w)/p_B(x)^w; the notebook's `NegationEBMDiffusionModel` uses `1.3` and `−0.3` | confident |

## Held claims

Navigation: ⬅️ [What the words are](#what-the-words-are) | 📋 [TOC](#table-of-contents) | [Next](#dead-ends) ➡️

| Claim | What is unresolved | What would settle it |
|---|---|---|
| 11. disentangling VAEs on the 2D toy as a window on Stable Diffusion's latent | unsupervised disentanglement is not identifiable without supervision or inductive bias (Locatello et al. 2019, confident), so a β-VAE on 2D data may not put A and B on separate axes; and a 2D latent from 2D data says something about coordinates (whether A and B fall on separate axes, the orthogonal-mean-difference condition of Bradley et al. 2025a, Lemma 8.1) rather than about a high-dimensional latent | claim 10's warp check first; then a 2D VAE whose latent is deliberately rotated 45° against A and B, composed in the latent, against one aligned with them |

## Dead ends

Navigation: ⬅️ [Held claims](#held-claims) | 📋 [TOC](#table-of-contents) | [Next](#checks-outstanding) ➡️

| Claim | The workaround | Why it failed |
|---|---|---|

## Checks outstanding

Navigation: ⬅️ [Dead ends](#dead-ends) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

| Claim | The check | What each outcome means |
|---|---|---|
| 4 | Ideal experts on the ring × bar pair, exact product on a grid, reverse diffusion with summed scores against annealed ULA, MALA and UHA; total variation of each sampler's histogram to the exact product | a gap from the exact product here is sampler error with zero model error |
| 2a | Quadrant toy: A is `x1 > 0`, B is `x2 > 0`, one Gaussian blob per quadrant at `(±0.5, ±0.5)`, std `0.15`. Pretrain one energy-parameterised MLP diffusion model on all four quadrants equally, copy it twice, fine-tune one copy on bottom-right only (A, not B) and one on top-left only (B, not A) for N in `0, 100, 300, 1k, 3k, 10k` steps; baseline is two experts trained from scratch on the same fine-tune data. Score each pair on a `241 × 241` grid at t = 0: share of exp(−(E_A + E_B)) within 3 std of the top-right blob's centre (a quadrant share would mislead, because two far-apart experts multiply into a blend at the origin that splits evenly across quadrants). No sampler, so claim 4 stays out of it | at N = 0 both experts are the pretrained four-quadrant density and the share is exactly `0.25` by symmetry; experts trained from scratch put `0.989` at the origin and `0.006` in the top-right blob (run 1). If leftover density spreads evenly over the three other quadrants (ε each), the share is about ε/2 and falls with N: pretraining support alone does not choose A∩B. If leftover sits only on the two quadrants next to each expert's own, top-right ties with bottom-left (neither A nor B) and the share approaches `0.5`. Falling to the from-scratch value within a few hundred steps means fine-tuning erases the support and rescue (a) needs a brake (early stop, a low-rank adapter, replayed pretraining data) |
| 8 | First B1's drain gap: task 10.1 of `01-measure-the-drain-gap.md` in `/home-mscluster/mmolefe/Playground/PhD/medical-imaging/compositional_diffusion/b-shortcut-through-a-device/b1-drain-shortcut-in-pneumothorax/plans/`, sensitivity at 90% specificity and AUROC for drain and drain-free films on held-out folds. Then train one pneumothorax classifier four ways: original NIH films; the real drain-free films upweighted; plus drain-removed edits of positive films (RadEdit-style masked edits); plus composed pneumothorax-without-drain films. Test all four on real drain-free pneumothorax films (NEATX labels) beside the drain films, and on the same films with only the drain removed, scored against a VAE round-trip of the original | a gap under B1's stop rule (under 5 points, or reversed) ends the drain case before any training. Upweighting closes it: no generated data is needed. Edits close it and composition adds nothing: editing is the tool. Composition closes it further: the untreated look matters and editing cannot supply it. None closes it: the shortcut is not fixed by data of any kind |
| sampler code | `anneal_samplers.py:34-35` divides the MALA proposal log-density by `2*ss**2` where the proposal variance is `2*ss` (should be `4*ss`); `anneal_samplers.py:39` accepts or rejects the whole batch with one `torch.rand(1)` against energies summed over the batch | fix both before any MALA number counts, or a "MALA fails" result is a code failure |

## Runs

Navigation: ⬅️ [Checks outstanding](#checks-outstanding) | 📋 [TOC](#table-of-contents) | [Next](#routes) ➡️

| # | Anchor | What it executed | State | Finding |
|---|---|---|---|---|
| 1 | claim 2, child 2a | [the four-quadrant answer key](runs/answer-key-quadrants/README.md): ideal blob-mixture experts multiplied on a `441 × 441` grid, construction (2) at leftover ε and construction (1) at A∩B data share r, 2 s in `co3` | done | construction (2): top-right share is ε/(2 − 4ε), `0.25` at ε = ¼, `0.006` at ε = 0 with `0.989` at the origin; leftover next door only gives `0.50` top-right and `0.50` bottom-left. Construction (1) with "contains" labels: `0.989` in the top-right blob for every r from `0.25` to `0.001`, and the origin blend at r = 0. Rescue 2b with built-in factorisation and no A∩B data: `0.9999` (plain) and `0.989` (background-divided) in the top-right blob |
| 2 | claim 2, children 2a to 2c | scene-cadence: [the three-rescues scene](https://claude.ai/artifact/9eFnG2rWEciPsceMwdem4u), 15 presses in five frames on one quadrant plot, numbers from `scene_key.py` (which reproduces run 1 exactly) in `artifacts/scenes/three-ways-a-product-reaches-a-and-b/` | done | the ε/(2 − 4ε) share counts same-corner terms only and "neither" gets the same share as top-right; `0.5` is an exact ceiling by symmetry (x → −x swaps the two experts); construction (1)'s normaliser is approximately r (`3.8e-7` at r = 0); construction (1) holds `0.9855` at r = 10⁻⁴, `0.9555` at 10⁻⁵, `0.7364` at 10⁻⁶, `0.0935` at 10⁻⁸; the origin blend splits evenly across the four quadrants and is none of the four kinds of film |
| 3 | claim 2, children 2a to 2c | the [Diffusion Models quiz](https://claude.ai/artifact/MyTdAaVfZW3wiqhbLKg4wu), version 7: a new section of 24 questions on the four-quadrant answer key and the three rescues, built on the scene's figures, every answer from the answer key or the scene key | done | 76 questions on the page; question 23 asks for the three rescues from memory, question 24 is a chest-film scenario that never says "support" |
| 4 | claim 2, children 2a to 2c | [Where the Product Lands](https://claude.ai/artifact/2A7YDnfKe8EPsS7yFDVcNT), the scene of the three rescues: 14 presses (support, 2a with two fine-tuned experts and with one labelled model, 2b, 2c, explore), every number from the answer key or `artifacts/scenes/when-a-product-reaches-a-and-b/checks/scene_key.py`, which reproduces all 18 key rows | done | Gaussian experts are never exactly zero, so "no support" means a density too small to matter, and two far-apart experts multiply into a blend at the origin rather than into nothing. One blob holds `1 − e^(−9/2) = 0.989` of its mass within 3 std, which is why `0.989` recurs |
| 5 | claim 2, child 2c; claim 4 | `/demonstrate`: reverse diffusion with summed scores, annealed ULA and UHA from `anneal_samplers.py`, and a fixed copy of annealed MALA, on construction (1) at r = `0.05` with exact noised mixture scores; 2,000 samples, 3 seeds, scored by top-right blob share against `0.989` and by total variation against exact samples; outputs in `runs/samplers-on-exact-experts/` | done | [the sampler card](runs/samplers-on-exact-experts/README.md). Total variation to the exact grid, mean ± sd over 3 seeds, with 2,000 exact samples as the floor at `0.133 ± 0.006`: annealed ULA `0.128 ± 0.012`, fixed MALA `0.124 ± 0.007` (acceptance `0.600`), UHA `0.132 ± 0.002`, reverse diffusion with summed scores `0.182 ± 0.003`, and `0.167 ± 0.012` at 1,000 levels. Top-right share barely separates them (`0.983` reverse diffusion against `0.989` exact); the miss is a shift of the blob toward the origin, centre `(0.465, 0.464)` against `(0.5, 0.5)`. The original MALA rejected every whole-batch proposal (variance stuck at `4.02` from 4 after 200 steps); the denominator bug alone drifts a chain at the target to variance `1.11` in 50 steps. MCMC used about 21 times reverse diffusion's score evaluations |
| 6 | claim 10 | [the warp check](runs/compose-before-or-after-a-warp/README.md): the quadrant toy in a latent space decoded by a fixed invertible warp at four strengths, composing before and after it, compared by total variation on an `801 × 801` grid | done | construction (1) and negation are unchanged by the warp (`4e-17`, `7e-17`); the plain product changes by `0.035`, `0.080`, `0.126` at largest area stretches of `2.6`, `9`, `49` |
| 7 | claim 4 | `/demonstrate`: a 2 × 2 of exact or trained energy-MLP experts by reverse diffusion or fixed annealed MALA, construction (1) at r = `0.05` and `0.001`, 3 seeds; sampler effect, expert effect and their interaction on total variation above the floor; outputs in `runs/sampler-by-expert-interaction/` | done | [the interaction card](runs/sampler-by-expert-interaction/README.md). The errors interact: MALA's gain over reverse diffusion shrinks with trained experts, an interaction of `+0.054 ± 0.016` total variation at r = `0.05` and `+0.232 ± 0.046` at r = `0.001` (all six seeds positive). At r = `0.001` reverse diffusion fails even with exact experts (top-right share `0.501`, `0.544` of samples at the origin, total variation `0.680`); trained MALA reaches the right share (`0.989`) at the wrong place (total variation `0.391`, centre `(0.441, 0.523)`). Training sets at r = `0.001` held 23 to 34 top-right samples for each conditional expert. The trained composed score's relative error is `0.34` (r = `0.05`) and `1.30` (r = `0.001`) at the lowest noise level, under `0.04` from level 30 up |

## Routes

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#sources) ➡️

| # | Anchor | The ask | Return line | State |
|---|---|---|---|---|
| 1 | claim 1 (and claim 3's two conditioning rules) | `/demonstrate`: procedural red-circle and blue-dots images, each titled with its stored label, hidden concepts boxed from the drawing parameters, and the piles each conditioning rule trains on; prompt in [the hidden-labels route prompt](routes/01-hidden-labels-pictured.prompt.md) | `routes/01-hidden-labels-pictured.return.md`: the figure paths, the count table, one line on what it changes for claims 1 and 3 | emitted; `launch` refused with "Workspace not trusted" until `claude` is run once in the repo root and the trust prompt accepted |
| 2 | claim 2, child 2a | thread: the pretraining-support check as a labelled teaching notebook (four quadrants, fine-tuning snapshots, the product at each step, why `0.25`, the ε/(2 − 4ε) curve, and construction (1) at rare and absent A∩B); prompt in [the pretraining-support notebook prompt](routes/02-pretraining-support-notebook.prompt.md) | `routes/02-pretraining-support-notebook.return.md`: notebook path, results folder, top-right share at N = 0, 300, 10000, construction (1) at r = 0.05 and 0 | kept as the specification route 3 reads; not launched |
| 3 | claim 2, child 2a | `/drip --code` walk building route 2's notebook one cell per message, with predictions before each number; prompt in [the pretraining-support drip prompt](routes/03-pretraining-support-drip-code.prompt.md) | `routes/03-pretraining-support-drip-code.return.md`: notebook path, measured top-right blob share at N = 0, 300, 10000, fitted ε, construction (1) at r = 0.05 and 0 | emitted |
| 4 | the two claim 2 scenes | thread: rework both scenes so each press reveals one taught concept and dims everything already taught; prompt in [the scene reveal prompt](routes/04-scene-progressive-reveal.prompt.md) | `routes/04-scene-progressive-reveal.return.md`: the republished URLs, the press list with the one concept each press reveals, and which scene is kept if they are merged | emitted |
| 5 | claim 2, child 2b | thread: is a treated-looking lung with no drain a chimera to avoid or the counterfactual a shortcut study needs, and which use (training augmentation or stress-testing) comes first; prompt in [the chimera-or-counterfactual prompt](routes/05-treated-lung-chimera-or-counterfactual.prompt.md) | `routes/05-treated-lung-chimera-or-counterfactual.return.md`: an answer per question with sources, and whether 2b's open risk is a risk or a feature by use | returned and integrated ([the route 5 return](routes/05-treated-lung-chimera-or-counterfactual.return.md)): 2b's risk is a feature for stress-testing and a risk for training only where the air rim vanishes; claims 6 and 8 revised |
| 6 | claim 8 | `/pressure-test`: is composition still worth pursuing for the drain shortcut against masked editing; prompt in [the composition-against-editing prompt](routes/06-composition-against-masked-editing.prompt.md) | `routes/06-composition-against-masked-editing.return.md`: verdict, papers that retrained on edits or composed medical films, whether claim 8 stands | emitted |
| 7 | claim 10 | `/pressure-test`: is it known that weighted products with exponents summing to 1 are unchanged by reparameterisation while the plain product is not, and does it matter for latent diffusion; prompt in [the exponent-sum prompt](routes/07-exponent-sum-invariance.prompt.md) | `routes/07-exponent-sum-invariance.return.md`: verdict, where it is stated, how claim 10 should be worded | emitted |
| 8 | claim 12 | its own `/drip-idea` walk: an SDXL-shaped model on 32 × 32 drawn shapes (tiny text encoder, cross-attention at two resolutions, pooled embedding) that reproduces this walk's 2D findings at image scale and then probes where hybrids come from; prompt in [the hybrid testbed prompt](routes/08-sdxl-shaped-hybrid-testbed.prompt.md); its map lives at `artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md` | `routes/08-sdxl-shaped-hybrid-testbed.return.md`: the testbed walk's verdict and route out, and how claim 12 should be marked | emitted |
| 9 | runs 1, 5, 6, 7 | `/drip --code --quiz`: rebuild the four experiments one notebook cell at a time, predicting each number before it appears, every word explained on first use; prompt in [the rebuild-by-hand prompt](routes/09-rebuild-the-walks-numbers-by-hand.prompt.md) | `routes/09-rebuild-the-walks-numbers-by-hand.return.md`: notebook path, any number that did not match, and the learner's own sentence for each number | emitted |

## Sources

Navigation: ⬅️ [Routes](#routes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Source | What it gives the idea | Confidence |
|---|---|---|
| Du et al. 2023, *Reduce, Reuse, Recycle* (`~/goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/sampling-composed-models/compositional-generation-with-energy-based-diffusion-models-and-mcmc.pdf`) | why summed scores under reverse diffusion miss the product; annealed ULA, MALA and HMC with an energy parameterisation | confident |
| NEATX (Non-Expert Annotations of Tubes in X-rays) | chest-drain labels from two non-expert annotators on pneumothorax-positive NIH ChestX-ray14 films: 3,543 films in the label file B1 scored and in FastDiME's label file, 1,637 with a drain and 1,906 without; NIH and SIIM-ACR ship no drain labels, so negatives need predicted labels | confident on the counts (B1's `pneumothorax-score-per-film.csv`); the 3,709 quoted by a secondary search is unchecked |
| Jiménez-Sánchez et al. 2023, *Detecting Shortcuts in Medical Images: A Case Study in Chest X-rays* | drain annotations for a subset of pneumothorax films, and the drain shortcut documented | likely, verify |
| Oakden-Rayner et al. 2019, *Hidden Stratification Causes Clinically Meaningful Failures* (arXiv 1909.12475) | Zech's DenseNet-121 reproduction of CheXNet scores AUC `0.940` on pneumothorax films with a drain and `0.770` without, on the NIH test set with a radiologist's drain labels, where 80% of test pneumothoraces had a drain; the radiologist's labels were never released, and every NEATX film sits in Zech's training fold | confident, full text read in B1 (B1's who-says-what file) |
| Thian et al. 2021 | a ChestX-ray14 and CheXpert model scores AUC 0.95 with a chest tube and 0.94 without at six outside hospitals, radiologist reference labels | likely, abstract read in B1 |
| RadEdit, Pérez-García et al. (arXiv 2312.12865, ECCV 2024) | masked diffusion editing of chest X-rays; on CANDID-PTX it edits the pneumothorax out and keeps the drain, and a classifier trained on CANDID-PTX scores 93.3 on real films and 17.9 on 629 edited films (one trained on SIIM-ACR: 93.7 and 81.7); it tests classifiers and never trains on its edits | confident, section 5.3 and Table 2 read in B1 |
| Weng et al., ECCV 2024, *Fast Diffusion-Based Counterfactuals for Shortcut Removal and Generation* (FastDiME, arXiv 2312.14223) | a diffusion model with a self-made mask adds or removes chest drains on NIH pneumothorax films and pacemakers on CheXpert; its label file covers the 3,543 NEATX films; whether it trains classifiers on its edits is unchecked | likely, about three quarters of the paper read in B1 |
| B1, `/home-mscluster/mmolefe/Playground/PhD/medical-imaging/compositional_diffusion/b-shortcut-through-a-device/` | TorchXRayVision's DenseNet scored on all 3,543 NEATX films: median `+0.001` between drain and drain-free films of the same patient over 237 pairs, and 514 of 1,906 drain-free against 180 of 1,637 drain films below its threshold; the drain-gap plan and the follow-up count of drain films without a pneumothorax label | the project's own runs |
| Liu et al. 2022, *Compositional Visual Generation with Composable Diffusion Models* | the AND and NOT operators built from conditional and unconditional scores | confident |
| `notebooks/simple_distributions.ipynb` | the JAX reference: `ProductEBMDiffusionModel` (cell 6) adds two separately trained energies, which is construction (2); `NegationEBMDiffusionModel` (cell 8); the 2D datasets (cells 14 to 20) | read in this session |
| `anneal_samplers.py` | the PyTorch annealed MALA, CHA, UHA and ULA samplers to reuse | read in this session, two MALA bugs under Checks outstanding |
| [the earlier walk on this idea](../poe-overlap-dial/IDEA_MAP.md) | runs on ideal and trained 2D experts: trained MLP experts within `0.04` nats of ideal ones; UHA within `0.02` total variation of the perfect-sampler floor on five composition kinds; reverse diffusion off on every kind but disjunction | its own runs, read in this session |

## Next step

Navigation: ⬅️ [Sources](#sources) | 📋 [TOC](#table-of-contents)

Claims 2 to 5 are settled on their repairs; 2a waits on route 3's notebook and claim 4's check is run 7. Claim 6 is current. The chest-drain case waits on B1's drain gap (the first step of claim 8's check), which decides whether a classifier trained on NIH uses the drain at all.
