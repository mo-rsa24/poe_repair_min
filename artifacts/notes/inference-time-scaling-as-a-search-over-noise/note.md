# Inference-time scaling as a search over the initial noise: Ma et al., arXiv 2501.09732, unpacked

Ma, Tong, Jia, Hu, Su, Zhang, Yang, Li, Jaakkola, Jia, Xie. "Inference-Time Scaling for Diffusion
Models beyond Scaling Denoising Steps." arXiv 2501.09732, v1 16 January 2025. Read 2026-09-05 from
the arXiv HTML rendering (abstract, sections 2 to 5, appendices A and B). No venue on the arXiv
page. The one thing the text does not carry is the formula that turns a pivot noise into its
neighbours; that gap is named in section 4 and again at the end.

Read for [the zero-order search plan](../../../plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/11-zero-order-search-over-the-initial-noise.md)
in scope 06.

## Table of contents

- [1. The problem](#1-the-problem)
- [2. Solution overview](#2-solution-overview)
- [3. Background](#3-background)
- [4. Methodology, step by step](#4-methodology-step-by-step)
- [5. Mathematics](#5-mathematics)
- [6. Results](#6-results)
- [7. Evaluation methodology](#7-evaluation-methodology)
- [What this breakdown could not establish](#what-this-breakdown-could-not-establish)

## 1. The problem

Navigation: 📋 [TOC](#table-of-contents) | [Next](#2-solution-overview) ➡️

A diffusion model given a prompt draws one Gaussian noise tensor and denoises it. Two draws of
that noise give two different pictures of different quality, and the only lever people pull to
spend more compute is more denoising steps, which stops paying off after a few dozen. On FLUX.1-dev
at 30 steps, one draw for a DrawBench prompt scores an ImageReward of 0.97 on average, and no number
of extra steps moves that. The paper exists to spend compute on the other source of variation: which
noise the run starts from.

## 2. Solution overview

Navigation: ⬅️ [1. The problem](#1-the-problem) | 📋 [TOC](#table-of-contents) | [Next](#3-background) ➡️

Treat generation as a search. Draw several starting noises, run the sampler on each, ask a
scoring function (the paper's word is verifier) which finished image is best, and keep it. Then
spend more by searching smarter: perturb the winner's noise and try again (zero-order search), or
branch part-way down the denoising path (search over paths). The whole thing is training-free; the
model is fixed and only the choice of noise changes. The paper's claim is that sample quality keeps
rising with the number of network evaluations spent on the search, on ImageNet and on
text-to-image, and that which verifier and which search algorithm fit a task is itself a design
choice.

## 3. Background

Navigation: ⬅️ [2. Solution overview](#2-solution-overview) | 📋 [TOC](#table-of-contents) | [Next](#4-methodology-step-by-step) ➡️

**A deterministic sampler makes the noise the whole story.** With an ordinary differential
equation solver (Heun, Euler, DDIM at eta 0), the map from initial noise to finished image is a
fixed function. The paper says it plainly: "because the model evaluations are inherently
deterministic, there is a fixed mapping from these noises to the final samples." So choosing a
noise is choosing an image.

**A network function evaluation (NFE)** is one forward pass of the denoiser. The paper accounts
for every cost in NFEs, and splits them into the NFEs spent generating the final image and the NFEs
spent searching.

**A verifier** is any function from a finished image (and optionally its condition) to one number.
Three families: an oracle that reads the benchmark metric itself, a supervised model (CLIP, DINO,
an aesthetic predictor, ImageReward), and a self-supervised proxy built from the model's own
features.

**The x-prediction** is the model's current best guess of the clean image at a noisy step, the
same object this project calls x0-hat. The paper uses it once, to score partway-denoised samples.

## 4. Methodology, step by step

Navigation: ⬅️ [3. Background](#3-background) | 📋 [TOC](#table-of-contents) | [Next](#5-mathematics) ➡️

Three search algorithms, section 3.2, in increasing order of structure.

**Random search (best of N).**

1. Draw N noises independently, each of shape [4, 128, 128] for a 1024-square latent model (the
   paper's models differ; the shape is illustrative).
2. Run the sampler on each to a finished image.
3. Score all N with the verifier and keep the highest.

Cost: N times the sampler's NFEs. The paper runs N in {2, 4, 8, ..., 256}.

**Zero-order search.**

1. Draw one noise n and call it the pivot.
2. Form N candidates in the pivot's neighbourhood, the set of noises y at distance λ from n:
   "S_n^λ = {y : d(y, n) = λ}". The text names no distance metric and no sampling formula, so how a
   candidate is actually drawn is not in the paper; see the gap note below.
3. Run the sampler on each candidate and score.
4. Make the best candidate the new pivot and repeat.

"When N gets larger, the algorithm will locate a more precise local optimum, and when λ increases,
the algorithm will have a larger stride." The paper fixes N 2 for text-to-image and tries N in
{2, 4} on ImageNet, finding "the effectiveness of increasing N is marginal, and N = 4 seems to
already be a good estimation of the local optimum". Appendix A.3, Figure 12: small λ is slightly
worse, large λ overfits the verifier once compute passes about 1000 NFEs.

**Search over paths.**

1. Draw N noises and denoise each to noise level σ (0.11 in the paper's units).
2. For each, draw M fresh noises and re-noise forward from σ to σ + Δf (Δf 0.78).
3. Denoise each back to σ + Δf − Δb (Δb 0.81, so each round nets a little cleaner), score with the
   verifier on the clean x-prediction because "the verifiers are typically not adapted to noisy
   input", keep the top N.
4. Repeat until σ reaches 0, then finish with random search over the survivors.

**A small worked instance of zero-order search, substituted for pedagogy** (the paper gives none).
Take a toy latent of shape [4, 8, 8], so 256 numbers. Pivot n is one draw from a standard normal.
Pick N 2 and a stride λ. Two candidates y1, y2 sit at distance λ from n. Denoise n, y1, y2 with a
20-step Euler solver: 60 NFEs. Score the three finished images. Suppose y2 wins. y2 becomes the
pivot, two new candidates are drawn around it, 60 more NFEs. After K rounds the pivot has moved K
strides toward whatever the verifier likes, and the final image is one more full sampler run from
it. The neighbourhood keeps every candidate close to a noise that already worked, which is what
the paper credits for holding diversity better than random search.

**The gap.** The paper text defines the neighbourhood only as "distance λ" and never writes how a
candidate is drawn. The one public reimplementation this note checked (sayakpaul/tt-scale-flux)
exposes a "threshold" of 0.95 on the candidates and gives no formula either. This project's plan
uses the variance-preserving form z' = (z + σ u) / sqrt(1 + σ²) with u standard normal, so every
candidate is still a unit-variance Gaussian and its cosine to the pivot is 1 / sqrt(1 + σ²): 0.995
at σ 0.1 and 0.958 at σ 0.3, the second within the reimplementation's 0.95 threshold. That choice
is this project's, not the paper's, and the plan says so.

## 5. Mathematics

Navigation: ⬅️ [4. Methodology, step by step](#4-methodology-step-by-step) | 📋 [TOC](#table-of-contents) | [Next](#6-results) ➡️

Two definitions carry the framing and nothing else is needed to run the method.

**Equation 1, the verifier.** V maps an image in R^{H×W×C} and a condition in R^d to one real
number. Read aloud: a verifier takes a picture and its prompt and returns a score.

**Equation 2, the algorithm.** f takes a verifier, the model D_θ, and N (noise, condition) pairs,
and returns one noise in R^{H×W×C}. Read aloud: a search algorithm looks at N starting noises
through the verifier and hands back the one to generate from.

The neighbourhood set in section 3.2, S_n^λ = {y : d(y, n) = λ}, is the only other formula. With
d the Euclidean norm and the noises of dimension 65,536, a standard normal draw has norm about 256,
so a λ of order 1 is a small angle; the paper does not state d, so this reading is an inference.

There is no loss and no derivation; the method changes nothing inside the model.

## 6. Results

Navigation: ⬅️ [5. Mathematics](#5-mathematics) | 📋 [TOC](#table-of-contents) | [Next](#7-evaluation-methodology) ➡️

**ImageNet, SiT-XL, oracle verifier (Figure 3).** FID over 50,000 samples falls steadily as the
search budget grows from 0 to about 8,000 NFEs; Inception Score rises across guidance weights. The
oracle reads the metric it is judged on, so this is the ceiling, not a usable method.

**Supervised verifiers and verifier hacking (Figures 4 and 13).** Searching against CLIP or DINO
class logits raises precision and lowers recall as the budget grows: the samples get individually
more classifiable and collectively less diverse. The paper names this verifier hacking and ties
it to random search's unconstrained space. Zero-order search and search over paths "manage to
alleviate the diversity issue of FID to some extent while maintaining a scaling Inception Score"
(Figure 6).

**Text-to-image, FLUX.1-dev on DrawBench (Table 2, 2,880 search NFEs against a 30-NFE
generation).** With a verifier ensemble and random search, Aesthetic 5.79 to 6.06, CLIPScore 0.71
to 0.77, ImageReward 0.97 to 1.41, LLM grader 84.29 to 88.18. ImageReward's typical range on this
prompt set is about 0 to 1.5, so 0.97 to 1.41 is a large move; the others are small.

**T2I-CompBench (Table 1).** Searching on ImageReward lifts every category; colour 0.7692 to
0.8303 is the largest.

**Compute trade (Table 4).** PixArt-Σ with search at about 2.6 times FLUX.1-dev's compute reaches
FLUX.1-dev's quality, and at about 0.09 times matches FLUX without search.

## 7. Evaluation methodology

Navigation: ⬅️ [6. Results](#6-results) | 📋 [TOC](#table-of-contents) | [Next](#what-this-breakdown-could-not-establish) ➡️

**7a. Training procedure.** There is none. "Most models used in our work are pre-trained"
(Appendix A.1): SiT-B, SiT-L and SiT-XL for ImageNet, FLUX.1-dev and PixArt-Σ for text-to-image,
plus the verifier models as released. The tuned quantities that stand in for training are the
search hyperparameters: λ and N for zero-order search (Figure 12, Figure 6), and σ 0.11, Δf 0.78,
Δb 0.81 for search over paths (Appendix A.3). The paper does not say how those were chosen or on
what split, which is the "training-free method with a tuned knob" trap; flagged.

**7b. Inference procedure.** Per Table 5 (Appendix A.2): SiT-XL uses a second-order Heun ODE
solver at 250 NFEs for the final image and 50 NFEs per search iteration, guidance 1.0; FLUX.1-dev
uses Euler at 30 NFEs both for the final image and per iteration, guidance 3.5; PixArt-Σ uses DDIM
at 30 NFEs, guidance 4.5, 1024 square. Search iterations run at a reduced step count and the winner
is regenerated at the full count. Guidance scales are disclosed and held fixed across search
budgets, so the comparisons along a scaling curve are fair on that axis.

**7c. Qualitative assessment.** Figure 7 shows FLUX.1-dev samples before and after search. No
statement of how those prompts or seeds were chosen was found in the text, so they are anecdote
until the code release says otherwise; flagged.

**7d. Quantitative assessment.** ImageNet: FID and Inception Score over 50,000 synthesised samples
at batch size 256 against the reference statistics of Karras et al. (Appendix A.5), plus
precision and recall in Appendix B. FID is the Fréchet distance between InceptionV3 feature
Gaussians of generated and reference sets; it rewards matching the reference distribution, so a
verifier that collapses diversity raises FID even as per-image scores improve, which is exactly the
hacking the paper reports. Text-to-image: DrawBench, 200 prompts in 11 categories, one image per
prompt; T2I-CompBench, 1,800 validation prompts, two images per prompt, with BLIP-VQA for colour,
shape and texture, UniDet for spatial and numeracy, and a weighted mix of BLIP-VQA, UniDet and CLIP
for the complex category (Appendix A.5). The LLM grader is Gemini 1.5 Flash scoring five aspects
(accuracy to prompt, originality, visual quality, internal consistency, emotional resonance) on 0
to 100 (section 4). The verifier ensemble ranks each sample by the unweighted mean of its Aesthetic,
CLIPScore and ImageReward ranks (section 4, Appendix A.4). Search budgets: 2,880 NFEs in Table 2,
3,840 in Figure 8. Baselines are the same models with no search; no other inference-time method is
compared. No seed variance is reported anywhere, and the ImageNet seed count is not stated. No human
evaluation. For a compositional reader, T2I-CompBench's detector-based numeracy and spatial
categories are the ones that can show binding, and colour is the largest gain, so the
compositional evidence rests on BLIP-VQA rather than on a count.

## What this breakdown could not establish

Navigation: ⬅️ [7. Evaluation methodology](#7-evaluation-methodology) | 📋 [TOC](#table-of-contents)

The formula that draws a zero-order candidate from its pivot, the distance metric behind λ, the λ
values Figure 12 tried, the number of seeds behind any number, and how Figure 7's examples were
chosen. The closest inference for the first is the variance-preserving mixture named in section 4,
which is what this project runs.
