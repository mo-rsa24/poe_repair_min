# 📚 Sources: poe_repair_min

Every external document and page this folder draws on. A provenance table cites an entry here by
number; the entry says what the source is, where it lives, and how far to trust it. Everything
else this folder cites is internal to this repository and is cited directly by its own path in
each file's "Where this came from" table, which is this format's convention for a repo-internal
claim.

No web page was fetched while building this folder: the repository's own documents (the paper
draft, `MASTER_PLAN.md`, `report/experiments-log.md`, the evidence and results-archive folders, and the
pipeline's own output files) were sufficient to ground every claim this pass made, per the
tier-1, read-only checks described in `~/.claude/skills/context-pulse/SKILL.md`.

## 1. Leveraging Low-Rank Adaptation for Multi-Concept Customization in Training-Free Diffusion Models

**What it is** a paper found by this project's literature scout, kept in the standing literature
folder. Full author list and venue not established by this pass.

**Where it lives** `plans/standing/literature/papers/Leveraging Low-Rank Adaptation for
Multi-Concept Customization in Training-Free Diffusion Models.pdf` (local copy only)

**Dated** unknown; not yet logged in a reading register (see
[00-INDEX.md § Still open](00-INDEX.md#still-open))

**What it is good for** background on LoRA-based multi-concept customisation in diffusion models,
judging only from the title. Not yet read: `pdftoppm` (needed to render its pages) is not
installed in this environment, so this pass could not open it past the filename.

**Read on** 2026-08-24 (title only)

## 2. Diffusion Explorer: Interactive Exploration of Diffusion Models

**What it is** Alec Helbling and Duen Horng Chau, arXiv 2507.01178. An educational browser tool
that trains small 2D diffusion and flow-matching models in TensorFlow.js and animates their
samples over time under a time slider, with contours, scatter, pathlines and a classifier-free
guidance page that draws the conditional and unconditional predictions as arrows on a sample.

**Where it lives** paper <https://arxiv.org/abs/2507.01178>; code
<https://github.com/helblazer811/Diffusion-Explorer>; live demo
<https://alechelbling.com/Diffusion-Explorer/>

**Dated** 2025

**What it is good for** the design idioms the landing figures borrow: precomputed trajectories
loaded from JSON, one time slider driving every panel, arrows drawn on the current sample. Not
good for running SDXL: its models are 2D toys and every renderer assumes a sample is two numbers.
The paper says extending to high-dimensional data and real images is open work.

Two posts pointed here. Dan Kornas (@DanKornas, 2026-06-06,
<https://x.com/DanKornas/status/2063131518868828375>): "Diffusion models are easier to
understand when you can see the geometry move. Diffusion Explorer is an interactive educational
tool for understanding the geometric intuition behind diffusion and flow-based generative
models. It helps you connect equations to model behavior by visualizing generated samples over
time, showing how samples change through training, and letting you train on custom hand-drawn
distributions." The rest of the post lists the tool's five features (flow and score objectives,
a sample-dynamics view, training-time visualisation, hand-drawn distributions, the rectified-flow
explainer). Alec Helbling (@alec_helbling, 2025-07-11,
<https://x.com/alec_helbling/status/1943658616151503043>), the author's own release post, a
reply under his announcement: "Paper: https://arxiv.org/abs/2507.01178 Demo:
https://alechelbling.com/Diffusion-Explorer/ Code:
https://github.com/helblazer811/Diffusion-Explorer", with a 15-second video of the tool. Neither
post claims anything beyond the toy setting; the "see the geometry move" line is the whole of what
this project borrowed.

**Read on** 2026-09-05 (README, the CFG page's source, and the paper via fetch; both posts via an
API mirror)

## 3. Diffusion Transformers with Representation Autoencoders

**What it is** Boyang Zheng, Nanye Ma, Shengbang Tong and Saining Xie, arXiv 2510.11690. Trains
ViT decoders that reconstruct pixels from frozen DINOv2, SigLIP2 and MAE features, so a point in
an encoder's space can be turned back into a picture.

**Where it lives** paper <https://arxiv.org/pdf/2510.11690>; code
<https://github.com/bytetriper/RAE>; decoder weights
<https://huggingface.co/nyu-visionx/RAE-collections>

**Dated** October 2025

**What it is good for** labelling the axes of an embedding-space figure with decoded pictures
instead of a legend (walk along a principal component, decode each point). Decoded frames are
reconstructions and carry the decoder's own bias; a chimera render must be reconstructed first to
see whether the decoder keeps it.

**Read on** 2026-09-05 (abstract and repository description via search; not yet run here)

## 4. Eigenfaces-style analysis on embedding models, an X post

**What it is** a post by Arnas Uselis (@a_uselis) showing walks along the top principal
components of pixel space, DINOv2-B and SigLIP2, each decoded through a representation
autoencoder, so the reader sees what each encoder's leading directions mean. Quotes a post by
Kosta Derpanis on eigenfaces. The author's own caveat: the pictures mix what the encoder cares
about with what the decoder introduces.

**Where it lives** <https://x.com/a_uselis/status/2096026369180205108>

**Dated** 2026

**What it is good for** the idea behind the axis pictures in the landing figures, and the
observation that pixel-space PCA walks change lighting and blur while DINOv2 walks change object
identity and layout, which is why the endpoint figures are drawn in DINOv2 space and not over
SDXL latents.

**Read on** 2026-09-05 (post text via an API mirror; the attached video was not viewable)

## 5. CDM: Contrastive Distribution Matching for discrete diffusion, and the two X posts that led to it

**What it is** Jaihoon Kim, Minhyuk Sung and co-authors, arXiv 2605.23346, project page
<https://cdm-smc.github.io/>. Reward-guided sampling for discrete diffusion by twisted
sequential Monte Carlo (many particles, each weighted by a twist function that estimates how
much reward lies downstream, then resampled). The contribution is learning the twist in advance
with a contrastive objective, raising it on samples from the target and lowering it on samples
from the current model, so discrete diffusion does not have to roll out every particle to score
it. Demonstrated on toxic-text avoidance, regulatory DNA, protein designability and diffusion
language-model alignment. Nothing in it touches images or composition.

Two posts pointed here. Minhyuk Sung (@MinhyukSung, 2026-09-01,
<https://x.com/MinhyukSung/status/2094798566091022441>): "SMC is a powerful approach for
diffusion reward guidance, but it can be extremely costly for discrete diffusion due to repeated
rollouts from every particle. CDM learns a twist function in advance with a contrastive
objective, amortizing this rollout cost while avoiding the train-test mismatch of previous
objectives." Jaihoon Kim (@KimJaihoon, quoted, <https://x.com/KimJaihoon/status/2093044703826743314>):
"Steering diffusion models toward a reward is slow. CDM makes it up to 50× faster, with less than
5% compute overhead."

**Dated** 2026

**What it is good for** the design of the twisted-SMC baseline in scope 06: set the base to
product-of-experts and the target to the joint prompt, then the optimal twist is the ratio of
the two diffused densities, and a classifier trained on samples of both at the same noise level
has that ratio as its logit. Also the reframing that the trained LoRA corrector is the gradient
of the same twist, learned by regression. Not good for its headline speed-up, which is a
discrete-diffusion result: continuous latents already get a cheap twist from the Tweedie
estimate, as the project page itself says.

**Read on** 2026-09-05 (both posts via an API mirror; the project page and abstract via fetch)
