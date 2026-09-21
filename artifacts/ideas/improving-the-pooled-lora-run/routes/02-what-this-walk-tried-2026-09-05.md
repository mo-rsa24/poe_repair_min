# What this walk tried on 2026-09-05, in order, and what each step left behind

Route 02 of [the idea map](../IDEA_MAP.md). One session, one question at the start (does more
training data close the held-out gap of the pooled rank 8 adapter?), four runs, one literature
sweep, two findings. This file is the trace: every step names what it asked, what it did, what
came back, and where the evidence sits. The findings themselves are in `report/`; this file never
restates a verdict, it points at one.

## Table of contents

- [Where the walk started](#where-the-walk-started)
- [Step 1: the pressure test](#step-1-the-pressure-test)
- [Step 2: the exposure-bias hypothesis, and run 2 that killed it](#step-2-the-exposure-bias-hypothesis-and-run-2-that-killed-it)
- [Step 3: the paper scout](#step-3-the-paper-scout)
- [Step 4: Mitra et al., unpacked](#step-4-mitra-et-al-unpacked)
- [Step 5: three uses of Mitra, two of which ran](#step-5-three-uses-of-mitra-two-of-which-ran)
- [Step 6: the report](#step-6-the-report)
- [The X post in the sources register](#the-x-post-in-the-sources-register)
- [What is left](#what-is-left)

## Where the walk started

Navigation: 📋 [TOC](#table-of-contents) | [Next](#step-1-the-pressure-test) ➡️

**The setting.** SDXL, "a cat" x "a dog", 50-step DDIM at guidance 7.5, plain product of experts
gives one blended animal on 8 of 8 seeds. One LoRA on the cross-attention projections (rank 8, 16
or 32) predicts the correction the product drops, from a cache of 11 pairs by 8 seeds by 50
steps, and is injected at every step. Held-out cat x dog composes on 7 of 8 seeds at the best
checkpoints.

**The question the walk opened with.** The idea map's claim 1: a render-time cosine of 0.54 on
training cells and 0.40 on held-out cells, flat from 10k steps, is data-limited, so more pairs
move it.

**What was already known when this session began.** Run 1 in the idea map had measured the fit
on cached states: 0.97 on training pairs, 0.80 held-out, gap widest late.

## Step 1: the pressure test

Navigation: ⬅️ [Previous](#where-the-walk-started) | 📋 [TOC](#table-of-contents) | [Next](#step-2-the-exposure-bias-hypothesis-and-run-2-that-killed-it) ➡️

**Asked.** Are the three remaining failures (the blend, the one-animal render, the late haze)
named in the composition literature, which fixes are sampler-side and which model-side, does any
prior work generalize a learned per-step correction across pairs, is the one-animal render a
property of the deterministic trajectory, and is late haze a known overfitting signature?

**Did.** Read the trainer to find one fact the brief had not stated: each adapter branch runs on
its own prompt and never sees the other concept. Wrote the seven-part verdict.

**Came back.** Promising but needs rework. The blend is the product target itself (Bradley et al.
2502.04549, §3.2), so exact samplers of the product cannot fix it. The one-animal render is
catastrophic neglect, a basin of the deterministic trajectory chosen in the first steps. The haze
is the no-decay, constant-learning-rate, saturated-loss signature.

**Where it sits.** [The pressure-test route](01-pressure-test-poe-failures.md), folded into
claim 1 of the idea map the same session.

## Step 2: the exposure-bias hypothesis, and run 2 that killed it

Navigation: ⬅️ [Previous](#step-1-the-pressure-test) | 📋 [TOC](#table-of-contents) | [Next](#step-3-the-paper-scout) ➡️

**Asked.** The cache holds states on the uncorrected trajectory and inference visits the corrected
one. Is the render-time 0.40 the cost of that mismatch?

**Hypothesis, written before the run.** On the adapter's own corrected path the fit starts near
0.9 and decays toward 0.4 as the state leaves the cache. If flat, the 0.40 measures something else.

**Did.** Wrote `scripts/showcase/on_policy_fit_cosine.py`: roll out cat x dog seeds 9 to 16 from
the cached initial noise with the correction at every step, and at each reached state compute the
true correction with a joint-prompt forward. Slurm job 49421 on mscluster72.

**Came back.** Flat. Fit on the corrected path 0.82 against 0.86 on the cache, while the state
drifts up to 86% away. Drift costs 0.04, not 0.4. In parallel the drip-idea session traced the
0.54 and 0.40 to `direction_metrics`, the cosine against the pool-mean correction, which a
state-specific target cannot push to 1. The number was retired.

**Where it sits.** Run 2 in the idea map; rung 1 and rung 3 of
[the fit, drift or pair finding](../../../../report/does-the-fix-reach-unseen-pairs/is-the-held-out-gap-a-fit-a-drift-or-a-pair-problem.md);
output under `/datasets/mmolefe/poe_repair_min/outputs/showcase/on_policy_fit_r8_030000/`.

**A check that came free.** The rollout's renders are pixel-identical to the showcase sampler's
renders for the same seeds and checkpoint, so the fit was measured on the real inference path.
Figure 3 of [the explainer](../../../results/is-the-held-out-gap-a-fit-a-drift-or-a-pair-problem/figure-explainer.md).

## Step 3: the paper scout

Navigation: ⬅️ [Previous](#step-2-the-exposure-bias-hypothesis-and-run-2-that-killed-it) | 📋 [TOC](#table-of-contents) | [Next](#step-4-mitra-et-al-unpacked) ➡️

**Asked.** What in the 2024 to 2026 literature on composition in diffusion and flow models covers
four components the walk might be missing: the mathematics of the dropped term, training data and
on-policy targets, corrective sampling and basins, adapter placement and regularization.

**Did.** About 180 abstracts over 13 arXiv queries, section text for 11 papers, one web search.
Ranked on whether a paper treats one of the four components.

**Came back.** Thirteen papers judged, three set aside. The five put in reading order: Mitra et
al. 2605.22596 (the bound and null-token dropout), Rectify Then Diffuse 2608.03135 (one step on
the initial noise beats CO3 on animal pairs on SDXL), Spectral Prior 2607.22091 (exposure bias
measured per timestep on SDXL), DanceOPD 2606.27377 (on-policy distillation), ACE 2512.10339
(when a product path is undefined mid-run).

**Where it sits.** The selection file at
`~/goal-setting/learning/t2i-compositional-failure/paper-scout/selection-2026-09-05.md`, with
load-bearing sections, measured results and confidence marks per paper; the paper-scout profile
memory lists them so they are not re-scouted.

## Step 4: Mitra et al., unpacked

Navigation: ⬅️ [Previous](#step-3-the-paper-scout) | 📋 [TOC](#table-of-contents) | [Next](#step-5-three-uses-of-mitra-two-of-which-ran) ➡️

**Asked.** What does the paper really do, and is it relevant?

**Came back, in four steps.** Bayes splits the conditional score into an unconditional part and a
task part. The task part splits across factors up to the gradient of g, the pointwise mutual
information between the factors given the state. If |g| ≤ G and its Hessian is bounded by M,
Landau's inequality gives ‖∇g‖ ≤ 2√(GM). That gradient is this project's correction: ∇g =
−r_t/σ_t. Their method keeps the additive sum and drops ∇g; their experiments show one shared
network with null-token dropout makes the dropped term negligible for factors that barely
interact (27 of 30 held-out combinations against 1 of 30 for separately trained experts).

**The relevance, decided.** SDXL's product of experts already is their factored model: one UNet,
a shared empty prompt, prompt dropout at pretraining. It fails on cat x dog, which through their
theorem says G is not small for similar animals. Their composition rule is this project's
baseline; their theorem names the quantity the correction measures; their held-out protocol
(unseen combinations of seen factors) is a test this project had not run.

**Where it sits.** The symbol table in [the pressure-test route](01-pressure-test-poe-failures.md)
and the chat record; nothing else was written for this step.

## Step 5: three uses of Mitra, two of which ran

Navigation: ⬅️ [Previous](#step-4-mitra-et-al-unpacked) | 📋 [TOC](#table-of-contents) | [Next](#step-6-the-report) ➡️

**Use 1, the theorem turned around (run 3).** Hypothesis: the lower bound on G·M from the cache
rises with the plain-PoE fail rate. Did: `scripts/showcase/interaction_bound_from_cache.py` over
25 pairs on the CPU. Came back: null. Every failing pair fails at bounds from 100 to 3000. Post
hoc, the pairs split at step 10 into near-synonyms (low) and distinct animals (high), a factor of
ten apart, with cat x dog high and most of the training pool low.
[The finding](../../../../report/does-the-fix-reach-unseen-pairs/does-interaction-strength-predict-which-pairs-blend.md).

**Use 2, the two-tier held-out (run 4).** Hypothesis: pairs of seen words in an unseen pairing
fit near the training 0.97 if the adapter learned per concept, near 0.80 if per pair. Did: run
1's fit script over eight excluded pairs at 12 seeds, Slurm job 49851 on mscluster52. Came back:
0.84 for both-words-seen, 0.77 for one-word-seen, against 0.97 and 0.80. Per pair.
[The finding](../../../../report/does-the-fix-reach-unseen-pairs/is-the-held-out-gap-a-fit-a-drift-or-a-pair-problem.md), rung 2.

**Use 3, null-token dropout in the adapter's training: not run, and withdrawn.** Mitra's dropout
identifies per-factor corrections because their network sees every factor slot at once. This
adapter runs each branch on its own prompt, so no branch can know another was dropped. Dropping
cat with the joint target would teach the dog branch to render cat and dog; dropping cat with the
frozen dog target would force the adapter to change nothing. Neither identifies anything. What
survives is a small-weight prior-preservation term against the late haze, which is a claim 3
experiment to cost against post-hoc EMA, not a Mitra experiment.

## Step 6: the report

Navigation: ⬅️ [Previous](#step-5-three-uses-of-mitra-two-of-which-ran) | 📋 [TOC](#table-of-contents) | [Next](#the-x-post-in-the-sources-register) ➡️

Two finding files in `report/`, each with numbered rungs, every number naming its file and field,
a "what this cannot tell you" section and a provenance table. Three figures filed under
`artifacts/results/` in two groupings, each with a README card and a figure explainer that reads
every image in plain words and links the documents that own each term. The index carries both
verdicts with their headline numbers.

## The X post in the sources register

Navigation: ⬅️ [Previous](#step-6-the-report) | 📋 [TOC](#table-of-contents) | [Next](#what-is-left) ➡️

The sources register holds one X post,
[eigenfaces-style walks through embedding models](../../../../context/sources.md#4-eigenfaces-style-analysis-on-embedding-models-an-x-post)
by Arnas Uselis: walks along the top principal components of pixel space, DINOv2-B and SigLIP2,
each decoded through a representation autoencoder, with the author's own caveat that the pictures
mix what the encoder cares about with what the decoder adds. It was read on 2026-09-05 through an
API mirror; the attached video was not viewable.

It did not feed any run in this walk. It shaped the sibling finding
[where does each condition land](../../../../report/when-does-the-outcome-lock-in/where-does-each-condition-land.md): the
observation that pixel-space walks change lighting and blur while DINOv2 walks change object
identity is why that finding's endpoint figures are drawn in DINOv2 space and not over SDXL
latents, and why its axis pictures are still an open slot. The connection to this walk is
indirect: the same eight cat x dog seeds appear there as points on axes and here as fit numbers
and as a contact sheet.

## What is left

Navigation: ⬅️ [Previous](#the-x-post-in-the-sources-register) | 📋 [TOC](#table-of-contents)

- Settle claim 1 in the idea map: the 20-pair cache is built from new pairings of the 22 seen
  words, drawn from the high band of the interaction figure, held-out words kept out.
- Claim 3, the haze: post-hoc EMA over the rank 32 checkpoints already on disk, before any
  training run.
- Two baselines the scout put on the table for scope 06: Rectify Then Diffuse on animal pairs,
  and ACE's path-existence criterion checked on SDXL's schedule at guidance 7.5.
- The backlinks marked ⬜ in both figure explainers.
- Fail rates for the five wider-cache pairs the interaction figure colours grey.
