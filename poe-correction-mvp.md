# PoE correction MVP — cat × dog, seed 42

## What this plan is

Three stages, each with a single sharp goal.

1. **Show the correction is possible.** With mono available at inference, inject the cached residual along the PoE trajectory and watch the hybrid morph into a clean co-occurrence.
2. **Diagnose the PoE / mono / residual relationship.** A small set of figures that explain *what* the residual is, *when* it is large, and *how* it relates to the two trajectories.
3. **Replace mono with a learner.** A bottom-up menu of techniques that produce the correction without ever calling mono at inference.

The hypothesis is qualitative. The single success criterion across all three stages is: *over the denoising trajectory, the chimera-like hybrid blob gradually morphs into a visually distinct cat and dog co-occurring on the canvas*. Loss curves, residual norms, cross-seed agreement, statistical significance — none of these decide success. They are diagnostics, not gates.

## Scope and constraints

- **Cell.** `a_cat__x__a_dog`, seed 42. One pair, one seed. No cross-seed. No cross-pair.
- **Reference samplers.** Plain PoE and plain mono. No sched-M2, no anchor windows, no schedule scans — those live in other documents and are not the contribution being tested here.
- **Mono usage.** Oracle and teacher only. Cached residuals come from a single mono rollout. Stage 1 is the only place mono is invoked at inference, and only as evidence that the correction is possible in principle.
- **Eval.** Qualitative inspection of decoded trajectories. Curves and bar plots appear inside diagnostic figures, never as the headline.
- **Residual definition.** `r_t = ε̃_J(z_t, t, e_J) − ε̃_PoE(z_t, t, e_A, e_B)` in guided-ε space, per [memory/residual_definition.md](.claude/projects/-home-mscluster-mmolefe-Playground-PhD-poe-repair-min/memory/residual_definition.md).

The qualitative success picture, in one sentence: at λ=0 you see a chimera; as λ→1 you see two distinct animals; the intermediate decoded latents show the chimera *progressively splitting* — not flickering, not collapsing, not being replaced by a re-generated scene. That gradual morph is the signal. Anything else is a negative result.

---

## Stage 1 — Show the correction is possible (mono allowed at inference)

**Goal.** Demonstrate qualitatively that injecting the oracle residual into the PoE trajectory walks the decoded image from chimera to clean co-occurrence. The existence proof — without it, no learner downstream can succeed. Mono is invoked at inference *for this stage only*.

**Hypothesis under test.** The cached residual `r_t = ε̃_J − ε̃_PoE` is a faithful and complete description of the correction PoE needs at every step. Injecting it gradually (λ : 0 → 1 along the PoE trajectory) walks PoE into mono. Subtracting it gradually (λ : 0 → 1 along the mono trajectory) walks mono back to PoE.

**Success.** A row of decoded thumbnails along the λ-walk shows a smooth, monotone morph from chimera (λ=0) to two distinct animals (λ=1). The trajectory should look like the hybrid *separating*, not like the hybrid being replaced by a freshly regenerated scene.

**Failure.** Either (a) the morph is non-monotone — image quality oscillates as λ increases — or (b) the endpoint at λ=1 is not visually equivalent to a mono rollout. Either outcome means the residual cache is wrong or the per-step injection is misformulated; fix that before Stage 2 or Stage 3.

### Stage 1 figures

**Figure 1.1 — PoE-anchored λ walk.** 11 decoded thumbnails (λ = 0.0, 0.1, …, 1.0) along the PoE trajectory with λ · r_t injected at every step. Existing data at [outputs/veracity/pairs/a_cat__x__a_dog/seed_42/teacher_residual_const_lam{000..100}/](outputs/veracity/pairs/a_cat__x__a_dog/seed_42/).

**Figure 1.2 — Mono-anchored λ walk.** 11 decoded thumbnails along the *mono* trajectory with λ · r_t *subtracted* at every step. λ=0 is mono; λ=1 should be PoE-like. Same tooling as 1.1 with the sign of the injection flipped. Not yet rendered.

Two rows of thumbnails. Nothing else in Stage 1.

---

## Stage 2 — Diagnose the PoE / mono / residual relationship

**Goal.** Build a small set of figures that explain *what* the residual does and *when*. These figures are not gates for downstream work — they are for understanding the phenomenon, and for the eventual write-up.

**Hypothesis under test.** The residual is (i) non-trivial in magnitude, (ii) concentrated in time around the commit window, (iii) spatially structured rather than noise-like, and (iv) interpretable in attention space — it lines up with where PoE and mono's cross-attention maps disagree about routing.

**Success.** Each figure makes a visually clear statement about one of those properties.

**Failure.** A figure produces a noisy mess with no pattern. That is also a useful result: it falsifies one of the properties and tells Stage 3 not to assume it.

### Stage 2 figures

**Figure 2.1 — Residual magnitude over time.** A single curve: `‖r_t‖` along the PoE trajectory with the commit window shaded. Source: per-step `.pt` tensors in `teacher_residual_const_lam000/residuals/`. Tests property (i) and (ii).

**Figure 2.2 — Decoded trajectory comparison.** Three rows (PoE-only, PoE + r_t at λ=1, mono), each showing decoded images at 5–8 intermediate timesteps. Lets the reader see *how* the residual reshapes the trajectory at every step, not just the endpoint. Tests that the morph is *progressive*, not endpoint-only.

**Figure 2.3 — Per-channel residual snapshot.** At a fixed timestep inside the commit window, show the four channels of `r_t` as heatmaps next to the decoded `z_t`. Tests property (iii) — spatial structure or its absence.

**Figure 2.4 — Cross-attention divergence.** At the same fixed timestep, show side-by-side cross-attention maps for the "cat" and "dog" tokens: PoE's versus mono's. Tests property (iv) — where in the canvas the two models disagree about routing.

That is the diagnostic set. Four figures, each making one statement. None of them gates Stage 3.

---

## Stage 3 — Replace mono with a learner

**Goal.** Train a network or adapter that produces the correction `r̂_t` at inference, *without* ever invoking mono. The deployed sampler is identical to Stage 1's λ=1 injection except `r_t` is replaced by `r̂_t = f_θ(z_t, t, e_J)` (or by the equivalent weight-space perturbation, depending on technique).

**Hypothesis under test.** The residual is a learnable function of `(z_t, t, e_J)` on this cell. Some technique in the menu below — possibly more than one — can recover enough of the residual to reproduce Stage 1's morph qualitatively.

**Success per technique.** The decoded trajectory from a learner-corrected PoE rollout matches Stage 1's λ=1 morph qualitatively. The chimera progressively becomes two distinct animals. Endpoint quality is comparable to mono.

**Failure per technique.** The decoded trajectory shows no morph, a morph that plateaus partway, or non-monotone behaviour. The technique has not captured the residual on this cell.

**Replay baseline (Stage 3 prerequisite).** Before any learner is judged, run a "cached-residual replay" baseline: at every step, inject the *cached* `r_t` directly with no learner involved. This must reproduce Stage 1 exactly. If replay fails, the cache is broken — fix that before training anything.

### How Stage 3 is organised

Techniques cluster into four groups by *what they modify*. Within each group, techniques are ordered from simplest to most expressive. Across groups, the recommended progression starts at the cheapest group (Group A) and escalates only when a group plateaus.

The four groups:

- **Group A — External score-space correctors.** A separate network produces an additive ε-correction; SDXL is frozen.
- **Group B — UNet weight-space adapters.** SDXL's projections are perturbed by trainable parameters; correction is the delta between perturbed and frozen forwards.
- **Group C — Conditioning-pathway additions.** SDXL weights are untouched; new conditioning tokens are appended to cross-attention and trained.
- **Group D — Attention-routing rewrites.** The attention map itself is overwritten at each cross-attention layer; SDXL weights are untouched.

Every technique trains against the same target (`r_t`), under the same default schedule (σ-windowed MSE — weight per-step loss by `‖r_t‖` or by `1[t ∈ commit_window]`). The σ-window is not a contingency; it is the default.

---

### Group A — External score-space correctors

**What the group does.** A small standalone network sits next to SDXL. At each denoising step it reads the current latent and the joint conditioning and outputs an additive correction in ε-space. SDXL is frozen end-to-end. The correction is added to PoE's prediction before DDIM consumes it.

**Shared operations.**
- Train on cached `(z_t, t, e_J, r_t)` tuples with σ-windowed MSE.
- Deployed sampler at every step: (1) compute `ε̃_PoE` via frozen SDXL, (2) run the group's network to produce `r̂_t`, (3) add, (4) hand the sum to DDIM.
- Output shape is always 4×128×128.

**What distinguishes techniques within the group.** What kind of network reads the state — a flat CNN, a small UNet, or a head on top of frozen SDXL features.

#### A1. Latent-CNN corrector

**What it is.** A 4–6 block convolutional network that operates on the raw latent. Conditioning on `t` and `e_J` is mixed in via FiLM at every block.

**Input.** Latent `z_t` (4×128×128), timestep `t`, joint embedding `e_J`.

**Output.** Correction `r̂_t` (4×128×128).

**Formulation.** Stacked blocks of `feature ← γ(t, e_J) · Conv(feature) + β(t, e_J)`. Final 1×1 conv to 4 channels. About a million parameters.

**Implementation.** Standard PyTorch module. FiLM projection: small MLP on `(sinusoidal(t), mean_pool(e_J))` producing per-channel scale and shift.

**Training.** Sample `t` (weighted by σ-window); retrieve cached `(z_t, r_t)`; forward; MSE; backprop into the CNN only. Hours per cell on one GPU.

**Focus.** Spatial structure of the residual at the latent's native resolution.

**What makes it different.** No pretrained component. No SDXL pass during training. Cheapest learner in the entire menu.

**Success.** Corrected rollout reproduces Stage 1's λ=1 morph.
**Failure.** Plateau where the corrected trajectory still shows a chimera. If the plateau looks scale-specific (fine details wrong) → escalate to A2. If the plateau looks input-bottlenecked (the latent alone doesn't carry enough signal) → escalate to A3.

#### A2. Latent-UNet corrector

**What it is.** Same role as A1, but the network is a small UNet (2–3 scales, skip connections, t-conditioned residual blocks).

**Input.** Same as A1.

**Output.** Same as A1.

**Formulation.** DDPM-style small UNet. 5–10M parameters depending on widths.

**Implementation.** Shrink an off-the-shelf small UNet (e.g. one used for toy CIFAR-10 diffusion experiments).

**Training.** Identical to A1.

**Focus.** Multi-scale residual structure — coarse "separate the animals" patterns plus fine-grained texture corrections, in the same forward.

**What makes it different from A1.** Skip connections expose the network to coarse and fine views at once. Useful when the residual is multi-scale; wasted capacity if it isn't.

**Success.** Same as A1.
**Failure.** If A2 still plateaus and the failure no longer looks scale-specific, the residual isn't extractable from the raw latent — escalate to A3.

#### A3. Frozen-feature MLP head

**What it is.** Run SDXL's frozen UNet on `(z_t, t, e_J)`; extract mid-block features; pass them through a small MLP that outputs the correction.

**Input.** Same as A1, routed through frozen SDXL first.

**Output.** Same as A1.

**Formulation.** `r̂_t = MLP(extract(SDXL_frozen(z_t, t, e_J)))`. Conditioning is implicit — SDXL already consumes `(t, e_J)` internally.

**Implementation.** Forward hook on the chosen module; flatten or pool spatially; 2–4 dense layers; reshape to 4×128×128.

**Training.** Per step: one frozen SDXL forward plus one MLP forward. Slower than A1/A2 because of the SDXL pass. Backprop into the MLP only.

**Focus.** Mapping SDXL's internal representation of `(z_t, t, e_J)` to the residual. Useful when the residual is a function of what SDXL already knows internally about the joint prompt.

**What makes it different from A1/A2.** Inherits SDXL's pretrained representation without modifying SDXL. The MLP doesn't have to learn what features the latent contains — frozen SDXL already extracted them.

**Success.** Same as A1.
**Failure.** Residual is not recoverable from frozen features. The next move is to *modify* SDXL — escalate to Group B.

---

### Group B — UNet weight-space adapters

**What the group does.** Modify SDXL's UNet projections via small trainable perturbations. The correction is the difference between the perturbed forward and the frozen forward. The base SDXL weights stay frozen end-to-end; only the adapter parameters train.

**Shared operations.**
- Two SDXL forwards per inference step: perturbed and frozen. `r̂_t = ε_perturbed − ε_frozen`.
- Train on cached `r_t` with σ-windowed MSE between `(ε_perturbed − ε_frozen)` and the target.
- Same deployed sampler structure as Group A: compute `ε̃_PoE`, compute `r̂_t`, add, DDIM.

**What distinguishes techniques within the group.** The parameter budget (per-channel scale → norm-and-bias → low-rank LoRA → full fine-tune) and the location of the perturbation (cross-attention vs self-attention vs everywhere).

#### B1. Per-channel scale adapter (IA³)

**What it is.** A learned vector that multiplies each chosen projection's output channel-wise. No matrix decomposition.

**Input.** Same as the UNet — `(z_t, t, e_J)`.

**Output.** Modified ε prediction; correction is perturbed minus frozen forward.

**Formulation.** `out = (W · in) ⊙ s` where `s ∈ R^{d_out}` is the learned channel-wise gain.

**Implementation.** Custom forward hook on each target projection; multiply output by a learned `nn.Parameter` of shape `(d_out,)`.

**Training.** σ-windowed MSE on cached `r_t`. A few thousand parameters in total. Converges fast.

**Focus.** Per-channel gain. Cannot rotate or mix channels — only rescale them.

**What makes it different.** Floor of parameter efficiency in the whole menu. Useful as a sanity check.

**Success.** Reproduces the morph. If this works, the failure has a very low-dimensional structure and richer adapters (B3, B4) would be overkill.
**Failure.** No morph. Escalate to B2.

#### B2. Norm-and-bias tuning (AdaGN / DiffFit)

**What it is.** Trains only `γ` and `β` of every GroupNorm / LayerNorm in the UNet, plus every Linear / Conv bias. Nothing else.

**Input.** Same as B1.

**Output.** Same as B1.

**Formulation.** Walk the UNet module tree; mark norm weights and biases plus linear/conv biases as trainable; freeze everything else.

**Implementation.** A `requires_grad` toggle pass over `unet.named_parameters()`.

**Training.** σ-windowed MSE on cached `r_t`. A few hundred thousand parameters.

**Focus.** Per-channel feature magnitude after normalisation, across the whole UNet — every layer's "channel-wise volume knob."

**What makes it different from B1.** Reaches every normalisation in the UNet, not just selected projections. Far more reach than IA³, still much smaller than LoRA.

**Success.** Reproduces the morph. If this works, the failure is feature scaling (not attention routing or text reading specifically).
**Failure.** No morph. Escalate to B3.

#### B3. Cross-attention LoRA *(current known-working baseline)*

**What it is.** Rank-8 low-rank adapters injected into `attn2.{to_q, to_k, to_v}`. Modifies how the UNet reads text conditioning.

**Input.** Same as B1.

**Output.** Same as B1.

**Formulation.** `W_eff = W_frozen + α A B` with `A: rank × d_in`, `B: d_out × rank`.

**Implementation.** `peft` / `diffusers` LoRA injection on the `attn2` regex; freeze everything else.

**Training.** σ-windowed MSE on cached `r_t`. About 5M trainable parameters.

**Focus.** The text-to-latent meeting point — where the prompt influences spatial features.

**What makes it different from B1/B2.** Targets the attention pathway specifically and at higher capacity. Inherits SDXL's pretrained prior in that pathway.

**Status.** Already shown to reproduce the qualitative morph on this cell. This was called "M5" in earlier documents — renamed here so the *operation* (low-rank perturbation of cross-attention) is visible in the name rather than hidden behind a project label.

**Success.** The morph reproduces — confirmed.
**Failure (counterfactual).** Would have meant the text-reading pathway isn't the right target on this cell and B4 or D1 should have been tried instead. Since B3 does reproduce the morph, the open question is whether *cheaper* groups (A, B1, B2) can also reproduce it, not whether B3 itself works.

#### B4. Self-attention LoRA

**What it is.** Identical to B3 in every respect except the hook target — adapters go on `attn1.{to_q, to_k, to_v}` instead of `attn2`. Self-attention layers decide how spatial positions interact with each other, independent of text.

**Input.** Same as B3.

**Output.** Same as B3.

**Formulation.** Same as B3 with one regex changed (`attn1` instead of `attn2`).

**Implementation.** Identical to B3 with the module pattern swapped.

**Training.** Identical to B3.

**Focus.** Spatial binding — how patches attend to each other on the canvas, independent of what the prompt is.

**What makes it different from B3.** Same machinery, different culprit. B3 bets the failure is *what is being named*; B4 bets the failure is *where things go*.

**Success.** Reproduces the morph. If both B3 and B4 reproduce it independently, the failure has two components; if combining them is additive, that itself is a finding.
**Failure.** No morph. The failure is not in self-attention.

#### B5. Full UNet retraining

**What it is.** Initialise from pretrained SDXL; train every UNet parameter; the model's output is `r̂_t` directly, not ε.

**Status.** Deferred indefinitely. No low-rank regularisation, severe single-cell overfitting risk, irreversible loss of SDXL's noise function. Listed for completeness only.

---

### Group C — Conditioning-pathway additions

**What the group does.** SDXL weights are completely untouched. New conditioning information is added — typically as learned tokens prepended to cross-attention key/value sequences — and SDXL learns to use it as if it were additional text.

**Shared operations.**
- Zero modification to SDXL weights anywhere. The only trainable parameters are the new conditioning tensors.
- Two SDXL forwards per training step (with and without the new tokens), differenced for the correction.

**What distinguishes techniques within the group.** Currently a single technique. Listed as a group because the *operation* — augment conditioning without touching weights — is structurally distinct and may grow.

#### C1. Prefix-token conditioning

**What it is.** Prepend 8–16 learned tokens to the K/V sequences at each cross-attention layer.

**Input.** Same as standard SDXL inputs; the prefix is added inside each cross-attention layer.

**Output.** Modified ε prediction; correction is with-prefix minus without-prefix forward.

**Formulation.** Per layer ℓ, learn `P_K^{(ℓ)}, P_V^{(ℓ)} ∈ R^{n_prefix × d}`. K and V at that layer become `concat([P_K, K])` and `concat([P_V, V])` along the sequence axis.

**Implementation.** Subclass diffusers' attention processor; do the concatenation inside the processor; mark only the prefix tensors as trainable.

**Training.** σ-windowed MSE on cached `r_t`. About `n_layers × n_prefix × d` parameters — typically smaller than rank-8 LoRA.

**Focus.** Adding new conditioning channels the model can attend to, without changing how it attends.

**What makes it different from Group B.** SDXL stays bit-identical. The mechanism is indirect — the model has to *interpret* the prefix as meaningful conditioning, which is a strength if the conditioning pathway is the right place to intervene and a weakness otherwise.

**Success.** Reproduces the morph. If it works, the failure is "missing conditioning" — the joint prompt's signal can be encoded as a few learned tokens without changing SDXL.
**Failure.** No morph. The conditioning pathway alone cannot carry the correction.

---

### Group D — Attention-routing rewrites

**What the group does.** Touch nothing in the score-space pipeline and nothing in SDXL's weights. Instead, intercept the cross-attention map at each layer and overwrite its spatial pattern with a target supplied by a small predictor.

**Shared operations.**
- A custom attention processor that, in capture mode, records joint-prompt attention; in inject mode, overwrites per-concept attention's spatial pattern using a target from the predictor.
- Predictor trains against cached joint-prompt attention maps with JSD or KL, not against ε-residuals.

**What distinguishes techniques within the group.** Currently a single deployable technique. Has its own thread in [routed-poe-attention.md](routed-poe-attention.md); listed here as the fourth group so the design space is closed.

#### D1. Attention-mask substitution (PRMP)

**What it is.** A small predictor reads `(z_t, t, E_cat, E_dog)` and outputs per-concept spatial masks at each cross-attention layer's resolution. The masks are written into the attention pattern at inference, replacing per-concept attention's spatial component with the predicted target.

**Input.** Per-concept text encodings, latent, timestep. No joint embedding (it is not consumed at inference).

**Output.** Per-concept spatial masks at every cross-attention layer's resolution (32×32 and 16×16 in SDXL).

**Formulation.** Replace `A_c[i,j,k]` with `mask_c^{(t,ℓ)}(i,j) · A_c[i,j,k] / Z`. Values `V` come from per-concept forwards; only the spatial pattern is overwritten.

**Implementation.** `set_attn_processor` on every `attn2` module. Predictor is a small conv encoder with per-resolution mask heads.

**Training.** Cache joint-prompt attention maps from one mono rollout; extract per-concept masks by summing the joint map over each concept's text tokens; train the predictor to match the cached masks under JSD or KL.

**Focus.** Spatial routing — *where* concepts land, not what they look like.

**What makes it different from Groups A–C.** Operates in attention space, not score space. No weight modification, no score-space network, no conditioning augmentation. The most surgical option in the menu.

**Success.** Reproduces the morph. If it works without help from Groups A–C, the failure is purely spatial routing.
**Failure.** No morph from D1 alone — the residual is not summarised by routing.

---

### Stage 3 recommended progression (bottom-up)

The order is the simplicity gradient. Each step is cheaper than the next; each escalation is licensed by a specific failure mode of the previous step.

1. **Cache-replay baseline.** Inject cached `r_t` directly at every step. Must reproduce Stage 1 exactly. If not, fix the cache before training anything.
2. **A1 — Latent-CNN corrector.** Cheapest learner in the menu. If it reproduces the morph, every further step is optional ablation rather than necessary research.
3. **A2 / A3.** Only if A1 plateaus and the plateau looks scale-specific (→ A2) or input-bottlenecked (→ A3).
4. **B1 / B2.** Parameter-efficiency floor of weight-space adapters. Either closes the morph (failure is low-dimensional) or rules out the "feature scaling is the issue" hypothesis.
5. **B3 — Cross-attention LoRA.** Already known to reproduce the morph. Run as the reference baseline against which cheaper groups are compared.
6. **B4 — Self-attention LoRA, alongside B3.** Adjudicates text-conditioning vs spatial-binding. Three runs (A1, B3, B4) on the same cell with the same target settle the locus-of-failure question.
7. **C1 — Prefix-token conditioning.** Only after B-level results land. Tests whether the correction can be delivered without touching SDXL weights at all.
8. **D1 — Attention-mask substitution.** Separate thread; not gated by Stage 3's order.
9. **B5.** Deferred indefinitely.

The three runs that produce the clearest research statement at minimum cost: **A1**, **B3**, **B4** — each with the same target, the same σ-windowed loss, the same qualitative success criterion. The pattern of which one morphs and which one doesn't says where in the model the correction belongs.

---

## What this plan does NOT include

- Cross-seed agreement, cross-pair generalisation, hold-one-out evaluation.
- Quantitative scoring (GroundingDINO, VQAScore, CLIP-on-Tweedie).
- Sched-M2, anchor-window samplers, schedule scans. Different contribution, different document.
- Hypernetworks emitting per-pair adapters. Multi-cell future work; this plan is one cell.
- Outcome-supervised training (DRaFT / DDPO). Addresses rollout drift across pairs, not single-cell residual fitting.
- Statistical comparisons across seeds. The hypothesis is qualitative; one seed is enough to test it.

---

## Order of operations

1. Render Stage 1 Figure 1.2 (mono-anchored λ walk). Existing tooling with the injection sign flipped. Verifies the morph runs in both directions.
2. Render Stage 2 Figures 2.1–2.4. All four are built from cached residuals plus a single attention-map capture.
3. Run the Stage 3 cache-replay baseline. Confirms the cached injection recovers Stage 1.
4. Train and qualitatively evaluate **A1** with σ-windowed loss.
5. Train **B3** (the existing cross-attention LoRA) and **B4** (the self-attention sibling) with the same target and schedule.
6. Compare the three decoded trajectories qualitatively. The pattern that emerges chooses what to do next.
