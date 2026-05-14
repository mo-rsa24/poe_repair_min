# PoE correction without mono — full design space, in plain English

## What this document is

Every option for fixing PoE at inference (without ever calling mono) lives somewhere in a three-axis design space:

- **Where** in the model the correction lives.
- **What form** the corrector takes.
- **What signal** trains it.

Each axis has multiple options. A concrete method is one choice from each axis. Most write-ups conflate the three; this document keeps them separate so each can be evaluated and falsified independently.

Every option below is described the same way:

- **What it is** — one-paragraph plain description.
- **The bet** — the hypothesis about PoE's failure that this option commits to.
- **Inputs** — what tensors and signals the option consumes.
- **Output** — what it produces, at the resolution the sampler consumes.
- **How it is formulated** — the math, simply.
- **How it is implemented in practice** — what you would write in code.
- **What happens during training** — forward, target, loss, backward.
- **What makes it different** — the nearest alternative and why this isn't that.

The architecture menu in [poe-correction-mvp.md](poe-correction-mvp.md) picks one path through this space. Everything not on that path is described here too, so the boundary is visible.

The deployed sampler is always the same: at every DDIM step, PoE produces `ε̃_PoE`, the corrector contributes something, the two combine into the final ε that DDIM consumes, and `z_{t-1}` follows. Mono is never called. What changes between options is only the corrector's location, form, and supervision.

---

## Decision A — Where the correction lives in the model

This decision is a bet about *why* PoE fails. Each location targets a different culprit.

### A1. Score-space sibling network

**What it is.** A small standalone network that lives next to SDXL but never touches it. At every denoising step, it produces a correction tensor that you add to PoE's prediction before the DDIM update.

**The bet.** PoE's score is wrong by an amount a small learnable function can predict from the current state. You don't need to change how SDXL works internally — you just need to add to its output.

**Inputs.** The current latent `z_t` (shape 4×128×128), the timestep `t` (scalar), and the joint text embedding `e_J` for "a cat and a dog" — the same conditioning a hypothetical joint call would have used.

**Output.** A correction tensor of the same shape as ε (4×128×128). Added to `ε̃_PoE` to produce the corrected prediction.

**How it is formulated.** `r̂_t = f_θ(z_t, t, e_J)`, then `ε̃_corrected = ε̃_PoE + r̂_t`. The function `f_θ` is a small CNN, UNet, or MLP-on-features (Decision B picks which).

**How it is implemented in practice.** A small PyTorch module is defined once. The sampler is wrapped so that after PoE computes its prediction at each step, the wrapper calls `f_θ(z_t, t, e_J)`, adds the output, and passes the sum to DDIM. SDXL stays frozen end-to-end.

**Training.** Cache `r_t = ε̃_J − ε̃_PoE` at every step along the PoE trajectory once (one full PoE rollout plus one full mono rollout with shared latents). At each training step: sample `t` uniformly, look up the cached `(z_t, r_t)`, run `f_θ`, compute MSE against `r_t`, backprop into `f_θ` only. About a million parameters, trains in hours.

**What it focuses on.** Closing the score-space gap directly. It does not look at attention, does not modify SDXL's weights, and has no opinion about *where in the UNet* the failure originates.

**What makes it different.** Unlike LoRA approaches, SDXL itself is never touched — memory is low and there is no risk of breaking SDXL's pretrained behaviour. Unlike attention-substitution approaches, the corrector doesn't need to know anything about attention maps; it just learns the score-space gap.

---

### A2. Cross-attention LoRA on the frozen SDXL UNet (M5)

**What it is.** Small low-rank weight matrices are inserted into SDXL's cross-attention projection layers (`attn2.to_q`, `attn2.to_k`, `attn2.to_v`). Running the UNet with these adapters active gives a slightly different prediction; the difference between LoRA-on and LoRA-off forwards is the correction.

**The bet.** PoE fails because the text-conditioning pathway is wrong. The cross-attention layers are where the text embedding meets the latent; perturbing those layers fixes the composition.

**Inputs.** Same as the standard SDXL UNet — `(z_t, t, e_J)`. Both forward passes (LoRA-on and LoRA-off) take the same inputs.

**Output.** A corrected ε prediction. The correction itself is `r̂_t = ε_LoRA(z_t, t, e_J) − ε_frozen(z_t, t, e_J)`.

**How it is formulated.** For each cross-attention projection `W`, replace `W` with `W + α A B` during the LoRA-on forward, where `A` is `rank × d_in`, `B` is `d_out × rank`, `rank` is typically 8, and `α` is a small scalar. The two forwards differ only in this substitution.

**How it is implemented in practice.** Use the existing `peft` or `diffusers` LoRA injection utilities. Identify all `attn2` modules in the UNet, attach rank-8 adapters, freeze everything else. The sampler is wrapped to run both forwards per step.

**Training.** Two SDXL forwards on cached `(z_t, t, e_J)` — one LoRA-on, one frozen. Target is cached `r_t`. MSE between `(ε_LoRA − ε_frozen)` and the target. Backprop into the `A` and `B` matrices only. About 5 million parameters; days per cell at the speeds we measured.

**What it focuses on.** Modifying how text tokens influence spatial features. The adapter weights live exactly where text-to-latent meeting happens.

**What makes it different.** Unlike the score-space sibling (A1), the corrector inherits SDXL's pretrained prior — it is learning a *perturbation* of an already-correct function rather than learning the function from scratch. Unlike a full fine-tune (A6), only the rank-r matrices change, which is a strong regularisation. Unlike self-attention LoRA (A3), it targets the text-reading pathway specifically.

---

### A3. Self-attention LoRA on the frozen SDXL UNet (new)

**What it is.** Identical to cross-attention LoRA in every respect except the hook target — adapters go on `attn1.to_q`, `attn1.to_k`, `attn1.to_v` instead. Self-attention layers decide how spatial positions interact with each other, independent of the text.

**The bet.** PoE fails because spatial binding is wrong — the two concepts collide on overlapping regions of the canvas because the latent's internal spatial structure is mis-arranged. Fixing how positions attend to each other fixes the composition, regardless of text conditioning.

**Inputs.** Same as A2.

**Output.** Same form as A2 — a corrected ε prediction. The correction is `ε_LoRA(z_t, t, e_J) − ε_frozen(z_t, t, e_J)` exactly as in A2.

**How it is formulated.** Same as A2 but with `attn1` in the hook set instead of `attn2`.

**How it is implemented in practice.** Identical code path to A2 with one regex changed (the module-name pattern that selects which layers get adapters).

**Training.** Identical to A2.

**What it focuses on.** Adjusting how spatial patches attend to each other. Self-attention is the pathway by which "this region of the canvas is one object, that region is another" gets internally represented.

**What makes it different.** A2 bets the failure is *what is being named*. A3 bets the failure is *where things go*. Running both as siblings against the same target is the cheapest way to adjudicate which hypothesis is right. If both help additively, the failure has two independent components — a separately publishable finding.

---

### A4. AdaGN / DiffFit — normalisations and biases only (new)

**What it is.** Don't touch any convolution or attention weight. Only tune the `γ` and `β` parameters of every GroupNorm and LayerNorm in the UNet, plus every linear and conv bias. A few hundred thousand parameters total — tiniest UNet-touching option.

**The bet.** PoE's outputs are scaled or shifted slightly wrong, channel-by-channel. The failure isn't in routing or attention or token interactions — it's that features come out at the wrong magnitudes after normalisation.

**Inputs.** Same as A2 — `(z_t, t, e_J)`.

**Output.** Same as A2 — a corrected ε. The correction is the difference between the DiffFit-on and DiffFit-off forwards.

**How it is formulated.** GroupNorm produces `γ · (x − μ)/σ + β`. DiffFit makes `γ` and `β` trainable while keeping all other parameters frozen. Add the linear biases to the trainable set.

**How it is implemented in practice.** Walk the UNet's module tree. For each `GroupNorm` and `LayerNorm`, mark `weight` and `bias` trainable. For each `Linear` and `Conv2d`, mark `bias` trainable. Everything else: `requires_grad = False`.

**Training.** Same target and loss as A2 (cached `r_t`, MSE). Backprop into only the few hundred thousand trainable parameters. Significantly faster per step than LoRA because gradient computation skips most weights.

**What it focuses on.** Per-channel feature magnitude. It cannot change what features the UNet computes — only how big each channel is allowed to be after normalisation.

**What makes it different.** Far fewer parameters than LoRA. If this works, the failure is a feature-scaling problem — a much simpler diagnosis than "attention routing is wrong." If it does nothing, that is positive evidence the failure needs the attention pathway specifically.

---

### A5. Attention-map substitution (PRMP / Routed-PoE)

**What it is.** Don't change any SDXL weights. At each cross-attention layer, intercept the attention map (the softmax output that decides which spatial position attends to which text token) and overwrite it with a target spatial pattern. A small predictor learns to produce that target from sub-prompts.

**The bet.** Only the spatial structure of attention is wrong. The values flowing through the attention head are fine; what's broken is *where* they get routed.

**Inputs.** The predictor sees `(z_t, t, E_cat, E_dog)` — per-concept text encodings, not the joint embedding. At inference there is no joint prompt available.

**Output.** Per-concept spatial masks at each cross-attention layer's resolution — `mask_cat^{(t, ℓ)}` and `mask_dog^{(t, ℓ)}`, each a probability distribution over spatial positions at that layer's resolution (32×32 or 16×16 in SDXL).

**How it is formulated.** Replace the attention softmax output `A_c[i,j,k]` with `mask_c^{(t,ℓ)}(i,j) · A_c[i,j,k] / Z`, where `Z` is the renormalisation. The values `V` come from the per-concept forward; only the spatial pattern is overwritten.

**How it is implemented in practice.** Use `diffusers`' `set_attn_processor` API to install a custom `AttnProcessor` on every `attn2` module. In capture mode (training data collection), the processor saves the joint prompt's attention maps. In inject mode (inference), it multiplies the per-concept attention by a mask supplied by the predictor.

**Training.** Cache the joint prompt's attention maps at every (step, layer) for a few seeds. Extract per-concept masks by summing the joint map over each concept's text tokens. Train the predictor (a small conv encoder plus per-resolution heads) to match the cached masks with JSD or KL loss.

**What it focuses on.** Spatial routing of concepts. Nothing else.

**What makes it different.** Operates in attention space, not score space — most closely matches the mechanism the compositional-generation literature (StructureDiffusion, Attend-and-Excite, Divide & Bind) identifies as the failure mode. Touches no SDXL weights. Requires careful infrastructure to hook every cross-attention layer and to synchronise the predictor's forward with the UNet's.

---

### A6. Full UNet fine-tune

**What it is.** Initialise from pretrained SDXL UNet. Train every parameter. The model's output is no longer noise prediction — it is `r̂_t` directly.

**The bet.** Nothing structural. This is the brute-force default if every cheaper option fails.

**Inputs.** Same as A2.

**Output.** Directly the correction tensor `r̂_t` (not an ε).

**How it is formulated.** `r̂_t = UNet_finetuned(z_t, t, e_J)`. No subtraction, no decomposition. The fine-tuned weights *are* the residual function.

**How it is implemented in practice.** Mark every UNet parameter trainable; load pretrained weights; train on cached `(z_t, t, e_J, r_t)` tuples.

**Training.** MSE against cached `r_t`. Hundreds of millions of trainable parameters on one cell's worth of data; severe overfitting risk.

**What it focuses on.** Everything. There is no inductive bias.

**What makes it different.** No regularisation from a low-rank constraint. Once trained, the model can no longer be used as a noise predictor — a separate frozen SDXL is still required for the PoE marginals. Memory cost and irreversibility are both higher than LoRA. Deferred indefinitely on the MVP.

---

## Decision B — What form the corrector takes

This axis is largely orthogonal to Decision A. The forms below are how the function is parameterised; the location decides what space it operates in.

### B1. Small from-scratch CNN (M2b)

**What it is.** A 4–6 block convolutional network with FiLM conditioning. About a million parameters. No pretrained component.

**Inputs.** Raw latent `z_t` (4×128×128), timestep `t`, joint text embedding `e_J`.

**Output.** Correction tensor (same shape as the latent at the score-space location).

**How it is formulated.** Each conv block computes `feature ← γ(t, e_J) · Conv(feature) + β(t, e_J)`. Four to six blocks stacked, finishing with a 1×1 conv that maps to 4 output channels.

**How it is implemented in practice.** Standard PyTorch `nn.Module`. FiLM projection is a small MLP that takes `(sinusoidal_embed(t), mean_pool(e_J))` and produces per-channel scale and shift for every block.

**Training.** MSE on cached `r_t`, uniform `t` sampling. Trains in a few hours per cell.

**What it focuses on.** Spatial structure of the residual at the latent's native resolution.

**What makes it different.** Cheapest learner that respects spatial structure. No pretrained dependency to manage. The downside is no prior — every weight must be learned from one trajectory's worth of data.

---

### B2. Small UNet from scratch

**What it is.** A 2- or 3-scale encoder–decoder with skip connections, t-conditioned. Trained without pretrained weights.

**Inputs.** Same as B1.

**Output.** Same as B1.

**How it is formulated.** Standard DDPM-style UNet, smaller — about 5–10M parameters depending on channel widths.

**How it is implemented in practice.** Either write from scratch or shrink an existing tiny-UNet implementation (the small UNets used in toy CIFAR-10 diffusion experiments are a good starting point).

**Training.** Same as B1.

**What it focuses on.** Multi-scale residual structure — coarse "separate the animals" patterns *and* finer texture corrections at the same time.

**What makes it different.** Skip connections give the network coarse and fine views simultaneously, which a flat CNN doesn't. Useful when the residual has multi-scale structure. More parameters than B1 without a pretrained prior; risk of overfitting if the residual isn't actually multi-scale.

---

### B3. MLP on frozen UNet features

**What it is.** Run SDXL's frozen UNet on `(z_t, t, e_J)` and extract intermediate activations from a chosen hook point (typically the mid-block). Feed those features through a small dense network that outputs the correction.

**Inputs.** Same as B1, but the features pass through the frozen SDXL first.

**Output.** Same as B1.

**How it is formulated.** `r̂_t = MLP(extract(SDXL_frozen(z_t, t, e_J)))`. Conditioning is implicit — the SDXL forward already consumes `(t, e_J)`.

**How it is implemented in practice.** Register a forward hook on the chosen module; save its activations; flatten or pool spatially; pass through 2–4 dense layers; reshape output to match `r_t`.

**Training.** Same target and loss as B1, but every training step pays for a full SDXL forward to extract features. Slower per step.

**What it focuses on.** Mapping SDXL's internal representation of `(z_t, t, e_J)` to a residual.

**What makes it different.** Leverages SDXL's pretrained representation without modifying it. If the residual is a function of what SDXL already computes internally — and we would expect it to be — this has the right inductive bias. Cost: hook-layer choice is an unobvious hyperparameter, and every training step pays for a full UNet forward.

---

### B4. LoRA adapter on the pretrained UNet

**What it is.** Rank-8 (typically) low-rank matrices added to selected projection layers of the frozen UNet. Only the rank-8 matrices train. See A2 / A3 for the location-specific instantiations.

**Inputs.** Same as the UNet — `(z_t, t, e_J)`.

**Output.** Either an ε prediction (if the LoRA modifies the noise function) or the difference between LoRA-on and LoRA-off forwards (if used as a corrector).

**How it is formulated.** `W_effective = W_frozen + α A B` with `A: rank × d_in`, `B: d_out × rank`. Two forwards per training step, sharing the frozen path.

**How it is implemented in practice.** Use `peft` or `diffusers` LoRA utilities. Specify which module pattern (regex) to attach to. Mark only `A` and `B` trainable.

**Training.** MSE on cached `r_t`. Backprop into `A` and `B`. About 5M parameters at rank 8.

**What it focuses on.** A low-rank perturbation of whichever projection layers it hooks. The rank constraint encodes "the right answer is close to SDXL's current behaviour."

**What makes it different.** Inherits SDXL's pretrained prior; low parameter count by structural constraint; easy to swap on and off at inference. Higher infra cost than B1–B3 because every training step runs the full UNet.

---

### B5. IA³ — per-channel scales (new)

**What it is.** Even smaller than LoRA. Each chosen projection layer gets a single learned vector that multiplies its output element-wise along the channel dimension. No matrix decomposition — just one scale per channel.

**Inputs.** Same as B4.

**Output.** Same as B4.

**How it is formulated.** `out = (W · in) · s`, where `s` is a learned vector of size `d_out` and the multiplication is element-wise. Compared to a rank-1 LoRA, IA³ has the same number of trainable values but no input-side projection.

**How it is implemented in practice.** Custom hook on each target projection: multiply its output by a learned `nn.Parameter` of shape `(d_out,)`. Mark only those parameters trainable.

**Training.** Same target and loss as B4. Typically converges fast because the parameter space is so small.

**What it focuses on.** Per-channel gain on each projection's output. Cannot rotate or mix channels — only rescale them.

**What makes it different.** Floor of parameter efficiency — usually 100× fewer parameters than LoRA at rank 8. Useful as a sanity check: if even IA³ closes some fraction of the gap, the failure has very low-dimensional structure and LoRA is overkill. If IA³ does nothing, the right to escalate to LoRA is earned.

---

### B6. Learned prefix tokens (new)

**What it is.** Don't modify SDXL at all. Instead, prepend a small set of learned tokens (8–16) to the key/value sequences going into cross-attention. SDXL reads them as if they were extra text tokens, but they're trained parameters with no fixed meaning.

**Inputs.** Same as standard SDXL inputs, but at each cross-attention layer the K/V sequences are augmented with the learned prefix.

**Output.** A modified ε prediction. The correction is the difference between prefix-on and prefix-off forwards.

**How it is formulated.** For each cross-attention layer `ℓ`, learn `P_K^{(ℓ)}, P_V^{(ℓ)}` of shape `(n_prefix, d)`. At forward time, the K and V sequences become `concat([P_K, K], dim=seq)` and `concat([P_V, V], dim=seq)`.

**How it is implemented in practice.** Subclass the diffusers attention processor; concatenate the learned prefix to K and V inside the processor; mark only the prefix tensors as trainable.

**Training.** Same target and loss as B4. About `n_layers × n_prefix × d` parameters — typically smaller than rank-8 LoRA.

**What it focuses on.** Adding new conditioning channels that the model can "attend to," without changing how it attends.

**What makes it different.** SDXL's weights are completely untouched — closer to textual inversion than to LoRA. The mechanism is indirect (the model has to interpret the prefix as meaningful conditioning), which is a strength if the conditioning pathway is the right place to intervene and a weakness otherwise.

---

### B7. Hypernetwork emitting per-cell parameters

**What it is.** One shared network sees the prompt pair `(e_A, e_B)` and the timestep, and outputs the parameters of a corrector (LoRA matrices, prefix tokens, AdaGN scales — whatever parameterisation you chose) tailored to *this* prompt pair.

**Inputs.** Per-concept text encodings and the timestep. The hypernet does not see the latent.

**Output.** A complete set of corrector parameters for the chosen form.

**How it is formulated.** `θ_corrector = HyperNet(e_A, e_B, t)`. The corrector then operates as in B4/B5/B6 but with hypernet-supplied weights.

**How it is implemented in practice.** Design the hypernet (a few transformer layers or a deep MLP). At each inference step (or at the start of sampling, if `t` is dropped from the conditioning), run the hypernet once to populate the corrector, then run the corrector through the sampler.

**Training.** Multi-pair training. Cache `r_t` for many pairs. The hypernet learns a mapping from prompt-pair embedding to corrector parameters. Train end-to-end.

**What it focuses on.** Generalising across pairs without forcing one fixed corrector to cover all of them.

**What makes it different.** Natural generalisation step after a per-cell parameterisation works. Lets one shared module cover many pairs without negative transfer (because the parameters change per pair). Risk: the hypernet might memorise the training pairs and produce nonsense for unseen ones. Out of scope until per-cell is solid.

---

### B8. Full fine-tune of SDXL

See A6 — the location and the form collapse into the same option when every weight in the UNet is trainable.

---

## Decision C — What signal trains the corrector

This axis is the most overlooked. The same architecture can succeed or fail depending on the loss.

### C1. Cached ε-residual `r_t = ε̃_J − ε̃_PoE`

**What it is.** The corrector regresses to the difference between mono's score and PoE's score at every step. The target is cached once from offline rollouts.

**Inputs to the loss.** Predicted `r̂_t` from the corrector; cached `r_t` from disk; the timestep `t`.

**Output of the loss.** A scalar MSE per step, averaged over the batch.

**How it is formulated.** `L = mean_t( ‖r̂_t − r_t‖² )`. Sample `t` uniformly per batch.

**How it is implemented in practice.** Pre-compute `r_t` at every step along the PoE trajectory by running both PoE and mono once with shared latents. Store per-step `.pt` files. The training loader samples `t`, retrieves `(z_t, r_t)`, computes the prediction, and backprops.

**Training behaviour.** Stable and tractable. The gradient is non-degenerate from step one because the target is non-zero almost everywhere.

**What it focuses on.** The specific tensor the project is organised around — the diagnostic `r_t`.

**What makes it different.** Matches the diagnostic object directly. The target is small in magnitude (it's a residual, not the full score), which makes optimisation easier and makes low-rank regularisers like LoRA's rank constraint more meaningful.

---

### C2. Cached joint score `ε̃_J` itself

**What it is.** Skip the residual decomposition. Train the corrector to predict mono's full score `ε̃_J` directly; subtract `ε̃_PoE` at inference.

**Inputs to the loss.** Predicted score; cached `ε̃_J`.

**Output.** MSE against the full score.

**How it is formulated.** `L = mean_t( ‖ε̂_J,t − ε̃_J,t‖² )`. At inference: `ε̃_corrected = ε̂_J` (no subtraction).

**How it is implemented in practice.** Same training pipeline as C1 but with a different target tensor.

**Training behaviour.** Larger target magnitudes; the network spends capacity learning structure that PoE already gets right. Worse optimisation surface for the same final behaviour.

**What it focuses on.** Reproducing mono's score wholesale.

**What makes it different.** Mathematically equivalent at convergence to C1, but slower and more wasteful. Listed for completeness — to make explicit why C1 is the default.

---

### C3. Attention-map distillation alongside ε-matching (new)

**What it is.** Keep the ε-MSE from C1, and add a penalty term: the corrector also pays a JSD (or KL) between PoE's cross-attention maps and mono's cross-attention maps at chosen layers.

**Inputs to the loss.** Predicted `r̂_t`, cached `r_t`, the cross-attention maps produced when the corrected model runs, and the cached cross-attention maps from mono.

**Output.** A combined loss `L = ‖r̂_t − r_t‖² + λ · JSD(A_corrected, A_mono)`.

**How it is formulated.** Two losses summed with a weighting hyperparameter `λ`. The JSD is computed per layer, per step, then averaged.

**How it is implemented in practice.** Hook every cross-attention layer to capture attention maps during the corrected forward pass. Compare against cached mono attention maps. Add the JSD term to the existing MSE.

**Training behaviour.** Same as C1 but with extra gradient signal flowing back from the attention pathway. The corrector is now optimised both to close the score gap *and* to make the resulting attention look like mono's.

**What it focuses on.** Bridging score-space and attention-space objectives in a single head.

**What makes it different.** Couples Decision A1 (score-space) and Decision A5 (attention-space) — even though A5 has its own training story, C3 is the version that fuses them. Useful if score-space alone plateaus and you suspect attention is the source of the residual structure the corrector can't capture. Adds one hyperparameter (`λ`) that must be tuned.

---

### C4. σ-windowed loss (new)

**What it is.** Don't change the target — change the weighting. Only supervise where `‖r_t‖` is large. In practice that's the commit window — the middle of the trajectory, where PoE and mono disagree most.

**Inputs to the loss.** Same as C1, plus a per-step weight derived from `‖r_t‖` (or from a hard mask `1[t ∈ commit_window]`).

**Output.** Weighted MSE.

**How it is formulated.** `L = mean_t( w_t · ‖r̂_t − r_t‖² )` where `w_t ∝ ‖r_t‖` or `w_t = 1[t ∈ commit]`.

**How it is implemented in practice.** Either reweight the per-step MSE in the loss computation, or sample `t` non-uniformly during training (importance sampling). Both produce the same expected gradient.

**Training behaviour.** The network spends capacity where the residual is non-trivial. Loss curves on commit-window steps drop faster; loss curves on extreme-σ steps stay flat — at whatever value initialised them — and that's fine, because the cached `r_t` is small there anyway.

**What it focuses on.** The middle of the trajectory, where the residual lives.

**What makes it different.** Free upgrade — no new parameters, no new infra, no new hyperparameters beyond the window definition. Removes wasted capacity. Currently in the MVP as a contingency; the recommendation is to make it default.

---

### C5. Outcome-supervised loss (DRaFT / DDPO)

**What it is.** Don't regress to any cached tensor. Run the full sampler with the corrector inserted; decode the final image; score it with a downstream grader (a CLIP-based or VQA-based reward model); backprop the gradient through the sampler back to the corrector.

**Inputs to the loss.** The decoded image at the end of sampling; a reward signal from the grader.

**Output.** A scalar reward, maximised by the corrector.

**How it is formulated.** DRaFT: `L = -reward(decode(sample(corrector)))`, backprop through the entire sampling chain. DDPO: same loss but treated as RL — gradient comes from policy-gradient updates rather than direct differentiation.

**How it is implemented in practice.** Differentiable sampling — accumulate gradients through every DDIM step, with gradient checkpointing for memory. Choose a grader: VQAScore, CLIPScore, GroundingDINO confidence.

**Training behaviour.** Slow and expensive. Each training step is a full inference rollout plus a grader forward. But the gradient sees the actual deployed trajectory, which addresses rollout drift directly.

**What it focuses on.** The final image, not any intermediate tensor.

**What makes it different.** Every other loss here trains on cached `(z_t, target_t)` tuples from the PoE trajectory. At inference, the corrected sampler walks a *different* trajectory, so the corrector sees latents it was never trained on (rollout drift). Outcome supervision closes that gap — the corrector is optimised for what it actually produces, not for a cached target it might not be able to reproduce. Out of MVP scope, but named because it's the only honest answer to "cached and rollout don't see the same latents."

---

## Putting it together — concrete methods are one cell from each column

A method is a tuple (Where, Form, Signal). The current plans and recommendations land at:

| Method | Where (A) | Form (B) | Signal (C) | Status |
|---|---|---|---|---|
| MVP M2b (current default) | A1 score-space | B1 from-scratch CNN | C1 ε-residual | running |
| MVP M5 (current LoRA bet) | A2 cross-attn | B4 LoRA | C1 ε-residual | running on single cell |
| MVP M2b + σ-window | A1 score-space | B1 from-scratch CNN | C1 + C4 | recommended addition |
| Self-attn LoRA sibling | A3 self-attn | B4 LoRA | C1 + C4 | recommended addition |
| AdaGN floor | A4 norms only | (norm tuning) | C1 + C4 | recommended addition |
| Routed-PoE / PRMP | A5 attention map | (mask predictor) | (JSD on cached masks) | separate thread |
| IA³ sanity floor | A2 or A3 | B5 IA³ | C1 | deferred |
| Prefix-token alt | A2 | B6 prefix tokens | C1 | deferred |
| Attention-aware M2b | A1 score-space | B1 from-scratch CNN | C1 + C3 | deferred |
| Hypernetwork (multi-cell future) | A2 or A3 | B7 hypernet → LoRA | C1 across many pairs | future |
| Outcome-supervised (drift fix) | any | any | C5 instead of C1 | future |
| Full fine-tune | A6 full UNet | B8 full | C1 | deferred indefinitely |

Three cheap recommended runs on the seed-42 beachhead: **MVP M2b + σ-window**, **cross-attn LoRA + σ-window**, **self-attn LoRA + σ-window**. Same target, same signal, three different bets about *where*. The pattern of which wins (or whether they win additively) settles the locus-of-failure question.

---

## What this document deliberately leaves out

- **Detection and outcome metrics.** GroundingDINO regimes, VQAScore, CLIP-on-Tweedie — these gate eval, not training, and live in [phase0-consolidated-plan.md](phase0-consolidated-plan.md) §4.
- **Cross-seed / cross-pair experiments.** Out of scope per the seed-42 beachhead. See [.claude/projects/-home-mscluster-mmolefe-Playground-PhD-poe-repair-min/memory/project_seed42_beachhead.md](.claude/projects/-home-mscluster-mmolefe-Playground-PhD-poe-repair-min/memory/project_seed42_beachhead.md).
- **Rollout-drift mitigation beyond outcome supervision.** Other answers exist (trajectory augmentation, on-policy caching) but aren't on the spine of "show correction is possible on one cell."
- **Mono usage at inference.** Forbidden by [memory/mono_usage_rules.md](.claude/projects/-home-mscluster-mmolefe-Playground-PhD-poe-repair-min/memory/mono_usage_rules.md). Every option here respects that — mono appears only as a cached oracle.
