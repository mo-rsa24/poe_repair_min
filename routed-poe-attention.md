# Routed-PoE — attention-routing hypothesis: diagnostic + architecture

> **North star:** PoE fails on "cat and dog" because it never sees the joint prompt's
> spatial routing — the per-concept cross-attention maps inside the UNet that decide
> *where the cat goes* and *where the dog goes*. The residual `Δ_t = ε̃_J − ε̃_PoE`
> is the image of that missing routing measured at the very end of the pipeline.
> This plan tests the routing hypothesis with a cheap substitution diagnostic and,
> if it passes, deploys a small predictor that supplies the missing routing at
> inference time from per-concept embeddings only — no mono call, no LoRA on SDXL.

Slot into [phase0-consolidated-plan.md](../../Playground/PhD/poe_repair_min/phase0-consolidated-plan.md)
as **Thread D**, parallel to Thread B (M2 learning) and Thread C (D-series
structure diagnostics). Thread D is licensed independently of Phase 1 of Thread B:
the diagnostic is half a day of GPU and tells you whether to keep building Phase 2
(LoRA-on-Δ_t in ε-space) or replace it with attention-routing prediction.

---

## 1. The hypothesis, precisely

PoE's score:
```
ε̃_PoE = ε(x_t, t | ∅) + Σ_{c∈{cat,dog}} [ ε(x_t, t | c) − ε(x_t, t | ∅) ]
```
Each per-concept forward `ε(x_t, t | c)` runs SDXL with `E(c)` as conditioning.
Inside SDXL, each cross-attention layer computes
```
A_c(x_t, t)[i,j,k] = softmax_k( Q(x_t)[i,j] · K(E(c))[k] / √d )
out_c(x_t, t)[i,j] = Σ_k A_c[i,j,k] · V(E(c))[k]
```
where `(i,j)` indexes a spatial position at the layer's resolution and `k` indexes
a token in the conditioning sequence. **The attention map `A_c` is what decides
where the concept lands spatially.** Per-concept forwards see only their own
prompt's `K` — they have no information about co-occurring concepts.

The joint forward `ε(x_t, t | E("a cat and a dog"))` produces a different attention
map `A_J` per layer. The joint prompt's `K` carries co-occurrence binding from
CLIP/OpenCLIP pretraining; `A_J` splits the canvas — cat-tokens attend to one
region, dog-tokens to another. PoE never has access to `A_J`.

**The hypothesis:**
> If we substitute `A_J`'s per-concept spatial routing into PoE's cross-attention
> layers — *only* the routing, keeping per-concept values — PoE flips the basin.

If true, the deployable method is a small head that predicts the per-concept
spatial routing from `(x_t, t, E(cat), E(dog))` at inference time. This is:
- **Not M1** (M1 predicts the full joint embedding `ê_J`; deploying it at every
  step is mono-at-inference; restricted to the anchor window only).
- **Not M2a** (M2a tried to make `UNet(p*) = Δ̂` for a linear-combination target;
  the new target is a single attention map, which lies on the per-layer attention
  manifold by construction — no closure problem).
- **Not Phase 2's UNet-LoRA on Δ_t** (Δ_t is the symptom at the output of the
  pipeline; routing is the cause inside the pipeline; predict the cause).

---

## 2. Why this is the right level of description

Three pieces of evidence from the literature converge on attention routing:

- **StructureDiffusion** (Feng et al., 2022) — keeps the **full joint prompt's
  attention map M^t and queries Q** but mixes in per-noun-phrase values V. The
  *where* comes from joint; the *what* comes from sub-prompts. Training-free,
  per layer, per step. This is exactly the WHERE/WHAT split this plan adapts to
  PoE.
- **Attend-and-Excite** (Chefer et al., 2023) — fixes "missing subject" failures
  by `z_t ← z_t − α · ∇_{z_t} max_s [1 − max_{i,j} A_t^{(s)}[i,j]]`. The intervention
  is on cross-attention activations, not on scores. Compositional failures route
  through attention.
- **Divide & Bind** (Li et al., 2023) — `L_attend = −min_s TV(A_t^{(s)})`, plus a
  binding loss `L_bind = JSD(Ã^{(r)} ∥ Ã^{(s)})`. Same x_t-gradient template;
  binding is enforced on attention maps.

Every published fix for the compositional failure mode PoE produces operates on
**attention maps**, not on scores. Score-space residual learning (Phase 2 as
currently written) is downstream of where the literature says the problem lives.

---

## 3. Thread D — the plan, in five phases

### Phase D0 — Re-orient (1 pomodoro, no GPU)

- Read [phase0-consolidated-plan.md §3](../../Playground/PhD/poe_repair_min/phase0-consolidated-plan.md)
  and the M2a falsification in `m2-residual-diagnostic-plan.md` §1.
- Three sentences to recite before any GPU work:
  1. "PoE has no joint attention; the routing literature says that is the failure
     mode for missing-subject and attribute-mixing compositions."
  2. "The substitution test in Phase D1 is a *zero-training* diagnostic; if it
     fails, the routing hypothesis is dead and Phase 2 of Thread B is reinstated."
  3. "The architecture in D3 is small and well-supervised; if D1 passes and D3
     fits, deploy alongside M1 — replace neither sched-M2 nor Δ_t-injection in the
     anchor window."

### Phase D1 — Attention-routing substitution diagnostic (½ GPU-day)

The single load-bearing experiment of this plan. **If this fails, the entire
routing-hypothesis thread is dead and we return to Phase 2 LoRA as written.**

**Setup:**

For cat × dog, seeds `{4, 42, 123}`, at every denoising step t ∈ {0..49}:

1. **Cache joint attention.** Forward `UNet(x_t, t, E("a cat and a dog"))` once;
   register PyTorch forward hooks on every `CrossAttention` module. Save the
   attention map `A_J^{(t, layer)}` of shape `(H_layer · W_layer, T_J)` to disk.
   SDXL has cross-attention at 32×32 and 16×16 latent resolutions across ~70
   transformer blocks; total cache ~50 steps × 70 layers × ~1k spatial × ~77 tokens
   × fp16 ≈ a few GB per seed. (Verify exact count via
   `for m in pipe.unet.named_modules(): if isinstance(m, CrossAttention): ...`).
2. **Token-to-concept assignment.** Tokenize "a cat and a dog" via the SDXL
   tokenizer; record which sequence positions correspond to `cat` (positions
   {1} or {1,2} depending on tokenization), `and` (one position), `dog` (one
   or two positions). Store as `{cat: [idxs], dog: [idxs], and: [idxs]}`.
3. **Extract per-concept spatial masks from `A_J`.**
   ```
   mask_cat^{(t, layer)}(i,j) = Σ_{k ∈ cat_tokens} A_J^{(t, layer)}[i, j, k]
   mask_dog^{(t, layer)}(i,j) = Σ_{k ∈ dog_tokens} A_J^{(t, layer)}[i, j, k]
   ```
   Renormalise each to a probability over spatial positions (softmax-style or
   just divide by sum-per-token across spatial), at each layer.

**The substitution.** Run PoE forward passes (one per concept) but at every
cross-attention layer, **override** the attention output:
```
out_c(x_t, t)[i,j] = mask_c^{(t,layer)}(i,j) · (Σ_k A_c[i,j,k] · V(E(c))[k])
```
Equivalently — and cleaner to implement — replace
`A_c[i,j,k] ← mask_c^{(t,layer)}(i,j) · A_c[i,j,k] / Σ_{(i',j')} mask_c · A_c`.
Either way, the per-concept *spatial* attention is replaced by the joint-prompt's
spatial mask for that concept, while *which tokens within the concept are
attended* remains the per-concept forward's own choice. The values V come from
per-concept embeddings, not joint.

Decode and score with the §4 protocol from
[phase0-consolidated-plan.md §4](../../Playground/PhD/poe_repair_min/phase0-consolidated-plan.md):
GroundingDINO regime + VQAScore min(p_yes) + CLIP-on-Tweedie (proxy).

**Pre-committed pass criterion (D1 verdict):**

> Routed-PoE achieves `both_distinct` AND VQAScore-min ≥ zero + 0.7·(mono − zero)
> on ≥ 2 of 3 seeds. (Same threshold structure as D4-A in Thread C.)

| D1 outcome | Verdict | Next |
|---|---|---|
| Pass on ≥ 2/3 seeds | **Routing alone is the fix.** | Phase D2 (locality ablation) |
| Halfway (e.g. flips regime to `both_overlapping` only) | **Routing is necessary but not sufficient.** | Phase D2 + supplement with V-mixing as in StructureDiffusion |
| Fails (Routed-PoE ≈ PoE) | **Routing hypothesis is wrong for PoE.** | Abandon Thread D; return to Thread B Phase 2 as written |

**Kill criterion:** If `mono` itself fails to flip the basin on the headline cell
(it shouldn't — sched-M2 + ê_J is val cos 0.997), the cache or the §4 protocol is
broken; debug before interpreting D1.

### Phase D2 — Locality ablation (½ GPU-day, conditional on D1 passing)

Where in space and time does the routing actually have to be supplied?

**D2-Layer.** Substitute masks at:
- only down-blocks (deepest cross-attention layers)
- only mid-block
- only up-blocks (shallowest)
- only 32×32 layers, only 16×16 layers
- all layers (= D1 reference)

Six conditions × 3 seeds. Score per §4. Bar chart of VQAScore-min by condition.

**D2-Time.** Substitute masks at:
- only commit window t ∈ [5, 25]
- only pre-commit t ∈ [0, 5]
- only post-commit t ∈ [25, 49]
- all timesteps (= D1 reference)

Four conditions × 3 seeds. Mirrors D4-A-t from Thread C.

**Why this matters:** A predictor only has to be accurate on the layers/steps
where the substitution does work. If only the commit-window 32×32 layers move
the basin, the architecture in D3 collapses to ≤ 10 layers × 20 timesteps × 2
concepts × 32 × 32 — small enough to overfit a single cell trivially.

### Phase D3 — Architecture: PRMP (3–5 days, conditional on D1 passing)

**PRMP — Per-Resolution Mask Predictor.** Predicts per-concept spatial masks at
each cross-attention layer from per-concept embeddings only, at inference time.

#### Inputs at each denoising step t

- `x_t` — current latent, `4 × 128 × 128` (SDXL latent space).
- `t` — scalar timestep.
- `E_cat`, `E_dog` — per-concept text encodings, `77 × 2048` each (SDXL OpenCLIP).
- `e_pooled_cat`, `e_pooled_dog` — per-concept pooled embeddings, `1 × 1280` each.
- *Optionally:* `ê_J_pooled` from the already-trained M1, as auxiliary
  conditioning — does not violate mono-usage rules since `ê_J` is learned from
  sub-prompts and never decoded via UNet at inference.

#### Outputs

For each cross-attention layer ℓ in SDXL UNet (~70 layers across 32×32 and
16×16 resolutions):
- `mask_cat^{(t, ℓ)}` — `H_ℓ × W_ℓ` spatial probability map
- `mask_dog^{(t, ℓ)}` — `H_ℓ × W_ℓ` spatial probability map

Two masks per layer; one head per resolution since the spatial shape is shared
across all layers at a given resolution.

#### Architecture (one concrete proposal — Variant A: shared backbone)

```
                   ┌─────────────────┐
   x_t  (4×128×128)│                 │
                   │  Conv encoder   │
                   │  (stride-2 ×2)  │── downsample to 32×32, 16×16, 8×8 features
   t   ──FiLM──────►│                 │
                   └────────┬────────┘
                            │
   E_cat, E_dog ──cross-attn ──► resolution-specific feature maps
   (per-concept queries)
                            │
                   ┌────────▼────────┐
                   │ Per-resolution  │── Output: 2 channels (cat, dog) per
                   │ mask heads      │   resolution; softmax over spatial
                   │ (conv + softmax)│
                   └─────────────────┘
```

- Conv encoder: small (e.g. `[4, 64, 128, 256, 512]` channels, ~5M params).
- Time conditioning: FiLM gain/bias from a sinusoidal-then-MLP timestep embedding.
- Concept conditioning: at each resolution, a single cross-attention block with
  Q = spatial feature, K=V = concatenated `[E_cat; E_dog]` (77 + 77 tokens) with
  a learned concept-type embedding added to disambiguate. This is *intentionally*
  symmetric: PRMP must decide which concept goes where without prompt order
  baked in.
- Mask heads: 1×1 conv → 2 channels → softmax-over-spatial (per concept), at
  each of {32×32, 16×16}.

Per-layer broadcast: the same `mask_cat^{(t, 32)}` is reused across all SDXL
cross-attention layers at 32×32 resolution. Justification: D2-Layer should
reveal whether per-layer specialisation is needed (it usually isn't — masks at
the same resolution tend to look alike in cached `A_J`).

#### Variant B (fallback if A is too coarse)

If D2-Layer shows that layers at the *same* resolution want materially different
masks (e.g., shallow layers attend more diffusely than deep layers), add a
per-layer adapter: a tiny 2-layer MLP per cross-attention block that takes the
shared resolution-level mask and a layer-id embedding and outputs a layer-specific
mask. Adds < 1M params total.

#### Training

- **Targets.** From D1's cached `mask_cat^{(t,ℓ)}` and `mask_dog^{(t,ℓ)}` for the
  joint forward, per seed, per step, per layer. Already computed in D1.
- **Loss per layer/step/concept.** Soft mask is a probability over spatial
  positions; use JSD or KL between predicted and target. Sum or average across
  (layer, step, concept).
- **Regularisers.** (1) Light TV on predicted masks to discourage speckle;
  (2) entropy floor so masks don't collapse to a single pixel; (3) symmetry
  penalty: swapping `(E_cat, E_dog)` inputs should swap `(mask_cat, mask_dog)`
  outputs.
- **Schedule.** Train on cat × dog seed 42 alone first (overfit cell);
  inspect predicted masks vs cached masks visually; only then train multi-seed
  on `{4, 42, 123}`.
- **Optimiser.** AdamW, lr 1e-4, batch = one (step, seed, layer-group) at a
  time. ~5k steps for single-cell overfit; ~20k for multi-seed.

#### Inference

For each PoE forward pass at step t:
1. Run PRMP once: `(x_t, t, E_cat, E_dog) → {mask_cat^{(t,ℓ)}, mask_dog^{(t,ℓ)}}`
   at each resolution.
2. Hook every SDXL cross-attention layer. Inside each hook, for the per-concept
   forward `c ∈ {cat, dog}`, apply the spatial mask `mask_c^{(t, layer)}` as in
   the D1 substitution.
3. Compose scores via standard PoE.

No mono call. No LoRA on SDXL. Frozen SDXL UNet weights throughout. Compatible
with M1's sched-M2 anchor — PRMP supplies routing at *every* step, M1 supplies
the joint pooled embedding *only inside the anchor window*; they are
complementary.

### Phase D4 — Integration + cross-seed (3–5 days, conditional on D3 fitting)

- **D4-A.** Run PRMP-substituted PoE on seeds `{4, 42, 123}`, headline cell.
  Compare to D1's cached-mask Routed-PoE (the ceiling). Pre-committed pass:
  PRMP closes ≥ 0.7 × (cached-mask − PoE) gap on VQAScore-min on ≥ 2/3 seeds.
- **D4-B.** Hold-one-seed-out. Train on `{4, 42}`, test on `123`; rotate. Pass:
  cross-seed transfer ≥ 0.5 × the within-seed ceiling.
- **D4-C.** Ablate against M1: PoE alone, PoE+M1-anchor, PoE+PRMP, PoE+M1+PRMP.
  Verify M1 and PRMP are complementary (joint anchor + per-step routing) rather
  than redundant.

---

## 4. Visualizations — what each diagnoses

| Figure | What you plot | What it diagnoses | Phase |
|---|---|---|---|
| **V1 — Attention map disagreement** | Side-by-side `A_PoE^{cat}` vs `A_J^{cat}` at chosen layers, t ∈ {5, 15, 25, 35}; same for dog. 2×4 grid per concept. | Whether PoE's routing is even wrong, and where. If PoE concentrates cat in the same spot as dog, you've literally seen the chimera mechanism. | D1 |
| **V2 — Mask divergence over time** | Two curves: KL(mask_PoE^{cat} ∥ mask_J^{cat}) per t, and same for dog. Average across layers; show 32×32 and 16×16 separately. Dashed line at commit window edges. | When in the trajectory the routing diverges. Expected: low early (broad masks), spikes inside commit window, decays after. Tells you which timesteps the predictor must be accurate at. | D1 |
| **V3 — Substitution result grid** | 3 rows (seeds 4/42/123) × 3 cols (PoE / Routed-PoE / Mono). Each cell: decoded 512×512 image + detection-regime disc + VQAScore-min number + CLIP-on-Tweedie (grey, small). | The headline result. Two filled discs and rising VQAScore-min from PoE → Routed-PoE → Mono = pass. | D1 |
| **V4 — Per-layer ablation bars** | One bar group per layer-substitution condition (down-only, mid-only, up-only, 32-only, 16-only, all). Y = VQAScore-min, averaged over seeds. Threshold line at 0.7·(mono − zero). | Which layer subset carries the routing signal. Determines whether PRMP needs per-layer heads. | D2-Layer |
| **V5 — Per-window ablation bars** | Four panels (pre-commit / commit / post-commit / all-steps), each with 3 seed groups × 4 conditions (PoE, Routed-PoE, mono, zero). Mirrors D4-A-t exactly. | When in the trajectory routing has to be in place. Determines whether PRMP needs to be accurate at all t or only inside commit. | D2-Time |
| **V6 — Predicted vs cached mask** | 4×2 grid: rows are chosen timesteps, columns are predicted-cat / target-cat / predicted-dog / target-dog. One row of difference maps below. | Whether PRMP has learned the target, layer by layer. Diagnostic during D3 training, not for the headline. | D3 |
| **V7 — PRMP vs ceiling vs PoE bars** | Three bar groups per seed (PoE baseline / PRMP-substituted PoE / cached-mask Routed-PoE / mono ceiling). Y = VQAScore-min. | How much of the routing ceiling PRMP captures. The closing-the-gap metric. | D4-A |
| **V8 — Complementarity grid** | 2×2: rows are M1-anchor on/off, cols are PRMP on/off. Each cell: VQAScore-min averaged across seeds + detection-regime strip. | Whether M1 and PRMP are independent contributions (off-diagonal not equal to diagonal) or redundant. | D4-C |

---

## 5. Honesty caveats

- **Token-axis mismatch.** Joint prompt tokens ≠ single prompt tokens. The
  proposal collapses the token axis by summing attention over concept-related
  tokens to get a spatial mask, then re-applies that mask multiplicatively to
  per-concept attention. This is a deliberate dimension reduction; *some*
  information in `A_J` (which token of "cat and a dog" the cat-region attends
  to most) is discarded. If D1 fails marginally, this projection is the first
  suspect — consider keeping token-level structure via a per-token mask family
  rather than a single spatial mask per concept.
- **"And"-token routing.** `A_J` has attention mass on the "and" token too;
  PRMP drops it. If the basin-flip is partially carried by the "and" token's
  spatial pattern (which would be a publishable mechanism finding on its own),
  the dropped-and ablation will reveal it. Add this as a sanity panel in V1.
- **SDXL has cross-attention at 32×32 and 16×16 only.** No 64×64 cross-attn.
  PRMP only needs heads at two resolutions. Verify in code before scaffolding.
- **GroundingDINO confidence is logit-style, not calibrated.** Carry the same
  caveat from Thread A captions.
- **N=3 seeds for cross-seed claims.** Pre-committed thresholds defend ≥ 2/3
  passes; marginal 2/3 is suggestive, not decided. Phase D4-B's hold-one-out is
  the only honest cross-seed test until more seeds are cached.
- **Mono use stays oracle-only.** D1 caches `A_J` from a single mono forward
  per step at training time. Inference (D4 onward) never calls mono. Consistent
  with [mono_usage_rules](../projects/-home-mscluster-mmolefe-Playground-PhD-poe-repair-min/memory/mono_usage_rules.md).
- **This is not StructureDiffusion.** StructureDiffusion uses the joint prompt at
  runtime. Routed-PoE doesn't — it learns to predict the joint's routing from
  sub-prompts. The mechanism is borrowed; the inference path is novel.

---

## 6. Decision matrix (cross-thread)

| Trigger | Verdict | Action |
|---|---|---|
| D1 passes on ≥ 2/3 seeds | Routing hypothesis confirmed | Continue to D2 + D3; **demote Thread B Phase 2 LoRA to backup** |
| D1 produces `both_overlapping` not `both_distinct` | Routing necessary but values matter too | D2 as planned, plus add V-mixing (StructureDiffusion-style average) as a second arm in D3 |
| D1 fails (Routed-PoE ≈ PoE) | Routing hypothesis dead | Abandon Thread D, reinstate Phase 2 LoRA on Δ_t |
| D2-Layer shows only deep 32×32 layers matter | Predictor is shallow | PRMP Variant A is sufficient |
| D2-Layer shows per-layer specialisation | Predictor needs per-layer heads | PRMP Variant B (per-layer adapter) |
| D2-Time shows commit-window-only suffices | Predictor only trains on commit-window targets | ~⅓ the supervision; correspondingly smaller PRMP |
| D3 single-cell PRMP JSD > 0.1 after 5k steps | Architecture binding | Try DiT-style backbone or larger; if still failing, the masks aren't predictable from sub-prompts and the deployment claim fails |
| D4-A PRMP-substituted PoE ≪ cached-mask Routed-PoE | Architecture caps the recoverable signal | Investigate: token-level masks instead of spatial-only, or distillation from cached masks with a stronger backbone |
| D4-C shows M1 + PRMP redundant | One of them is doing the other's job | Drop the redundant one from deployment; report finding |
| D4-C shows additive gains | M1 anchors basin commit; PRMP shapes per-step routing | Deploy both; this is the new contribution |

---

## 7. Cost summary

| Item | Compute | Wallclock |
|---|---|---|
| D0 re-orient | none | 30 min |
| D1 — cache `A_J` (3 seeds × 50 steps × 70 layers via hooks) | 1 GPU | ~1 h |
| D1 — substitution sampling (3 seeds × 3 conditions) | 1 GPU | ~2 h |
| D1 — §4 scoring (GroundingDINO + LLaVA already installed for Thread A) | 1 GPU | ~30 min |
| **D1 total** | **1 GPU** | **~½ day** |
| D2-Layer (6 conditions × 3 seeds) | 1 GPU | ~3 h |
| D2-Time (4 conditions × 3 seeds, reuse latent schedule) | 1 GPU | ~2 h |
| D3 — PRMP single-cell overfit (cat × dog seed 42, 5k steps) | 1 GPU | ~½ day |
| D3 — PRMP multi-seed train (20k steps) | 1 GPU | ~1 day |
| D4-A — eval on 3 seeds | 1 GPU | ~2 h |
| D4-B — hold-one-out × 3 rotations | 1 GPU | ~½ day |
| D4-C — complementarity grid | 1 GPU | ~3 h |
| **Total to a deployable PRMP + integration** | **1 GPU** | **~5–7 days** |

D1 alone is the load-bearing spend. ½ GPU-day decides whether to keep going.

---

## 8. Implementation notes

- **Hooking cross-attention in SDXL diffusers.** The `CrossAttention` /
  `Attention` modules in `diffusers.models.attention_processor` accept custom
  processors via `unet.set_attn_processor(...)`. Write a `RoutingHookProcessor`
  that (a) in capture mode, records `A` per layer to a dict keyed by `(t, layer_id)`;
  (b) in inject mode, replaces the spatial pattern of `A` with a supplied mask
  before the `attn @ V` multiplication. Use a context manager to switch modes.
  Reference: `diffusers/src/diffusers/models/attention_processor.py`, the
  `AttnProcessor2_0.__call__` signature.
- **Layer enumeration.** SDXL UNet exposes `unet.attn_processors` keyed by
  string paths like `down_blocks.1.attentions.0.transformer_blocks.0.attn2`.
  Only `attn2` modules are cross-attention; `attn1` is self-attention — do not
  hook self-attention.
- **Latent schedule pinning.** Same wiring requirement as D4-A-t in Thread C:
  every condition in D1 / D2 must share initial latent and DDIM step sequence
  per seed, or inter-condition variation in the `zero` reference will produce
  visible artifacts. Pin via the same mechanism as
  [scripts/run_veracity_phase_c.sh](../../Playground/PhD/poe_repair_min/scripts/run_veracity_phase_c.sh).
- **Cache layout.**
  ```
  outputs/routed_poe/cache/{seed}/A_J/step_{t:03d}/layer_{ℓ:03d}.pt
                                                  keys: A, token_idx_cat, token_idx_dog, token_idx_and
  outputs/routed_poe/cache/{seed}/masks/{t:03d}/{layer:03d}.pt
                                                  keys: mask_cat, mask_dog
  outputs/routed_poe/figures/{V1..V8}.png
  outputs/routed_poe/verdicts/d1.json, d2.json, d3.json, d4.json
  ```
- **PRMP code home.** New module
  `poe_repair/composers/routed_poe/`, mirroring the existing
  `poe_repair/composers/` layout. Trainer in `poe_repair/students/prmp/`.
  Reuse the §4 eval helpers in `poe_repair/experiments/veracity/metrics.py` —
  do not duplicate.
- **Caching script.** New `scripts/cache_joint_attention.py` taking `--seed`,
  `--prompt`, `--out_dir`; runs the joint forward and dumps all `A_J^{(t,ℓ)}`
  via the `RoutingHookProcessor` in capture mode.
- **D1 orchestrator.** New
  `poe_repair/experiments/routed_poe_d1/main.py` — runs the three conditions
  (PoE / Routed-PoE / mono), scores them, writes `d1.json` + the V1–V3 figures.

---

## 9. Quick links

- Parent plan: [phase0-consolidated-plan.md](../../Playground/PhD/poe_repair_min/phase0-consolidated-plan.md)
- M2a falsification rationale: `.claude/plans/m2-residual-diagnostic-plan.md` §1
- Shared eval protocol: phase0-consolidated-plan.md §4
- Detection + VQA helpers: [poe_repair/experiments/veracity/metrics.py](../../Playground/PhD/poe_repair_min/poe_repair/experiments/veracity/metrics.py)
- diffusers attention-processor API: `diffusers.models.attention_processor.AttnProcessor2_0`
- StructureDiffusion (the WHERE/WHAT split this plan adapts): https://arxiv.org/abs/2212.05032
- Attend-and-Excite (attention-loss precedent): https://arxiv.org/abs/2301.13826
- Divide & Bind (attention-loss precedent): https://arxiv.org/abs/2307.10864
- Bounded Attention (region-routing precedent): https://arxiv.org/abs/2403.16990

---

## 10. The one-sentence summary

Cache the joint prompt's per-concept spatial attention masks, substitute them
into PoE at every cross-attention layer, and if that flips the basin on ≥ 2/3
seeds, train a small per-resolution mask predictor that supplies those masks at
inference from per-concept embeddings only — replacing the score-space LoRA in
Thread B Phase 2 with an attention-space predictor that addresses the cause
rather than the symptom.
