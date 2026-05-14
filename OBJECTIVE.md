# Objective

## Problem

PoE composition assumes pixel-level conditional independence given the
concept set:

$$p(x \mid c_A, c_B) \propto p(x \mid c_A)\,p(x \mid c_B) / p(x).$$

In epsilon space:

$$\hat\varepsilon_{\text{PoE}} = \tilde\varepsilon_A + \tilde\varepsilon_B - \varepsilon_\emptyset.$$

This is wrong. The true joint $p(x \mid c_A, c_B)$ has structure that the
product of marginals omits. We call the gap the **interaction term**:

$$r_t \;=\; \tilde\varepsilon_J - \tilde\varepsilon_{\text{PoE}} \;=\; \tilde\varepsilon(x_t, t, e_J) - \tilde\varepsilon(x_t, t, e_A, e_B).$$

A primary failure mode of PoE is **slot collision** (a.k.a. hybridization):
both subjects receive strong attention but at the same spatial region, so
the model renders one chimera. Cat × dog at most seeds is the canonical
example.

## Hypothesis

We can train a small **synthesizer** $\hat e_J(e_A, e_B, e_\emptyset)$ on
pairs $(c_A, c_B)$ that have a literal joint encoding $e_J$ available.
Used at inference, $\hat e_J$ acts as a **proxy for the joint embedding**:
plugged into a sched-anchored sampler, it provides the interaction-term
signal that PoE omits, *without requiring* the joint prompt at deployment
time.

The scientific question is **generalization to held-out pair compositions**:

> When SDXL has the joint in its prior (so Mono works) but PoE alone does
> not access it, can $\hat e_J$ — *trained without seeing the held-out
> pair* — reconstruct enough of the joint signal to repair PoE?

If yes, we have shown that compositional structure is *predictable from
marginals* via a learned mapping. If no, the synthesizer collapses to the
PoE quality (no claim).

A reviewer's hard question: *"What about pairs SDXL itself has never
composed?"* — Mono fails on those too. Our scope is the **marginals → joint
gap when the joint exists in the model's prior but PoE doesn't access it.**

## Method

For each pair $(c_A, c_B)$ at inference, we compute one of:

| Method             | Anchor source     | Failure mode it addresses |
|---|---|---|
| PoE                | —                 | (the failure mode itself)  |
| Mono(literal e_J)  | `encode("a cat and a dog")` | upper bound; requires the joint prompt |
| sched-M2(literal)  | `encode("a cat and a dog")` | sanity (does the schedule architecture work?) |
| **sched-M2(ê_J)**  | `synthesizer(e_A, e_B, e_∅)` | **the headline claim** |
| sched-M2(ê_J)+FOCUS | ê_J + JS-divergence velocity correction | residual collision after ê_J |
| sched-M2(ê_J)+AAE | ê_J + iterative latent refinement | residual neglect after ê_J |
| CO3                 | runs unmodified per their pipeline | external SDXL-native baseline |

Published correction methods (FOCUS, AAE, CO3) fix Mono's residual failure
modes. We run them on top of $\hat e_J$ as well — the claim being that
*what fixes Mono should also fix $\hat e_J$* if $\hat e_J$ is a faithful
proxy for $e_J$.

## Faithfulness doctrine

We use the canonical implementations under `composition/` as the ground
truth for each method's algorithm.

- **CO3** is SDXL-native; we call `composition/debottam_co3/composers/Co3`
  directly without modification. Inputs follow their convention
  (`prompt_orig`, `prompt = "A+B"`, `seeds`).
- **FOCUS** ships against SD3 / FLUX. We re-implemented its algorithm
  faithfully on SDXL: JS-divergence loss with cubed-Q peakifier and σ=1.0
  Gaussian blur (matches `controller.py:return_storage`), velocity
  correction $\lambda \sigma(t)(1-t) \nabla_x f$ (matches
  `sd1/pipeline.py:1097-1103`), aggregation at `attn_res=16` (matches
  `controller.py:save_cross_attention`).
- **AAE** ships against SD 1.5 / 2.1. We re-implemented faithfully on
  SDXL: threshold ramp `{0:0.05, 10:0.5, 20:0.8}`, `max_iter_to_alter=25`,
  `max_refinement_steps=20`, scale_factor=20, scale_range=(1.0, 0.5) with
  $\sqrt{\cdot}$ (matches `pipeline_attend_and_excite.py:558`), σ=0.5
  attention smoothing, BOS exclusion + index shift (matches their
  `_compute_max_attention_per_index`).
- **P2P** ships against SD 1.5. The full P2P Re-weight machinery (2-prompt
  batch + cross-attn replacement) was *not* ported — instead we use the
  simpler single-prompt amplifier from the `AttentionReweight` cell of
  their notebook, gated by `cross_replace_steps=0.8`. This is documented
  in `composers/p2p.py` and may be dropped if reviewer faithfulness on P2P
  is critical.

Experiment **E4** runs AAE's *unmodified* SD 2.1 pipeline as a faithfulness
sanity check against our SDXL port.

## Code structure

The repository is organised one-to-one with the experiments:

```
poe_repair/
  config.py                  Run config + paths
  runtime.py                 SDXL primitives (encode_prompt, latents, scheduler)
  _sdxl/                     SDXL model loading
  embeddings/                Synthesizer training + inference (the M2 we improve)
  diagnostics/residual.py    attention_overlap (one function)
  methods/_sampling.py       7 reference samplers + helpers (one file, ~1100 LoC)
  composers/                 Thin wrappers — one per method
    mono.py poe.py sched_m2.py aae.py focus.py p2p.py co3.py
  experiments/
    e1_held_out.py           Headline result
    e2_synth_audit.py        Synthesizer reconstruction quality (embed + UNet)
    e3_mono_methods.py       Mono + corrections sanity (the ceiling diagnostic)
    e4_appendix_xstack.py    AAE on SD2.1 native — faithfulness check
  run.py                     Cached baseline dispatcher (solo_a/b, poe, mono)
checkpoints/synthesizer/     Trained synthesizer weights (load-bearing, 121MB)
composition/                 Cloned reference repos (read-only)
  aae/ focus/ p2p/ debottam_co3/
outputs/                     Generated artifacts (regenerated per run)
```

Total package: ~2,500 LoC.

## Run sequence

1. **E3 first** — confirm each correction method fixes Mono in our setup.
2. **E2 next** — measure synthesizer reconstruction quality (embed + UNet) on
   held-out pairs. If `seq_cosine` is < 0.95 or `unet_rmse` is implausibly
   high, the synthesizer is the bottleneck and the headline experiment will
   under-perform. Iterate on synthesizer training before E1.
3. **E1** — the headline. Held-out pairs × seeds × 7 methods.
4. **E4 (optional)** — appendix-grade AAE-on-SD2.1 vs AAE-on-our-SDXL
   qualitative comparison.
