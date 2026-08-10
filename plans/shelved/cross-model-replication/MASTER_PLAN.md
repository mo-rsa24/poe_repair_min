# Cross-Model-Replication

> Shelved: an empty scaffold for background work (zero plan files). Comes back when its parent
> plan is promoted out of the background pool into the paper table; /populate-plans it then.

## Mission
If the does-the-correction-cause-composition story holds only on SDXL with DDIM, it is a fact about
one model. This scope repeats the causal core on SD 1.5 and SD 2.1 and across
samplers, testing that the term exists, is noise-level-localized, and is
causally sufficient wherever text-to-image diffusion composes by PoE.

## Objectives
1. Per-model caches: residual trajectories on SD 1.5 and SD 2.1 (inference
   only, config parity with the SDXL cells, pinned latents per cell).
2. Per-model dose tests: the oracle λ-sweep with its controls on each model.
3. Sampler sweep: DDIM, DDPM, Euler at λ=0 and λ=1; the window compared in
   noise-level coordinates, not step index.
4. Stochastic extras: full per-concept density traces on the SDE runs, and
   the does-noise-alone-rescue read at λ=0.

## Goals
1. Caches pass the same bulk smoke as the SDXL cache.
2. One dose curve per model: support if the oracle rises while random stays
   flat on each model; divergence between models is reported, not hidden.
3. Window-in-SNR overlay: support if the peak band sits at the same noise
   levels across samplers; a sampler-dependent band falsifies universality
   and is reported as its own finding.
4. Density traces delivered for the SDE runs; the stochastic-rescue rate at
   λ=0 recorded either way.

## Expected Outcome
The paper's universality section: the same story on three models and three
samplers, or a precise statement of where it breaks.

## Definition of Done
1. SD 1.5 and SD 2.1 caches built and smoke-checked.
2. Dose curves per model with controls.
3. The sampler window overlay in log-SNR coordinates.
4. SDE density traces and the λ=0 stochastic-rescue read.

## Sub-Scopes
(none)

## Plans
(to be populated)

## Environment Context
See `docs/ENVIRONMENT.md` for this project's environment/architecture facts.
Read before drafting or checking any plan in this scope.
