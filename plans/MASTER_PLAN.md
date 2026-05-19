# Master Plan — Fixing PoE

This folder is the canonical orchestrator for the PoE-repair project.
Every research thread lives as a numbered sub-plan; the numbering is the
order of operations. Each sub-plan stands on its own and follows the
same shape: research question, the code we have, the commands to run,
and a taxonomy (poor / bad / unknown / good) for reading the result.

## The goal in one sentence

When SDXL is asked to compose two trained concepts ("a cat", "a dog")
via Product-of-Experts, it usually fails — the result is a chimera, a
single concept, or noise. We want to fix it.

## The line of reasoning

We work bottom-up. Each phase only justifies the next phase if it
passes. If a phase fails, we know which downstream phases stop making
sense.

1. **Phase 1 — Veracity.** Is the gap between PoE and Mono fixable *in
   principle*? We compute the oracle residual `r_t = ε̃_Mono − ε̃_PoE`
   and inject it during sampling. If injecting it walks PoE → Mono in
   image space, the repair problem is well-posed.
2. **Phase 2 — Residual diagnostics.** What is `r_t` made of? When in
   the trajectory does it matter? Where in image space does it live?
   This characterises the *target* a learner would have to predict.
3. **Phase 3 — Conditioning-window baseline.** Before claiming any LoRA
   contribution, we need to know how much SDXL alone — with the prompt
   masked at various step ranges — can already do. This is the
   no-residual baseline.
4. **Phase 4 — Single-seed LoRA.** The deployable artifact. A rank-8
   cross-attention LoRA trained on the cached PoE↔Mono residual for
   `cat × dog, seed 42`. Inference is Mono-free. This is the headline
   success.
5. **Phase 5 — Group-A failures.** Three external-corrector
   architectures (latent CNN / latent UNet / frozen-feature MLP) that
   try to predict the residual *outside* the UNet. Reported negatively.
6. **Phase 6 — Internal-force failures.** Two Mono-free PoE-internal
   forces (attention-overlap repulsion, score-alignment damping)
   reported negatively.
7. **Phase 7 — Cross-seed residual diagnostics.** Take the Phase-2
   characterisation and ask whether `r_t` is the *same kind of object*
   across seeds. Decides whether a single LoRA could plausibly cover
   many seeds.
8. **Phase 8 — Cross-seed LoRA.** Train one LoRA on a pool of seeds and
   evaluate on held-out seeds. Either pooling works (and we know *what*
   the LoRA learned, from Phase 7) or it doesn't.

## How the phases gate each other

```
Phase 1 (veracity)
   │  proves the gap is reachable from the PoE trajectory
   ▼
Phase 2 (diagnostics)
   │  pins down the target's shape, timing, and locus
   ▼
Phase 3 (CFG-mask baseline)
   │  measures what the prompt alone can already do
   ▼
Phase 4 (LoRA, single seed)
   │  proves the residual is learnable inside the UNet
   ▼
Phases 5 + 6 (failure cases, in parallel)
   │  contrast — external + internal alternatives don't reach the LoRA
   ▼
Phase 7 (cross-seed diagnostics)
   │  decides whether the target generalises across seeds
   ▼
Phase 8 (cross-seed LoRA)
        deployable cross-seed corrector — or a documented dead end
```

## What we keep fixed

- **Pair.** `a cat × a dog` only. Other pairs are out of scope for
  this project.
- **Seed-axis discipline.** Single seed 42 for Phases 1–6 (the
  beachhead). Phases 7–8 broaden along the seed axis but keep the pair
  fixed.
- **Definition of the residual.** Always in *guided* ε-space:
  `r_t = ε̃_J − ε̃_PoE`. Never raw UNet output. Equivalent up to the
  guidance scale by the PMI identity.
- **Mono is a diagnostic ceiling, not a deployable method.** It uses
  the literal joint prompt `e_J` at inference; the whole point of
  composition is to avoid that.
- **Cache root.** Training caches live under
  `/datasets/mmolefe/poe_repair_min/outputs/training_cache/`, resolved
  via `POE_REPAIR_TRAINING_CACHE`.

## What's done as of 2026-05-19

| Phase | Status |
|---|---|
| 1 Veracity | Cached on disk + figures rendered. |
| 2 Residual diagnostics | Existence + CLIP-window subexperiments executed. |
| 3 Conditioning-window | Schedule sweep + inspector route landed. |
| 4 LoRA single seed | Trained, probes saved, inspector route landed (`outputs/lora/cat_dog/seed_42/`). |
| 5 Group-A failures | Three architectures run; reported negatively. |
| 6 Internal-force failures | Two forces run; reported negatively. |
| 7 Cross-seed Δ_t structure | Executed at N=8; `landing_6` (Δ_t consistent with seed noise at cross-seed mean). |
| 8 Cross-seed LoRA pooling | Plan written; not yet executed. |

## How to read this folder

- [MASTER_PLAN.md](MASTER_PLAN.md) — this file.
- [01-veracity.md](01-veracity.md) — oracle correction sanity check.
- [02-residual-diagnostics.md](02-residual-diagnostics.md) — existence + CLIP-window characterisation.
- [03-conditioning-window-baseline.md](03-conditioning-window-baseline.md) — no-LoRA CFG-mask sweep.
- [04-lora-single-seed.md](04-lora-single-seed.md) — the deployed single-seed LoRA.
- [05-group-a-failure.md](05-group-a-failure.md) — external corrector failures.
- [06-internal-force-failure.md](06-internal-force-failure.md) — PoE-internal force failures.
- [07-cross-seed-delta-structure.md](07-cross-seed-delta-structure.md) — is Δ_t a seed-invariant signal?
- [08-cross-seed-lora-pooling.md](08-cross-seed-lora-pooling.md) — pooled LoRA across seeds.

Each sub-plan has the same five sections: **Question**, **Why this
phase exists**, **Code**, **Commands**, **How to read the result
(poor / bad / unknown / good)**. Anything that doesn't fit those five
sections is not in the sub-plan.

## Sources superseded by this folder

- `lora-success.md` (root) — superseded by 04.
- `residual-diagnostics.md` (root) — superseded by 02.
- `group-a-failure.md` (root) — superseded by 05.
- `internal-force-failure.md` (root) — superseded by 06.
- `conditioning-window.md` (root) — superseded by 03.
- `.claude/plans/cross-seed-lora-pooling.md` — superseded by 08.
- `.claude/plans/delta-t-structure-or-noise.md` — superseded by 07.
- `claude/veracity-figure-plan.md` — superseded by 01.

The originals are kept as reference; this folder is the operational
spec.
