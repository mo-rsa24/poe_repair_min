# Hierarchical Plan — LoRA across the composition taxonomy

This is the master orchestrator for the *next* arc of LoRA work, built
on top of the single-seed and cross-seed results that already exist in
[04-lora-single-seed.md](04-lora-single-seed.md) and
[08-cross-seed-lora-pooling.md](08-cross-seed-lora-pooling.md). The two
existing plans cover one concept pair only (`a cat × a dog`, Group 6 of
the taxonomy below). The three new sub-plans push the LoRA mechanism
outward along two axes: *concept difficulty* (the taxonomy) and *seed
generalisation*.

## The arc in one sentence

Show that the rank-8 LoRA residual corrector, the one that closed the
PoE gap on the Group-6 beachhead, is a *mechanism* that travels across
the taxonomy and across seeds, not a Group-6-specific accident.

## The composition taxonomy

We partition concept pairs into six groups, ordered by (approximately)
increasing composition difficulty / PoE gap. The ordering itself is a
finding from prior diagnostic work; the new plans treat it as fixed.

| # | Group | Definition | Representative pair |
|---|---|---|---|
| 1 | Training support / natural co-occurrence | Pairs likely present in SD + CLIP training data, no attention competition. | `a dolphin × an ocean wave` |
| 2 | Factorization | Object + style. | `a dog × oil painting style` |
| 3 | Object + scene | Foreground + background; similar to G1 but unlikely in training data. | `a mailbox × a snowfield` |
| 4 | Dual-object composition | Two non-conflicting objects competing for spatial attention. | `a typewriter × a cactus` |
| 5 | Concept-pair entanglement | One concept drags in an entangled third concept (often a human). | *(deferred — see note below)* |
| 6 | Concept collision | Competing concepts produce chimeras. | `a cat × a dog` *(beachhead, done)* |

**G5 is deliberately not run in this arc.** The candidate pair
`a tuxedo × a flamingo` (and the other G5 entries — `a_wedding_dress__x__a_lobster`,
`a_fur_coat__x__a_goldfish`, etc.) carry a vague composition target:
the entangled third concept is not specified by the prompt, so the
ground truth "what should the joint trajectory look like" is itself
under-determined. Spending GPU-hours on a cell where the reference
itself is ambiguous would muddy every downstream read. G5 is kept in
the taxonomy for completeness but excluded from Plans 09–11; revisiting
it requires first nailing down a more precise prompt grammar for
entanglement pairs.

Pair source on disk:
`/datasets/mmolefe/neurips2026/pilot_5seeds_interaction/seed_<N>/group<G>_<slug>/<pair>/`.
The directory layout is **seed → group → pair**. Pair slugs use the
`a_X__x__a_Y` form (e.g. `a_typewriter__x__a_cactus`). The Group-6
beachhead `a cat × a dog` is *not* in the pilot tree — its cache and
LoRA artefact live under `outputs/lora/a_cat__x__a_dog/seed_42/` from
[04-lora-single-seed.md](04-lora-single-seed.md) and remain the
canonical Group-6 datum.

## The three new experiments

The arc is bottom-up along the taxonomy then outward along seeds.

```
        [04 single-seed cat×dog]   [08 cross-seed cat×dog]
                  │                          │
                  ▼                          ▼
        09 single-seed across taxonomy   (existing cross-seed work)
                  │
                  │  generalises along the *concept* axis (1 pair × 6 groups)
                  ▼
        10 cross-seed LoRA per group
                  │
                  │  generalises within each group along the *seed* axis
                  │  + within-group held-out *pair* transfer
                  ▼
        11 cross-pair × multi-seed LoRA
                       broadest claim: one LoRA, all groups, all seeds,
                       held out along both axes
```

Each new plan stands alone and follows the established five-section
shape (**Question**, **Why this phase exists**, **Code**, **Commands**,
**How to read the result**). Each also carries a **Research-objective
alignment** section justifying why the experiment is needed and what
claim it underwrites.

| Plan | Scope | Held-outs | Headline claim it supports |
|---|---|---|---|
| [09-lora-taxonomy-single-seed.md](09-lora-taxonomy-single-seed.md) | 1 representative pair per group × seed 42 — five pairs (G1–G4, G6; G5 deferred). | None (overfit per cell). | The LoRA mechanism is taxonomy-wide. The closed-PoE-gap result is not Group-6 specific. |
| [10-cross-seed-lora-per-group.md](10-cross-seed-lora-per-group.md) | 1 representative pair per group × multi-seed pool (G1–G4, G6); held-out-pair CLI flag for within-group transfer. | Held-out seeds + (optional) held-out pairs from same group. | Pooled LoRA generalises within each difficulty group along the seed axis, and transfers (or doesn't) to siblings of the trained pair. |
| [11-lora-cross-pair-cross-seed.md](11-lora-cross-pair-cross-seed.md) | Five groups (G1–G4, G6) × all available pairs × all seeds in pool; single LoRA. | Held-out pairs + held-out seeds (two-axis crossbar). | One LoRA generalises across the studied taxonomy as a whole. The corrector is a single object, not a per-cell library. |

## Shared infrastructure (new contracts)

All three new plans share the following pieces. They are listed once
here and not repeated in the sub-plans.

### MDS / PCA trajectory panels in the residual inspector

[04-lora-single-seed.md](04-lora-single-seed.md) introduced
`scripts/build_lora_inspector_mds.py` (five stages — `collect-static`,
`collect-cells`, `project`, `render`, `update-manifest`) and a residual
inspector tab that swaps pre-rendered PNGs in response to `(epoch, λ)`
slider changes. Plans 09–11 inherit and extend this contract:

- **One PNG per `(epoch, λ)` cell**, pre-rendered to disk. The inspector
  performs *no* in-browser computation.
- **Panel contents per cell.** Four denoising trajectories projected
  jointly into a shared 2D plane.
  - **Static** (do not change with sliders): `A`, `B`, `A ∧ B`. These do
    not depend on the LoRA, so they are baked once per pair from the
    no-LoRA samplers.
  - **Dynamic** (only thing that moves): `PoE + λ · R`. As the user
    scrubs the epoch and λ sliders, this arm visibly bends toward
    `A ∧ B` with more training and larger λ.
- **Style.** Reuse the look of
  `/home-mscluster/mmolefe/Playground/PhD/neurips2026/paper/figures/cut/trajectory_g1g4.png`
  per the reconstructed prompt — same palette, axis treatment, path
  styling, terminal letter labels. The MDS pre-renderer already uses
  the matched style from `render_taxonomy_paper_figure.py`.
- **Inspector placement.** Residual tab (tab 2 — the "LoRA residual"
  tab). The panel sits directly below the existing PoE-baseline /
  PoE-LoRA / mono image row and is wired to the *same* `epoch` and `λ`
  sliders that drive the row above.
- **Per-pair independence.** For every pair under
  `outputs/lora/<pair>/seed_42/results/`, the MDS panels are cached,
  rendered, and manifested **independently**:
  - Cache lives at `outputs/lora/<pair>/seed_42/results/mds_cache/`.
  - PNGs land at `outputs/lora/<pair>/seed_42/results/mds_probes/<epoch>/<λ>/mds.png`.
  - `inspector_manifest.json` for that pair carries its own `mds_cells`
    block.
  - The inspector's pair dropdown selects the pair; that surfaces *that
    pair's* MDS panels alongside *that pair's* decoded image row. No
    cross-pair coupling.

A one-shot wrapper that loops over every pair under
`outputs/lora/*/seed_42/` and runs the five-stage pipeline is on the
TODO list of [04-lora-single-seed.md](04-lora-single-seed.md); it is
the natural driver for plan 09.

### Pair-slug discipline

Single-seed LoRA artefacts go under
`outputs/lora/<pair-slug>/seed_42/results/`. Cross-seed pooled LoRA
artefacts go under
`outputs/cross_seed_lora_pooling/<pair-slug>/{task_b,task_c,...}/`.
Cross-pair pooled LoRA artefacts go under
`outputs/cross_pair_lora_pooling/<run-id>/`. The pair slug is the disk
slug from the pilot tree (e.g. `a_typewriter__x__a_cactus`), with the
historical exception of `a_cat__x__a_dog` which keeps its short name to avoid
breaking the existing manifest.

### Cache prerequisites

| Plan | Cache cells needed |
|---|---|
| 09 | Five single-pair caches at seed 42 — one per G1–G4, G6 representative pair (G5 deferred). |
| 10 | Per-group multi-seed pools (e.g. 8 seeds × representative pair) for each of G1–G4 and G6; plus held-out pair cells if `--heldout-pair` is exercised. |
| 11 | Full cross product — all pairs in scope × all seeds in pool, across G1–G4 and G6. |

The cache root resolves through `POE_REPAIR_TRAINING_CACHE`
(`/datasets/mmolefe/poe_repair_min/outputs/training_cache` on the
cluster). Cells are produced upstream and rsync'd, not regenerated in
this repo.

## What this arc keeps fixed

- **Residual definition.** Always in guided ε-space, `r_t = ε̃_J − ε̃_PoE`
  (see [residual_definition](../.claude/projects/-home-mscluster-mmolefe-Playground-PhD-poe-repair-min/memory/residual_definition.md)).
- **Mono is the diagnostic ceiling, never deployed.** All taxonomy and
  cross-seed/pair claims are against PoE; mono only appears as the
  static `A ∧ B` endpoint in MDS panels and as the offline cache target.
- **Inference is Mono-free** at every λ for every LoRA in plans 09–11.
- **Inspector remains pre-render-only.** Computation is in
  `build_lora_inspector_mds.py` (or its analogue); the running app
  only swaps PNGs.

## What this arc does *not* attempt

- Outcome supervision (DRaFT / DDPO / hypernet LoRAs).
- Architectural changes to the LoRA (rank, target modules, adapters)
  beyond what plans 10/11 sweep as optional follow-ons.
- Cross-task transfer outside the six-group taxonomy.
- Quantitative VQA / GroundingDINO gating as a primary read — eyeball
  contact sheets and MDS panels remain the headline; metrics are
  optional confirmation.

## Status — 2026-05-25

Disk-verified roll-up across plans 09, 10, 11 and the held-out-pair eval
that 12 carves off the G6 cross-seed checkpoint. Caches are confirmed in
`/datasets/mmolefe/poe_repair_min/outputs/training_cache/heldout/` and
artefacts under `outputs/lora/`, `outputs/cross_seed_lora_pooling/`,
`/datasets/.../cross_seed_lora_pooling/`, and
`outputs/cross_pair_lora_pooling/`.

| Sub-plan | Status | Evidence on disk |
|---|---|---|
| 09 — Single-seed across taxonomy | **In progress.** All five representative-pair caches at seed 42 exist (G1–G4 + G6, seeds {1..12, 42}); only G4 (`a_typewriter__x__a_cactus`) and G6 (`a_cat__x__a_dog`) have trained LoRAs. G1/G2/G3 training not started. MDS not run on G4. `build_taxonomy_lora.sh` wrapper not written. | `/datasets/.../training_cache/heldout/{a_dolphin,…,a_cat__x__a_dog}/seed_*` all present; `outputs/lora/a_typewriter__x__a_cactus/seed_42/results/checkpoints/lora_step_080000.pt` trained but `inspector_manifest.json` has no `mds_cells`; `outputs/lora/a_cat__x__a_dog/.../mds_probes/` has 47 (epoch, λ) cells. |
| 10 — Cross-seed per group | **In progress.** G6 done (`k04__ep2000_resumed`, verdict ok, ep 2000 reached). G1/G2/G3/G4 per-pair k04 runs are part-trained (last checkpoint ≈ step 50–60 k of 100 k, i.e. ≈ ep 1000–1200 of 2000) with no `verdict.json` yet. No `step0_prescreen` runs for G2/G3/G4. No Task C/Task D/contact sheets on any per-pair run. `sample_heldout --heldout-pair` flag landed, but no sibling-pair caches built and no held-out-pair driver script. | Per-pair runs at `/datasets/.../cross_seed_lora_pooling/{a_dolphin,a_dog__x__oil_painting_style,a_mailbox,a_typewriter}/task_b_learning_curve/k04__ep2000/checkpoints/` (6–7 ckpts each, no `verdict.json`, no `history.json`); G6 final ckpt at `/datasets/.../k04__ep2000_resumed/checkpoints/lora_step_100000.pt` with `verdict.json = "ok"`. |
| 11 — Cross-pair × multi-seed | **Not started.** No module, no scripts, no pool YAMLs, no run dirs. | `outputs/cross_pair_lora_pooling/` exists but empty; `poe_repair/experiments/cross_pair_lora_pooling/` does not exist; `scripts/cross_pair_lora_pooling/` does not exist. |
| 12 — G6 held-out-pair eval (carved off of 10) | **Code ready, runs not started.** Patches landed (`build_eval_cache.py`, `sample_heldout --heldout-pair`, `render_per_epoch --pair-slug-override`); no sibling-pair caches built; no held-out-pair samples or per-seed-summary figures rendered. | `scripts/build_eval_cache.py` present; `--heldout-pair` and `--pair-slug-override` flags wired in their files; `/datasets/.../training_cache/heldout/a_wolf__x__a_husky` absent; `samples/heldout_pair/`, `samples/per_epoch_heldout/`, `samples/per_seed_summary_heldout/` absent under the pueuo7bl run dir. |

### Cross-cutting state

- **MDS pipeline (shared infra).** Implemented and used on `a_cat__x__a_dog`
  (`outputs/lora/a_cat__x__a_dog/seed_42/results/mds_probes/` with 47 epoch dirs,
  `mds_cells` block in its manifest). **Not run on any other pair**, so
  the residual-inspector "PoE + λ·R bends toward A∧B" panel is currently
  visible only for G6. This is the unfinished MDS work — the latest
  `build_lora_inspector_mds.py` has not been driven over the G1–G4
  representative pairs.
- **Inspector.** `lora_inspector.py` still runs on port 5050. The pair
  dropdown surfaces whichever pairs have manifests; today that is
  a_cat__x__a_dog, typewriter (no MDS), camel (exploratory, not plan-09), and
  whatever else lives under `outputs/lora/*/seed_42/results/`.

### Three suspicions, evidence-checked

1. **"MDS work is incomplete — the latest pipeline command was not run to
   produce the new residual-inspector tab on the new pairs."** **Confirmed.**
   Only `a_cat__x__a_dog`'s manifest carries `mds_cells`. The G4 trained pair
   (`a_typewriter__x__a_cactus`) and the exploratory `a_camel` pair both
   have plain manifests without `mds_cells`. The MDS pre-renderer has
   not been invoked on any of the G1–G4 representatives.
2. **"Unsure whether the per-pair sampling / eval commands for plans 09
   and 10 have been executed."** **Partially executed.** Plan 09 has
   only G4 + G6 trained (no MDS on G4). Plan 10 has the four per-pair
   `k04__ep2000` runs *in progress* (≈ ep 1000–1200 of 2000) with no
   verdict, no Task C, no Task D, no contact sheets, and no
   step0_prescreen for G2/G3/G4.
3. **"Cross-pair LoRA (plan 11) has not been started."** **Confirmed.**
   `outputs/cross_pair_lora_pooling/` is empty, no
   `poe_repair.experiments.cross_pair_lora_pooling` namespace exists,
   no driver script. Plan 12 (held-out-pair eval of pueuo7bl) is the
   only cross-pair work in flight, and only as patched code — none of
   its sampling steps have been run.

## How to read this folder

- [PHASE_MAP.md](PHASE_MAP.md) — the original eight-phase
  orchestrator covering Phases 1–8 on `cat × dog`.
- [LORA_TAXONOMY_PLAN.md](LORA_TAXONOMY_PLAN.md) — this file. The
  next-arc orchestrator covering Plans 09–11.
- [09-lora-taxonomy-single-seed.md](09-lora-taxonomy-single-seed.md)
- [10-cross-seed-lora-per-group.md](10-cross-seed-lora-per-group.md)
- [11-lora-cross-pair-cross-seed.md](11-lora-cross-pair-cross-seed.md)
