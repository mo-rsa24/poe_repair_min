# Decision Timeline — Fixing PoE with a LoRA residual corrector

The linear account of what the objective was, what ran, what it said, and the decision it forced. This is the spine; downstream plans and write-ups backfill against it.

**Thesis in one line.** SDXL Product-of-Experts composition (`ε̃_PoE = ε̃_A + ε̃_B − ε̃_∅`) usually fails (chimera / single concept / noise). A rank-8 cross-attention LoRA, trained on the cached guided residual `r_t = ε̃_J − ε̃_PoE`, closes the PoE→Mono gap at inference **without ever encoding the joint prompt** (Mono-free).

**The pyramid (deployment-unit question at each tier).** Overfit → Survive-Noise → Cross-Pair → Group-Wise → Scale. Each tier widens the held-out set and asks: what is the reusable unit — a cell, a pair, a group, or one LoRA for the whole taxonomy?

## How to read this log

- Each gate: **Question / Ran / Result / Decision / Fired next**, tagged `[tier · date · status]`.
- **Status:** ✅ done · ◑ partial/in-progress · ⏸ trained-not-evaluated · ⚠️ open · ❎ negative result (kept as evidence).
- **Dates:** firm from 2026-05-14 on (checkpoint/cache mtimes + git). Early-May research gates (G01–G03) are `~approx` — their artifacts are regenerable and not in git, so dates are reconstructed from the cache-build era and are confidence-low. Operations gates are firm (2026-07-21).
- **Append-only.** Later results supersede earlier ones with a banner; nothing is rewritten.
- Noise runs (false-starts, benign sync-fatals) are **excluded** unless they changed direction; the one that did (`0y9un0o4`) is called out in G11.

---

## G01 — Is the PoE→Mono gap reachable at all? [Overfit · ~2026-05-06 · ✅]

- **Question.** Is the gap between broken PoE and the Mono ceiling fixable *in principle* — can you walk PoE → Mono from the PoE trajectory?
- **Ran.** Oracle correction: compute `r_t = ε̃_Mono − ε̃_PoE` and inject it during sampling (veracity experiment; `plans/01-veracity.md`).
- **Result.** Injecting the oracle residual walks PoE → Mono in image space. The repair problem is well-posed. *(Date approximate; artifacts regenerable, not in git.)*
- **Decision.** A learner that predicts `r_t` is worth building. Proceed to characterise the target.
- **Fired next.** G02.

## G02 — What is the residual made of, and when does it matter? [Overfit · ~2026-05-07 · ✅]

- **Question.** What is `r_t`'s shape, timing, and locus — i.e. what would a learner have to predict?
- **Ran.** Residual diagnostics: existence (well-defined + structured) and CLIP-window (commitment window) sub-experiments (`plans/02-residual-diagnostics.md`).
- **Result.** `r_t` is structured (not noise) and has a commitment window in the trajectory. Defines the target and the magnitude curve later used to sanity-check the LoRA.
- **Decision.** The target is learnable-shaped. Before crediting a LoRA, measure what the prompt alone can already do.
- **Fired next.** G03, G04.

## G03 — How much can the prompt alone do, with no residual? [Overfit · ~2026-05-13 · ✅]

- **Question.** With CFG masked at various step ranges on clean SDXL (no LoRA, no Mono), how much composition is already reachable? The no-residual floor.
- **Ran.** Conditioning-window CFG on/off mask sweep, cat×dog seed 42 (`plans/03-conditioning-window-baseline.md`; code restored under `conditioning_window` at git C13, 2026-05-18).
- **Result.** A minimum conditioning window still yields a recognisable cat+dog; establishes the floor the LoRA's marginal effect is measured against (same x_T reused).
- **Decision.** Floor established. Any LoRA claim must beat this, not just beat raw PoE.
- **Fired next.** G04.

## G04 — Can a rank-8 cross-attn LoRA close the gap, Mono-free? [Overfit · 2026-05-14 · ✅ HEADLINE]

- **Question.** Can a rank-8 LoRA on `attn2.{to_q,to_k,to_v}`, trained only on the cached guided residual, drive PoE → Mono at inference without ever encoding the joint prompt?
- **Ran.** Train on cat×dog seed 42; loss `MSE(ε̃_PoE_lora, ε̃_J_cached)`; deploy via `run_lora_residual_inject` (adapter off/on per step, `Δ̂ = ε̃_PoE_lora − ε̃_PoE_frozen`, `ε = ε̃_PoE_frozen + λ·Δ̂`). Headline checkpoint `lora_step_062500.pt` (mtime 2026-05-14), ~600 epochs (`plans/04-lora-single-seed.md`).
- **Result.** λ=0 is byte-identical to vanilla PoE (canary passes). λ=1 morphs the chimera into two distinct animals by ~epoch 500–600. Masked sampler at `all_on` ≡ standard sampler at λ=1 (Δ=0). Reaches ~40% of the PoE→Mono latent distance and is still moving. **The deployable result exists on the beachhead cell.**
- **Decision.** The residual is learnable inside the UNet. This is the project's headline artifact. Now (a) prove it's not trivially achievable by alternatives, and (b) test whether it travels.
- **Fired next.** G05 (contrast), G06 (seed-axis diagnostic).
- **Evidence.** `artifacts/rung1-overfit/lora/a_cat__x__a_dog/seed_42/run__local/checkpoints/lora_step_062500.pt` — loads, 420 LoRA keys, shape (8,640) (verified `inventory/scripts/03_integrity.py`, 2026-07-21).

## G05 — Do external / internal alternatives reach the LoRA? [Overfit · ~2026-05-17 · ❎ negative]

- **Question.** Can the residual be predicted *outside* the UNet (external correctors) or via Mono-free PoE-internal forces, instead of a LoRA?
- **Ran.** Group-A: latent-CNN, latent-UNet, frozen-feature-MLP external correctors (`plans/05-`docs/results-archive/group-a-failure.md`, W&B `poe-repair-group-a`, three runs finished). Internal-force: attention-overlap repulsion + score-alignment damping (`plans/06-internal-force-failure.md`).
- **Result.** All five fail to close the gap. The training runs completed; the *finding* is failure.
- **Decision.** The LoRA result is meaningful — not something any corrector achieves. Report the failures negatively as the contrast set; scale the LoRA, not the alternatives.
- **Fired next.** Justifies G07+ scaling of the LoRA specifically.

## G06 — Is the residual the same object across seeds? [Survive-Noise · ~2026-05-19 · ◑]

- **Question.** Is `Δ_t` a seed-invariant signal (so one LoRA could cover many seeds), or seed-idiosyncratic?
- **Ran.** Cross-seed Δ_t structure diagnostic at N=8 (`plans/07-cross-seed-delta-structure.md`).
- **Result.** Landed close to "seed noise" at the cross-seed mean (`landing_6`): detectable but small shared structure.
- **Decision.** Pooling might only recover a weak seed-mean. Don't assume it works — test pooling empirically before believing cross-seed generalisation.
- **Fired next.** G07.

## G07 — Does a seed-pooled LoRA generalise to held-out seeds? [Survive-Noise · 2026-05-20 · ✅ on G6]

- **Question.** Train one LoRA on a pool of cat×dog seeds; does it compose on held-out seeds, and is it recovering the seed-mean or something seed-conditional?
- **Ran.** Pooled trainer (`cross_seed_lora_pooling`), G6 run `k04__ep2000_resumed` (W&B `pueuo7bl`, seeds {1..4} pooled, ep 2000 / step 100000), `verdict.json = "ok"` (`plans/08-cross-seed-lora-pooling.md`).
- **Result.** The pooled run completed cleanly to ep 2000 (verdict ok); checkpoint intact. *(The held-out-seed compose read is the eval; Task-D bridge / held-out-pair eval carved to Plan 12, code-ready but not run.)*
- **Decision.** Cross-seed pooling is viable enough on G6 to test per-group. The per-group cross-seed and within-group cross-pair questions become the next tier.
- **Fired next.** G08, G09.
- **Evidence.** `artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/taskB__k04_ep2000_resumed__wandb-pueuo7bl/` (verdict ok; ckpt loads).

## G08 — Is the single-seed mechanism pair-generic or cat×dog-specific? [Overfit breadth · 2026-05-20 · ◑ partial]

- **Question.** Does the Plan-04 mechanism close the gap on one representative pair per taxonomy group (G1–G4, G6), or is it a Group-6 (concept-collision) artefact ("split the chimera")?
- **Ran.** Single-seed LoRA per representative pair (`plans/09-lora-taxonomy-single-seed.md`). Trained: G4 `a_typewriter__x__a_cactus` (step 80000, mtime 2026-05-20) and G6 (inherited). Also exploratory `a_camel__x__a_desert_landscape` (W&B `8p1spi5b`).
- **Result.** **Partial.** Only G4 + G6 trained; G1–G3 not trained; MDS trajectory panels not run on G4. The "not Group-6-specific" claim is **not yet supported** across the taxonomy.
- **Decision.** Cannot retire the Group-6-specific worry yet. G1–G3 single-seed training still owed before the taxonomy-wide mechanism claim lands.
- **Fired next.** Blocks the clean read of G09/G10/G11.

## G09 — Does cross-seed pooling hold per group? [Survive-Noise per group · 2026-05-23 · ◑ in progress]

- **Question.** Per group (G1–G4, G6): does a seed-pooled LoRA on that group's representative pair generalise across seeds, and transfer to a sibling pair (`--heldout-pair`)?
- **Ran.** Per-pair `k04__ep2000` banks for G1–G4 (`cross_seed_lora_pooling/<pair>/`, mtime 2026-05-23) (`plans/10-cross-seed-lora-per-group.md`).
- **Result.** **In progress.** G1–G4 per-pair runs are part-trained (last ckpt ≈ step 50–60k of 100k, ≈ ep 1000–1200 of 2000); **no `verdict.json`, no Task-C ceiling, no Task-D, no contact sheets, no step0 prescreen for G2/G3/G4.** Held-out-pair driver landed but no sibling caches built.
- **Decision.** No per-group landing yet. The per-group cross-seed claim is unresolved; runs need finishing before any bucket read.
- **Fired next.** Feeds G10/G11 but is not yet a gate that closed.

## G10 — Is "group" a deployable pooling unit? [Group-Wise · 2026-05-27 · ◑ g6-smoke only]

- **Question.** Train one LoRA on 7 within-group pairs × all 12 seeds, hold out 3 pairs of the same group (pair axis only). Does it compose on held-out pairs, beating Plan-10's single-pair sibling smoke?
- **Ran.** `cross_pair_lora_pooling` within-group, group **g6 only** (`within_group/g6/main`, trained ~30k steps; mid-training crossbar eval `eval_crossbar/step_020000`, `in_in`+`out_in`, manifest `n_cells_planned = n_cells_sampled = 43`, 43 jsonl rows) (`plans/16-within-group-cross-pair-cross-seed.md`).
- **Result.** **g6 smoke only.** g1–g4 have YAMLs but no run dirs. The mid-training eval produced 43/43 cells; the full sample→contact-sheet→Task-D tail did not run.
- **Decision.** Rung not read — g6 is a smoke, not a per-group verdict. "Group is a deployable unit" remains untested for 4 of 5 groups.
- **Fired next.** Disambiguates G11 only once g1 (+ others) run.
- **Evidence.** `artifacts/rung3-group-wise/cross_pair/within_group/g6/main__wandb-ow1jo0xq/` (ckpt step 30000 loads; 43-cell crossbar verified).

## G11 — Can ONE LoRA span the taxonomy, held out on both axes? [Scale · 2026-05-26 · ⏸ trained-not-evaluated]

- **Question.** Train one rank-8 LoRA on 5 representative pairs × 8 seeds (40 cells); evaluate the four-quadrant crossbar (pair×seed, in/out). Headline quadrant: held-pair × held-seed (`out_out`).
- **Ran.** `cross_pair_lora_pooling/all_groups/main` (W&B main `2em6frqv`), per-step multi-pair trainer, trained to `lora_step_030000.pt` (`plans/11`, `plans/15-cross-pair-cross-seed-lora-mscluster.md`). A later run `0y9un0o4` **died early** — the one noise run that changed direction (below).
- **Result.** **Trained, never crossbar-evaluated.** `samples/` holds only `per_epoch/`; **no `cells.jsonl`** → `sample_crossbar` / `contact_sheet` / `task_d_bridge` never ran on it. The strongest quadrant (`out_out`) was **never sampled anywhere**. The checkpoint itself loads (bytes intact).
- **Decision.** **RE-RUN (finish training + run the crossbar).** The taxonomy-spanning "one LoRA" claim — the strongest form of the deployable contribution — **does not yet exist as a result**. The died-early `0y9un0o4` is load-bearing precisely because it left this state: trained-but-unevaluated.
- **Fired next.** Open re-run item; recorded in `plans/standing/artifact-reconciliation/`.
- **Evidence.** `artifacts/rung4-scale/cross_pair/all_groups/main__wandb-2em6frqv/` — ckpt loads; `samples/` has no `cells.jsonl` (verified 2026-07-21).

## G12 — Raw-latent MDS measures appearance, not co-occurrence [side-thread · ~2026-05-25 · ◑ reframed]

- **Question.** Does the latent-trajectory MDS panel show PoE+λ·R approaching Mono in the property we care about (co-occurrence of both animals)?
- **Ran.** MDS on flattened `z_t` (`build_lora_inspector_mds.py`); known failure case examined (λ=0.5 ep900 single fluffy cat sits *near* mono; λ=1.0 ep1600 chimera sits *farther*).
- **Result.** Raw-latent L2 orders configurations by pixel-appearance, not concept overlap — the ordering is backwards for the claim. Semantic (DINOv2 x̂₀) and latent-manifold (LSO pullback-metric) reframings proposed (`plans/13-semantic-mds-toggle.md`, `plans/15-latent-manifold-geometry.md`). `manifold_cache` reports **0/7 complete** (all cells missing eps).
- **Decision.** Don't trust raw-latent MDS as evidence of convergence. Reframe as semantic/manifold geometry; the geometry backfill is parked (0/7). Visualization side-thread, not on the corrector's critical path.
- **Fired next.** Parked; only `cat_dog` MDS panels exist (inspector).

---

## Operations tier — artifact reconciliation [all 2026-07-21]

## OPS-01 — What exists, who made it, what's its status? [✅]

- **Question.** Catalogue every run artifact across both roots and reconcile against W&B; which runs are suspect?
- **Ran.** `/data-inventory` + two-root classified sweep (`inventory/01-artifact-inventory.md`, `inventory/02-two-root-classified.md`, `inventory/scripts/01_inventory.py`).
- **Result.** 22G / 13,214 files; 4 W&B projects (`poe-repair-{lora,cross-seed,cross-pair,group-a}`). Flagged: 3 empty-looking `lora/` pair dirs, died-early (`0y9un0o4`, `ow1jo0xq`), false-starts (`lu7g7svh`, `9ux1sm67`), benign sync-fatal (`d5b2706v`). Heavy seed banks live on `/datasets`, light eval samples in repo (same pair name, different stage — not duplicates).
- **Decision.** Defines the "kept" set and the suspect list that OPS-02 integrity-checks.
- **Fired next.** OPS-02.

## OPS-02 — Do the kept artifacts load / are caches complete? [✅]

- **Question.** Load-test kept checkpoints + all suspects; caches complete; result sets match manifests?
- **Ran.** `/data-integrity-check` (`inventory/scripts/03_integrity.py`, `03b_cache_check.py`; report `inventory/03-integrity-and-disposition.md`).
- **Result.** **16/16 checkpoints load** (420 LoRA keys, shape (8,640)) — including all four suspects (bytes intact; suspect = run completeness, not corruption; `d5b2706v` intact at step 16510). Caches: 645 cells; 24 "1-shard" cells are **by-design eval stubs** (`build_eval_cache.py`), not truncation. Top-level manifests **stale** (declare 3 cells vs 645). `all_groups/main` **never crossbar-evaluated** (confirms G11). `manifold_cache` **0/7** (confirms G12). **Correction (supersedes OPS-01's "0-byte stubs" wording):** the three `lora/` stubs are **live symlink views** to cross_seed banks (kept); only `training_cache_overfit_catdog` is a genuinely-broken view (quarantined). `du`/`find -type f` had read symlink-only dirs as 0 bytes.
- **Decision.** **Keep** most; **RE-RUN** `all_groups` crossbar + `manifold_cache`; **regenerate** stale manifests; **quarantine** discards (false-start, dryrun, superseded, broken view). Read-only — recommend, don't delete.
- **Fired next.** OPS-03.

## OPS-03 — Make the artifact tree navigable + canonical [✅]

- **Question.** One naming/directory scheme keyed by rung→experiment→pair→seed so both roots are navigable, without breaking references?
- **Ran.** `/rename`: `inventory/04-canonical-layout-move-plan.md` + guarded `inventory/scripts/04_apply_layout.sh` (dry-run → apply); then a code-side `cat_dog → a_cat__x__a_dog` pass (paths, inspector slug + discovery, docs).
- **Result.** `artifacts/{rung1-overfit…rung4-scale, caches, _shared, _quarantine}` applied on both roots; within-root moves + compat symlinks at every old path; discards quarantined (reversible); re-run reports **0 residual actions**. Code repointed to canonical paths; inspector discovers/labels `a_cat__x__a_dog` once (no phantom); `cat_dog` fully retired in code + docs (memory `catdog-slug-shared-key`).
- **Decision.** Canonical spine established. Spawned the **`artifact-reconciliation` sub-scope** (`plans/standing/artifact-reconciliation/`) with 01–04 as done tasks + a standing "re-sweep on new runs" node, so this stays current.
- **Fired next.** Standing re-sweep (`plans/standing/artifact-reconciliation/plans/05-resweep-on-new-runs.md`).

---

## Coverage manifest (what this spine does NOT yet cover)

Honest gaps — do not read the log as "everything landed":

- **Open re-runs forced but not done:** G11 `all_groups` crossbar eval (`out_out` never sampled); G12 `manifold_cache` backfill (0/7).
- **In-progress tiers with no verdict:** G09 (per-group cross-seed, G1–G4 part-trained, no verdicts); G10 (within-group, g6 smoke only, g1–g4 not started); G08 (G1–G3 single-seed not trained, MDS not run on G4).
- **Quantitative claims without a regenerated real table here:** the G04 "~40% of PoE→Mono distance" and "two distinct animals by ep500–600" are eyeball/plan-stated, not re-verified with a metric table in this log. GroundingDINO/VQAScore gating was deferred in Plan 04. If a supervisor needs the number, that exhibit is `pending`.
- **Dates G01–G03 are approximate** (regenerable artifacts, not in git) — confidence low; firm dates begin at G04 (2026-05-14).
- **W&B not queried live** — statuses inherited from local run-dir reconciliation (OPS-01).
- **Not tiered here:** G5 (entanglement) deferred throughout; conditioning_window_lora "rescue" companion; the inspector web app (evidence surface, not a decision).

_Generated 2026-07-21. Append-only; supersede with a banner, never rewrite._

---

## OPS-04 — Where does non-rung output go? [✅ decided, ⚠️ move pending]

- **Question.** Eight experiment top-dirs sit outside the canonical `artifacts/` scheme. Do the diagnostics and mechanism-study dirs belong in rung1-4, or somewhere else?
- **Ran.** Re-sweep detection (`inventory/sweeps/2026-08-04-resweep-detection.md`, read-only) → scope call + load tests (`inventory/sweeps/2026-08-04-scope-call.md`) → `04_apply_layout.sh` extended, dry-run only.
- **Result.** Load tests all PASS. `animals_compose_transfer` headline `lora_step_100000.pt` is intact: 420 LoRA keys, rank-8, `step=100000`, `epoch=2000`. **Structural note that supersedes OPS-02's flat-420-key contract for this run:** the tensors sit under a `lora_state` sub-dict alongside optimizer and scaler state, so a loader written against the flat layout fails on it. `conditioning_window{,_lora}` hold no `.pt` files (figures only). `residual_diagnostics/delta_structure_unguided/tensors.pt` holds `delta` / `eps_poe` / `eps_mono` — the same quantity the interaction-term scope calls r_t.
- **Decision.** **Rung1-4 stays for ladder experiments only.** Three buckets added: `artifacts/diagnostics/` (`residual_diagnostics`, `conditioning_window{,_lora}`, `group_a_failure`), `artifacts/scopes/<scope>/` (`animals_compose_transfer`, `compose_scorer`, `poe`), and `artifacts/_shared/presentation/`. All eight **keep**; `group_a_failure` keeps as a reference negative (7G, revisit once the negative claims are written). Dry-run: 8 MOVE+LINK, 32 REFUSE (already filed), 3 SKIP (quarantined). The move itself is not yet applied.
- **Fired next.** `APPLY=1` on the layout script, then DoD-4 closes.

_Appended 2026-08-04._
