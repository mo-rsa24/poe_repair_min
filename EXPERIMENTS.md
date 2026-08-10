> **Superseded.** This registry predates the plan tree. Experiments are designed in
> `plans/<scope>/plans/`, judged in `plans/<scope>/review/`, and governed by
> `~/.claude/EXPERIMENT_CONVENTIONS.md`. Kept because seven files cite its EXP numbers.

# EXPERIMENTS — LoRA-Fixes-PoE (publishable bar)

**Generated**: 2026-07-22 (experiment-planner over docs/RESULTS_SUMMARY.md + master Definition of Done).
**Schema**: provisional (no canonical EXPERIMENTS.md format present; reconcile if one appears). One block per pyramid rung.
**Claims**: no CLAIMS.md yet, so each experiment carries a `local:` claim tied to a master Objective.

## Locked claim (from the master mission)

A rank-8 cross-attention LoRA trained on the cached guided PoE→Mono residual `r_t = ε̃_J − ε̃_PoE` makes Product-of-Experts composition co-occur like Mono at inference **without the joint prompt (Mono-free)**, and this fix generalises along the seed and pair axes to a **characterisable reach** (at minimum a per-group catalogue, at most one taxonomy-spanning LoRA).

**Global three-way rule** (each rung sharpens it): *support* = Mono-free composition holds and generalises to held-out seeds for ≥1 group; *null* = the LoRA fails to compose at λ=1 on the trained cell, or held-out-seed pass ≤1/4 across all groups; *inconclusive* = composes on trained cells but held-out lands in-between → rerun with more seeds/pairs, do not narrate into a win.

## The publishable bar (the decision this file locks)

The binding constraint is **delivery, not transfer**: the trained cell plateaus at ~40% of the PoE→Mono distance. So the defensible paper is **not** "one LoRA spans the taxonomy" (Scale is likely a fallback). The **minimal publishable unit** is the arc:

> the fix works Mono-free (Rung 1) → survives seed noise per group (Rung 2) → transfers to unseen pairs within a difficulty group (Rung 4) → and the deployment unit is decided by the crossbar read (Rung 5), even if the honest answer is "ship the per-group catalogue."

| Rung | DoD | Publishable bar | Verdict | Gates the paper? |
|---|---|---|---|---|
| 1 Overfit | 1 | Mono-free composition on cat×dog **+ ≥3 more groups**, λ=0 canary byte-identical, MDS bend | ⚠️ cat×dog only + G4 (MDS owed) | **MANDATORY** |
| 2 Survive-Noise | 2 | Pooled LoRA composes on **≥3/4 held-out seeds for ≥3 groups** | ◑ G6 trained, held-out enactment pending | **MANDATORY** |
| 3 Cross-Pair | 3 | (smoke only; confounded per the plan) | ⚠️ not run | **OPTIONAL — recommend downgrading DoD-3** |
| 4 Group-Wise | 4 | Within-group pool composes on **≥2/3 held-out concept-disjoint pairs for ≥1 group**, beats single-pair, Task D cosine > baseline | ◑ G6 smoke only | **MANDATORY (the transfer claim)** |
| 5 Scale | 5 | Four-quadrant crossbar **evaluated and classified**; deployment unit chosen. A null (single-LoRA underperforms → catalogue) is publishable. | ⏸ crossbar never run | **MANDATORY to READ, not to win** |

**Recommended DoD change to lock**: downgrade DoD-3 (Cross-Pair) from a completion gate to an optional smoke. The plan file itself calls single-pair→sibling confounded and names Group-Wise (Rung 4) as the reviewer-credible transfer test. Keeping DoD-3 mandatory spends ~a day of runs on evidence a reviewer discounts. Your call; the blocks below treat it as optional.

## Must be re-run (crashed / missing / pending) — the execute set

Ordered by dependency. This is what steps 5–8 of the runbook schedule.

1. **Rung 5 crashed run** — `0y9un0o4` died early; **resume** `all_groups` from `2em6frqv/lora_step_030000.pt` (do NOT restart, the pool spec is unchanged), then run the crossbar (`cells.jsonl` absent → the gap).
2. **Rung 2 pending enactment** — the G6 held-out-seed samples (seeds 9–12) that make DoD-2 real; `pueuo7bl` is trained (verdict ok) but the held-out-seed proof was never rendered.
3. **Rung 1 missing MDS** — G4 `a_typewriter__x__a_cactus` MDS pre-render (owed from the old Plan 08).
4. **Rung 1 missing trainings** — G1/G2/G3 single-seed LoRAs (currently 0-byte stubs).
5. **Rung 2 unfinished** — G1–G4 pooled runs to ep2000 + verdicts (part-trained ~ep1000–1200).
6. **Rung 4 missing** — build the ~520 Plan-16 cache cells, then G1 (most-opposite to G6) end-to-end + G6 full tail (final-checkpoint crossbar + Task D).

Everything else in the rung plans is either done or optional (method extensions: Attend-and-Excite grafts, SLERP-merge, orthogonal adaptation).

---

## EXP-01: Overfit — Mono-free repair, pair-generic
- claim_id: local:obj1 (Overfit)
- independent_var: concept pair (cat×dog, then one representative per group G1–G4) and λ (0→1)
- dependent_var: recognisable composition (two concepts visible by eye / two-object VQA), PoE→Mono distance reached, MDS trajectory bend toward the joint target
- ablation_rows: λ=0 canary (must be byte-identical to plain PoE) · λ=1 full correction · negative controls (group-A external correctors, internal forces — already ✅, they fail)
- metric: recognisable-composition rate across the ≥5 pairs; the λ=0 canary is the Mono-free proof (DoD-6). Must move if the claim is true: a residual-trained LoRA that composes is the whole foundation.
- sample_size: 5 representative pairs (G1–G4, G6), seed 42, λ ∈ {0, 0.25, 0.5, 0.75, 1.0}; probes every 50 epochs to ~600
- falsify_condition: **support** if ≥4/5 groups compose by eye at λ=1 with byte-identical λ=0; **null** if cat×dog fails to separate at any λ, or ≤1/5 groups compose; **inconclusive** between → train the remaining groups before reading
- figures: per-pair epoch×λ morph (inspector LoRA-residual tab) + MDS bend (anti-corroboration: a pair whose corrected path does NOT bend toward the joint target)
- compute: mscluster, 1×GPU per pair, ~few h each; checkpoints under `/datasets/.../artifacts/rung1-overfit/`; MDS pre-render ~30 min
- status: ⚠️ pending (cat×dog ✅, G4 trained/MDS owed, G1–G3 owed)

## EXP-02: Survive-Noise — the fix outlasts the seed
- claim_id: local:obj2 (Survive-Noise)
- independent_var: seed (train pool {1–8}, held-out {9–12}), per group
- dependent_var: held-out-seed composition pass rate (of 4)
- ablation_rows: pooled-k LoRA on held-out seeds (Task B) · per-seed ceiling (Task C, is the held-out seed just hard) · Δ̄_t bridge (Task D)
- metric: held-out-seed pass rate ≥3/4. The training `verdict.json="ok"` is necessary but NOT sufficient — it is a training verdict, not a held-out read.
- sample_size: 5 pairs (one per group), 4 held-out seeds each, pooled over ≥4 train seeds
- falsify_condition: **support** if pooled composes on ≥3/4 held-out seeds for ≥3 groups; **null** if ≤1/4 for all groups (fix is seed-luck → the core claim fails here); **inconclusive** at 2/4 → add held-out seeds
- figures: held-out-seed contact sheet per group (render_seed_summary) + anti-corroboration: a group where Task C shows the held-out seeds were simply easy
- compute: mscluster; G6 held-out sampling is inference (~1 h); G1–G4 resume-to-ep2000 ~few h each
- status: ⚠️ pending (G6 pool trained verdict ok; held-out-seed enactment pending; G1–G4 part-trained, no verdicts)

## EXP-03: Cross-Pair — sibling transfer (SMOKE, optional)
- claim_id: local:obj3 (Cross-Pair)
- independent_var: eval pair (a cousin the LoRA never trained on), holding the LoRA fixed
- dependent_var: composition on the unseen sibling across held-out seeds
- ablation_rows: G6 LoRA on wolf×husky / lion×dog (note: lion×dog shares "dog" → near-freebie, not real transfer)
- metric: sibling composition ≥2/4 seeds — **but discount it**: a single-pair LoRA saw no variety, so a hit can't be told from a memorised correction that fits. Reviewer-credible transfer is EXP-04.
- sample_size: 1–2 siblings per group, seeds 9–12
- falsify_condition: not a publication gate. If it fails, it only nominates the sibling as a Scale-pool candidate. If it passes, EXP-04 still has to confirm with concept-disjoint pairs.
- figures: `<train>__heldout__<eval>` triptych (sibling beside PoE/Mono refs)
- compute: inference-only over `pueuo7bl`, ~1 h; **run only if cheap/time permits**
- status: ⚠️ optional (code ready, not run)

## EXP-04: Group-Wise — is a difficulty group a deployable unit
- claim_id: local:obj4 (Group-Wise)
- independent_var: held-out pair within a group (train 7 pairs, hold out 3), seed held or not
- dependent_var: held-out-pair composition (`out_in`), two-tier — (a) deployable: image composes; (b) scientific: Task D cosine of Δ̂_t to the group-mean Δ̄_t^(G)
- ablation_rows: within-group pool vs single-pair sibling (EXP-03 baseline) · concept-disjoint held-out (wolf×husky) vs shared-concept (lion×dog) · Task-D pre-screen (do the group's single-pair nudges line up before the ~30 h train)
- metric: held-out-pair pass ≥2/3 with **concept-disjoint** siblings, beating the single-pair test; Task D cosine above a same-group-vs-cross-group baseline. Both tiers reported separately — never sell a high cosine as "it works."
- sample_size: per group, 7 train pairs × up to 12 seeds, 3 held-out pairs; start G1 (most-opposite to G6) + G6 full tail
- falsify_condition: **support** if ≥1 group composes on ≥2/3 concept-disjoint held-out pairs AND Task D > baseline; **null** if ≤1/3 concept-disjoint AND low Task D (group is not a transfer unit → claim narrows to per-pair); **inconclusive** → widen held-out pairs (degradation curve: rate vs fraction held out)
- figures: `contact_sheet_out_in.png` per group + degradation curve (rate vs held-out fraction) + anti-corroboration: a failed transfer cell diagnosed (magnitude / timing / group-coarseness / off-manifold) landing below the ~40% plateau
- compute: mscluster; Plan-16 cache build ~11 h GPU (once); per-group run ~30 h; Task-D pre-screen is forward-passes only (cheap, run FIRST to skip dead groups)
- status: ⚠️ pending (G6 smoke only, 43 cells; G1–G4 not started; ~520 cache cells missing)

## EXP-05: Scale — one LoRA, the crossbar, or the catalogue fallback
- claim_id: local:obj5 (Scale)
- independent_var: pair seen/unseen × seed seen/unseen (2×2 crossbar); headline cell `out_out` (both new)
- dependent_var: per-quadrant composition + Task D, classified per group
- ablation_rows: `in_in` / `in_out` / `out_in` / `out_out` · plain all-groups LoRA vs cross-group orthogonal-adaptation variant (method extension, only if plain is mixed)
- metric: the crossbar **read** is the deliverable, not a win. `out_out` composing with per-group structure = support for one-LoRA-spans-taxonomy; `out_out` failing while EXP-04 per-group succeeds = null for the single LoRA → **ship the per-group catalogue** (a publishable decision).
- sample_size: 5 pairs × 8 seeds = 40 train cells; crossbar over held pairs × held seeds
- falsify_condition: **support (one LoRA)** if `out_out` composes across most groups with per-group Task-D structure; **null (→ catalogue)** if `out_out` fails while per-group holds; **inconclusive** if training is short — resume before reading. Either way the DoD-5 deliverable (cells.jsonl + `out_out` sheet + classification + unit chosen) is **mandatory**.
- figures: four-quadrant contact sheets; the held-pair×held-seed (`out_out`) sheet is the paper figure; Task D bridge across (pair, seed)
- compute: mscluster; resume `2em6frqv` from step 30000 (do not restart), then `sample_crossbar` all four quadrants; ~several h train tail + sampling
- status: ⚠️ pending (trained to 30k, crossbar never run, `cells.jsonl` absent; `0y9un0o4` crashed)

---

## Pre-registration (dated 2026-07-22, do not revise post-hoc)

- **If the claim is right**: EXP-01 composes on ≥4/5 groups (λ=0 canary byte-identical); EXP-02 ≥3/4 held-out seeds on ≥3 groups; EXP-04 ≥2/3 concept-disjoint held-out pairs on ≥1 group; EXP-05 crossbar read yields a deployment unit (single LoRA or catalogue).
- **If it's wrong**: EXP-01 cat×dog fails to separate at any λ (kills everything), or EXP-02 ≤1/4 held-out across all groups (seed-luck), or EXP-04 ≤1/3 with concept-disjoint pairs (no group-level transfer → per-pair only).
- **In-between**: rerun with more seeds/pairs per the three-way rule. The ~40% delivery plateau is the known confound — a failed transfer cell must be diagnosed (Task D norm/timing/coarseness) before it is read as "transfer failed" rather than "delivery failed."

## EXP-06: Mechanism — does the LoRA's attention shift match test-time attention optimization
- claim_id: local:mechanism-attn (not yet in a CLAIMS.md; reconcile if one appears)
- independent_var: correction method {PoE λ=0, LoRA λ=1 (`lora_step_062500.pt`), Attend-and-Excite-equivalent (test-time, no training)} × seed (cat×dog, train {1..8} + held-out {9..12}, from `outputs/cross_seed_lora_pooling/seed_pool.yaml`)
- dependent_var: Δ_attn(method, seed) = commitment-window-averaged attention mass on the dropped concept's token, minus that seed's own λ=0 baseline, using the existing `_CrossAttnRecorder` AAE-canon aggregation (cross-attn layers with query_len ≤ 32², bilinear-resized to 16×16, `agg_resolution=16`); paired with the existing visual/eyeball composition label per seed
- ablation_rows: A) plain PoE λ=0 (baseline) · B) LoRA λ=1 (the trained Mono-free fix) · C) Attend-and-Excite-equivalent (test-time attention-loss intervention, `keep_grad=True` recorder path, no UNet training) · D) visual composition label (reused from existing rung-1/rung-2 convention, not a new run)
- metric: Δ_attn(LoRA) vs Δ_attn(AAE), sign + magnitude ratio, restricted to the commitment window from G02/residual-diagnostics.md (outside that window the residual is already known near-zero, so attention differences there are noise). Original (pre-pressure-test) metric — Δ_attn correlates with visual success — is retained as a prerequisite check but is not the headline claim per the pressure-test verdict (that correlation alone echoes Attend-and-Excite's founding premise, not novel).
- sample_size: 12 seeds (8 train + 4 held-out), inherited from the existing rung-2 seed pool, not newly chosen for power. Small by benchmark standards; justified because the object under study is per-seed mechanism on an already-completed rung's cells, not a compositional-eval sweep.
- falsify_condition (three-way, pressure-test-upgraded): **support** if ≥3 visually-successful LoRA seeds show Δ_attn(LoRA) matching Δ_attn(AAE) in sign and within 2x magnitude (commitment-window-restricted) → same mechanism, cheaper delivery; **null** if the majority of visually-successful seeds show sign-mismatch or >4x magnitude divergence between LoRA and AAE → different mechanism, reported as its own finding, not a failed experiment; **inconclusive** if fewer than 3 seeds show visual success at all (known ~40% delivery ceiling from rung 1), or the AAE baseline itself is noisy across seeds → add a second pair from the compose-by-default bucket (G1–G3), do not loosen the threshold on the same 12 seeds.
  - Prerequisite check (not the headline, but gates it): does Δ_attn(LoRA) correlate with the visual label (row D) across the 12 seeds at all — support if yes, null if attention shifts regardless of visual outcome, inconclusive if noisy. If this prerequisite is null, the AAE comparison above is moot (there's no attention signature to compare).
- figures: three-trajectory attention-shift plot per seed (PoE / LoRA / AAE, timestep on x-axis, attention mass on y-axis) for 2-3 representative seeds (one clear support case, one clear null/divergent case if any) + a 12-seed summary scatter (Δ_attn(LoRA) vs Δ_attn(AAE), commitment-window-averaged, colored by visual success) as the headline comparison figure + anti-corroboration: a seed where the LoRA visually succeeds but Δ_attn is near-zero (shift-free success, would undermine the whole attention-mechanism story if common)
- compute: mscluster, single GPU, no training. Rows A/B: pure inference forward passes, ~50 steps × 12 seeds each, well under 1h total. Row C: adds backward-through-latent at commitment-window steps only (fixed 1-2 grad steps per intervention step, not swept), ~2h total. No Slurm array needed at this scale; disk guard still applies (attention `.pt` files accumulate, ~7,200 small files total across all rows, sized like the existing `veracity_attn` cache). Target: `/datasets/mmolefe/poe_repair_min/outputs/attn_mechanism/` (new subtree, mirrors existing `veracity_attn` layout).
- status: ⚠️ pending — capture hook (`_CrossAttnRecorder`) and save logic already exist and work (proven on `teacher_residual`, not yet wired to `run_lora_residual_inject`); AAE-equivalent baseline is new code (~80-120 lines, reuses existing `keep_grad=True` recorder path + existing decode/visual-read utilities, not a new module tree). First do-able step: row A (plain PoE, λ=0, all 12 seeds) requires zero new code — `run_lora_residual_inject` already supports `disable_adapters()` — checkpoint is a 12×50 sanity table of Δ_attn(λ=0, seed, t) before any new capture wiring or AAE code is written.

## Cluster notes

- Checkpoints target `/datasets/mmolefe/poe_repair_min/artifacts/...`, NOT `/home-mscluster` (which has hit 100% and silently killed checkpointing). Keep a `df` guard in every job preamble.
- Confirm partition/QOS before emitting a Slurm array; for the `hippo` single box use a sequential/GNU-parallel loop over configs, not an array.

``bash
# Disk guard preamble (paste into every job; abort before a full FS eats the run's tail)
CKPT_DIR=${CKPT_DIR:-/datasets/mmolefe/poe_repair_min/artifacts}
USED=$(df --output=pcent "$CKPT_DIR" | tail -1 | tr -dc '0-9')
[ "${USED:-100}" -ge 90 ] && { echo "ABORT: $CKPT_DIR ${USED}% full — checkpointing will fail." >&2; exit 1; }
``
