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

## Cluster notes

- Checkpoints target `/datasets/mmolefe/poe_repair_min/artifacts/...`, NOT `/home-mscluster` (which has hit 100% and silently killed checkpointing). Keep a `df` guard in every job preamble.
- Confirm partition/QOS before emitting a Slurm array; for the `hippo` single box use a sequential/GNU-parallel loop over configs, not an array.

```bash
# Disk guard preamble (paste into every job; abort before a full FS eats the run's tail)
CKPT_DIR=${CKPT_DIR:-/datasets/mmolefe/poe_repair_min/artifacts}
USED=$(df --output=pcent "$CKPT_DIR" | tail -1 | tr -dc '0-9')
[ "${USED:-100}" -ge 90 ] && { echo "ABORT: $CKPT_DIR ${USED}% full — checkpointing will fail." >&2; exit 1; }
```
