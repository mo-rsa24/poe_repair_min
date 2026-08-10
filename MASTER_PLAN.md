# LoRA-Fixes-PoE

## Do this next

Open [plans/interaction-term/plans/03-dose-response.md](plans/interaction-term/plans/03-dose-response.md)
and re-score the dose sweep. The 480 cells are generated and the result holds
(oracle 7% to 93% across λ, both controls flat near 6%), but the scorer globbed
the whole tree, so λ=0 and λ=1 are scored over 44 cells while the middle doses
use 32. Pin the root to this sweep's seeds, choose a confidence floor that
rejects the 162px sliver, re-read the curves. The paper's headline figure waits
on this.

## Running order

One row per plan across every scope and level. Plan numbering is per folder and
is not the order. Sub-scopes under `interaction-term` have no plan files yet, so
they carry no rows.

| Step | Plan | What it does | Status | Waits on |
|---|---|---|---|---|
| 1 | compose-scorer/01-anchors | three reference anchors per validation pair | ✅ | |
| 2 | compose-scorer/02-build-scorer | instance-count scorer; embedding reads nulled | ✅ | 1 |
| 3 | compose-scorer/03-validate-emit-contract | 10/10 validated, emits scorer_validated.json | ✅ | 2 |
| 4 | artifact-reconciliation/01-data-inventory | catalogue every run artifact | ✅ | |
| 5 | artifact-reconciliation/02-two-root-classified-sweep | classify across both filesystems | ✅ | 4 |
| 6 | artifact-reconciliation/03-data-integrity-check | checkpoints load, manifests agree | ✅ | 5 |
| 7 | artifact-reconciliation/04-canonical-layout-reorg | move artifacts to the canonical layout | ✅ | 6 |
| 8 | animals-compose-transfer/01-pool-and-precondition | curate pair_pool.yaml by fail-rate | ✅ | 3 |
| 9 | interaction-term/00-build-the-instruments | 13 instruments built and smoked | ✅ | |
| 10 | interaction-term/01-preregister-normalization | relative_norm fixed before any read | ✅ | 9 |
| 11 | interaction-term/02-mechanism-reprobe | value channel not attention, 64 cells, median 1.52x | ✅ | 10 |
| 12 | interaction-term/03-dose-response | the causal headline: λ sweep with two controls | ◑ | 11 |
| 13 | interaction-term/05-cache-analyses | SVD, SNR, fork curve. No GPU, no queue | ⚠️ | 10 |
| 14 | interaction-term/04-window-pair | when in the trajectory the term matters | ⚠️ | 12 |
| 15 | animals-compose-transfer/02-wire-scorer-eval-hook | scorer into the eval hook, three live W&B curves | ⚠️ | 8 |
| 16 | animals-compose-transfer/03a-phase1-pooled | one pooled LoRA, held-out transfer read | ⚠️ | 15 |
| 17 | animals-compose-transfer/03-run-A-leave-one-pair-out | 15 LoRAs, leaderboard, degradation curve | ⚠️ | 16 |
| 18 | animals-compose-transfer/04-run-B-contrast | size-matched mixed pool against animals | ⚠️ | 17 |
| 19 | interaction-term/06-corroborations | the independent checks on the causal claim | ⚠️ | 12, 14 |
| 20 | interaction-term/07-composition-type | does the term behave the same across composition types | ⚠️ | 19 |
| 21 | interaction-term/08-replication | second model, second sampler | ⚠️ | 19 |
| 22 | interaction-term/09-print-gates | the two /pressure-test gates before anything is written | ⚠️ | 20, 21 |
| 23 | interaction-term/10-figures | the paper figures from this scope, via /design-figure | ⚠️ | 22 |
| 24 | animals-compose-transfer/05-figures | the transfer evidence cascade, F2 to F5 | ⚠️ | 18 |
| 25 | interaction-term/11-inspector | the interactive read of the interaction term | ⚠️ | 23 |
| 26 | paper-iclr/00-compile-the-template | tectonic build works, de-stub, figure-path rule | ◑ | |
| 27 | paper-iclr/01-title-and-spine | the claim in one line, section order | ⚠️ | 26 |
| 28 | paper-iclr/02-figure-layout | which figure goes where, and the run order it implies | ⚠️ | 23, 24 |
| 29 | paper-iclr/03-draft-method-and-intro | method and intro prose | ⚠️ | 27 |
| 30 | paper-iclr/05-results-skeleton | placeholders, not prose | ⚠️ | 28 |
| 31 | paper-iclr/06-mechanism-and-caveats | the mechanism section, honest about what did not replicate | ⚠️ | 11, 22 |
| 32 | paper-iclr/04-abstract | written last, from the spine and the method | ⚠️ | 29, 30 |
| 33 | artifact-reconciliation/05-resweep-on-new-runs | standing: re-catalogue whenever new runs land | ⚠️ recurring | |
| 34 | literature/01-reading-register | standing: what the field knows, and the source behind every tried idea | ⚠️ recurring | |

## Mission
Does a LoRA make PoE co-occur like Mono, and does that fix carry to unseen pairs?
When SDXL composes two concepts by Product-of-Experts it usually fails (chimera /
single concept / noise). We train a rank-8 cross-attention LoRA on the cached
guided residual r_t = ε̃_J − ε̃_PoE so that, at inference and without ever
encoding the joint prompt (Mono-free), the corrected PoE prediction moves toward
the Mono ceiling, far enough to separate the concepts by eye on the beachhead
cell (~40% of the PoE→Mono distance). The program asks how far that fix reaches:
one cell, one pair, one difficulty class, or the whole studied taxonomy.

## Objectives
(The five pyramid rungs — direction. Each widens the held-out set.)
1. **Overfit** — a rank-8 cross-attn LoRA on the cached residual makes PoE
   co-occur like Mono at inference (Mono-free), and the mechanism is
   pair-generic, not concept-collision-specific.
2. **Survive-Noise** — the fix survives seed variation: one LoRA pooled over
   seeds generalises to held-out seeds, per group.
3. **Cross-Pair** — a pair-trained fix transfers to an unseen sibling pair of the
   same group (the cheap within-group transfer probe).
4. **Group-Wise** — "group" is a deployable pooling unit: one LoRA per difficulty
   class, trained on within-group pairs, generalises to held-out pairs.
5. **Scale** — a single LoRA spans the studied taxonomy held out on BOTH pair and
   seed axes (the deployment crossbar) — or the deployable artefact is the
   per-group catalogue, established with evidence.

## Goals
(Checkpoints — measurable. Status from docs/DECISION_TIMELINE.md.)
1. Overfit: cat×dog seed 42 — λ=0 byte-identical to PoE, λ=1 two distinct animals
   by ~ep600 [✅ G04]; one representative pair per G1–G4+G6 closes the gap
   single-seed [◑ only G4+G6 trained; G1–G3 owed].
2. Survive-Noise: pooled LoRA composes on ≥3/4 held-out seeds for the
   representative pair, per group [G6: pool trained to convergence ✅ (verdict
   ok), BUT composes-on-held-out-seeds ⧗ pending (enactment generating, job
   recap_g6); ◑ G1–G4 part-trained, no verdicts].
3. Cross-Pair: a group-G LoRA composes on a held-out sibling on ≥2/4 held-out
   seeds [⚠️ code ready (Plan 12), not run].
4. Group-Wise: within-group 7-pair pool composes on the held-out 3 pairs,
   matching or beating the single-pair sibling smoke [◑ g6 smoke only; g1–g4 not
   started].
5. Scale: one LoRA on 5 pairs × 8 seeds — held-pair×held-seed quadrant composes
   with per-group structure, OR documented fallback to per-group catalogue
   [⏸ trained to step 30k, crossbar never evaluated].

## Expected Outcome
A deployable, Mono-free PoE corrector whose reach is characterised: at minimum a
per-pair/per-group catalogue backed by evidence, at most a single
taxonomy-spanning LoRA. The alternatives that do NOT work (external correctors,
PoE-internal forces) are documented as negative controls, and every landing is
recorded in the decision timeline.

## Definition of Done
1. ⚠️ Overfit read across G1–G4+G6 single-seed (gap closed by eyeball + MDS bend).
2. ⚠️ Survive-Noise: per-group pooled LoRAs have held-out-seed verdicts.
3. ➖ Cross-Pair (OPTIONAL smoke — downgraded 2026-07-22, not a publication gate): single-pair→sibling transfer is confounded; the reviewer-credible transfer test is DoD-4 (Group-Wise) with concept-disjoint pairs. See EXPERIMENTS.md EXP-03.
4. ⚠️ Group-Wise: within-group pooled LoRAs read for G1–G4+G6 (or the honest subset).
5. ⚠️ Scale: four-quadrant crossbar evaluated; held-pair×held-seed classified;
   deployment unit chosen (single LoRA vs per-group catalogue).
6. ✅ Negative controls (group-A, internal-force) reported; Mono-free property holds
   at every λ; docs/DECISION_TIMELINE.md reflects each landing.
7. ✅ G5 (entanglement) explicitly deferred with rationale.

## Sub-Scopes
- ⚠️ plans/artifact-reconciliation/ — "keep run artifacts catalogued, integrity-checked, canonically organised" (standing: carries a recurring re-sweep node)
- ⚠️ plans/compose-scorer/ — reusable instrument: a 3-anchor scorer that tells a two-animal composition from a chimera blend; emits scorer_validated.json (the cross-scope contract)
- ⚠️ plans/animals-compose-transfer/ — animals-only hard-pair LoRA transfer (leave-one-pair-out + size-matched-mixed contrast); DEPENDS ON compose-scorer's scorer_validated.json
- ⚠️ plans/literature/ — standing: what the field already knows, and the source behind every idea-trying run
- ⚠️ plans/paper-iclr/ — the ICLR manuscript in `paper/iclr/`; no GPU, no queue

## Plans
(One plan file per pyramid rung, grouped under `plans/rungs/`. Detailed phase
files are archived under `plans/phases/` and referenced from each rung plan;
`plans/phases/PHASE_MAP.md` is the retired 8-phase orchestrator.)
- ⚠️ rungs/01-overfit.md — beachhead + taxonomy breadth + negative controls (DoD 1, 6)
- ⚠️ rungs/02-survive-noise.md — seed-pooled LoRA, held-out seeds, per group (DoD 2)
- ⚠️ rungs/03-cross-pair.md — held-out-pair transfer probe (DoD 3)
- ⚠️ rungs/04-group-wise.md — within-group pooling, is "group" a unit (DoD 4)
- ⚠️ rungs/05-scale.md — one LoRA, four-quadrant crossbar, or catalogue fallback (DoD 5)
- ⚠️ rungs/06-supervisor-briefing.md — communicate current state + name next moves, plain-speak'd (delivery, not a rung)

## Environment Context
See `docs/ENVIRONMENT.md` for this project's environment/architecture facts.
Read before drafting or checking any plan in any scope. `CLAUDE.md` at the repo
root holds the rules for how runs are classified, recorded, and allowed to move
a plan.

## 🖥️ Viewing results (web apps)

The fastest way to *see* the results is the **LoRA Inspector** — a Flask app with four
tabs: **CFG-mask ablation** (no-LoRA floor) · **LoRA residual** (the epoch × λ morph +
MDS trajectory) · **MDS large** · **LoRA + CFG-mask**. Pair dropdown top-right.

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
$PY scripts/build_lora_manifest.py        # (re)build the manifest the inspector reads
bash scripts/run_lora_inspector.sh        # serves 127.0.0.1:5050 and prints the exact tunnel line
#   from your laptop:  ssh -L 5050:localhost:5050 <cluster-node>   (it prints the node)
#   then open:         http://localhost:5050
```

Live checkpoint viewer (group-A student runs, writes side-by-side PoE|Mono|student PNGs):

```bash
$PY -m scripts.watch_and_visualize --ckpt-dir <ckpt-dir> --pair "a cat|a dog" --seed 42
```

## Glossary

The kept terms, one plain line each. The plan prose uses these; the commands and Δ/ε notation stay exact.

- **PoE (Product-of-Experts):** composing "a cat and a dog" by adding the model's separate opinions about each prompt. It usually fails and fuses them.
- **Chimera:** that failure. One animal with parts of both, instead of two animals.
- **Mono / the ceiling:** the cheat that works — give the model the literal joined prompt "a cat and a dog". It composes fine, but it defeats the point, so we only use it as the target.
- **Mono-free:** at test time the LoRA never sees the joined prompt. That is the whole point.
- **The residual (`r_t = ε̃_J − ε̃_PoE`, also `Δ_t`):** the step-by-step correction, the gap from broken-PoE toward the Mono target.
- **LoRA (rank-8, cross-attention / `attn2`):** a small set of extra weights bolted onto the layer where the text prompt enters. What we train.
- **λ (lambda), `PoE+λ·R`:** the dial for how much of the correction to add. 0 is off (identical to plain PoE), 1 is full.
- **Seed vs pair:** a seed re-rolls only the starting noise; a pair swaps the two concepts. A new pair is the harder test.
- **Cell:** one (pair, seed) training or evaluation point.
- **Crossbar / `in_in` / `out_in` / `out_out`:** the 2×2 test grid — pair seen/unseen crossed with seed seen/unseen. `out_out` (both new) is the hardest.
- **Task D:** the check of whether the LoRA's correction points along the shared group direction (a cosine), separate from whether the picture composes.
- **MDS:** a 2-D plot of the denoising paths, used to see the corrected path bend toward the joint target.
