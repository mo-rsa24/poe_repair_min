# LoRA-Fixes-PoE

## Where things stand

This block is a snapshot; the live version prints at session start, or on demand with
`python3 scripts/plan_pulse.py --brief`.

- **What is going on:** nothing is running; the GPU is free.
- **The last thing we did:** one parent scope for the paper, its cadence stated: figures first,
  write at 5 to 10 resolved register slots.
- **Do this next:** the dose re-score. Read
  [the procedure](plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/procedures/hypothesis-02-recheck-the-headline-numbers.md)
  to completion and answer the two open questions in
  [the review file](plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/review/hypothesis-02-more-correction-more-composition.md).
  It resolves F2, the paper's headline slot.

## The paper: what has to land

An order, so every row has a step number and says what it waits on. Only unfinished rows
appear: 11 earlier steps are done and their evidence lives in the `review/` files and in git,
not here. This table is meant to be readable in under a minute, and finished rows do not help
with that.

| Step | Plan | What it does | Status | Waits on |
|---|---|---|---|---|
| 1 | does-the-correction-cause-composition/hypothesis-02-more-correction-more-composition | the headline causal result: more correction, more composition, with two flat controls | ◑ re-score owed | |
| 2 | does-the-correction-cause-composition/hypothesis-04-what-the-cached-runs-already-show | the analyses needing no GPU and no queue, five of six tasks left | ⚠️ | |
| 3 | does-the-correction-cause-composition/hypothesis-03-when-in-the-run-it-matters | when in the denoising process the correction matters | ⚠️ | 1 |
| 4 | does-the-fix-reach-unseen-pairs/instrument-02-three-live-curves-while-training | the one-epoch GPU smoke confirming three live curves | ⚠️ | |
| 5 | does-the-fix-reach-unseen-pairs/hypothesis-01-does-one-pooled-fix-transfer-at-all | finish the read: steps 70k to 100k unscored, go/no-go note owed | ⚠️ | 4 |
| 6 | does-the-fix-reach-unseen-pairs/hypothesis-02-transfer-as-a-rate-over-fifteen-pairs | 15 LoRAs, leaderboard, degradation curve | ⚠️ | 5 |
| 7 | does-the-fix-reach-unseen-pairs/baseline-01-the-size-matched-control-pool | size-matched mixed pool against animals-only | ⚠️ | 6 |
| 8 | does-the-correction-cause-composition/hypothesis-05-the-same-story-from-three-sides | the independent checks on the causal claim | ⚠️ | 1, 3 |
| 9 | does-the-correction-cause-composition/gate-01-two-literature-checks-before-print | the two /pressure-test passes before anything is written | ⚠️ | 8 |
| 10 | does-the-correction-cause-composition/figure-01-the-seven-paper-figures | the figures this scope owes the paper | ⚠️ | 9 |
| 11 | does-the-fix-reach-unseen-pairs/figure-01-the-transfer-figures | the transfer evidence figures | ⚠️ | 7 |
| 12 | writing-the-paper/writing-01-make-the-template-build | tectonic build, de-stub, the figure-path rule | ◑ | |
| 13 | writing-the-paper/writing-02-the-title-and-the-section-spine | the claim in one line, section order | ⚠️ | 12 |
| 14 | writing-the-paper/writing-03-where-each-figure-goes | which figure goes where, and the run order that implies | ⚠️ | 10, 11 |
| 15 | writing-the-paper/writing-04-method-and-introduction | method and intro prose | ⚠️ | 13 |
| 16 | writing-the-paper/writing-05-the-results-skeleton | placeholders, not prose: empty tables and XX numbers, one per figure-register slot | ⚠️ | |
| 17 | writing-the-paper/writing-06-mechanism-and-limitations | the mechanism section, honest about what did not replicate | ⚠️ | 9 |
| 18 | writing-the-paper/writing-07-the-abstract-written-last | written last, from the spine and the method | ⚠️ | 15, 16 |
| 19 | does-the-correction-cause-composition/hypothesis-01-what-the-fix-changes-inside-the-model | one task left: the /pair-figure decision, per-seed points against pair-level means | ⚠️ | |

Step 19 is unblocked and cheap despite its number. Step numbers are positions in the
order, and `sync-plan-tree` renumbers on its next pass.

**Owed, and not a table row.** Three `compose-scorer` plans have every task unticked while
that scope's summary says all three are finished and `scorer_validated.json` was emitted. The
work happened and the ticking did not. That is a harvest job for `/sync-plan-tree`, which ticks
from the output rather than from the scope's own claim.

## Reading, in the background

A pool, not an order. Pull from it when a claim needs backing or a method needs a source.
Found with `/paper-scout`, read with `/unpack-paper` or `/drip --paper`, registered in
[plans/standing/literature/](plans/standing/literature/). No row here blocks a row above.

| Paper | Why it matters to us | Which claim it touches | Read |
|---|---|---|---|
| (the 7 already reconciled on the does-the-correction-cause-composition question) | establishes that the residual IS the term PoE drops | does-the-correction-cause-composition, the causal claim | ✅ back-fill owed into the register |

## Experiments running in the background

A pool. Every row is a run that tries an idea, so no row here may change a claim: a striking
number earns the right to propose an experiment and nothing more. Results land in
`PARKING_LOT.md`.

| Run | What it would earn | State |
|---|---|---|
| does-the-correction-cause-composition/idea-01-does-it-hold-for-attribute-pairs | whether the correction behaves the same for attribute pairs as for object pairs, which would widen the claim's reach | ⚠️ not started |
| does-the-correction-cause-composition/generalization-01-other-models-and-samplers | the same result on a second model and sampler, which is a likely reviewer ask but not a claim we make | ⚠️ not started |

## Standing jobs

No order and no end. Re-entered rather than closed.

- [plans/standing/artifact-reconciliation/plans/05-resweep-on-new-runs.md](plans/standing/artifact-reconciliation/plans/05-resweep-on-new-runs.md): re-catalogue and integrity-check whenever new runs land. Every `✓ verified` tag points at a path on a filesystem that is not under version control.
- [plans/standing/literature/plans/01-reading-register.md](plans/standing/literature/plans/01-reading-register.md): keep the reading table above current, and make sure every idea-trying run names the paper it came from.

## The scopes, and the state of each

What each folder under `plans/` is and whether to open it. Live means it has rows in the lists
above. Standing means it is re-entered, never finished. Done means its output exists and is in
use. Nothing here is ambiguous on purpose: a scope that cannot say its state in one line gets
shelved until it can.

The listing itself now says the state: a bare folder is live, and everything else sits in a
container named for its state.

| Folder | State | One line |
|---|---|---|
| `closing-the-compositional-gap/` | live, the paper | one parent scope for the manuscript and its two result claims: `does-the-correction-cause-composition` (causal), `does-the-fix-reach-unseen-pairs` (transfer), `writing-the-paper` (the draft) |
| `standing/literature/` | standing | the reading register |
| `standing/artifact-reconciliation/` | standing | artifacts catalogued and integrity-checked |
| `completed/compose-scorer/` | done | delivered `scorer_validated.json`, in use by both result scopes |
| `shelved/` | the shelf | `rungs`, `mechanism-study`, `phases`; one line at the top of each says what would bring it back |

## One plan, one table

Every live plan appears in exactly one of the four lists above, and all four live in this file.
No scope keeps a list of its own. When a background experiment starts feeding the paper it
**moves** into the paper table and gets a step number, which is how a promotion becomes visible
instead of being a quiet field change. `plan_pulse --checks=8` fails if a live plan is in none
of them or in more than one.

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
- ⚠️ plans/standing/artifact-reconciliation/ — "keep run artifacts catalogued, integrity-checked, canonically organised" (standing: carries a recurring re-sweep node)
- ⚠️ plans/completed/compose-scorer/ — reusable instrument: a 3-anchor scorer that tells a two-animal composition from a chimera blend; emits scorer_validated.json (the cross-scope contract)
- ⚠️ plans/closing-the-compositional-gap/plans/does-the-fix-reach-unseen-pairs/ — animals-only hard-pair LoRA transfer (leave-one-pair-out + size-matched-mixed contrast); DEPENDS ON compose-scorer's scorer_validated.json
- ⚠️ plans/standing/literature/ — standing: what the field already knows, and the source behind every idea-trying run
- ⚠️ plans/closing-the-compositional-gap/plans/writing-the-paper/ — the ICLR manuscript in `paper/iclr/`; no GPU, no queue

## Plans
(One plan file per pyramid rung, grouped under `plans/rungs/`. Detailed phase
files are archived under `plans/shelved/phases/` and referenced from each rung plan;
`plans/shelved/phases/PHASE_MAP.md` is the retired 8-phase orchestrator.)
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
