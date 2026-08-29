# Renames

Old path to new path, so a stale reference still resolves. One section per sitting. Grouped by
cluster, not by individual file, except where a single file's destination isn't implied by its
folder's row.

## Sitting 2026-08-24 (the retrofit sweep, stages 1-4)

### Run families (`outputs/` and `artifacts/rung*`), from commit `bf235b8` and earlier this sitting

| Old | New |
|---|---|
| `artifacts/rung1-overfit/lora/` | `artifacts/results/can-lora-learn-a-residual-that-corrects-poe/one-pair-one-seed/` |
| `outputs/animals_compose_transfer/` | `artifacts/results/does-the-fix-reach-unseen-pairs/` |
| `outputs/group_a_failure/` | `artifacts/results/residual-dynamics/correction-outside-the-unet/` |
| `outputs/residual_diagnostics/` | `artifacts/results/residual-dynamics/residual-between-mono-and-poe/` |
| `outputs/conditioning_window/` | `artifacts/results/when-the-correction-must-arrive/cfg-window-without-lora/` |
| `outputs/conditioning_window_lora/` | `artifacts/results/when-the-correction-must-arrive/cfg-window-with-lora/` |
| `outputs/poe/` | `artifacts/results/poe-blends-instead-of-composing/poe-baseline-samples/` |
| `outputs/compose_scorer/` | `artifacts/results/can-we-trust-the-compose-score/compose-scorer-validation/` |
| `pairs.py` | `poe_repair/pairs.py` |

**Held, not moved.** `artifacts/rung2-survive-noise/cross_seed/` stays under its old name: the
mount holds a second, disjoint four-seed pooled set under the same name, and this sweep does not
write onto the mount. `artifacts/rung3-group-wise/` and `artifacts/rung4-scale/` stay, cut by
group by design. `outputs/interaction_term/` stays: its rename table row is a merge onto the
mount, which this sweep does not perform. `outputs/presentation/` stays: marked cold, undecided,
owed a look pass. See "Still open" in the closing report.

### The evidence set (`evidence/f2-lambda1-audit/`), from commit `9406934`

| Old | New |
|---|---|
| `evidence/f2-lambda1-audit/` | `artifacts/results/can-we-trust-the-compose-score/do-the-successful-cells-contain-both-animals/` |
| `evidence/f2-lambda1-audit/01-both-there/` | `.../both-animals-there/` |
| `evidence/f2-lambda1-audit/02-two-of-one/` | `.../two-of-the-same-animal/` |
| `evidence/f2-lambda1-audit/03-cannot-call/` | `.../cannot-decide/` |
| `evidence/f2-lambda1-audit/04-look-alike-by-design/` | `.../the-two-animals-look-alike/` |
| `evidence/f2-lambda1-audit/05-scored-failure/` | `.../scored-as-a-failure/` |

Referrers to the old path in `context/`, `RESEARCH_GUIDELINES.md`, and several plan files under
`plans/closing-the-compositional-gap/plans/can-we-trust-the-compose-rate/` had never been updated
when the folder moved; this sitting repointed all of them.

### Duplicates and scenes, from commit `9406934`

| Old | New |
|---|---|
| `recap/` | `artifacts/_quarantine/recap/` |
| `paper/iclr/figures/F1b-two-regimes-seed42.{png,pdf}` | `artifacts/_quarantine/paper-figure-duplicates/` (dropped: named by nothing) |
| `paper/iclr/figures/F2-correction-strength.png` | `artifacts/_quarantine/paper-figure-duplicates/` (dropped: byte-identical to `compose-rate-as-correction-rises.png`) |
| `evidence/f5b-trajectory-divergence/how-far-the-corrected-run-separates-from-the-uncorrected-one.png` | dropped (byte-identical to the `paper/iclr/figures/` original, which survives) |
| `plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/scene/{src,public,loader,package.json,...}` | `artifacts/scenes/how-much-correction-is-needed/` |
| `dl-scene/app/{src,public,...}` | `artifacts/scenes/sdxl-vae-architecture-map/app/` |
| `results/*.log` | `logs/` |

**Fixed this sitting.** `plans/.../scene/dist` and `plans/.../scene/node_modules`, and
`dl-scene/app/dist` and `dl-scene/app/node_modules`, were accidentally committed into git by the
interrupted checkpoint instead of being left untracked. `git rm --cached` only; the bytes are
still on disk at the old path, untracked, exactly where Instruction 5's manual deletion still
finds them.

### `docs/` decommission, this sitting

| Old | New |
|---|---|
| `docs/DECISION_TIMELINE.md` | `DECISION_TIMELINE.md` (repo root) |
| `docs/RESEARCH_GUIDELINES.md` | `RESEARCH_GUIDELINES.md` (repo root) |
| `docs/EXPERIMENT_ERROR_CATALOG.md` | dropped: confirmed-superseded duplicate of `environment/known-failures.md`, which already recorded migrating it |
| `docs/RESULTS_SUMMARY.md` | `report/RESULTS_SUMMARY.md` |
| `docs/instrument_smoke.md` | `report/instrument_smoke.md` |
| `docs/normalization_preregistration.md` | `report/normalization_preregistration.md` |
| `docs/reading-register.md` | `plans/standing/literature/reading-register.md` |
| `docs/results-archive/` | `artifacts/_quarantine/results-archive/` |
| `docs/figures/what-each-figure-argues.md` | `paper/iclr/what-each-figure-argues.md` |
| `docs/figures/scene-logsnr/` | `artifacts/scenes/logsnr-explainer/` |
| `docs/evidence/INDEX.md` | `report/paper-evidence-index.md` |
| `docs/evidence/F2-dose-response/scorer-count-caveat.md` | `artifacts/results/can-we-trust-the-compose-score/the-scorers-instance-count-is-not-a-count.md` |
| `docs/evidence/plausibility_climb.png` | `artifacts/results/how-much-correction-is-needed/plausibility_climb.png` |
| `docs/evidence/F6-subspace-vs-transfer/` | `artifacts/results/which-way-the-correction-points/does-the-subspace-test-predict-transfer/` |
| `docs/evidence/F6-what-the-spectrum-measures/` | `artifacts/results/which-way-the-correction-points/what-the-spectrum-measures/` |
| `docs/evidence/F7-mechanism-reprobe/` | `artifacts/results/residual-dynamics/content-change-relative-to-attention-change/` |
| `docs/evidence/F8-animals-hard-vs-easy/` | `artifacts/results/does-the-fix-reach-unseen-pairs/hard-vs-easy-transfer/` |
| `docs/evidence/EXP01-commitment-step/` | `artifacts/results/when-the-correction-must-arrive/commitment-step/` |
| `docs/evidence/EXP04-window-vs-commitment/` | `artifacts/results/when-the-correction-must-arrive/window-vs-commitment/` |

**Not moved.** `docs/IMMERSE_PoE_Foundations.md` stays: its home is the `poe-derivation-foundations`
learning journey at `/home-mscluster/mmolefe/goal-setting/learning/poe-derivation-foundations/`,
which this sweep cannot write into without a paired run (`retrofit-repo --paired <path>`). `docs/`
is therefore not yet empty.

### The five remaining loose folders, this sitting

| Old | New |
|---|---|
| `evidence/subspace-vs-transfer/{demo.py,figure.py,test_demo.py}` | merged into `artifacts/results/which-way-the-correction-points/does-the-subspace-test-predict-transfer/` |
| `evidence/f6-what-the-spectrum-measures/{control.py,result.json}` | merged into `artifacts/results/which-way-the-correction-points/what-the-spectrum-measures/` |
| `evidence/h05-three-sides/` | `artifacts/results/does-the-interaction-term-cause-composition/five-checks-from-three-sides/` |
| `evidence/f5b-trajectory-divergence/` | `artifacts/results/how-far-the-corrected-run-separates-from-the-uncorrected-one/` |
| `show-me/batch-shape-nondeterminism/` | `artifacts/notes/batch-shape-nondeterminism/` |
| `captures/coind-conditional-independence-loss.md`, `captures/build.sh`, `captures/tufte.css` | `artifacts/notes/coind-conditional-independence-loss/` |
| `flow-map-images/rung-01-conditional-independence.png` | `artifacts/notes/coind-conditional-independence-loss/rung-01-conditional-independence.png` |
| `diagrams/figures/why-this-plan-exists.png` | `plans/closing-the-compositional-gap/plans/does-the-fix-reach-unseen-pairs/plans/diagrams/figures/why-this-plan-exists.png` |

### Record folders, this sitting

| Old | New |
|---|---|
| `inventory/` | `plans/standing/artifact-reconciliation/inventory/` |
| `learning-captures/` | `artifacts/_quarantine/learning-captures/` |
| `todoist-staging/` | `artifacts/_quarantine/todoist-staging/` (was untracked/gitignored; still is, now under the `artifacts/` blanket ignore) |

### `dl-scene/` and `pressure-tests/`, this sitting

| Old | New |
|---|---|
| `pressure-tests/` | `artifacts/notes/interaction-term-as-pmi-gradient/` |

**Held, not moved.** `dl-scene/app/{node_modules,dist}` (its source already moved above); held
for Instruction 5's manual deletion.

### Loose root files, this sitting

| Old | New |
|---|---|
| `midrun_separation_example.png` | `artifacts/results/when-the-correction-must-arrive/commitment-step/midrun_separation_example.png` |
| `text_orthogonality_probe.png` | `artifacts/results/does-text-alone-predict-composition/text_orthogonality_probe.png` |

## What this sweep did not touch

The mount (`/datasets/mmolefe/poe_repair_min/`): every one of "The mount's eleven families" in
`plans/retrofit-poe-repair-min.md`'s rename table is unrenamed, and no merge of
`outputs/interaction_term/` onto it has happened. This sweep does not move, rename, or delete
anything on that filesystem; it only references and cards it. Resuming that half is its own
sitting.

`scripts/build_*.py` (9 files) into `scripts/build/`, and the ~50 remaining flat `scripts/*.py`
into experiment-named groupings: listed in the plan's rename table, not attempted this sitting.

### Root files absorbed into their natural homes, this sitting

| Old | New |
|---|---|
| `DECISION_TIMELINE.md` (repo root) | `report/decision-timeline.md` |
| `EXPERIMENTS.md` (repo root) | `report/experiments-log.md` |
| `RESEARCH_GUIDELINES.md` (repo root) | `context/research-guidelines.md` |
| `PARKING_LOT.md` (repo root) | dropped: the routing practice it served (idea-runs land here) is
  retired, not relocated — a striking result from an idea-run now becomes a row or task in the
  plan tree directly, and the ~10 plans that pointed at it were edited to say so |

## 2026-08-27: idea walks moved off the root (drip-idea run, tidy-repo census walk)

| Old path | New path | Why |
|---|---|---|
| ideas/text-embedding-of-the-residual/ | artifacts/ideas/text-embedding-of-the-residual/ | idea walks live under artifacts/ideas/, never at the root |
| ideas/merge-supervisor-structure/ | artifacts/ideas/merge-supervisor-structure/ | same rule; 5 inbound references rewritten |
| tmp/ | removed (was empty) | empty untracked directory |

## 2026-08-27: tidy-repo pass after the sync commit (two deletions, no moves)

| Old path | New path | Why |
|---|---|---|
| cat and dog.jpg (repo root) | deleted | untracked, no referrer; had served as a one-off style reference in a codex render test (noted in artifacts/ideas/codex-diagram-render-queue/IDEA_MAP.md:103) |
| temp/a_frog__x__a_toad/ | deleted (754M) | byte-identical duplicate of /datasets/mmolefe/poe_repair_min/outputs/interaction_term/window/pairs/a_frog__x__a_toad/, which stays; verified by per-seed file counts and sampled md5s including latents |
| plans/closing-the-compositional-gap/plans/showcase-the-trained-adapter/decisions-taken-here.md | plans/closing-the-compositional-gap/plans/showcase-the-trained-lora/decisions-taken-here.md | absorbed into the merged scope ledger; original kept at artifacts/drips/showcase-the-trained-adapter/reconciled-ledger-now-absorbed.md |
| plans/closing-the-compositional-gap/plans/showcase-the-trained-lora/ | plans/closing-the-compositional-gap/plans/showcase-the-trained-adapter/ | the two parallel showcase scopes merged under the adapter name by the user's call; the five plan files, four review files, diagrams and ledger moved intact; the residual frame (old MASTER_PLAN.md) is at artifacts/plans/archived/showcase-the-trained-lora/ |
| plans/closing-the-compositional-gap/plans/showcase-the-trained-adapter/ | plans/closing-the-compositional-gap/plans/showcase-the-trained-lora/ | renamed back per the arbitration in DRIFT_LOG.md once the authoring session went quiet; contents moved intact |

## 2026-08-29: the paper's sub-scopes promoted to numbered top-level scopes

The parent scope `closing-the-compositional-gap/` held six sub-scopes one hop down, while a
seventh peer sat at the top level, so `ls plans/` showed neither the whole set nor any order.
Every scope is now standalone directly under `plans/`, numbered in the order a reader meets them.
The number is a reading order, not the step order, which stays in the root `## Running order`.

| Old path | New path | Why |
|---|---|---|
| `plans/closing-the-compositional-gap/plans/showcase-the-trained-lora/` | `plans/01-showcase-the-trained-lora/` | the trained adapter is the artifact the paper ships and the other scopes measure against |
| `plans/closing-the-compositional-gap/plans/can-we-trust-the-compose-rate/` | `plans/02-can-we-trust-the-compose-rate/` | the instrument every number in 03 and 04 leans on |
| `plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/` | `plans/03-does-the-correction-cause-composition/` | the causal claim |
| `plans/closing-the-compositional-gap/plans/does-the-fix-reach-unseen-pairs/` | `plans/04-does-the-fix-reach-unseen-pairs/` | the transfer claim, which the causal claim precedes |
| `plans/closing-the-compositional-gap/plans/when-does-the-outcome-lock-in/` | `plans/05-when-does-the-outcome-lock-in/` | the mechanism question the two claims raise |
| `plans/is-the-gap-the-samplers-or-the-models/` | `plans/06-is-the-gap-the-samplers-or-the-models/` | already top-level; numbered into the same order |
| `plans/closing-the-compositional-gap/plans/writing-the-paper/` | `plans/07-writing-the-paper/` | consumes everything above it, so it reads last |
| `plans/closing-the-compositional-gap/diagram-prompts.md` | `plans/diagram-prompts.md` | the parent's illustrated map becomes the project map |
| `plans/closing-the-compositional-gap/diagrams/` | `plans/diagrams/` | its renders and process history travel with it |
| `plans/closing-the-compositional-gap/plans/diagrams/why-this-plan-exists.prompt.md` | `plans/04-does-the-fix-reach-unseen-pairs/diagrams/` | it illustrates the transfer scope's sweep, so it sits beside that plan |
| `plans/closing-the-compositional-gap/plans/diagrams/figure-coverage-prompt.md` | `plans/diagrams/figure-coverage-prompt.md` | generic across scopes; belongs with the project map |
| `plans/closing-the-compositional-gap/MASTER_PLAN.md` | `artifacts/plans/archived/closing-the-compositional-gap/MASTER_PLAN.md` | its mission, running-order pointer and scope table were absorbed into the root `MASTER_PLAN.md`; keeping a pointer-only scope would have preserved the hop the promotion removes |

290 link edits across 109 files, resolved against each file's old location and re-emitted from its
new one rather than string-swapped, so depth changes inside the promoted scopes were corrected too.
Four references in the learning journeys (`sampler-correctors-for-composition`,
`trajectory-manifold-by-hand`) were repointed in the same pass. Dangling-link count measured
before and after: no reference that resolved before the move fails after it.
