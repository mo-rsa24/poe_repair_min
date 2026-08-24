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
