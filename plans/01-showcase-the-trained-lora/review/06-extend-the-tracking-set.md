# 🔬 Review: does the extended tracking set log everything, before anything launches?

Nothing has run yet. This file judges [the design](../plans/tools/06-extend-the-tracking-set.md). Run
kind: measuring tool (a first short run proving the wiring; no science question of its own).

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/tools/06-extend-the-tracking-set.md) | the four reads, the admission rule, the freeze |
| this file | the verdict on that short run, and the manifest hash |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [Still open](#still-open)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

- **The tracking set**: the frozen list of renders and metrics produced at every checkpoint of
  experiments A and B; frozen means the manifest hash in the W&B config never changes mid-run.
- **Teacher-forced**: computed on cached PoE states, immune to fp16 closed-loop drift.

## Runs

Navigation: ⬅️ [Words](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Date | Run id | What ran | Wall time | Outcome |
|---|---|---|---|---|
| 2026-09-01 | `onhzjfa1` (v1, `mscluster108` device 1) | Full ledger scope: 152 cells (all 19 pairs × 8 seeds), 50 DDIM steps | ~1h (killed, not converged on cost) | Killed: at ~49s/cell measured mid-run, a full checkpoint's eval pass projected to ~2h, ~20h added across a 100k-step run — infeasible |
| 2026-09-01 | `s3chdlbv` (v2, `mscluster108` device 1) | 38 cells (2 seeds × 19 pairs), streamed per-cell logging added, bucket boundaries hardcoded from the ledger prose (early 0-10, commit 18-36, late 37-49) | 33m25s | Completed; all seven families non-null, but bucket boundaries disagreed with the codebase's own `cfg.probe.commit_window` (5-25), already used by `train/loss_bucket/*` |
| 2026-09-01 | `xrmmlhck` (v3, `mscluster108` device 1) | 4 cells (one in-pair `a_wolf__x__a_husky`, one out-pair `a_cat__x__a_dog`, 2 seeds each), buckets read from `cfg.probe.commit_window`/`where_applied_steps` | 5m46s | **Completed clean; this is the run judged below.** |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

**This is the one question whose failure blocks A and B.**

- [x] ⚠️ **Do all seven metric families log at least one non-null value per eval step on the
  first short run?** The threshold: yes for all seven (three inherited, four new), else no launch.
  **Yes**, confirmed on `xrmmlhck` (v3): `direction_cosine`, `frac_distance_reached`,
  `compose_rate` (already wired) and `learned_actual_cosine`, `embedding_drift`,
  `divergence_profile`, `spectral_share` (new) all logged real, non-null values —
  e.g. `eval/tracking/learned_actual_cosine/bucket_commit/mean = 0.090`,
  `eval/tracking/embedding_drift/dino/mean = 0.371`,
  `eval/tracking/spectral_share/top1_share = 0.381`.

## Written before the run, answered after

Navigation: ⬅️ [The bar](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

- [x] ⚠️ Did the eval pass's wall time grow, and by how much, against the band measured by
  `instrument-02`'s own short run (the admission rule promises roughly zero growth)?
  **It grew, and the admission rule's promise did not hold at full scope.** At the original
  152-cell scope (v1), the eval pass alone projected to ~2h per checkpoint — the four new
  reads are not free (`embedding_drift` alone adds two embedder calls per cell that
  `instrument-02` never ran). This is why the tracking set's *scope* (not the metric set)
  was cut to one in-pair, one out-pair. At that reduced scope (v3), the whole run — training
  plus eval pass — took 5m46s; not directly comparable to `instrument-02`'s own band since
  the cell count differs, but small enough that eval-pass growth is no longer the binding
  cost for A or B.
- [x] ⚠️ Is the manifest hash visible in the run config, and does re-running startup reproduce it?
  **Yes.** `tracking_set_hash` appears in `xrmmlhck`'s W&B config
  (`f08f322d4542c3f7d405dec8c1a1cd2e79c09109a9a144de7a2d014c6eb54eb4`), matching
  `tracking_set.json` on disk exactly. Reproduction follows directly from the mechanism
  (the trainer reads the frozen file and reports its hash verbatim, no computation in
  between) rather than from a second identical run — only one v3 run was made.

## Could the answer be an artefact

Navigation: ⬅️ [Before/after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- [x] ⚠️ **Was the comparison fair?** The short run used the frozen manifest, not a subset.
  **Yes, relative to what's now frozen** — `xrmmlhck` used `tracking_set.json` exactly as
  written, unmodified mid-run. Note the manifest itself is no longer what the ledger
  originally specified: it names one in-pair and one out-pair (4 cells) rather than the
  ledger's "four F9 renders" plus "cat×dog seed 1" concept, a scope cut made live during
  this plan's execution for cost (see `tracking_set.json`'s own `deviates_from_ledger`
  field for the full reasoning).
- [x] ⚠️ **Was the measuring tool sound?** The inherited three curves still match the healthy shapes
  that `instrument-02` recorded.
  **Not directly comparable, and that's expected.** `xrmmlhck` trained a fresh LoRA for one
  epoch (50 steps), so `direction_cosine/mean = 0.035` and `frac_distance_reached/mean =
  0.021` are near-zero, consistent with an essentially untrained adapter — not
  `instrument-02`'s plateaued ~0.4 shape from a converged run. Soundness here means the
  three curves logged real numbers in a sane range, not NaN or constant, which they did;
  it does not mean they traced the trained shape, since nothing here trained long enough to.
- [x] ⚠️ **Did the run respect the environment?** Free-device launch per protocol; outputs on
  `/datasets`.
  **Yes.** `mscluster108` device 1, confirmed free via `nvidia-smi` immediately before launch
  (per `execution-protocol.md`'s shared-device path); `run_dir` under
  `/datasets/mmolefe/poe_repair_min/outputs/showcase/tracking_smoke`.

## Still open

Navigation: ⬅️ [Artefact checks](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents)

- **The panel screenshot instruction 2.2 asks for is still not captured.** Blocked on the
  same thing `runbook/looking-at-what-a-run-produced/reading-a-training-run.md` §3 already flags: neither the W&B nor the
  Playwright MCP connector is usable non-interactively here (Playwright's browser binary
  isn't installed on this node; installing one wasn't done unilaterally). Needs a human step
  (a connector authorization, or a manual browser install) before this can close.
- **Whether experiments A and B should also switch to 50 sampling steps** (up from the
  original launcher's 25) to make the new tracking metrics meaningful at production scale —
  a direct consequence of this plan's bucket design (`cfg.probe.commit_window` needs steps up
  to 25, `where_applied_steps` needs step 22), not yet confirmed against A/B's actual launch
  scripts, which don't exist yet.
