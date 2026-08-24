# Plan 10 — Cross-seed LoRA, per group, with held-out-pair evaluation

> Parent: [LORA_TAXONOMY_PLAN.md](LORA_TAXONOMY_PLAN.md). Generalises
> [08-cross-seed-lora-pooling.md](08-cross-seed-lora-pooling.md) along
> the *group* axis and adds a within-group held-out-pair evaluator.
> Depends on the per-group single-seed artefacts produced by
> [09-lora-taxonomy-single-seed.md](09-lora-taxonomy-single-seed.md).

## Question

For each studied group of the composition taxonomy (G1–G4 and G6; G5
deferred — see Plan 09), take the representative pair from Plan 09,
train a pooled LoRA across the seed axis (same recipe as
[08-cross-seed-lora-pooling.md](08-cross-seed-lora-pooling.md)), and
ask two things:

1. **Within-pair, cross-seed transfer.** Does pooling across seeds for
   that group's representative pair give recognisable composition on
   held-out seeds? Identical question to Plan 08, but asked per group.
2. **Within-group, cross-pair transfer.** Does a LoRA trained on the
   representative pair of group `G` work on a *different* pair from
   group `G` (e.g. a `cat × dog`-trained LoRA evaluated on
   `wolf × husky`)? Exposed via a `--heldout-pair` CLI flag.

The unit of analysis is the *group*, not the pair.

## Why this phase exists

Plan 08 left an open question: is "cross-seed LoRA pooling" a property
of `cat × dog` specifically, or a property of the LoRA mechanism? Plan
09 will (if `Good`) show the *single-seed* mechanism is taxonomy-wide,
but seed-axis generalisation could still be Group-6 specific.

Plan 10 makes the per-group cross-seed claim explicit, and adds the
cheapest possible cross-pair probe — does a group-G LoRA, with no
retraining, work on a sibling pair of the same group? If yes, "group"
is a meaningful unit. If no, the group abstraction is not strong enough
to pool over and Plan 11 (full cross-pair) becomes the only way to make
a "broad" claim.

## Research-objective alignment

The five-thread project framing
([project_framing](../.claude/projects/-home-mscluster-mmolefe-Playground-PhD-poe-repair-min/memory/project_framing.md))
treats LoRA as the deployed thread. For that thread to scale to a
real-world deployment story, the LoRA must (a) work on more than one
seed and (b) be re-usable on similar pairs without per-pair retraining.

This plan is the per-group test of both. Whichever way it lands, the
result is interpretable:

- **Pooling generalises within every group, including across pairs.**
  The deployment story is "one LoRA per difficulty class," a tractable
  catalogue. Strong support for the paper's deployable claim.
- **Pooling generalises within groups for the *same pair* but does not
  transfer to sibling pairs.** The deployment story collapses to "one
  LoRA per pair." Still useful, but Plan 11 becomes the only path to a
  taxonomy-level deployable artefact.
- **Pooling fails on some groups.** Honest negative result — read
  through the lens of
  [framing_discipline](../.claude/projects/-home-mscluster-mmolefe-Playground-PhD-poe-repair-min/memory/framing_discipline.md):
  group-A architectures are reported negatively, and so are LoRA
  groups that fail to pool. This narrows the deployment scope to the
  groups that succeed.

The held-out-pair flag is the cheapest possible diagnostic for whether
the taxonomy ordering carries deployment-relevant information. It
doubles as a hook for Plan 11's two-axis crossbar: any held-out pair
that fails here is a natural inclusion candidate for Plan 11's training
pool.

## Representative pairs (one per group; same as Plan 09)

| # | Group | Pair slug (training) | Sibling pair for `--heldout-pair` smoke test |
|---|---|---|---|
| 1 | Co-occurrence | `a_dolphin__x__an_ocean_wave` | `a_polar_bear__x__an_iceberg` |
| 2 | Factorization | `a_dog__x__oil_painting_style` | `a_cat__x__charcoal_drawing_style` |
| 3 | Object + scene | `a_mailbox__x__a_snowfield` | `a_fire_hydrant__x__a_snowfield` |
| 4 | Dual-object | `a_typewriter__x__a_cactus` | `a_drum_set__x__a_snowman` |
| 5 | Entanglement | *(deferred — not used in this plan)* | — |
| 6 | Concept collision | `a_cat__x__a_dog` *(short slug)* | `a_wolf__x__a_husky` |

Sibling pairs are chosen by hand from the pilot tree to be plausible
matches in difficulty *within their group*. They are exposed via a CLI
flag, not pinned in code — `--heldout-pair <slug>` accepts any pair
from the same group's cache.

**G5 is deferred** for the same reason as in Plan 09: the entangled
third concept is not specified by the joint prompt, so neither the
training target nor the "did it transfer to a sibling" read is
trustworthy. G5 remains in the taxonomy for completeness but is not
trained, evaluated, or cached in this plan.

## Code

This plan reuses Plan 08's pooled-LoRA modules verbatim but generalises
them along the pair axis. The output root for the pooled artefact moves
from `outputs/cross_seed_lora_pooling/` to
`outputs/cross_seed_lora_pooling/<pair-slug>/`, and a new
`--heldout-pair` flag is added to the held-out evaluator.

| Module | Reused / changed | Role |
|---|---|---|
| `poe_repair.experiments.held_out_seeds.seed_pool` | reused | YAML seed-pool loader + leak guard. |
| `poe_repair.experiments.held_out_seeds.train_pooled` | reused; takes `--pair <slug>` | Pooled trainer per pair. |
| `poe_repair.experiments.held_out_seeds.sample_heldout` | **changed**: adds `--heldout-pair <slug>` | Held-out evaluator; if `--heldout-pair` differs from training pair, attaches the trained LoRA and samples on the held-out pair's cache. |
| `poe_repair.experiments.held_out_seeds.step0_prescreen` | reused | Inference-time mono-average pre-screen per pair. |
| `poe_repair.experiments.held_out_seeds.task_d_bridge` | reused | Δ̄_t bridge per pair. |
| `poe_repair.experiments.held_out_seeds.contact_sheet` | reused; takes `--pair` | Renders task B/C/heldout-pair grids per pair. |
| `scripts/cross_seed_lora_pooling/task_b_learning_curve.sh` | reused via `PAIR=` env var | Wraps `train_pooled` + `sample_heldout`. |
| `scripts/cross_seed_lora_pooling/task_c_per_seed_ceiling.sh` | reused via `PAIR=` env var | Per-seed ceiling. |
| `scripts/cross_seed_lora_pooling/heldout_pair.sh` *(new)* | new | Loops over groups (`LORA_GROUPS=G1..G6`), resolves each group's pooled checkpoint via `checkpoints/latest.json`, runs `sample_heldout --heldout-pair <sibling>` for seeds {9..12}. Per-group ckpt override via `G{N}_CHECKPOINT=`. |
| `scripts/cross_seed_lora_pooling/build_sibling_caches.sh` *(new)* | new | Wraps `scripts/build_eval_cache.py` over the five sibling pairs × seeds {9..12}. Required precondition for `heldout_pair.sh`. |
| `scripts/cross_seed_lora_pooling/render_seed_summary.py` | existing | Stays — emits the per-pair seed-summary grid surfaced by the residual inspector. |

The `--heldout-pair` flag is the only sampler-side new code. Its
semantics:

- Load the checkpoint trained on `--pair P_train` with seed-pool
  `train_pool`.
- Resolve the held-out pair `--heldout-pair P_eval`'s cache cells.
- For each held-out seed in `seed_pool.held_out`, sample the held-out
  pair's `(e_A, e_B, e_∅)` through `run_lora_residual_inject` with the
  trained adapter attached.
- Compare against vanilla PoE on the *same* `P_eval` cells — that is
  the headline panel.
- If `P_eval == P_train`, the behaviour is identical to Plan 08 (no
  regression).

## Seed pool

Reuse the same `seed_pool.yaml` shape as Plan 08:

```yaml
train_pool: [1, 2, 3, 4, 5, 6, 7, 8]
held_out:   [9, 10, 11, 12]
```

per-pair file under
`outputs/cross_seed_lora_pooling/<pair-slug>/seed_pool.yaml`. Leak
guard remains `train_pool ∩ held_out = ∅`.

## Cache prerequisites

For each of the five representative pairs (G1–G4, G6):

- 8 cache cells in `train_pool` × 1 pair = 8 cells per group.
- 4 cache cells in `held_out` × 1 pair = 4 cells per group.
- For each sibling pair under `--heldout-pair`, only the `held_out`
  seeds' cells are needed (4 cells per pair).

These live under
`/datasets/mmolefe/poe_repair_min/outputs/training_cache/heldout/<pair-slug>/seed_<N>/`,
resolved through `POE_REPAIR_TRAINING_CACHE`. Group 6 inherits the
existing Plan-08 cells under
`/datasets/mmolefe/poe_repair_min/outputs/training_cache/heldout/a_cat__x__a_dog/`.
Group 4 inherits the existing 13-seed cache at
`/datasets/.../training_cache/heldout/a_typewriter__x__a_cactus/`.
Groups 1, 2, 3 require cache production from scratch (`embeddings.pt`
+ `residuals/step_*.pt`) using `scripts/build_training_cache.py`.

## Commands

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
export CUDA_VISIBLE_DEVICES=1
export POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/outputs/training_cache
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
```

### Step 0 — per-pair pre-screen (no training)

```bash
for PAIR in a_dolphin__x__an_ocean_wave a_dog__x__oil_painting_style \
            a_mailbox__x__a_snowfield a_typewriter__x__a_cactus \
            a_cat__x__a_dog; do
  $PY -m poe_repair.experiments.held_out_seeds.step0_prescreen \
      --pair $PAIR \
      --output-root outputs/cross_seed_lora_pooling/$PAIR/step0_prescreen
done
```

Per pair, the pre-screen writes a contact sheet (PoE / r̄_t-inject /
oracle / mono) over the four held-out seeds. If the r̄_t-inject column
already looks like the joint, pooled training is mostly confirmatory.

### Task B — pooled learning curve, per pair

```bash
for PAIR in a_dolphin__x__an_ocean_wave a_dog__x__oil_painting_style \
            a_mailbox__x__a_snowfield a_typewriter__x__a_cactus \
            a_cat__x__a_dog; do
  PAIR=$PAIR EPOCHS=1600 KSET="1a 4 8" \
      bash scripts/cross_seed_lora_pooling/task_b_learning_curve.sh
done
```

Each call writes to
`outputs/cross_seed_lora_pooling/<pair>/task_b_learning_curve/k<label>__ep1600/`.
Each pooled checkpoint is immediately sampled on its own pair's
held-out seeds with `--record-eps` (for Task D).

### Task C — per-seed ceiling, per pair

```bash
for PAIR in a_dolphin__x__an_ocean_wave a_dog__x__oil_painting_style \
            a_mailbox__x__a_snowfield a_typewriter__x__a_cactus \
            a_cat__x__a_dog; do
  PAIR=$PAIR CEILING_SEEDS="9 10 11 12" EPOCHS=1600 \
      bash scripts/cross_seed_lora_pooling/task_c_per_seed_ceiling.sh
done
```

### Held-out-pair evaluation

Build the five sibling-pair eval-only caches first (seeds 9..12):

```bash
POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/outputs/training_cache \
    bash scripts/cross_seed_lora_pooling/build_sibling_caches.sh
```

Then run the held-out-pair driver:

```bash
POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/outputs/training_cache \
    bash scripts/cross_seed_lora_pooling/heldout_pair.sh
```

For each group, loads that group's pooled checkpoint (k=8, ep=1600) and
calls `sample_heldout` with `--heldout-pair <sibling>` on the four
held-out seeds. Output lands at
`outputs/cross_seed_lora_pooling/<pair>/heldout_pair/<sibling>/`.

Direct invocation:

```bash
$PY -m poe_repair.experiments.held_out_seeds.sample_heldout \
    --checkpoint outputs/cross_seed_lora_pooling/a_cat__x__a_dog/task_b_learning_curve/k08__ep1600/checkpoints/lora_step_<...>.pt \
    --pair       a_cat__x__a_dog \
    --heldout-pair a_wolf__x__a_husky \
    --out-dir    outputs/cross_seed_lora_pooling/a_cat__x__a_dog/heldout_pair/a_wolf__x__a_husky
```

### Task D — Δ̄_t bridge per pair

```bash
for PAIR in a_dolphin__x__an_ocean_wave a_dog__x__oil_painting_style \
            a_mailbox__x__a_snowfield a_typewriter__x__a_cactus \
            a_cat__x__a_dog; do
  $PY -m poe_repair.experiments.held_out_seeds.task_d_bridge \
      --pooled-run outputs/cross_seed_lora_pooling/$PAIR/task_b_learning_curve/k08__ep1600
done
```

### Contact sheets (the final per-group read)

```bash
for PAIR in a_dolphin__x__an_ocean_wave a_dog__x__oil_painting_style \
            a_mailbox__x__a_snowfield a_typewriter__x__a_cactus \
            a_cat__x__a_dog; do
  $PY -m poe_repair.experiments.held_out_seeds.contact_sheet \
      --pair $PAIR --task B
  $PY -m poe_repair.experiments.held_out_seeds.contact_sheet \
      --pair $PAIR --task C
  $PY -m poe_repair.experiments.held_out_seeds.contact_sheet \
      --pair $PAIR --task heldout_pair
done
```

## How to read the result

We classify each *group* into a bucket, then aggregate across groups.

| Bucket (per group) | What you see on this group | Means |
|---|---|---|
| **Poor** | Leak guard does not fire on a deliberately broken seed-pool YAML for that group. | Tooling regression in the per-pair output-root rewrite. Fix before reading. |
| **Bad** | Pooled-k8 on the representative pair fails on ≥ 3 of 4 held-out seeds, and per-seed ceiling (Task C) *also* fails. | The held-out seeds are themselves hard for the representative pair. Note as a seed-difficulty case for that group; do not generalise. |
| **Unknown** | Pooled-k8 works on 2 of 4 held-out seeds; held-out-pair sample is ambiguous. Task D cosine alignment to Δ̄_t is in [0.4, 0.6]. | Real but small effect; do not over-claim within-group transfer. |
| **Good** | Pooled-k8 produces recognisable composition on ≥ 3 of 4 held-out seeds. Held-out-pair sample on the sibling shows composition (not necessarily as cleanly as the trained pair) on ≥ 2 of 4 held-out seeds. Task D cosine to Δ̄_t exceeds the random-seed-pair baseline. | The pooled LoRA generalises along the seed axis *and* transfers within the group. The group is a deployable unit. |
| **Surprising-good** | Pooled-k8 beats per-seed ceiling on most held-outs and held-out-pair sample is as good as the trained pair. | Group-level transfer is genuine; the LoRA is not learning the pair, it's learning the group. Strong evidence for Plan 11. |

Aggregate read across the five studied groups (G1–G4, G6):

- **All five Good or Surprising-good.** Per-group pooling is the
  deployment unit. Plan 11's "one LoRA across all studied groups" is
  the natural next step and likely to work.
- **Mixed.** Per-group pooling works for the "easier" groups (likely
  G1–G3) and degrades on G4/G6 (or vice versa). Report the group-level
  outcome as the taxonomy story; Plan 11's all-groups LoRA may collapse
  to the working subset.
- **All Bad.** Cross-seed pooling does not survive moving off
  `cat × dog`. The Plan-08 result is pair-specific; Plan 11 is unlikely
  to succeed.

## What this plan does *not* do

- **Cross-group transfer.** A G4 LoRA on
  `a_typewriter__x__a_cactus` is *not* evaluated on a G5 sibling.
  That confound goes through Plan 11's two-axis crossbar.
- **Architectural sweeps.** Rank-8, `attn2` targets, same trainer.
  Sweep S1 (rank ∈ {16, 32}) from Plan 08 is *optional* per group and
  only run if Task C indicates the bottleneck is capacity, not data.
- **Outcome supervision.** As in Plan 08, none here.

## Status — 2026-05-23

| Item | Done | To do |
|---|:---:|:---:|
| Plan 08 modules consumed by `--pair <slug>` | ✅ (Plan 08 ships them) | |
| G5 explicitly deferred from this arc | ✅ | |
| Per-pair output roots `outputs/cross_seed_lora_pooling/<pair>/` | | ⬜ (small refactor of the runners) |
| `sample_heldout --heldout-pair <slug>` flag | | ⬜ (new CLI; ~30 lines) |
| `scripts/cross_seed_lora_pooling/heldout_pair.sh` driver | ✅ (2026-05-25; resolves checkpoints via `latest.json`, dry-runs cleanly on G1–G4 k04 ckpts + G6 legacy k04__ep2000_resumed) | |
| `scripts/cross_seed_lora_pooling/build_sibling_caches.sh` helper | ✅ (2026-05-25) | |
| Per-pair `seed_pool.yaml` (5 files; same shape as Plan 08) | | ⬜ |
| Cache cells for the 3 missing representative pairs (G1, G2, G3) at seeds {1..12} | | ⬜ (data side; G4, G6 already present) |
| Cache cells for the 5 sibling held-out-pair slugs at seeds {9..12} | | ⬜ (data side — driver `build_sibling_caches.sh` is ready; awaiting GPU window) |
| Step 0 pre-screen × 5 pairs | | ⬜ |
| Task B (k ∈ {1a, 4, 8}, ep 1600) × 5 pairs | | ⬜ (~5 × 5 h GPU) |
| Task C (4 ceilings, ep 1600) × 5 pairs | | ⬜ (~5 × 4 h GPU) |
| Held-out-pair samples × 5 groups | | ⬜ |
| Task D × 5 pairs | | ⬜ |
| Per-pair contact sheets (B / C / heldout_pair) | | ⬜ |
| Per-group classification table | | ⬜ |
