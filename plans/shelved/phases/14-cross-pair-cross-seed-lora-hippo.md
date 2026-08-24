# Plan 14 — Cross-pair × cross-seed LoRA on hippo (taxonomy-spanning deployable)

> Parent: [LORA_TAXONOMY_PLAN.md](LORA_TAXONOMY_PLAN.md). Sibling to
> [11-lora-cross-pair-cross-seed.md](11-lora-cross-pair-cross-seed.md):
> same research question, but targeted at the **hippo** Blackwell node
> rather than mscluster. Hippo does not share a filesystem with the
> login node; the only bridge is git. This plan therefore specifies the
> namespace, the pooling/eval design, **and** a self-contained
> bring-up procedure that takes a clean hippo shell to a finished run.

## Research question

Can a **single** rank-8 LoRA, trained on five representative pairs × eight
train-pool seeds (40 cells), cover the studied composition taxonomy
(G1–G4 and G6; G5 deferred per Plan 09)? Or does the deployable artefact
have to remain a per-group catalogue, as in Plan 10?

Held-out at evaluation time along **both** axes:

- **Pair axis**: five sibling pairs, one per studied group, never seen
  at training time.
- **Seed axis**: seeds `{9, 10, 11, 12}`, never seen at training time
  for any pair.

The headline quadrant is **held-pair × held-seed** — the deployment
crossbar. The other three quadrants are calibration: they bound how to
read the headline.

## Why this plan exists separately from Plan 11

Plan 11 specified the experiment for the mscluster RTX 8000 + shared
`/datasets/` filesystem. Hippo is:

- A separate Blackwell node (32 GB VRAM).
- No shared filesystem with mscluster106 — no `/datasets/`, no shared
  `outputs/`, no shared conda env.
- Only bridge to mscluster is `git`. Caches do **not** transfer via
  rsync in this plan because hippo can build them locally faster than
  rsync from mscluster106 can ship 40+ cells over the WAN.

Plan 11's commands assume `/datasets/mmolefe/...` is mounted and that
the cache cells already exist. On hippo neither is true. This plan is
the version that can be handed to a fresh hippo checkout and run.

## Namespace

New code under `poe_repair.experiments.cross_pair_lora_pooling/`:

| Module / file | Status | Role |
|---|---|---|
| `poe_repair.experiments.cross_pair_lora_pooling.pair_pool` | new | Loads `pair_pool.yaml`. Enforces `train ∩ heldout = ∅`. Mirrors the seed-pool loader's contract. `--check-only` mode for the leak guard. |
| `poe_repair.experiments.cross_pair_lora_pooling.train_pooled` | new | Wraps the Plan-08 trainer body. Replaces the seed-only loader with a `(pair, seed)` pool iterator. Inherits LoRA-rank/lr/epoch/checkpoint args verbatim. Adds `--pair-pool` and reuses `--seed-pool-path`. |
| `poe_repair.experiments.cross_pair_lora_pooling.sample_crossbar` | new | Four-quadrant sampling. For each `(pair, seed)` chosen by the quadrant policy (`in/in`, `in/out`, `out/in`, `out/out`), calls `run_lora_residual_inject` with the trained adapter; tags each output dir with `quadrant=<...>`. |
| `poe_repair.experiments.cross_pair_lora_pooling.contact_sheet` | new | Renders the held-pair × held-seed sheet (the paper figure) and the three calibration sheets. Rows = pairs, cols = `{PoE, pooled-LoRA, per-group-LoRA from Plan 10 if available, per-pair-LoRA from Plan 09 if available, mono}`. |
| `scripts/cross_pair_lora_pooling/train_all_groups.sh` | new | One-shot wrapper: validates pools → trains → samples crossbar → renders contact sheets. Sized for one 32 GB Blackwell GPU. |
| `poe_repair.experiments.held_out_seeds.seed_pool` | reused verbatim | Seed-pool YAML loader. |
| `poe_repair.experiments.held_out_seeds.task_d_bridge` | reused with `--pair`/`--seed` | Δ̄_t bridge per `(pair, seed)`. Inputs are per-cell. |
| `scripts/build_training_cache.py`, `scripts/build_eval_cache.py` | reused | Cache builders. Same scripts used on mscluster. |
| `scripts/cross_seed_lora_pooling/build_sibling_caches.sh` | reused | Coordinated with Plan 10's driver task: builds the five sibling-pair caches at seeds `{9..12}`. Idempotent. |

The trainer **never** writes into Plan 09 (`outputs/lora/<pair>/seed_42/`)
or Plan 10 (`outputs/cross_seed_lora_pooling/<pair>/`) trees — those are
read-only references for the cross-comparison columns of the contact
sheets.

## Pool design

```yaml
# outputs/cross_pair_lora_pooling/pair_pool.yaml
# G5 deliberately omitted — see Plan 09 for the deferral rationale.
train:
  - a_dolphin__x__an_ocean_wave           # G1 representative
  - a_dog__x__oil_painting_style          # G2 representative
  - a_mailbox__x__a_snowfield             # G3 representative
  - a_typewriter__x__a_cactus             # G4 representative
  - a_cat__x__a_dog                       # G6 representative
heldout:
  - a_polar_bear__x__an_iceberg           # G1 sibling
  - a_cat__x__charcoal_drawing_style      # G2 sibling
  - a_fire_hydrant__x__a_snowfield        # G3 sibling
  - a_drum_set__x__a_snowman              # G4 sibling
  - a_wolf__x__a_husky                    # G6 sibling
```

```yaml
# outputs/cross_pair_lora_pooling/seed_pool.yaml
train_pool: [1, 2, 3, 4, 5, 6, 7, 8]
held_out:   [9, 10, 11, 12]
```

### Seed-count resolution

Eight training seeds chosen to match Plan 08's k=8 result (the cell at
which seed-pool pooling first generalised on `cat × dog`). Held-out
seeds `{9..12}` chosen to **share the cache** with Plan 10's
held-out-pair driver and Plan 12's held-pair evaluator — building these
cells once serves three plans.

### Leak-guard contract

Both pair-level and seed-level leak guards must fire at load:

- `pair_pool.load_pair_pool` aborts if `set(train) & set(heldout)` is
  non-empty.
- `seed_pool.load_seed_pool` (already exists) aborts if
  `set(train_pool) & set(held_out)` is non-empty.
- The trainer entry point calls **both** loaders before any GPU work.
  A deliberately broken pair YAML must abort within ~1 s with a clear
  message naming the offending pair slug.

Task A below verifies the guard with intentional bad YAMLs.

### Optional richer-pool v2

Triggered **only** if the 5-pair baseline lands `Good` or `Surprising-good`
in the held-pair × held-seed quadrant (see "How to read the result"
below). v2 extends `train` to 2–3 pairs per group (10–15 pairs total),
keeping the held-out siblings out. Cache cost: ~5–10 extra pairs ×
8 seeds = 40–80 extra cells. Run only if Task C's hinge result
warrants it.

## Evaluation design

### Four-quadrant crossbar layout

Two axes:

- **Pair axis** (rows of the sheet): `in-pair` = pair was in
  `pair_pool.train`; `held-pair` = pair was in `pair_pool.heldout`.
- **Seed axis** (columns of the sheet): `in-seed` = seed was in
  `seed_pool.train_pool`; `held-seed` = seed was in `seed_pool.held_out`.

The four quadrants:

| Quadrant | Pair | Seed | n_cells sampled | What it tests |
|---|---|---|---|---|
| `in/in` | seen | seen | 5 (one per group, one seed each) | Training sanity floor. Pooled LoRA must at least match Plan-09 single-cell quality. |
| `in/out` | seen | unseen | 5 × 4 = 20 | Pure seed generalisation. Should match Plan-10's per-group pooled result on the same pair. |
| `out/in` | unseen | seen | 5 × 8 = 40 (subsample 5×2=10) | Pure within-group pair transfer, without seed confound. Mirrors Plan-10's `--heldout-pair` smoke. |
| `out/out` | unseen | unseen | 5 × 4 = 20 | **The deployment crossbar.** Headline. |

### Held-pair × held-seed contact sheet

One PNG per quadrant. The `out/out` sheet is the paper figure. Layout:

```
rows    = 10 pairs (5 train + 5 heldout, stacked train-block-then-heldout-block)
columns = [PoE | pooled-LoRA (this plan) | per-group LoRA (Plan 10) | per-pair LoRA (Plan 09) | mono]
```

Cells with no Plan-10 / Plan-09 reference render a placeholder. Mono is
the diagnostic ceiling (per
[mono_usage_rules](../.claude/projects/-home-mscluster-mmolefe-Playground-PhD-poe-repair-min/memory/mono_usage_rules.md))
— present in the sheet but labelled as the ceiling, never deployed.

### Per-quadrant classification table

For each quadrant, count `n_recognisable / n_cells` by eyeball, using
the Phase-4 acceptance criterion (subject of the prompt visibly
composed; not the chimera). Report:

| Quadrant | n_cells | n_recognisable | % | Notes |
|---|---|---|---|---|

Persist as `outputs/cross_pair_lora_pooling/all_groups/main/quadrant_table.csv`
plus a rendered Markdown copy alongside the contact sheets.

### Decision rule for v2 (optional richer-pool variant)

The **hinge quadrant is `out/in`** (held-pair × in-seed). Reasoning:

- If `out/in` is `Good` (≥ 60% recognisable, per-group structure intact)
  but `out/out` degrades, the limitation is the seed axis, not pair
  coverage — v2 (more pairs) will not help.
- If `out/in` is `Mixed` or worse, the LoRA is failing pair transfer
  even before the seed held-out is layered on. v2 (more pairs per group)
  is the right intervention — it's a coverage problem, not a capacity
  problem.

Hard rule: **run v2 iff `out/in` lands `Mixed` AND `in/out` lands `Good`**.
Any other combination either makes v2 redundant (`out/in` already `Good`)
or makes it premature (`in/out` failing means the training itself didn't
take, so fix that first).

## Hippo bring-up procedure

Sequential. Each step assumes the previous step landed; nothing is
parallel. Times are wall-clock estimates on hippo's 32 GB Blackwell.

### 0. State assumed at the login node (mscluster106)

Code is on `main` (or a feature branch); cache for the 5 representative
pairs at seeds `{1..8}` exists or is in flight under
`/datasets/mmolefe/poe_repair_min/outputs/training_cache/heldout/`.

### 1. Push code from mscluster106 to git remote

```bash
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
git status                                  # confirm clean / commit local changes
git push origin main                        # or the feature branch
```

The remote is `git@github.com:mo-rsa24/poe_repair_min.git` (confirmed
via `git remote -v`).

### 2. On hippo: clone (or pull) the repo

Agreed working directory on hippo (proposed):
`~/Playground/poe_repair_min` — same shape as mscluster, no
`PhD` segment because hippo isn't a research-account host.

```bash
# Fresh:
mkdir -p ~/Playground && cd ~/Playground
git clone git@github.com:mo-rsa24/poe_repair_min.git
cd poe_repair_min

# Existing checkout:
cd ~/Playground/poe_repair_min
git fetch origin
git checkout main
git pull --ff-only
```

### 3. Resolve the dataset location on hippo

`/datasets/mmolefe/...` does **not** exist on hippo. Pick a hippo-local
cache root and export it via `POE_REPAIR_TRAINING_CACHE`, per the
[storage_layout](../.claude/projects/-home-mscluster-mmolefe-Playground-PhD-poe-repair-min/memory/storage_layout.md)
contract. Proposed path (confirm during bring-up):

```bash
export POE_REPAIR_TRAINING_CACHE=$HOME/data/poe_repair_min/training_cache
mkdir -p "$POE_REPAIR_TRAINING_CACHE"
```

Pin this in `~/.bashrc` on hippo so future shells inherit it.

Expected disk footprint (sized from the mscluster cache):

- Per cell: ~50 MB (residuals + embeddings + meta).
- Training cells: 5 pairs × 8 seeds = 40 cells ≈ **2 GB**.
- Eval cells (sibling pairs at `{9..12}`): 5 × 4 = 20 cells ≈ **1 GB**.
- v2 extension: +40–80 cells ≈ +2–4 GB.
- Sample outputs + checkpoints: ~5–10 GB.
- **Total budget: ≤ 20 GB on hippo `$HOME`.**

If `$HOME` is quota-limited, route to a scratch volume hippo provides
and re-`export` accordingly.

### 4. Python environment on hippo

The mscluster path `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`
is not visible from hippo. Mirror the env locally:

```bash
# Once, on hippo:
curl -L -o ~/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash ~/miniforge.sh -b -p ~/miniforge3
~/miniforge3/bin/conda create -n co3 python=3.10 -y
~/miniforge3/envs/co3/bin/pip install -e .   # from the repo root
~/miniforge3/envs/co3/bin/pip install -r requirements.txt   # if present
```

Blackwell-specific: install a PyTorch build with CUDA 12.4+ support
(Blackwell sm_100). On the version of torch shipped in the existing
env, check `torch.cuda.is_available()` and a 1024×1024 SDXL forward
pass before committing to a long run.

Export the python path used by the shell runners:

```bash
export PY=$HOME/miniforge3/envs/co3/bin/python
```

### 5. Build the training cache on hippo (5 pairs × 8 seeds)

```bash
cd ~/Playground/poe_repair_min
export CUDA_VISIBLE_DEVICES=0

# One pair shown; loop over the five representative pairs in pair_pool.train.
for spec in \
  "a dolphin|an ocean wave|a dolphin and an ocean wave" \
  "a dog|oil painting style|a dog in oil painting style" \
  "a mailbox|a snowfield|a mailbox in a snowfield" \
  "a typewriter|a cactus|a typewriter and a cactus" \
  "a cat|a dog|a cat and a dog"
do
  IFS='|' read -r A B J <<< "$spec"
  for s in 1 2 3 4 5 6 7 8; do
    $PY -m scripts.build_training_cache \
        --prompt-a "$A" --prompt-b "$B" --joint-prompt "$J" \
        --seed "$s" --split heldout
  done
done
```

Per cell ~75 s on Blackwell ⇒ 40 cells ≈ **50 min** total.
`build_training_cache.py` is idempotent — re-running skips cells whose
final `residuals/step_<N-1>.pt` already exists.

### 6. Build the eval cache on hippo (5 sibling pairs × 4 held-out seeds)

Reuse the existing driver shared with Plan 10 / Plan 12:

```bash
cd ~/Playground/poe_repair_min
CUDA_VISIBLE_DEVICES=0 SEEDS="9 10 11 12" \
    bash scripts/cross_seed_lora_pooling/build_sibling_caches.sh
```

20 cells ≈ **25 min**. Pass `SKIP_REFS=1` to skip the PoE/Mono PNGs
(~1 s/cell instead of ~75 s) if the reference panels are not needed.

### 7. Build the pool YAMLs and verify the leak guards (Task A)

```bash
mkdir -p outputs/cross_pair_lora_pooling
# Author the two YAMLs from the templates in "Pool design" above, then:

$PY -m poe_repair.experiments.cross_pair_lora_pooling.pair_pool \
    --pair-pool outputs/cross_pair_lora_pooling/pair_pool.yaml \
    --check-only
$PY -m poe_repair.experiments.held_out_seeds.seed_pool \
    --seed-pool-path outputs/cross_pair_lora_pooling/seed_pool.yaml \
    --check-only

# Negative test:
cat > /tmp/leak_pair.yaml <<'EOF'
train:    [a_cat__x__a_dog]
heldout:  [a_cat__x__a_dog]
EOF
$PY -m poe_repair.experiments.cross_pair_lora_pooling.pair_pool \
    --pair-pool /tmp/leak_pair.yaml --check-only && echo BUG || echo OK
```

### 8. Training launch (Task B) — sized for 32 GB Blackwell

```bash
cd ~/Playground/poe_repair_min
export CUDA_VISIBLE_DEVICES=0
export POE_REPAIR_TRAINING_CACHE=$HOME/data/poe_repair_min/training_cache
export PY=$HOME/miniforge3/envs/co3/bin/python

$PY -m poe_repair.experiments.cross_pair_lora_pooling.train_pooled \
    --pair-pool outputs/cross_pair_lora_pooling/pair_pool.yaml \
    --seed-pool-path outputs/cross_pair_lora_pooling/seed_pool.yaml \
    --total-epochs 2400 \
    --epoch-size 50 \
    --train-batch-size 1 \
    --lora-rank 8 --lora-alpha 8 --lr 1e-4 \
    --dtype bfloat16 \
    --xformers \
    --ckpt-every-epochs 200 \
    --log-every-epochs 100 \
    --output-root outputs/cross_pair_lora_pooling/all_groups \
    --run-id main \
    --wandb-mode disabled
```

**Sizing notes for 32 GB Blackwell:**

- `--train-batch-size 1` matches Plan 08; one SDXL UNet forward at
  1024² with rank-8 LoRA fits in ~22 GB at `bf16` (Plan 08 ran in fp16
  at ~24 GB on the RTX 8000). bf16 is preferred on Blackwell — no loss
  scaler, more stable for the small rank-8 adapter.
- No gradient accumulation. The pool already gives 40 effective cells
  per epoch; accumulation buys nothing and lengthens wall time.
- Mixed precision: `--dtype bfloat16`. The trainer's
  `make_grad_scaler` no-ops on bf16. Rank-8 doesn't need fp32 master
  weights.
- `--xformers` if available (the flag silently no-ops if the kernel
  isn't built for sm_100; verify with one short epoch before the
  full run).
- Epoch budget: 2400 epochs ≈ Plan 09's per-cell Phase-4 (~600) × 4,
  amortising across the 40-cell pool. Checkpoints every 200 epochs;
  Plan 11's contact-sheet code accepts the latest checkpoint by
  default.

Expected wall time: **~20 h** for 2400 epochs at 40 cells/epoch on
Blackwell. Use `nohup` or `tmux` — hippo SSH sessions should not be
relied upon to stay open for 20 h.

### 9. Sampling + contact sheets (Task C)

```bash
CKPT=$(ls -t outputs/cross_pair_lora_pooling/all_groups/main/checkpoints/lora_step_*.pt | head -1)

$PY -m poe_repair.experiments.cross_pair_lora_pooling.sample_crossbar \
    --checkpoint "$CKPT" \
    --pair-pool outputs/cross_pair_lora_pooling/pair_pool.yaml \
    --seed-pool-path outputs/cross_pair_lora_pooling/seed_pool.yaml \
    --out-dir outputs/cross_pair_lora_pooling/all_groups/main/samples

$PY -m poe_repair.experiments.cross_pair_lora_pooling.contact_sheet \
    --pooled-run outputs/cross_pair_lora_pooling/all_groups/main
```

Sampling: 4 quadrants × ~10 cells/quadrant × 30 s/cell ≈ **20 min**.
Contact sheets: ~2 min.

### 10. Task D — Δ̄_t bridge

```bash
$PY -m poe_repair.experiments.held_out_seeds.task_d_bridge \
    --pooled-run outputs/cross_pair_lora_pooling/all_groups/main \
    --cells outputs/cross_pair_lora_pooling/all_groups/main/samples/cells.jsonl
```

Per cell, cosine alignment of pooled `ε_PoE_lora,t` against (a) that
cell's own `Δ_t`, (b) the pair-averaged `Δ̄_t^(P)`, (c) the
seed-averaged `Δ̄_t^(s)`. Pulls apart whether the LoRA learned a
pair-conditional or seed-conditional correction (or neither). This is
the diagnostic that distinguishes `Good` from `Surprising-good` in
the bucket table.

### 11. Pull artefacts back to mscluster

Channel: **rsync over SSH** (not git — checkpoints and PNGs are
binary and large). Recommended `.rsync-filter` ships only what's
needed for the figure:

```bash
# From mscluster106, pulling from hippo:
rsync -avzP --include='*/' \
    --include='samples/**/*.png' \
    --include='quadrant_table.csv' \
    --include='contact_sheet*.png' \
    --include='task_d_*.csv' \
    --include='task_d_*.png' \
    --include='checkpoints/lora_step_*.pt' \
    --exclude='*' \
    hippo:Playground/poe_repair_min/outputs/cross_pair_lora_pooling/all_groups/main/ \
    /datasets/mmolefe/poe_repair_min/outputs/cross_pair_lora_pooling/all_groups/main_hippo/
```

The pulled run lands under `..._hippo/` to avoid colliding with any
future mscluster-trained run with the same `--run-id`.

Checkpoint size: ~150 MB for rank-8 SDXL. Bring back **one** —
the final one. Earlier checkpoints stay on hippo for resume.

### 12. Smoke test the artefact on mscluster

Before declaring the experiment done, sample one held-pair × held-seed
cell on mscluster using the pulled checkpoint, and visually compare
to the hippo-rendered PNG of the same cell. They should be
bit-equal modulo cuBLAS nondeterminism. Catches any silent dtype /
runtime drift between Blackwell and the RTX 8000.

## Task breakdown (in scope; defined inline)

### Task A — pool leak-guards (~30 min)

Implement `poe_repair.experiments.cross_pair_lora_pooling.pair_pool`
with `load_pair_pool(path) -> PairPool` and a `--check-only` CLI.
Mirror the existing `seed_pool.py` contract: dataclass with `train`
and `heldout` list-of-slugs, `__post_init__` asserts disjoint, raises
with the offending slug in the message. Add the negative-test
fixture in `tests/` (or as a one-liner in this plan's Step 7) and
confirm both YAMLs fail loudly when broken.

**Done when:** valid YAMLs load silently; broken YAMLs abort in < 1 s
with a slug-named error; the seed-pool guard is also exercised on the
same `--seed-pool-path` used by the trainer.

### Task B — pooled training across the 40 cells (~20 h GPU)

Implement `poe_repair.experiments.cross_pair_lora_pooling.train_pooled`
as a thin wrapper around the Plan-08 trainer body. Only the loader
changes: instead of iterating `(cell, seed)` for one pair, iterate
`(pair, seed)` across `pair_pool.train × seed_pool.train_pool`. All
gradients, schedules, checkpointing, and W&B wiring are reused
verbatim. Add the `--pair-pool` flag, forward everything else.

Launch the run via Step 8's command. Monitor via
`tail -F outputs/cross_pair_lora_pooling/all_groups/main/train.log`.

**Done when:** a final checkpoint exists at
`outputs/.../checkpoints/lora_step_<final>.pt`, and the training loss
curve is monotonically decreasing through the first 600 epochs (sanity
that the loader is correctly batching across pairs, not collapsing to
one pair). Failure to decrease == loader bug; stop and fix before
sampling.

### Task C — four-quadrant samples + contact sheets (~30 min)

Implement `sample_crossbar` (Step 9) and `contact_sheet` (Step 9).
`sample_crossbar` picks cells per the quadrant policy and writes a
`cells.jsonl` that `contact_sheet` and `task_d_bridge` both consume.

**Done when:** four `contact_sheet_<quadrant>.png` files exist;
`quadrant_table.csv` has 4 rows; the held-pair × held-seed sheet is
visually inspected and classified into one of the buckets below.

### Task D — Δ̄_t bridge across `(pair, seed)` (~20 min)

Reuse `poe_repair.experiments.held_out_seeds.task_d_bridge`
with the new `--cells` argument (already supported by Plan 08's
version; verify before launch). Per cell, emits three cosine curves
vs. t: against own `Δ_t`, against `Δ̄_t^(P)` (pair-mean), against
`Δ̄_t^(s)` (seed-mean).

**Done when:** the held-pair × held-seed quadrant has either
`cos(ε_LoRA, own Δ_t) > cos(ε_LoRA, Δ̄_t^(P))` (LoRA learned
something pair-conditional — supports `Good`), or the cosines are
indistinguishable (LoRA learned the pair-mean — supports `Mixed`).

## How to read the result

Per-quadrant expected behaviour for each bucket landing (eyeball
criterion; not a fitted metric):

| Quadrant | `Good` | `Mixed` | `Bad` |
|---|---|---|---|
| `in/in` | ≥ 90% recognisable | ≥ 70% | < 50% — training collapsed; stop. |
| `in/out` | ≥ 70%; matches Plan-10 per-pair pooled | ≥ 50% on easier groups; degrades on G6 | mostly PoE-equivalent. |
| `out/in` | ≥ 60%; within-group transfer learned at scale | group-dependent; strong on G1–G3, weak on G4/G6 | no transfer — LoRA is per-pair. |
| **`out/out`** | ≥ 50%, with per-group structure matching Plans 09–10 | per-group structure visible but absolute scores below Plan-10's per-group pooled LoRA | indistinguishable from PoE. |

Whole-experiment buckets:

| Bucket | Landing pattern | Means |
|---|---|---|
| **Poor** | `in/in` fails. | Loader bug — pooling across pairs collapsed to one pair, or the seed pool wasn't loaded. Fix and re-run. |
| **Bad** | All non-trivial quadrants ≤ 30% recognisable; pooled LoRA ≈ PoE. | One LoRA cannot span the taxonomy. Catalogue at Plan-10's per-group granularity is the deployable. |
| **Unknown** | Pooled LoRA matches per-group LoRAs on easy groups, degrades on hard ones (G4/G6). | Taxonomy ordering carries deployment-relevant information. Honest finding: "one LoRA covers G1–G3; G4/G6 need their own." |
| **Good** | `out/out` visually ties per-group LoRAs across all five studied groups; Task D shows cos to own `Δ_t` exceeding cos to `Δ̄_t^(P)` and `Δ̄_t^(s)`. | One LoRA covers the taxonomy. Strongest deployable claim. |
| **Surprising-good** | Pooled LoRA beats Plan-09 per-pair LoRAs on `in/in`, and `out/out` matches per-group LoRAs. | The single LoRA learned a *general* corrector — pair pooling acted as residual-signal augmentation. Motivates a sequel on bigger pools (v2 + beyond). |

## What this plan does *not* do

- **Architecture sweeps.** Rank 8, `attn2` targets, same as the rest
  of the arc. If the run lands `Unknown` or `Bad`, the *first*
  follow-up (flagged, not scoped here) is rank ∈ {16, 32} and
  `attn1 + attn2` targets — not a new trainer.
- **Outcome supervision.** No DRaFT / DDPO / hypernet. Listed in
  Plan 04's scope discussion as future work.
- **Cross-cell aggregation metrics as the headline.** Eyeball contact
  sheets remain primary; Task D and per-quadrant counts are supporting
  evidence. Follows
  [framing_discipline](../.claude/projects/-home-mscluster-mmolefe-Playground-PhD-poe-repair-min/memory/framing_discipline.md).
- **Cross-host artefact sharing other than git + targeted rsync.**
  Hippo and mscluster do not share `/datasets/`; this plan does not
  try to make them.
- **Generalisation outside the five studied groups.** G5 deferred per
  Plan 09; pair pool bounded by the pilot tree.

## Risk register (hippo-specific items first)

| Risk | Trigger | Response |
|---|---|---|
| `torch.cuda.is_available()` is `False` on Blackwell with the shipped torch | First import check in Step 4 | Install a torch wheel with CUDA 12.4+ (sm_100) support; pin in `requirements-hippo.txt`. |
| `xformers` kernel missing for sm_100 | First epoch silently slow | Drop `--xformers`; verify SDPA backend is `EFFICIENT_ATTENTION`. |
| `bf16` loss diverges (rare on rank-8 but possible) | Loss `nan` within first 100 epochs | Fall back to `--dtype float16` with the existing grad scaler. |
| 40-cell pool OOMs at batch=1 on Blackwell at 1024² | First-epoch OOM | Drop to 768² training (sample at 1024² unchanged) — verified to fit in < 20 GB on RTX 8000 in Plan 08 history. |
| Hippo `$HOME` quota too small for cache + outputs | `mkdir` or first cache cell ENOSPC | Move `POE_REPAIR_TRAINING_CACHE` to hippo's scratch volume; document the path in the run log. |
| Cache rebuild on hippo diverges bit-wise from mscluster | Smoke test in Step 12 fails | Acceptable if the *visual* result is unchanged; cuBLAS nondeterminism on sm_100 vs sm_75 is expected. If results diverge visibly, treat as a runtime bug and triage before trusting the trained LoRA. |
| 20 h SSH session drops mid-train | `nohup` not used | Trainer checkpoints every 200 epochs and supports `--resume-from`; restart from the latest checkpoint. |
| Pair pool too small — LoRA overfits to 5 pairs | `out/out` collapses while `in/in` thrives | Run v2 per the decision rule above. |
| Cross-pair pool shifts the per-cell residual norm distribution | Task D shows cos collapse on all three references | Add a per-cell residual normaliser to the loader (out of scope here; flag for a follow-up plan). |

## Status — 2026-05-25

| Item | Done | To do |
|---|:---:|:---:|
| Pair pool design (G1–G4, G6; G5 deferred) | ✅ (above) | |
| Seed pool design (`{1..8}` train, `{9..12}` heldout) | ✅ (above) | |
| Hippo bring-up procedure | ✅ (this plan) | |
| Cache cells: 5 representative pairs at seeds `{1..8}` on **mscluster** | partial (G4/G6 cached at `{1..12, 42}`) | ⬜ for G1, G2, G3 on mscluster |
| Cache cells: 5 representative pairs at seeds `{1..8}` on **hippo** | | ⬜ (Step 5; ~50 min) |
| Cache cells: 5 sibling pairs at seeds `{9..12}` on **hippo** | | ⬜ (Step 6; shared with Plan 10/12 driver) |
| `poe_repair.experiments.cross_pair_lora_pooling.pair_pool` | | ⬜ (Task A) |
| `poe_repair.experiments.cross_pair_lora_pooling.train_pooled` | | ⬜ (Task B) |
| `poe_repair.experiments.cross_pair_lora_pooling.sample_crossbar` | | ⬜ (Task C) |
| `poe_repair.experiments.cross_pair_lora_pooling.contact_sheet` | | ⬜ (Task C) |
| `scripts/cross_pair_lora_pooling/train_all_groups.sh` | | ⬜ |
| Task A — pool YAMLs + leak guards | | ⬜ |
| Task B — train pooled LoRA across 40 cells on hippo | | ⬜ (~20 h Blackwell) |
| Task C — four-quadrant evaluation + contact sheets | | ⬜ |
| Task D — Δ̄_t bridge across `(pair, seed)` | | ⬜ |
| Artefact pull-back to mscluster (rsync over SSH) | | ⬜ (Step 11) |
| Cross-host smoke test on one held cell | | ⬜ (Step 12) |
| Per-quadrant classification table | | ⬜ |
| Optional richer-pool v2 | | ⬜ (conditional on `out/in = Mixed AND in/out = Good`) |
