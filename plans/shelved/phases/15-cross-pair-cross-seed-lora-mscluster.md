# Plan 15 — Cross-pair × cross-seed LoRA, executed on mscluster106

> Parent: [LORA_TAXONOMY_PLAN.md](LORA_TAXONOMY_PLAN.md).
> Re-targeted twin of [14-cross-pair-cross-seed-lora-hippo.md](14-cross-pair-cross-seed-lora-hippo.md).
> Same research question, same pool design, same evaluation. Drops every
> hippo-only piece (clone, env install, rsync, cross-host smoke) because
> the run is staying on mscluster106. Adds the code-level detail that
> Plan 14 glossed: the existing trainer is **not** plug-compatible with
> cross-pair training — its `train_epoch` binds one pair's text
> embeddings for the whole epoch. This plan owns the wrapper, the new
> multi-pair train function, the sampler, the contact sheet, and the
> Δ̄_t bridge extension.

## Research question

Unchanged from Plan 14. Can a **single** rank-8 LoRA, trained on 5
representative pairs × 8 train-pool seeds (40 cells), span the studied
taxonomy (G1–G4 and G6; G5 deferred per Plan 09)? Headline quadrant:
**held-pair × held-seed**.

## Why this plan exists separately from Plan 14

Plan 14 was written as a hippo bring-up document — its longest section
is the procedure for getting a fresh Blackwell shell to a finished run.
On mscluster106:

- Caches for the 5 train pairs at seeds `{1..12, 42}` already exist
  under [/datasets/.../training_cache/heldout/](../../../../../../datasets/mmolefe/poe_repair_min/outputs/training_cache/heldout/) — verified on
  2026-05-26.
- Caches for the 5 held-out sibling pairs at seeds `{9..12}` also
  exist on `/datasets/` — verified on 2026-05-26.
- Python env, `POE_REPAIR_TRAINING_CACHE`, git remote, all already
  configured. Steps 1–6 of Plan 14 collapse to zero work.
- The launch hardware is the RTX 8000 (Plan 08 reference), not
  Blackwell — fp16 with `make_grad_scaler` (not bf16), no `--xformers`
  flag, expected wall time ~30–35 h for 2400 epochs across the 40-cell
  pool (vs. Plan 14's 20 h Blackwell estimate).

The hippo plan stays valid for the day hippo opens up. This plan is the
one to execute now.

## What Plan 14 got wrong about "thin wrapper"

Plan 14, Task B: *"Implement `train_pooled` as a thin wrapper around
the Plan-08 trainer body. Only the loader changes."*

This is incorrect. Concrete evidence from
[poe_repair/experiments/lora/trainer.py:400-440](../poe_repair/experiments/lora/trainer.py#L400-L440):

```python
def train_epoch(
    ...,
    seq_a: torch.Tensor, pool_a: torch.Tensor,
    seq_b: torch.Tensor, pool_b: torch.Tensor,
    seq_e: torch.Tensor, pool_e: torch.Tensor,
    ...
):
```

`train_epoch` takes one frozen tuple of `(A, B, ∅)` text embeddings.
These are looked up once at run start by
[encode_all_prompts](../poe_repair/experiments/lora/main.py#L320) from
`cfg.cell.prompt_a / prompt_b / joint_prompt` — single-pair fields.
The seed-only `train_pooled` works because all 8 seeds share one pair
and therefore one set of embeddings. Cross-pair training needs a
**per-step** embedding lookup. The cached step itself already knows
its seed (`CachedStep.source_seed`); it does not yet know its pair.

Two options:

1. **Per-pair round-robin epochs.** Each epoch trains on one pair's
   cells only; cycle through the 5 pairs across 5 sub-epochs.
   Embeddings swap between sub-epochs. No new `train_epoch` variant.
   Downside: per-pair effective epoch_size shrinks; less fine-grained
   mixing; one bad pair's gradient lingers for a whole sub-epoch.
2. **`train_epoch_multi_pair` with per-step embedding lookup.** Tag
   each cached step with its pair slug; pass a `dict[pair_slug →
   embeddings]` into a new train function; pick the right embeddings
   per step. Plan-08 trainer body untouched; new function lives in
   the cross-pair namespace.

This plan picks **option 2**. Rationale: option 1 would make the
`out/out` headline harder to interpret (gradient ordering becomes a
confounder); option 2 keeps the comparison clean against Plan 10's
seed-only pooled LoRA (which sees uniform random sampling across cells).
The implementation cost is one extra ~150-line file.

## Pool design

Verbatim from Plan 14 — kept here for self-containment. G5 deliberately
omitted per Plan 09.

```yaml
# outputs/cross_pair_lora_pooling/pair_pool.yaml
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
# Note: no pair_slug field — this YAML is shared across all pairs in
# pair_pool. The seed-pool loader's existing pair_slug requirement is
# relaxed for cross-pair use (see Task A2).
train_pool: [1, 2, 3, 4, 5, 6, 7, 8]
held_out:   [9, 10, 11, 12]
```

Each pair also needs its prompts (A, B, joint). These come from the
existing per-pair config dictionary used by
[scripts/build_training_cache.py](../scripts/build_training_cache.py)
and friends. Task A3 below extracts them into a small YAML so the
cross-pair trainer doesn't depend on a Python prompt map.

## Namespace

New code under `poe_repair.experiments.cross_pair_lora_pooling/`:

| Module / file | Status | Role |
|---|---|---|
| `poe_repair/experiments/cross_pair_lora_pooling/__init__.py` | new | Package marker. |
| `poe_repair/experiments/cross_pair_lora_pooling/pair_pool.py` | new | YAML loader with disjoint guard. Mirrors `seed_pool.py`. `--check-only` CLI. |
| `poe_repair/experiments/cross_pair_lora_pooling/pair_prompts.py` | new | Map `pair_slug → (prompt_a, prompt_b, joint_prompt)`. Loads from `outputs/cross_pair_lora_pooling/pair_prompts.yaml`. |
| `poe_repair/experiments/cross_pair_lora_pooling/seed_pool.py` | new | Cross-pair-compatible seed loader: same shape as the cross-seed `seed_pool.py` but the `pair_slug` field is **optional**. Reuses the same disjoint guard. |
| `poe_repair/experiments/cross_pair_lora_pooling/multi_pair_trainer.py` | new | `train_epoch_multi_pair(unet, scheduler, optimizer, dataset, embeddings_by_pair, cfg, state, device, train_dtype, rng, grad_scaler, logger_callback)`. Per-step embedding lookup by `CachedStep.source_pair`. |
| `poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py` | new | The entry point. Loads both pools + prompts; iterates `pair_pool.train × seed_pool.train_pool` to resolve 40 cells; tags each step with `source_pair`; encodes all 5 pairs' embeddings once; calls `train_epoch_multi_pair`. Reuses checkpointing / W&B / config logic from `cross_seed_lora_pooling/train_pooled.py`. |
| `poe_repair/experiments/cross_pair_lora_pooling/sample_crossbar.py` | new | Four-quadrant sampler. For each `(pair, seed)` in the chosen quadrant, swaps the LoRA-loaded UNet's prompts and runs `run_lora_residual_inject`. Emits `cells.jsonl` consumed by the contact sheet and Δ̄_t. |
| `poe_repair/experiments/cross_pair_lora_pooling/contact_sheet.py` | new | Rows × cols layout per quadrant; reference columns from Plans 09 / 10 if found. |
| `poe_repair/experiments/cross_pair_lora_pooling/task_d_bridge.py` | new | Thin wrapper around `cross_seed_lora_pooling/task_d_bridge.py` that consumes `cells.jsonl` and computes the three cosines per `(pair, seed)`. If the underlying bridge already supports the inputs we need, this becomes a 10-line CLI shim. |
| `scripts/cross_pair_lora_pooling/run_all.sh` | new | One-shot driver: validate pools → train → sample → render. mscluster RTX 8000 sizing. |
| `poe_repair/experiments/lora/trainer.py` | **modified, +1 line** | Add `source_pair: str = ""` to `CachedStep`. Backwards-compatible default. |
| `poe_repair/experiments/cross_seed_lora_pooling.seed_pool.load_seed_pool` | reused | The existing loader's `pair_slug` field is required. For cross-pair YAMLs without a `pair_slug`, the new `cross_pair_lora_pooling.seed_pool` loads them; the cross-seed loader is left untouched. |
| `poe_repair/training_cache.resolve_cells` | reused | Called once per pair in `train_pooled.py`. |
| `poe_repair/runtime`, `lora/trainer.attach_lora`, `lora/main.encode_all_prompts`, `make_grad_scaler`, `make_optimizer`, etc. | reused | All untouched. |
| `outputs/cross_pair_lora_pooling/{pair_pool,seed_pool,pair_prompts}.yaml` | new | The three configs that drive the run. |

The new trainer **never** writes into Plan 09 / Plan 10 trees — those
are read-only references for the contact sheets' comparison columns.

## Tasks

Each task is sized to be a coherent unit of work. Sequential.

### Task A — Pools, prompts, and leak guards  (~45 min, code only)

**A1.** Add `source_pair: str = ""` to `CachedStep` in
[poe_repair/experiments/lora/trainer.py:47-57](../poe_repair/experiments/lora/trainer.py#L47-L57).
Default empty string keeps every existing caller compatible.

**A2.** Create [poe_repair/experiments/cross_pair_lora_pooling/__init__.py](../poe_repair/experiments/cross_pair_lora_pooling/__init__.py) (empty file).

**A3.** Create [pair_pool.py](../poe_repair/experiments/cross_pair_lora_pooling/pair_pool.py).
Interface mirrors [seed_pool.py](../poe_repair/experiments/cross_seed_lora_pooling/seed_pool.py):

```python
@dataclass(frozen=True)
class PairPool:
    train: tuple[str, ...]
    heldout: tuple[str, ...]
    source_path: Path

    def assert_disjoint(self) -> None: ...
    def persist_alongside(self, out_dir: Path) -> Path: ...

def load_pair_pool(path: Path | None = None) -> PairPool: ...
```

`assert_disjoint` raises with the offending slug in the message.
`__main__` block prints summary + asserts; used as `--check-only`.

**A4.** Create [seed_pool.py](../poe_repair/experiments/cross_pair_lora_pooling/seed_pool.py)
— a cross-pair-aware variant. Same `SeedPool` shape minus the
required `pair_slug` field (cross-pair YAMLs share one seed pool
across pairs). Copy the existing class; mark `pair_slug` optional
and default to empty. Re-use `assert_disjoint`.

**A5.** Create [pair_prompts.py](../poe_repair/experiments/cross_pair_lora_pooling/pair_prompts.py)
with `load_pair_prompts(path) → dict[pair_slug, PairPrompts]`. The
YAML lives at `outputs/cross_pair_lora_pooling/pair_prompts.yaml`:

```yaml
a_dolphin__x__an_ocean_wave:
  prompt_a: "a dolphin"
  prompt_b: "an ocean wave"
  joint_prompt: "a dolphin and an ocean wave"
a_dog__x__oil_painting_style:
  prompt_a: "a dog"
  prompt_b: "oil painting style"
  joint_prompt: "a dog in oil painting style"
# … repeat for all 10 pairs (5 train + 5 heldout)
```

`PairPrompts` is a dataclass with three string fields. `load_pair_prompts`
verifies every slug in `pair_pool.train ∪ pair_pool.heldout` has an
entry.

**A6.** Write the three YAMLs under [outputs/cross_pair_lora_pooling/](../outputs/cross_pair_lora_pooling/).

**A7.** Negative tests inline (no pytest needed):

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python

# valid
$PY -m poe_repair.experiments.cross_pair_lora_pooling.pair_pool \
    --pair-pool outputs/cross_pair_lora_pooling/pair_pool.yaml --check-only

# broken pair
cat > /tmp/leak_pair.yaml <<'EOF'
train:    [a_cat__x__a_dog]
heldout:  [a_cat__x__a_dog]
EOF
$PY -m poe_repair.experiments.cross_pair_lora_pooling.pair_pool \
    --pair-pool /tmp/leak_pair.yaml --check-only && echo BUG || echo OK

# missing prompt
cat > /tmp/bad_prompts.yaml <<'EOF'
a_cat__x__a_dog: {prompt_a: "a cat", prompt_b: "a dog", joint_prompt: "a cat and a dog"}
EOF
$PY -m poe_repair.experiments.cross_pair_lora_pooling.pair_prompts \
    --prompts /tmp/bad_prompts.yaml \
    --pair-pool outputs/cross_pair_lora_pooling/pair_pool.yaml \
    --check-only && echo BUG || echo OK
```

**Done when**: valid configs load silently; broken configs abort in
< 1 s with a slug-named error.

### Task B — Multi-pair trainer  (~3 h code, ~30 h GPU)

**B1.** Create [multi_pair_trainer.py](../poe_repair/experiments/cross_pair_lora_pooling/multi_pair_trainer.py).
This is the only non-trivial new code. Skeleton:

```python
def train_epoch_multi_pair(
    *,
    unet, scheduler, optimizer, dataset,
    embeddings_by_pair: dict[str, dict[str, torch.Tensor]],
    cfg, state, device, train_dtype, rng, grad_scaler,
    logger_callback,
) -> bool:
    """Same loop as lora.trainer.train_epoch, but picks the right
    (seq_a, pool_a, seq_b, pool_b, seq_e, pool_e) per step based on
    CachedStep.source_pair.

    The cached step's source_pair must be a key in embeddings_by_pair;
    a KeyError aborts the run (loader bug).
    """
```

Implementation: copy the body of `lora.trainer.train_epoch`, refactor
the per-step inner block to:

1. Sample step index uniformly over `len(dataset)` via `rng`.
2. `step = dataset[idx]`.
3. `emb = embeddings_by_pair[step.source_pair]`.
4. Pass `emb["seq_a"], emb["pool_a"], …` into the same forward /
   backward as today.

Everything else (bucket loss running, kill criterion, logger callback)
is unchanged. Keep this file under ~200 lines; do not re-implement
LoRA attach, scheduler init, or anything in the original trainer.

**B2.** Create [train_pooled.py](../poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py).
Structure follows [cross_seed_lora_pooling/train_pooled.py](../poe_repair/experiments/cross_seed_lora_pooling/train_pooled.py)
closely. Differences:

- CLI flags: `--pair-pool PATH` (required), `--seed-pool-path PATH`
  (required), `--pair-prompts PATH` (required). Drop `--k`,
  `--single-seed-pick`.
- `pool = load_pair_pool(...)`, `sp = load_seed_pool(...)`,
  `prompts = load_pair_prompts(...)`.
- Build cells:

  ```python
  all_cells = []
  pair_of_cell = {}
  for pair in pool.train:
      cells = resolve_cells(pair, sp.train_pool, cache_root=cache_root)
      for c in cells:
          pair_of_cell[id(c)] = pair
      all_cells.extend(cells)
  ```

- Build the cached step list and **tag with `source_pair`**:

  ```python
  dataset = []
  for c in all_cells:
      steps = lora_trainer.load_cached_steps(c, guidance_scale=...)
      for s in steps:
          s.source_pair = pair_of_cell[id(c)]
      dataset.extend(steps)
  ```

  (Don't reuse `load_cached_steps_pooled` — it doesn't know about
  pairs. The block above does the per-cell pair tagging.)

- Build embeddings_by_pair:

  ```python
  embeddings_by_pair = {}
  for pair in set(pool.train):
      cfg.cell.pair_slug = pair                      # transient
      cfg.cell.prompt_a = prompts[pair].prompt_a
      cfg.cell.prompt_b = prompts[pair].prompt_b
      cfg.cell.joint_prompt = prompts[pair].joint_prompt
      embeddings_by_pair[pair] = encode_all_prompts(cfg, models, device, dtype)
  cfg.cell.pair_slug = "all_groups"                   # for downstream display
  ```

- Replace the `train_epoch` call with `train_epoch_multi_pair`.
- `cfg.run_id` default: `cross_pair_pooled__r{rank}__lr{lr}__ep{total_epochs}__{ts}`.
- `cfg.run_dir` default: `outputs/cross_pair_lora_pooling/all_groups/{run_id}`.
- Skip inline sampling for the first run (keep `--sample-every-epochs 0`);
  it can be added later but the existing `_inline_sampling` builder is
  pair-bound and would need its own multi-pair pass.

**B3.** Launch command for mscluster106 (RTX 8000):

```bash
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
export CUDA_VISIBLE_DEVICES=0
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python

nohup $PY -m poe_repair.experiments.cross_pair_lora_pooling.train_pooled \
    --pair-pool outputs/cross_pair_lora_pooling/pair_pool.yaml \
    --seed-pool-path outputs/cross_pair_lora_pooling/seed_pool.yaml \
    --pair-prompts outputs/cross_pair_lora_pooling/pair_prompts.yaml \
    --total-epochs 2400 \
    --epoch-size 50 \
    --train-batch-size 1 \
    --lora-rank 8 --lora-alpha 8 --lr 1e-4 \
    --dtype float16 \
    --ckpt-every-epochs 200 \
    --log-every-epochs 100 \
    --output-root outputs/cross_pair_lora_pooling/all_groups \
    --run-id main \
    --wandb-mode disabled \
    > outputs/cross_pair_lora_pooling/all_groups/main.log 2>&1 &
```

**Sizing notes (RTX 8000):**

- `--dtype float16` (not bf16): RTX 8000 lacks bf16 tensor cores; fp16
  with `make_grad_scaler` is the verified Plan-08 recipe.
- `--train-batch-size 1`: SDXL at 1024² with rank-8 LoRA fits in
  ~24 GB on the RTX 8000's 48 GB at fp16.
- No `--xformers`: the env-shipped kernel works fine for sm_75; the
  flag is omitted to match Plan 08's verified setup.
- Epoch budget: 2400 epochs × 50 steps/epoch = 120k optimizer steps.
  Per cell ≈ 75 s of training time isn't the right unit here — per
  step is what matters. Plan 08 measured ~0.9 s/step at fp16 batch=1
  on RTX 8000 → **~30 h wall time for 2400 epochs**. Use `nohup` or
  `tmux`; do not rely on the SSH session.

**Sanity gate during training**: tail the log and confirm the running
bucket loss decreases monotonically through the first 600 epochs. If
loss flatlines or oscillates, the per-step embedding lookup is wrong
(likely cause: every step picked the same pair). Kill and fix before
continuing.

**Done when**: a final checkpoint exists at
`outputs/cross_pair_lora_pooling/all_groups/main/checkpoints/lora_step_<final>.pt`
and the running loss curve in `main.log` decreases monotonically
through the first 600 epochs.

### Task C — Four-quadrant sampler + contact sheets  (~2 h code, ~30 min GPU)

**C1.** Create [sample_crossbar.py](../poe_repair/experiments/cross_pair_lora_pooling/sample_crossbar.py).
For each of the 4 quadrants `{in/in, in/out, out/in, out/out}`, pick
cells per the policy below, call `run_lora_residual_inject` with the
trained adapter, write the PNG into a quadrant-tagged subdir.

Cell-selection policy (matches Plan 14, slightly reduced for runtime):

| Quadrant | Pair | Seed | n_cells |
|---|---|---|---|
| `in/in` | each train pair | one train-pool seed (`pool.train_pool[0]`) | 5 |
| `in/out` | each train pair | all 4 held-out seeds | 5 × 4 = 20 |
| `out/in` | each held-out pair | 2 train-pool seeds (`pool.train_pool[0:2]`) | 5 × 2 = 10 |
| `out/out` | each held-out pair | all 4 held-out seeds | 5 × 4 = 20 |

Total: 55 cells × ~30 s/cell ≈ **30 min**.

Per cell, the sampler:

1. Loads the LoRA checkpoint into a fresh UNet.
2. Encodes that pair's prompts (from `pair_prompts.yaml`).
3. Runs the existing `run_lora_residual_inject` (or whatever the
   trained-LoRA inference entry point is in this repo; verify before
   writing — `sample_heldout.py` calls one).
4. Writes `samples/{quadrant}/{pair_slug}/seed_{S}/lora.png`.
5. Appends one JSON line to `samples/cells.jsonl` with
   `{quadrant, pair_slug, seed, png_path}`.

Output dir layout:

```
outputs/cross_pair_lora_pooling/all_groups/main/samples/
  cells.jsonl
  in_in/   {pair_slug}/seed_{S}/lora.png
  in_out/  ...
  out_in/  ...
  out_out/ ...
```

**C2.** Create [contact_sheet.py](../poe_repair/experiments/cross_pair_lora_pooling/contact_sheet.py).
One PNG per quadrant. Layout from Plan 14:

```
rows    = pairs in that quadrant (5 or 10), grouped train-then-heldout
columns = [PoE | pooled-LoRA (this plan) | per-group LoRA (Plan 10 if found)
           | per-pair LoRA (Plan 09 if found) | mono]
```

For each cell:
- **PoE**: look under `/datasets/.../training_cache/heldout/{pair}/seed_{s}/refs/poe.png` (the existing eval cache populates these via `build_sibling_caches.sh`'s ref-PNG step).
- **pooled-LoRA**: the PNG just written by `sample_crossbar.py`.
- **per-group LoRA (Plan 10)**: `outputs/cross_seed_lora_pooling/{pair}/heldout_pair/seed_{s}/lora.png` if it exists; else a labelled placeholder tile.
- **per-pair LoRA (Plan 09)**: `outputs/lora/{pair}/seed_42/.../lora.png` if found.
- **mono**: from `refs/mono.png` in the cache.

Render each PNG to a 320 px thumbnail; stack into a contact sheet with
row labels (pair_slug) and column labels (method).

Outputs:
- `contact_sheet_in_in.png`
- `contact_sheet_in_out.png`
- `contact_sheet_out_in.png`
- `contact_sheet_out_out.png`  ← paper figure

Plus `quadrant_table.csv` with one row per quadrant:

```
quadrant,n_cells,n_recognisable,pct,notes
in_in,5,,,
in_out,20,,,
out_in,10,,,
out_out,20,,,
```

`n_recognisable` is left blank for human eyeball — that's Phase-4's
acceptance criterion per
[mono_usage_rules](../.claude/projects/-home-mscluster-mmolefe-Playground-PhD-poe-repair-min/memory/mono_usage_rules.md)
discipline, not an auto-fitted metric.

**Done when**: the four contact sheets exist; `cells.jsonl` has 55
lines; `quadrant_table.csv` exists with skeleton rows.

### Task D — Δ̄_t bridge across (pair, seed)  (~30 min)

**D1.** Audit
[cross_seed_lora_pooling/task_d_bridge.py](../poe_repair/experiments/cross_seed_lora_pooling/task_d_bridge.py).
CLI today: `--pooled-run`, `--out-dir`, `--seed-pool-path`. Plan 14
mentions a `--cells` flag — verify whether it exists. If it does,
write a thin wrapper that forwards arguments. If it doesn't, choose
one of:

- Extend the existing bridge with `--cells PATH` (preferred — single
  source of truth across plans).
- Write a fresh `task_d_bridge.py` in the cross-pair namespace that
  re-uses the per-step cosine math but reads the `cells.jsonl`
  produced by `sample_crossbar.py`.

Whichever path is taken, the bridge must compute three cosines per
cell:

1. `cos(ε_LoRA, own Δ_t)` — does the LoRA prediction match this cell's
   stored residual?
2. `cos(ε_LoRA, Δ̄_t^(P))` — does it match the **pair-mean** residual
   (averaged over that pair's 8 seeds)?
3. `cos(ε_LoRA, Δ̄_t^(s))` — does it match the **seed-mean** residual
   (averaged over the 5 pairs at that seed)?

Outputs:
- `task_d_cosines.csv` with columns `pair_slug, seed, quadrant, t, cos_own, cos_pair_mean, cos_seed_mean`.
- `task_d_cosines.png`: one panel per quadrant, three cosine curves
  vs. `t` (mean ± std across cells in that quadrant).

**Done when**: the CSV exists with one row per `(cell, t)`; the PNG
shows the three curves per quadrant. Eyeball the `out/out` panel
against the decision rule below.

### Task E — Read the result, decide on v2  (~30 min, no code)

Apply the per-quadrant classification table from Plan 14 verbatim.
Whole-experiment bucket (Poor / Bad / Unknown / Good / Surprising-good)
flows from the `out/out` quadrant landing.

**Decision for v2 (more train pairs per group)**: hard rule from
Plan 14 — *run v2 iff `out/in` lands `Mixed` AND `in/out` lands `Good`.*
Any other combination is either redundant (`out/in` already `Good`) or
premature (`in/out` failing means the training itself didn't take).

If v2 is triggered, append to `pair_pool.yaml` (2–3 pairs per group,
10–15 total). The 40-cell → ~100-cell scale-up means ~12 GB extra
cache (already free on `/datasets/`) and ~30 h → ~75 h training. v2 is
a separate run-id (`main_v2`), not a resume.

## Risk register (mscluster-specific)

| Risk | Trigger | Response |
|---|---|---|
| `train_epoch_multi_pair` loop binds wrong embeddings per step | First-epoch loss curve has wrong shape (very flat or oscillating) | Kill within 1 epoch. Add an assertion that `dataset[i].source_pair == ` the pair whose embeddings were used. Re-launch. |
| `CachedStep.source_pair` not populated → KeyError on first step | Run dies in `train_epoch_multi_pair` step 1 | Loader bug. Verify the per-cell tagging block in `train_pooled.py` runs before the dataset is fed in. Cheap to catch — fail-fast in the new train function. |
| 40-cell dataset OOMs at load (collectively ~10 GB of cached residuals in RAM) | `load_cached_steps` first call ENOMEM | Cells are ~250 MB each at fp32 → 40 × 250 MB ≈ 10 GB. mscluster106's RAM is fine, but worth confirming `free -g` shows ≥ 20 GB before launch. If tight, change `load_cached_steps` to lazy-load per step (out of scope — flag for follow-up). |
| Pair pool too small → LoRA overfits 5 pairs | `out/out` collapses while `in/in` thrives | Run v2 per the decision rule. |
| Cross-pair pool shifts per-cell residual norm distribution | Task D shows cos collapse on all three references | Add a per-cell residual normaliser to the loader (out of scope here; flag follow-up). |
| W&B mode silently records under wrong run if `--run-id main` collides with an earlier run | Second launch overwrites first | Use `--wandb-mode disabled` for the long run; persist artefacts purely on disk. |
| Long-running `nohup` killed by login-node housekeeping | `main.log` stops writing mid-train | Trainer checkpoints every 200 epochs. Inspect `latest.json`, relaunch with `--resume-from <latest>`. Verify the existing seed-pool resume path works for the new entry point (it should — both use the same `lora_trainer.attach_lora` + checkpoint format). |
| Plan 10 / Plan 09 PNGs missing for some cells (sparse reference columns) | Contact sheet has placeholder tiles | Acceptable — the rendered placeholder is labelled "no plan-10 ref"; the headline column is the pooled LoRA, not the references. |
| The cross-pair `seed_pool.yaml` (no `pair_slug` field) breaks the existing cross-seed loader if accidentally pointed at it | RuntimeError on `pair_slug` missing | Keep the cross-pair loader strictly inside `cross_pair_lora_pooling.seed_pool`; do not import the cross-seed one. The two YAMLs live in different directories. |

## What this plan does *not* do

Verbatim from Plan 14:

- **Architecture sweeps.** Rank 8, `attn2` targets, same as the rest
  of the arc. If the run lands `Unknown` or `Bad`, the *first*
  follow-up (flagged, not scoped here) is rank ∈ {16, 32} and
  `attn1 + attn2` targets — not a new trainer.
- **Outcome supervision.** No DRaFT / DDPO / hypernet.
- **Cross-cell aggregation metrics as the headline.** Eyeball contact
  sheets remain primary; Task D and per-quadrant counts are supporting
  evidence. Follows
  [framing_discipline](../.claude/projects/-home-mscluster-mmolefe-Playground-PhD-poe-repair-min/memory/framing_discipline.md).
- **Generalisation outside the five studied groups.** G5 deferred per
  Plan 09.

mscluster-specific exclusions:

- **No hippo bring-up.** Plan 14 keeps that role.
- **No inline sampling during training.** The existing
  `_inline_sampling.build_context` is pair-bound. Skip for run 1; add
  in a follow-up if the long run is worth interactive monitoring.

## Estimated total wall time

| Phase | Code | GPU | Human |
|---|---:|---:|---:|
| Task A | 45 min | 0 | — |
| Task B (write) | 3 h | 0 | — |
| Task B (train) | 0 | ~30 h | — |
| Task C (write) | 2 h | 0 | — |
| Task C (sample) | 0 | ~30 min | — |
| Task D | 30 min | ~5 min | — |
| Task E (read result) | 0 | 0 | ~30 min |
| **Total** | **~6 h** | **~31 h** | **~30 min** |

Code work can land before kicking off Task B's long GPU run, so the
critical path is **~6 h human + ~31 h wall**.

## Status — 2026-05-26

| Item | Done | To do |
|---|:---:|:---:|
| Pool design (5 train + 5 heldout, G5 deferred) | ✅ (inherited from Plan 14) | |
| Seed pool design (`{1..8}` train, `{9..12}` heldout) | ✅ | |
| Cache cells: 5 train pairs × seeds `{1..8}` on mscluster | ✅ (verified `{1..12, 42}` present) | |
| Cache cells: 5 sibling pairs × seeds `{9..12}` on mscluster | ✅ (verified) | |
| Env, `POE_REPAIR_TRAINING_CACHE`, git remote | ✅ | |
| `CachedStep.source_pair` field | | ⬜ (Task A1) |
| `cross_pair_lora_pooling/__init__.py` + `pair_pool.py` | | ⬜ (Task A2–A3) |
| `cross_pair_lora_pooling/seed_pool.py` + `pair_prompts.py` | | ⬜ (Task A4–A5) |
| `outputs/cross_pair_lora_pooling/{pair_pool,seed_pool,pair_prompts}.yaml` | | ⬜ (Task A6) |
| Leak-guard smoke (positive + negative) | | ⬜ (Task A7) |
| `multi_pair_trainer.train_epoch_multi_pair` | | ⬜ (Task B1) |
| `cross_pair_lora_pooling/train_pooled.py` | | ⬜ (Task B2) |
| Train pooled LoRA across 40 cells on mscluster | | ⬜ (Task B3, ~30 h) |
| `sample_crossbar.py` + `cells.jsonl` | | ⬜ (Task C1) |
| `contact_sheet.py` + four quadrant PNGs + `quadrant_table.csv` | | ⬜ (Task C2) |
| Task D bridge (audit + extend or shim) | | ⬜ (Task D1) |
| Per-quadrant classification + bucket landing | | ⬜ (Task E) |
| Optional richer-pool v2 | | ⬜ (conditional on `out/in = Mixed AND in/out = Good`) |
