# 🔧 Instrument — wire attention capture into the LoRA inference path

## Description
`_CrossAttnRecorder` already exists and already works — it's what captured attention for the
`teacher_residual` (oracle Mono−PoE) composer, cached at
`/datasets/mmolefe/poe_repair_min/outputs/veracity_attn/pairs/a_cat__x__a_dog/seed_42/`.
It has never been wired to `run_lora_residual_inject`, the actual LoRA deployment path. This
plan adds that wiring (same `unet(...)` forward-pass shape, confirmed by repo check — PEFT
adapter toggling via `disable_adapters()`/`enable_adapters()` is orthogonal to forward hooks),
then runs capture across cat×dog's full seed set at both λ=0 (plain PoE) and λ=1
(LoRA-corrected).

## Purpose
Serves Objective 1 (Instrument) and Definition-of-Done items 1–2. Without this, there is no
LoRA attention data to compare against anything — every downstream plan in this scope depends
on it.

## Goal
Attention `.pt` files exist for both λ=0 and λ=1, all 12 cat×dog seeds, same schema as the
existing `veracity_attn` cache, plus a 12×50 sanity table (seed × timestep) of attention mass
on the dropped concept's token confirming the capture pipeline behaves.

## Tasks
- [x] ⚠️ **[publishable-bar]** Run capture on plain PoE (λ=0), all 12 seeds — zero new code,
  `run_lora_residual_inject` already supports `disable_adapters()`. This is the cheapest
  possible first step and should run before touching the λ=1 wiring.
  DONE 2026-07-27: driver `poe_repair/experiments/mechanism_study/capture_attention.py`
  (`--lambda 0 --capture frozen --seeds 1-12`). 12 seed dirs × 100 files (50 steps ×
  {cat,dog}_branch_poe), schema matches veracity_attn. Δ_sum=0 confirms the adapter is
  never invoked (exact plain PoE). One fix was needed vs the "zero new code" expectation:
  `run_lora_residual_inject` crashed at λ=0 with no adapter attached
  (`disable_adapters()` → "No adapter loaded"); the adapter toggles are now guarded.
  Capture wiring (`attn_capture_dir`/`attn_token_indices`/`attn_capture_lora`) was added to
  `run_lora_residual_inject`, which also satisfies the task-3 wiring below.
  Prompt: reuse the existing `attn_capture_dir`/`attn_token_indices` args already threaded
  through `teacher_residual`'s driver code (`poe_repair/methods/_sampling.py:428-496`), calling
  `run_lora_residual_inject` with the adapter disabled and the same token-index spec used by
  `teacher_residual` (cat/dog token indices, branch mapping updated for the 3-branch
  {A, B, ∅} case since there's no J branch here).
  Output: `/datasets/mmolefe/poe_repair_min/outputs/attn_mechanism/plain_poe/a_cat__x__a_dog/seed_<N>/`
- [x] ⚠️ **[publishable-bar]** Build a small script that loads the 12 seeds' `.pt` files and
  prints/plots `A_missing(λ=0, seed, t)` as a 12×50 table — the checkpoint that confirms the
  capture pipeline works before any new wiring or code is written.
  DONE 2026-07-27: `sanity_table.py` prints the full 12×50 seed×timestep table (no missing
  cells) + a heatmap PNG per token; `view_attention.py` adds the qualitative view (per-step
  32×32 map strip + grounded overlay on the decoded sample). Seed-1 strip shows both cat and
  dog tokens attending to a single central blob — the expected PoE mode-collapse signature,
  the λ=0 baseline the LoRA (task 4) is read against.
- [x] ⚠️ **[publishable-bar]** Wire `attn_capture_dir`/`attn_token_indices` into
  `run_lora_residual_inject`'s LoRA-enabled (λ=1) forward pass — mechanical addition, wrap the
  `_three_branch_forward()` call in the same `_CrossAttnRecorder(unet, keep_grad=False)`
  context manager `teacher_residual` already uses (`_sampling.py:1351`, used at line 469).
  DONE 2026-07-27 (landed together with the λ=0 task): `_three_branch_forward_capture()` wraps
  the forward in `_CrossAttnRecorder`; `attn_capture_lora=True` selects the adapter-ON forward.
  Still to run: the λ=1 capture itself (next task) needs the trained checkpoint
  `lora_step_062500.pt`, which is NOT yet on disk under outputs/ — locate/produce it first.
- [x] ⚠️ **[publishable-bar]** Run capture on the LoRA-corrected pass (λ=1, checkpoint
  `lora_step_062500.pt`), all 12 seeds.
  Output: `/datasets/mmolefe/poe_repair_min/outputs/attn_mechanism/lora_lambda1/a_cat__x__a_dog/seed_<N>/`
  DONE 2026-07-27. Checkpoint is NOT under outputs/ (latest.json points to a stale path); it lives
  under `artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/taskB__k04_ep2000_resumed__wandb-pueuo7bl/checkpoints/lora_step_062500.pt`
  (pooled seeds 1-4, verdict "ok", 420 LoRA tensors). Ran
  `capture_attention.py --lambda 1 --capture lora --checkpoint <that>`. 12 seed dirs × 100 files.
  Δ_sum ≈ 900-1730 per seed (nonzero → adapter active), vs Δ_sum=0 at λ=0.

  PRELIMINARY FINDING (feeds DoD-3 prerequisite check): the LoRA changes attention (~13% late-window
  movement, peaks shift 6-13px between regimes) but does NOT separate the concepts on a
  peak-location measure: cat-token peak and dog-token peak stay ~2-4px apart in BOTH λ=0 and λ=1
  (median 3.2px, only 34% of steps ≥6px, mostly early-noise). Both tokens keep attending to one
  object. Meanwhile the decoded image DOES change (37/255 mean pixel diff λ0→λ1). So on this measure
  the fix is not visible as attention concept-separation. NOTE: whole-map cosine is saturated
  (~1.000 both regimes) and is the WRONG metric; the real study metric is the commitment-window
  Δ_attn in plan 03. Interactive inspector (both regimes, honest metrics, decoded λ0/λ1 samples):
  claude.ai artifact 0ca0dc95-424d-485f-bb89-1c3f98c3536d.

## What is actually happening here (plain read, 2026-07-27)

The plan asked one question: when the LoRA fixes a cat×dog picture, does it fix it by moving
where the words "cat" and "dog" look on the canvas? We now have the data to answer, and the
answer is not the clean one we hoped for.

Three plain facts, each with the number that backs it:

1. The LoRA is doing real work. With the LoRA off, the correction it adds is exactly zero
   (Δ_sum = 0, by construction). With it on, the correction is large (Δ_sum ≈ 900–1730 per
   seed). So the adapter is active, not a no-op.

2. The LoRA does change where attention goes. The attention maps move by about 13% between
   off and on, and their brightest spot shifts 6–13 pixels. So it is not the case that
   attention is untouched.

3. But it does not split the two words apart. The brightest spot for "cat" and the brightest
   spot for "dog" sit almost on top of each other (2–4 pixels apart) whether the LoRA is on or
   off. Both words keep looking at the same one animal. On this measure, the two concepts never
   separate.

Now the tension. The picture clearly does change (the pixels differ by 37/255 between off and
on), and in the training sweep on seed 9 the picture goes from a single chimera to two clean
animals somewhere around checkpoint step 20000. So the fix is real in the output. It just does
not show up as "cat attention here, dog attention there."

My best guess at what this means: the LoRA is not repairing composition by re-aiming attention
at two separate places. It is doing something else — most likely changing what each pixel's
attention *carries* (the content the model writes at a location) rather than *where* the words
point. The attention peak can stay in one place while the thing being painted there splits into
two objects. If that guess holds, the "does the LoRA reinvent test-time attention steering"
question (the whole scope's headline) is leaning toward NO: it fixes the same failure by a
different route than Attend-and-Excite, which is itself a finding worth having.

Two honest caveats before anyone leans on this:
- The peak-location and whole-map-cosine measures are crude. Cosine is saturated (~1.000 in
  both regimes) and cannot tell the two apart; peak-distance is noisy once two animals exist
  because the single brightest pixel hops between them. The real measure is the
  commitment-window Δ_attn in plan 03; treat this as the prerequisite check, not the verdict.
- This is one pair (cat×dog) and, for the training sweep, one seed (9). It shows the shape of
  the answer, not its generality.

What this unblocks: DoD-3's gating question ("does Δ_attn(LoRA) track the visual label at all")
now has a first answer — weakly, on this crude measure. That is enough to justify building the
proper Δ_attn metric (plan 03) and only then running the Attend-and-Excite comparison, rather
than the other way round.

Interactive views: plain-vs-LoRA inspector — claude.ai artifact
0ca0dc95-424d-485f-bb89-1c3f98c3536d; training sweep (seed 9, collapsed→separated) — claude.ai
artifact f2f2938e-99d6-4407-a7dc-ef6641545dbe.

## Recommended skill
▶ `/run-experiment` ✅ — GPU preflight + smoke test, then the capture runs across 12 seeds
   (inference-only, no training loop, but still a cluster job worth the mandatory smoke test
   before committing all 12 seeds).

## Engagement Instructions
```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
ls /datasets/mmolefe/poe_repair_min/outputs/attn_mechanism/plain_poe/a_cat__x__a_dog/ | wc -l   # expect 12 seed dirs
ls /datasets/mmolefe/poe_repair_min/outputs/attn_mechanism/lora_lambda1/a_cat__x__a_dog/ | wc -l   # expect 12 seed dirs
$PY -c "import torch; sd=torch.load('<one .pt path>', weights_only=True); print(sd.keys())"   # loads, has 'map'/'spec'/'step_index'/'timestep'
```
