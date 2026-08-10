# 🎯 Overfit — the LoRA closes the PoE gap Mono-free, and the mechanism is pair-generic

## Description
Show the fix works on one case, then that it is not luck from a single concept pair. We train a
small add-on (a rank-8 LoRA on the cross-attention layers) on the saved correction. After training,
the broken method that just adds two prompts together (PoE) shows both concepts instead of one
blur, and it does so without ever seeing the two prompts joined into one at test time (Mono-free).
Do cat×dog at seed 42 first, then one representative pair from each difficulty group (G1–G4, G6).
This plan absorbs the old phase files 01–04 and 09 (setup, first success, breadth) and the two
failed-corrector controls 05–06, now archived under `plans/shelved/phases/`.

## Purpose
Serves Objective 1 (Overfit) and Definition-of-Done items 1 and 6. This is the foundation. If the
fix only works because cat and dog collide in a special way, or if a simpler method matches it,
every rung above collapses. So we check both.

## Goal
Five single-pair LoRAs (G1–G4, G6). Each one turns the merged single blob into two separate concepts
you can see by eye, and bends the corrected path (PoE+λ·R) toward the target in the MDS plot. Plus
the written-up failure of the external and internal correctors, which is what makes the LoRA result
mean something.

## Latest status + how to see it
**As of 2026-07-22.** Beachhead landed on cat×dog seed 42; G4 (typewriter×cactus) trained, MDS panels owed; G1–G3 single-seed LoRAs owed (0-byte stubs); negative controls done. The trained cell plateaus at ~40% of the PoE→Mono distance (delivery is the first-order limiter).

Owning artifacts (repo root):
- `artifacts/rung1-overfit/lora/a_cat__x__a_dog/seed_42/run__local/checkpoints/lora_step_062500.pt` — 420 LoRA keys. W&B: local run (no id).
- `artifacts/rung1-overfit/lora/a_typewriter__x__a_cactus/seed_42/run__wandb-wag4z592/` — step 80000. W&B `wag4z592`. Do NOT confuse with `xcp40234` (typewriter *cross-seed* pool, rung 2).
- Unlisted, needs classifying: `outputs/lora/a_camel__x__a_desert_landscape` (945M, trained).
- Negative controls: `outputs/group_a_failure/{latent_cnn,latent_unet,frozen_feature_mlp}`.

See it:
```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
CUDA_VISIBLE_DEVICES="" $PY -c "import torch; sd=torch.load('artifacts/rung1-overfit/lora/a_cat__x__a_dog/seed_42/run__local/checkpoints/lora_step_062500.pt',map_location='cpu',weights_only=True); print(len([k for k in sd.get('lora_state',sd) if 'lora' in k.lower()]),'lora keys')"   # 420
$PY scripts/build_lora_manifest.py && bash scripts/run_lora_inspector.sh          # → http://localhost:5050
```

## Tasks
- [x] ✅ Groundwork: veracity (gap reachable), residual diagnostics (target structured), CFG-mask floor. → G01/G02/G03 landed (`plans/shelved/phases/01-03`).
- [x] ✅ Train single-seed LoRA on cat×dog seed 42. → G04 headline: λ=0 byte-identical to PoE, λ=1 two distinct animals by ~ep600. Artifact `artifacts/rung1-overfit/lora/a_cat__x__a_dog/seed_42/run__local/checkpoints/lora_step_062500.pt`.  ✓ verified (loads, 420 lora keys)
- [x] ✅ Train G4 `a_typewriter__x__a_cactus` single-seed. → trained to step 80000 (`artifacts/rung1-overfit/lora/a_typewriter__x__a_cactus/seed_42/run__wandb-wag4z592/`); MDS panels NOT run.
- [ ] ⚠️ **[publishable-bar]** Train G1/G2/G3 single-seed LoRAs, then build their inspector manifests.
  Prompt (`/run-experiment`): `for PAIR in a_dolphin__x__an_ocean_wave a_dog__x__oil_painting_style a_mailbox__x__a_snowfield; do $PY -m poe_repair.experiments.lora --pair $PAIR --seed 42 --split heldout --total-epochs 600 --probe-every-epochs 50 --lr 1e-4 --lora-rank 8; $PY scripts/build_lora_manifest.py --results-root artifacts/rung1-overfit/lora/$PAIR/seed_42/run__local; done` (needs `training_cache/heldout/$PAIR/seed_42/` — build first if absent).
- [ ] ⚠️ **[publishable-bar]** Run the MDS pre-render on G4 (owed from G08): `$PY scripts/build_lora_inspector_mds.py --results-root artifacts/rung1-overfit/lora/a_typewriter__x__a_cactus/seed_42/run__wandb-wag4z592 --pair-slug a_typewriter__x__a_cactus --epochs all --lambdas all --stages collect-static,collect-cells,project,render,update-manifest`
- [ ] ⚠️ **[publishable-bar]** Run the MDS pre-render for G1/G2/G3 once trained (MDS is a separate step from the manifest build): `for PAIR in a_dolphin__x__an_ocean_wave a_dog__x__oil_painting_style a_mailbox__x__a_snowfield; do $PY scripts/build_lora_inspector_mds.py --results-root artifacts/rung1-overfit/lora/$PAIR/seed_42/run__local --pair-slug $PAIR --epochs all --lambdas all --stages collect-static,collect-cells,project,render,update-manifest; done` → closes the MDS-bend half of DoD-1 across the taxonomy.
- [x] ✅ Negative controls: group-A external correctors (latent CNN/UNet/frozen-MLP) + internal-force (attention-overlap, score-alignment). → G05, all fail; the LoRA result is meaningful (`plans/shelved/phases/05-06`, `outputs/group_a_failure/`).  ✓ verified (outputs/group_a_failure/{latent_cnn,latent_unet,frozen_feature_mlp})
- [ ] ⚠️ **[publishable-bar]** Read the five-pair contact sheet; classify each poor/bad/unknown/good; retire or confirm the "Group-6-specific" worry.
- [ ] ⚠️ **[publishable-bar]** Mono-free canary (closes Definition-of-Done item 6): confirm the deploy sampler never uses the joined prompt — at λ=0 the output is byte-identical to plain PoE, on every rung's sampler.

## Recommended skill
▶ `/run-experiment` ✅ — drives the LoRA trainer per pair and captures the probe.
   alt: `/analyze-save` to read each checkpoint's probe against the failure-modes catalog.

## Engagement Instructions
```
$ CUDA_VISIBLE_DEVICES="" $PY -c "import torch; sd=torch.load('artifacts/rung1-overfit/lora/a_cat__x__a_dog/seed_42/run__local/checkpoints/lora_step_062500.pt',map_location='cpu',weights_only=True); print(len([k for k in sd.get('lora_state',sd) if 'lora' in k.lower()]),'lora keys')"   # 420
# Per new pair: probes at artifacts/rung1-overfit/lora/<pair>/seed_42/run__local/probes/epoch_0600/lambda_1.00/ show two concepts; λ=0 canary byte-identical to PoE.
```

**▶ View in the web app:** the **LoRA residual** and **MDS large** tabs of the inspector show this rung's epoch × λ morph and trajectory bend, per pair.
```bash
$PY scripts/build_lora_manifest.py && bash scripts/run_lora_inspector.sh   # → http://localhost:5050 (tunnel line printed)
```
