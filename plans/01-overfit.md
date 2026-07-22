# 🎯 Overfit — the LoRA closes the PoE gap Mono-free, and the mechanism is pair-generic

## Description
Prove the fix works on the first cell, then that it is not a fluke of one concept pair. Train a
small add-on (a rank-8 LoRA on cross-attention) on the cached correction so the failing
add-the-two-prompts method (PoE) instead shows both concepts, and does so without ever feeding it
the joined prompt at test time (Mono-free). Do it first on cat×dog seed 42, then on one
representative pair from each difficulty group (G1–G4, G6). Folds the old phase files 01–04 and 09
(groundwork, headline, breadth) and the two failed-corrector controls 05–06, now archived under
`plans/phases/`.

## Purpose
Serves Objective 1 (Overfit) and Definition-of-Done items 1 and 6. This is the foundation. If the
fix only works because cat and dog collide in a special way, or if a simpler method matches it,
every rung above collapses. So we check both.

## Goal
Five single-pair LoRAs (G1–G4, G6), each turning the fused chimera into two separate concepts by
eye, and each bending the corrected path (PoE+λ·R) toward the joint target in the MDS plot. Plus the
written-up failure of the external and internal correctors, which is what makes the LoRA result
mean something.

## Tasks
- [x] ✅ Groundwork: veracity (gap reachable), residual diagnostics (target structured), CFG-mask floor. → G01/G02/G03 landed (`plans/phases/01-03`).
- [x] ✅ Train single-seed LoRA on cat×dog seed 42. → G04 headline: λ=0 byte-identical to PoE, λ=1 two distinct animals by ~ep600. Artifact `artifacts/rung1-overfit/lora/a_cat__x__a_dog/seed_42/run__local/checkpoints/lora_step_062500.pt`.  ✓ verified (loads, 420 lora keys)
- [x] ✅ Train G4 `a_typewriter__x__a_cactus` single-seed. → trained to step 80000 (`artifacts/rung1-overfit/lora/a_typewriter__x__a_cactus/seed_42/run__wandb-wag4z592/`); MDS panels NOT run.
- [ ] ⚠️ Train G1/G2/G3 single-seed LoRAs, then build their inspector manifests.
  Prompt (`/run-experiment`): `for PAIR in a_dolphin__x__an_ocean_wave a_dog__x__oil_painting_style a_mailbox__x__a_snowfield; do $PY -m poe_repair.experiments.lora --pair $PAIR --seed 42 --split heldout --total-epochs 600 --probe-every-epochs 50 --lr 1e-4 --lora-rank 8; $PY scripts/build_lora_manifest.py --results-root artifacts/rung1-overfit/lora/$PAIR/seed_42/run__local; done` (needs `training_cache/heldout/$PAIR/seed_42/` — build first if absent).
- [ ] ⚠️ Run the MDS pre-render on G4 (owed from G08): `$PY scripts/build_lora_inspector_mds.py --results-root artifacts/rung1-overfit/lora/a_typewriter__x__a_cactus/seed_42/run__wandb-wag4z592 --pair-slug a_typewriter__x__a_cactus --epochs all --lambdas all --stages collect-static,collect-cells,project,render,update-manifest`
- [x] ✅ Negative controls: group-A external correctors (latent CNN/UNet/frozen-MLP) + internal-force (attention-overlap, score-alignment). → G05, all fail; the LoRA result is meaningful (`plans/phases/05-06`, `outputs/group_a_failure/`).  ✓ verified (outputs/group_a_failure/{latent_cnn,latent_unet,frozen_feature_mlp})
- [ ] ⚠️ Read the five-pair contact sheet; classify each poor/bad/unknown/good; retire or confirm the "Group-6-specific" worry.
- [ ] ⚠️ Mono-free canary (closes Definition-of-Done item 6): confirm the deploy sampler never uses the joined prompt — at λ=0 the output is byte-identical to plain PoE, on every rung's sampler.

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
