#!/usr/bin/env bash
# Resume m5 LoRA SDXL from epoch 800 for another 800 epochs.
# Picks up from m5_perarm_resume300_plus500/checkpoints/lora_step_040000.pt.
set -euo pipefail

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m poe_repair.experiments.m5_lora_sdxl \
  --pair a_cat__x__a_dog --seed 42 \
  --total-epochs 800 \
  --lambda-grid "0.0,0.5,1.0" \
  --num-inference-steps 25 \
  --resume-from outputs/m5_lora_sdxl/a_cat__x__a_dog/seed_42/m5_perarm_resume300_plus500/checkpoints/lora_step_040000.pt \
  --run-id m5_perarm_resume800_plus800 \
  2>&1 | tee logs/m5_perarm_resume800_plus800.log
