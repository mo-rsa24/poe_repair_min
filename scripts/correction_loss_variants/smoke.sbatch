#!/usr/bin/env bash
#SBATCH --job-name=corrloss_smoke
#SBATCH --partition=bigbatch
#SBATCH --time=00:40:00
#SBATCH --output=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/logs/corrloss_smoke_%j.out
#SBATCH --error=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/logs/corrloss_smoke_%j.err
set -uo pipefail
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
export SMOKE=1 GPU=0 ARM="${ARM:?set ARM}"
exec bash scripts/correction_loss_variants/run_arm.sh
