#!/bin/bash
#SBATCH --job-name=render_teacher_cache
#SBATCH --partition=bigbatch
#SBATCH --nodelist=mscluster58
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/showcase/logs/render_teacher_cache_%j.log
# V6 tasks 1.1 and 1.2: encode the picked mono renders of cells_v57 once.
# mscluster58 probed healthy on 2026-09-17 and is on the 2026-09-12 datasets=OK list.
set -uo pipefail
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python \
  scripts/showcase/build_render_teacher_cache.py \
  --cells artifacts/_shared/cross_pair_pool_configs/cells_v57.json
