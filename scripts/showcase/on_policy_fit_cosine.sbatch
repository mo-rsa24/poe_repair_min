#!/bin/bash
#SBATCH --job-name=onpolicy_fit_r8
#SBATCH --partition=bigbatch
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/showcase/on_policy_fit_r8_030000/slurm_%j.log
set -euo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/showcase/on_policy_fit_r8_030000
CKPT=$REPO/artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k/checkpoints/lora_step_030000.pt
cd "$REPO"; mkdir -p "$OUT"
USE_PCT=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ "$USE_PCT" -ge 90 ] && { echo "disk guard: $OUT at ${USE_PCT}%" >&2; exit 3; }
[ -x "$PY" ] || { echo "co3 python not found at $PY" >&2; exit 2; }
[ -f "$CKPT" ] || { echo "checkpoint missing: $CKPT" >&2; exit 2; }
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv || { echo "no GPU visible" >&2; exit 4; }
echo "[$(date -u +%H:%M:%S)] node=$(hostname) disk_use=${USE_PCT}%"
$PY scripts/showcase/on_policy_fit_cosine.py --checkpoint "$CKPT" --out "$OUT" --rank 8 --alpha 8
echo "[$(date -u +%H:%M:%S)] DONE exit=$?"
