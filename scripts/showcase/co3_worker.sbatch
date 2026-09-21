#!/bin/bash
#SBATCH --job-name=co3
#SBATCH --partition=bigbatch
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster65,mscluster74,mscluster83
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/logs/co3_%j.log
# One CO3 worker on one bigbatch RTX 3090. bigbatch has no GRES, so pin the node with -w.
#   W=2 sbatch -w mscluster42 scripts/showcase/co3_worker.sbatch
set -uo pipefail
echo "=== co3 worker ${W:?set W} on $(hostname), job $SLURM_JOB_ID, $(date -Is)"
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
GPU=0 bash /datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/w${W}.sh
RC=$?
echo "=== co3 worker ${W} exit $RC $(date -Is)"
exit $RC
