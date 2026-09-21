#!/bin/bash
#SBATCH --job-name=objbelt
#SBATCH --partition=bigbatch
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster65,mscluster74,mscluster83
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/logs/objbelt_%j.log
# One shard of the object x object conveyor on one bigbatch RTX 3090.
#   SHARD=3/4 sbatch -w mscluster42 scripts/showcase/objbelt_shard.sbatch
set -uo pipefail
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
echo "=== object belt shard ${SHARD:?set SHARD} on $(hostname), job $SLURM_JOB_ID, $(date -Is)"
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd "$REPO"
GPU=0 PROPOSER=propose_object_pairs.py BUILDER=build_new_animal_cells_pyenv.sh \
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python \
ROUNDS="${ROUNDS:-10}" PER="${PER:-9}" MAXWAIT="${MAXWAIT:-600}" SHARD="$SHARD" \
  bash scripts/showcase/conveyor.sh
RC=$?
echo "=== object belt shard ${SHARD} exit $RC $(date -Is)"
exit $RC
