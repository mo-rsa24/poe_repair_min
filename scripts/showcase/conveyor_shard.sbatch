#!/bin/bash
#SBATCH --job-name=belt
#SBATCH --partition=bigbatch
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster65,mscluster74,mscluster83
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/logs/belt_%j.log
# One shard of the conveyor on one bigbatch RTX 3090.
#
#   SHARD=0/6 ROUNDS=8 PER=9 sbatch -w mscluster42 scripts/showcase/conveyor_shard.sbatch
#
# bigbatch has no GRES configured, so Slurm does not arbitrate the GPUs and two jobs can land on
# one card. Pin a node per shard with -w. The belt's own guard waits for a busy card as a backstop.
#
# The shard makes this belt's candidate pairs disjoint from every other shard's, so several may run
# at once without two nodes rendering the same cell. The excluded nodes have dead GPUs; a job
# landing there dies on the render guard in one second.
set -uo pipefail
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
echo "=== belt shard ${SHARD:-all} on $(hostname), job $SLURM_JOB_ID, $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
cd "$REPO"
# Slurm pins the allocated card as device 0. The 3090s run the co3 environment, and the _pyenv
# builder is the one that takes PY from here rather than hardcoding the Blackwell python.
GPU=0 \
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python \
BUILDER=build_new_animal_cells_pyenv.sh \
SHARD="${SHARD:?set SHARD, e.g. 0/6}" \
ROUNDS="${ROUNDS:-8}" PER="${PER:-9}" MAXWAIT="${MAXWAIT:-600}" \
  bash scripts/showcase/conveyor.sh
RC=$?
echo "=== belt shard ${SHARD} exit $RC  $(date -Is)"
exit $RC
