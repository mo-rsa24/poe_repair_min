#!/bin/bash
#SBATCH --job-name=pool43
#SBATCH --partition=bigbatch
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster65,mscluster74,mscluster83
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/logs/%x_%j.log
# One child arm on a bigbatch RTX 3090.
#   RUN=pool43-animals CELLS=<json> [STEPRANGE="0 25"] [EXTRA="--orth-weight 3.0"] sbatch ...
#
# The settings shared by every arm are written here rather than passed through --export, because
# --export takes a comma-separated list and SAMPLETRAIN's own value contains commas: passing it
# that way silently splits it into bogus variables.
#
# CKPT=1 is required on these 24 GB cards; gradient checkpointing recomputes activations and
# leaves the gradients identical, so it costs wall clock and nothing else.
set -uo pipefail
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min || exit 7
echo "=== ${RUN:?set RUN} on $(hostname), job $SLURM_JOB_ID, $(date -Is)"
GPU=0 CKPT=1 \
POOLGEN=v57 WD=1e-2 BATCH=1 EPOCHS=600 SAMPLEEVERY=25 \
SAMPLETRAIN="a_lion__x__a_meerkat,a_typewriter__x__a_cactus" SAMPLETRAINCELLS=1 \
SAMPLEHELD="a_cat__x__a_dog,an_elephant__x__a_penguin" SAMPLEHELDCELLS=2 \
RUN="$RUN" CELLS="${CELLS:?set CELLS}" STEPRANGE="${STEPRANGE:-}" EXTRA="${EXTRA:-}" \
  bash scripts/showcase/train_pool_run.sh
echo "=== $RUN exit $? $(date -Is)"
