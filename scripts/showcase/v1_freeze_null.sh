#!/usr/bin/env bash
# V1 of the correction-loss variations: freeze the empty branch (--freeze-null).
# Scope 09, plan 04. Launched outside Slurm because biggpu allows one job per user.
#
#   GPU=0 bash scripts/showcase/v1_freeze_null.sh
#
# Rank 32 on a 49 GB A6000, so no gradient checkpointing: rank 32 holds about 25 GB and only
# needs the flag on a 24 GB card.
# Step range 0 25 matches the four 2026-09-12 experiments and 00b, which is the only lineage
# this run can be read against; the phase1_* runs used all 50 and a different pool.
# Weight decay 0 matches phase1_r32_100k, the run whose samples actually compose.
set -uo pipefail
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
export GPU="${GPU:-0}"
export RUN="${RUN:-v1_freeze_null_r16_s0_25}"
export CELLS=$REPO/artifacts/_shared/cross_pair_pool_configs/cells_v57.json
export POOLGEN=v57
export OUTROOT=/datasets/mmolefe/poe_repair_min/outputs/correction_loss_variants
export RANK=16 ALPHA=16
export STEPRANGE="0 25"
export EPOCHS=600 EPOCHSIZE=50
export SAMPLEEVERY=25
export WD=0
export SAMPLETRAIN="a_lion__x__a_meerkat,a_typewriter__x__a_cactus"
export EXTRA="--freeze-null"
bash $REPO/scripts/showcase/train_pool_run.sh
