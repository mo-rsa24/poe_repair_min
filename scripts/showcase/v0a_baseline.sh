#!/usr/bin/env bash
# V0a of the correction-loss variations: the objective running today, moved onto the scope pool.
# Scope 09, plan 04, task 3.2. This is the baseline every later variation is read against, and
# until this run exists there is nothing to read them against: V1, V3, V6 and V6a have all run
# and V0a never has.
#
#   GPU=0 nohup bash scripts/showcase/v0a_baseline.sh > logs/v0a.log 2>&1 &
#
# mscluster109 (RTX A6000, 49 GB), because that is where V1 ran and a matched set never straddles
# device models. biggpu allows one job per user through Slurm, so this goes outside Slurm.
# Rank 16 and steps 0 to 25 match the lineage: every rank-32 attempt on this pool died on
# 2026-09-17 and no one has read the logs.
set -uo pipefail
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
export GPU="${GPU:-0}"
# co3, not co3_bw: mscluster109 is an RTX A6000. co3_bw is for the Blackwell cards on
# mscluster110 to 112 only, and a co3 CUDA op there produces no output rather than an error.
export PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
export RUN="${RUN:-v0a_baseline_r16_s0_25}"
export CELLS=$REPO/artifacts/_shared/cross_pair_pool_configs/cells_v57.json
export POOLGEN=v57
export OUTROOT=/datasets/mmolefe/poe_repair_min/outputs/correction_loss_variants
export RANK=16 ALPHA=16
export STEPRANGE="0 25"
export EPOCHS=600 EPOCHSIZE=50
export SAMPLEEVERY=25
export WD=0
export SAMPLETRAIN="a_lion__x__a_meerkat,a_typewriter__x__a_cactus"
export EXTRA=""
bash $REPO/scripts/showcase/train_pool_run.sh
