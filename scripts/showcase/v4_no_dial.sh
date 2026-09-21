#!/usr/bin/env bash
# V4 of the correction-loss variations: the guidance weight dropped from both sides, on top of
# V1's frozen empty branch. Scope 09, plan 04, task 3.3.
#
#   GPU=1 nohup bash scripts/showcase/v4_no_dial.sh > logs/v4.log 2>&1 &
#
# Needs the trainer fix at one_pair_one_seed/trainer.py's _no_dial branch, which now reads
# _freeze_null. Before that fix this pair of switches trained with the null free while the
# sampler detached it. With it, this is V1 up to the constant w^2, and the run is the check
# that the cancellation the scope's derivation claims is what the code does.
#
# mscluster109 device 1, beside V0a on device 0, same device model as the V1 it is read against.
set -uo pipefail
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
export GPU="${GPU:-0}"
# co3, not co3_bw: mscluster109 is an RTX A6000. co3_bw is for the Blackwell cards on
# mscluster110 to 112 only, and a co3 CUDA op there produces no output rather than an error.
export PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
export RUN="${RUN:-v4_no_dial_frozen_r16_s0_25}"
export CELLS=$REPO/artifacts/_shared/cross_pair_pool_configs/cells_v57.json
export POOLGEN=v57
export OUTROOT=/datasets/mmolefe/poe_repair_min/outputs/correction_loss_variants
export RANK=16 ALPHA=16
export STEPRANGE="0 25"
export EPOCHS=600 EPOCHSIZE=50
export SAMPLEEVERY=25
export WD=0
export SAMPLETRAIN="a_lion__x__a_meerkat,a_typewriter__x__a_cactus"
export EXTRA="--no-dial --freeze-null"
bash $REPO/scripts/showcase/train_pool_run.sh
