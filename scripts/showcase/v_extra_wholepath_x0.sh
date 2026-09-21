#!/usr/bin/env bash
# Variation 08: fitting the whole path and handing back the tail. Scope 09, plan 06, task 3.2.
# Two changes over V1, deliberately together, both aimed at the late denoising steps where fine
# detail is drawn: every cached step trains instead of the first 25, and the error is scored on
# the predicted clean image rather than on the predicted noise.
#
#   GPU=0 nohup bash scripts/showcase/v_extra_wholepath_x0.sh > logs/v_extra.log 2>&1 &
#
# Do not launch this until plan 06's stage A has fallen short of its bar. The hand-off sweep
# costs an hour on checkpoints that already exist and may answer the question without training.
#
# mscluster109 once plan 04's two runs have finished, to stay on one device model.
set -uo pipefail
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
export GPU="${GPU:-0}"
# co3, not co3_bw: mscluster109 is an RTX A6000. co3_bw is for the Blackwell cards on
# mscluster110 to 112 only, and a co3 CUDA op there produces no output rather than an error.
export PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
export RUN="${RUN:-v_extra_wholepath_x0_r16}"
export CELLS=$REPO/artifacts/_shared/cross_pair_pool_configs/cells_v57.json
export POOLGEN=v57
export OUTROOT=/datasets/mmolefe/poe_repair_min/outputs/correction_loss_variants
export RANK=16 ALPHA=16
# No STEPRANGE: this run trains on all 50 cached denoising steps, which is the
# point of it. The epoch is still EPOCHSIZE optimizer steps, so this costs the
# same wall clock as the 0-to-25 runs; it changes which cached steps are sampled.
export EPOCHS=600 EPOCHSIZE=50
export SAMPLEEVERY=25
export WD=0
export SAMPLETRAIN="a_lion__x__a_meerkat,a_typewriter__x__a_cactus"
export EXTRA="--freeze-null --loss-space x0"
bash $REPO/scripts/showcase/train_pool_run.sh
