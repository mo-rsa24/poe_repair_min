#!/usr/bin/env bash
# V6 of the correction-loss variations: a better teacher (--render-teacher).
# Scope 09, plan 05. The target stops being the joint-prompt PREDICTION and becomes the noise
# added to the joint-prompt RENDER a person judged by eye to show both concepts. Same teacher as
# V0, used only where it is visibly right.
#
#   GPU=0 bash scripts/showcase/v6_render_teacher.sh
#
# Runs on mscluster112, an RTX PRO 6000 Blackwell, so PY is co3_bw rather than co3: co3's CUDA
# ops produce no output at all on these cards rather than an error.
# MAXUSED is raised from the launcher's 1024 MiB default because 112 carries another user's
# ~7.6 GB of 97.9. That leaves about 90 GB free, so a rank-32 run at ~25 GB shares comfortably.
# This is the one knob the launcher asks you to raise deliberately, and this is the reason.
#
# Blackwell reads the same uncorrected cell differently from the RTX 8000 and the A6000, so V6's
# numbers do not compare across devices to V1 and V3 on mscluster109. V6 changes the target
# anyway, so a comparison against them was never going to be clean on one axis.
set -uo pipefail
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
export GPU="${GPU:-0}"
export PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python
export RUN="${RUN:-v6_render_teacher_r16_s0_25}"
export CELLS=$REPO/artifacts/_shared/cross_pair_pool_configs/cells_v57.json
export POOLGEN=v57
export OUTROOT=/datasets/mmolefe/poe_repair_min/outputs/correction_loss_variants
export RANK=16 ALPHA=16
export STEPRANGE="0 25"
export EPOCHS=600 EPOCHSIZE=50
export SAMPLEEVERY=25
export WD=0
export MAXUSED=20000
export SAMPLETRAIN="a_lion__x__a_meerkat,a_typewriter__x__a_cactus"
export EXTRA="--render-teacher"
bash $REPO/scripts/showcase/train_pool_run.sh
