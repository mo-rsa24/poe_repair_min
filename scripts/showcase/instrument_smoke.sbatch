#!/bin/bash
#SBATCH --job-name=instrument_smoke
#SBATCH --partition=bigbatch
#SBATCH --nodelist=mscluster56
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/showcase/logs/instrument_smoke_%x_%j.log
# Task 2.1 of plan 01 in scope 09 (09-designing-the-correction-loss/plans/tools/01-...).
#
# Two epochs on the scope pool, proving all four instruments at once. It carries no claim, so it
# is rank 16 on a 24 GB card rather than the scope's rank 32 on a 49 GB one.
#
# What each instrument shows, and where to look:
#   1.1/1.2  W&B carries train/loss_undialled beside train/loss_fit, non-null and varying
#   1.3      a checkpoint directory beside the render directory at the same epoch
#   1.4      config.json names train_step_range, orth_weight and cells_file
#   1.5      dataset_meta.json reads split=train for every cell, including the eight
#            a_typewriter__x__a_cactus cells that also exist under heldout/
#
# SAMPLEEVERY=1 deliberately: the launcher defaults to 25, and at two epochs that writes no
# checkpoint and no render set at all, so instrument 1.3 would have nothing to show.
#
# mscluster56 is pinned rather than excluded-into: on 2026-09-17 eight of seventeen idle bigbatch
# nodes reported "Unable to determine the device handle for GPU0", 48 and 62 among them, which the
# known-failures list did not yet carry. 56 probed healthy and is on the 2026-09-12 datasets=OK list.
set -uo pipefail
export GPU=0
export RUN=instrument_smoke_$(date +%Y%m%d_%H%M)
export CELLS=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/artifacts/_shared/cross_pair_pool_configs/cells_v57.json
export POOLGEN=v57
export OUTROOT=/datasets/mmolefe/poe_repair_min/outputs/showcase/instrument_smoke
export RANK=16 ALPHA=16
export STEPRANGE="0 25"
export EPOCHS=2 EPOCHSIZE=50
export SAMPLEEVERY=1
export WD=0
# Rank 16 without checkpointing went out of memory on this 24 GB 3090 (job 56021, 2026-09-17):
# 23.53 of 23.55 GiB in use, dying in the training forward at multi_pair_trainer.py:93, before
# it reached a render. The scope-01 energy arms hit the same wall at rank 32 and pass this flag.
export CKPT=1
# The launcher's SAMPLETRAIN default is a v54 pair list and a_giraffe__x__a_lion is not in v57,
# so inline sampling died on KeyError in _inline_sampling.py:374 (job 56026, 2026-09-17).
# Both of these are in v57. SAMPLEHELD's default (cat x dog, elephant x penguin) is fine: both
# are v57's held-out pairs, and cat x dog is the triptych worth looking at.
export SAMPLETRAIN="a_lion__x__a_meerkat,a_typewriter__x__a_cactus"
bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/showcase/train_pool_run.sh
