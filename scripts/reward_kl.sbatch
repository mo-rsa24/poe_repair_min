#!/bin/bash
#SBATCH --job-name=refl-kl
#SBATCH --partition=bigbatch
#SBATCH --time=2-00:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster59,mscluster60,mscluster62,mscluster65,mscluster72,mscluster73,mscluster74,mscluster75,mscluster76,mscluster83
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/reward_finetune/logs/%x-%j.out
#SBATCH --error=/datasets/mmolefe/poe_repair_min/outputs/reward_finetune/logs/%x-%j.err
#
# K0 or K1 of scripts/launch_reward_kl.sh through Slurm, for when no biggpu card is free. The
# allocation gives one RTX 3090, which is device 0 inside it; the launcher's own guards (the
# /datasets check, the device check, the torch device assert) still run. Both runs on the same
# partition, so the anchor stays the only difference apart from which 3090 each lands on.
#
#   sbatch --job-name=refl-kl-K0 --export=ALL,RUN=K0 scripts/reward_kl.sbatch
#   sbatch --job-name=refl-kl-K1 --export=ALL,RUN=K1 scripts/reward_kl.sbatch
set -euo pipefail
RUN=${RUN:?set RUN=K0 or K1}
GPU=0 exec bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/launch_reward_kl.sh "$RUN"
