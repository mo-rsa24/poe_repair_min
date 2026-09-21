#!/usr/bin/env bash
#SBATCH --job-name=corrloss
#SBATCH --partition=bigbatch
#SBATCH --time=48:00:00
#SBATCH --output=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/logs/corrloss_%j.out
#SBATCH --error=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/logs/corrloss_%j.err
# One full arm of the correction-loss variant comparison. 600 epochs x 50 = 30,000 steps at
# about 0.79 steps/s, so roughly 10.5 hours. Checkpointing stays on: rank 16 without it went
# out of memory at 23.5 GB on a 3090 (job 53318), and with it peaks at 11.2 GB.
#
#   sbatch --nodelist=<node verified under Slurm> --export=ALL,ARM=<arm>[,MU=10] full.sbatch
#
# The node MUST be verified with a two-minute sbatch that tests /datasets, never with ssh:
# mscluster72 and mscluster73 pass an ssh test and then cannot see /datasets from inside a job.
set -uo pipefail
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
export GPU=0 ARM="${ARM:?set ARM}"
exec bash scripts/correction_loss_variants/run_arm.sh
