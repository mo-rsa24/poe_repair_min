#!/bin/bash
# Slurm wrapper for plan 15 (01-showcase-the-trained-lora) on the bigbatch partition, used when
# every biggpu device is occupied. The node's single 24 GB card is device 0 and is free on a fresh
# allocation, so the shared-device launcher's guards pass and it runs the stages in order.
#   sbatch --exclude=mscluster50,mscluster51,mscluster65 scripts/showcase/tangent_projection.sbatch
# (or --nodelist=<node> after probing the idle nodes' GPUs over SSH; mscluster65's card is faulted too)
#SBATCH --job-name=tangent_projection
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/showcase/logs/tangent_projection_%j.log
export GPU=0
export STAGE="${STAGE:-smoke_then_all}"
exec bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/showcase/tangent_projection_shared_device.sh
