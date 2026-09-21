#!/bin/bash
#SBATCH --job-name=overdose-cells
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --nodelist=mscluster46
#SBATCH --output=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/logs/overdose-cells-%j.log
# The four over-correction columns for the dose strip. Falls here because every
# biggpu device was busy on 2026-09-10. Pinned rather than excluded: the faulted
# set on bigbatch grows between probes, and job 52566 landed on mscluster76,
# which no exclude list named. mscluster46 was probed healthy the same day, per
# environment/hpc/nodes.md. Reprobe and repin rather than trusting this node.
bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/launch_overdose_cells.sh 0
