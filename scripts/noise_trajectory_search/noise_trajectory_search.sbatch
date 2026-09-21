#!/bin/bash
#SBATCH --job-name=nts
#SBATCH --partition=bigbatch
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/logs/nts-%j.out
#SBATCH --error=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/logs/nts-%j.err
# Scope 01, plan 16: the noise trajectory search as a Slurm job on a bigbatch node (the fallback
# when no biggpu device is free). The launcher's own guards (disk, python by node, device in use,
# faulted device, torch.cuda) run inside the job. mscluster44, 45, 50, 51 and 65 read idle with a
# faulted GPU (poe-launch-002), so name a node that passed before: 52, 53, 54 or 60.
#
# Usage:  sbatch --nodelist=<idle bigbatch node> scripts/noise_trajectory_search/noise_trajectory_search.sbatch {smoke|full} [extra args...]
set -euo pipefail
MODE="${1:-full}"; shift || true
exec env GPU=0 bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/noise_trajectory_search/run_noise_trajectory_search.sh "$MODE" "$@"
