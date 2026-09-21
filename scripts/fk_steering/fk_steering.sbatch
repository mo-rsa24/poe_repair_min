#!/bin/bash
#SBATCH --job-name=fk_steer
#SBATCH --partition=bigbatch
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/logs/fk_steer-%j.out
#SBATCH --error=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/logs/fk_steer-%j.err
# Scope 06, plans 10 and 11: Feynman-Kac steering as a Slurm job on a bigbatch node (the fallback
# when no biggpu device is free). The launcher's own guards (disk, python by node, device in use,
# torch.cuda) run inside the job.
#
# Usage:  sbatch --nodelist=<idle bigbatch node> scripts/fk_steering/fk_steering.sbatch {smoke|full|adapter} [extra args...]
set -euo pipefail
MODE="${1:-adapter}"; shift || true
exec env GPU=0 bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/fk_steering/run_fk_steering.sh "$MODE" "$@"
