#!/bin/bash
# Re-apply the CO3 keep list while jobs are still writing into the tree.
H=$(dirname "$(readlink -f "$0")")
R=/datasets/mmolefe/poe_repair_min/outputs/co3_probe/algo-co3-hybrid_
L=/home/molef/PhD/poe_repair_min/artifacts/results/what-co3-does-on-our-pairs/tiles
for i in $(seq 1 80); do
  timeout 150 ssh -o ConnectTimeout=60 mscluster84 "bash -s $R" < "$H/prune_co3.sh" >/dev/null 2>&1
  bash "$H/prune_co3.sh" "$L" >/dev/null 2>&1
  sleep 150
done
