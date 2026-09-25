#!/usr/bin/env bash
# One line per card and one per process on it, for scripts/local/run_queue.py. Run on a GPU node:
#   ssh <node> bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/gpu_probe.sh
#
#   GPU <node> <index> <used MiB> <total MiB> <utilisation %>     ([N/A] marks a faulted card)
#   APP <node> <index> <pid> <user> <used MiB>
n=$(hostname -s)
declare -A IDX
while IFS=', ' read -r i u m t z; do
  IDX[$u]=$i
  echo "GPU $n $i $m $t $z"
done < <(nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
while IFS=', ' read -r u p m; do
  [ -n "$p" ] && echo "APP $n ${IDX[$u]} $p $(ps -o user= -p "$p" | tr -d ' ') $m"
done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits)
