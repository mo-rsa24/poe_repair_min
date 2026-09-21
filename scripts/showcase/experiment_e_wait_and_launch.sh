#!/bin/bash
# Waits for a free device on the biggpu nodes and launches experiment_e_x0_loss.sh on it, once.
# Runs on the session node under nohup; polls every POLL_S seconds (default 300) for up to
# MAX_HOURS (default 48). A device counts as free by the same rule the launcher's guard applies:
# at most 4096 MiB in use and at most 5% utilisation, read twice 60 s apart so a device between
# two of someone's jobs is not mistaken for idle.
#
#   nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/showcase/experiment_e_wait_and_launch.sh \
#       > /datasets/mmolefe/poe_repair_min/outputs/showcase/logs/experiment_e_wait.log 2>&1 &
set -uo pipefail

REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
LOGS=/datasets/mmolefe/poe_repair_min/outputs/showcase/logs
LAUNCHER="$REPO/scripts/showcase/experiment_e_x0_loss.sh"
CANDIDATES="${CANDIDATES:-mscluster106:1 mscluster106:0 mscluster108:1 mscluster109:1 mscluster109:0 mscluster108:0}"
POLL_S="${POLL_S:-300}"
MAX_HOURS="${MAX_HOURS:-48}"
CLAIMED="$LOGS/experiment_e_claimed_device.txt"
mkdir -p "$LOGS"

free_on() {  # node index -> 0 if free
  local node="$1" idx="$2" out used util
  out=$(timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=8 "$node" \
        "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i $idx" 2>/dev/null) || return 1
  used=$(echo "$out" | cut -d, -f1 | tr -dc '0-9'); util=$(echo "$out" | cut -d, -f2 | tr -dc '0-9')
  [ -n "$used" ] && [ -n "$util" ] && [ "$used" -le 4096 ] && [ "$util" -le 5 ]
}

echo "[$(date -u '+%F %T')] waiting for a free device among: $CANDIDATES (poll ${POLL_S}s, max ${MAX_HOURS}h)"
deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  if [ -e "$CLAIMED" ]; then echo "already claimed: $(cat "$CLAIMED"); exiting"; exit 0; fi
  for cand in $CANDIDATES; do
    node="${cand%%:*}"; idx="${cand##*:}"
    if free_on "$node" "$idx"; then
      echo "[$(date -u '+%F %T')] $node device $idx reads free; confirming in 60 s"
      sleep 60
      if free_on "$node" "$idx"; then
        echo "[$(date -u '+%F %T')] claiming $node device $idx"
        echo "$node $idx $(date -u '+%F %T')" > "$CLAIMED"
        ssh -o BatchMode=yes "$node" "nohup bash $LAUNCHER $idx > $LOGS/experiment_e_x0loss.log 2>&1 &"
        sleep 20
        ssh -o BatchMode=yes "$node" "pgrep -af 'experiment_e_x0_loss|train_pooled' | grep -v pgrep | cut -c1-200; tail -5 $LOGS/experiment_e_x0loss.log"
        exit 0
      fi
      echo "[$(date -u '+%F %T')] $node device $idx was taken in the meantime"
    fi
  done
  sleep "$POLL_S"
done
echo "[$(date -u '+%F %T')] gave up after ${MAX_HOURS}h with no free device"
exit 2
