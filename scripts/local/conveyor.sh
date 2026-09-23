#!/bin/bash
# Pull every joint-prompt render as it lands on the cluster, and rebuild the contact sheet
# whenever the count changes. Runs for about eight hours; start it again for a longer sitting.
REMOTE=mscluster84:/datasets/mmolefe/poe_repair_min/outputs/pair_audition/
LOCAL=/home/molef/PhD/poe_repair_min/artifacts/results/which-new-pairs-compose-cleanly/tiles
SHEET=/home/molef/PhD/poe_repair_min/artifacts/results/which-new-pairs-compose-cleanly/pair-audition.png
HERE=$(dirname "$(readlink -f "$0")")
mkdir -p "$LOCAL"; last=0
for i in $(seq 1 240); do
  timeout 240 rsync -a -e "ssh -o ConnectTimeout=60" --exclude logs --exclude rejected --exclude '*.py' "$REMOTE" "$LOCAL" 2>/dev/null
  n=$(find "$LOCAL" -name mono.png | wc -l)
  if [ "$n" != "$last" ]; then
    python3 "$HERE/build_sheet.py" "$LOCAL" "$SHEET" >/dev/null 2>&1
    echo "$(date +%H:%M) $n tiles"; last=$n
  fi
  sleep 120
done
