#!/bin/bash
# Pull the sweep's tiles as they land, rebuild the per-run sheets, and keep a dated copy of every
# state the sheets passed through. Open `latest/` at any moment; `history/` holds what it looked
# like at each earlier point, so a checkpoint judged yesterday can be found again.
#
#   bash scripts/local/sweep_watch.sh            # runs for about eight hours
#
# Nothing is deleted on either machine: rsync only fetches files the laptop does not have.
set -uo pipefail
REMOTE=mscluster84:/datasets/mmolefe/poe_repair_min/outputs/ablation_grid/
ROOT=/home/molef/PhD/poe_repair_min/artifacts/results/which-checkpoint-to-take-each-figure-cell-from
TILES="$ROOT/tiles"
LATEST="$ROOT/latest"
HISTORY="$ROOT/history"
LOG="$ROOT/BUILDS.md"
REPO=/home/molef/PhD/poe_repair_min

mkdir -p "$TILES" "$LATEST" "$HISTORY"
[ -f "$LOG" ] || cat > "$LOG" <<'HEADER'
# Sweep builds

Every rebuild of the per-run checkpoint sheets, newest last. A row appears only when the tile count
changed, so a run of identical rows means nothing new landed. `latest/` always holds the newest
sheets and `history/<stamp>/` holds the sheets as they stood at that moment.

| When | Probe tiles | Runs | Snapshot |
|---|---|---|---|
HEADER

last=0
for i in $(seq 1 240); do
  timeout 300 rsync -a -e "ssh -o ConnectTimeout=60" --exclude logs --exclude '*.py' \
      "$REMOTE" "$TILES" 2>/dev/null
  n=$(find "$TILES" -name 'ours_*_s*_w50.png' | wc -l)
  if [ "$n" != "$last" ]; then
    if python3 "$REPO/scripts/sweep_sheets.py" "$TILES" "$LATEST" > /tmp/sweep_sheets.$$ 2>&1; then
      stamp=$(date +%Y%m%d-%H%M)
      mkdir -p "$HISTORY/$stamp"
      cp "$LATEST"/*.png "$LATEST"/*.json "$HISTORY/$stamp/" 2>/dev/null
      runs=$(grep -c 'checkpoints ->' /tmp/sweep_sheets.$$ || echo 0)
      echo "| $(date '+%Y-%m-%d %H:%M') | $n | $runs | [$stamp](history/$stamp/) |" >> "$LOG"
      echo "$(date +%H:%M) $n tiles, $runs runs, snapshot $stamp"
    fi
    rm -f /tmp/sweep_sheets.$$
    last=$n
  fi
  sleep 120
done
