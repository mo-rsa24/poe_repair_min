#!/usr/bin/env bash
# Wait for this node's current cache build to finish, then run the next batch on the same card.
# Polls rather than assuming: two builds on one GPU would contend for memory and both slow down.
set -uo pipefail
echo "=== waiting for the card to free on $(hostname), $(date -Is)"
while pgrep -f build_training_cache > /dev/null; do sleep 20; done
echo "=== card free at $(date -Is), starting the queued batch"
exec bash "$NEXT" 
