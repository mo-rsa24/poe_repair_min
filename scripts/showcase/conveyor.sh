#!/usr/bin/env bash
# Keep proposing pairs, rendering them at two seeds, and building a sheet per batch.
#
#   GPU=<idx> [PY=<python>] [ROUNDS=n] [PER=n] bash conveyor.sh
#
# One round is: wait for this card to be free, propose PER pairs that pass every filter, render
# seeds 1 and 2 for each, build the sheet, append its full path to the manifest. A batch that
# proposes nothing ends the belt, which is how it stops on its own when the candidates run out.
set -uo pipefail
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32
G=$REPO/artifacts/results/which-joint-prompt-targets-can-the-adapter-learn-from
# Six belts append here at once. One file per shard keeps two appends from interleaving on NFS;
# conveyor_status.sh reads them all as one list.
SHARD="${SHARD:-}"
MANIFEST=$G/conveyor-sheets${SHARD:+-shard$(echo "$SHARD" | tr / of)}.txt
PY="${PY:-/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python}"
# The Blackwells (110-112) run co3_bw; the A6000 and RTX 8000 nodes run co3, and their launcher is
# the one that takes PY from the environment. Pick the launcher to match the card.
BUILDER="${BUILDER:-build_new_animal_cells.sh}"
# The pair generator itself is swappable, so the same belt mechanics run an object-pair belt
# alongside the animal-pair one.
PROPOSER="${PROPOSER:-propose_pairs.py}"
GPU="${GPU:-0}"; ROUNDS="${ROUNDS:-8}"; PER="${PER:-9}"
# How long to wait for the card before giving up. An hour suits a belt queued behind another belt;
# a belt queued behind a training run needs to outlast the run, so pass a longer value.
MAXWAIT="${MAXWAIT:-3600}"
# Where this belt's sheets go, relative to the evidence folder.
SHEETDIR="${SHEETDIR:-inspection-grids}"
# Which seeds each proposed pair is rendered at. The belt used to hardcode "1 2", so 783 of the 785
# cached pairs carry only those two starting noises. Same-pair cross-seed cosine is 0.002 in this
# repo's own measurement, so a second seed is a genuinely different correction, not a repeat: seed
# breadth is as cheap a source of variety as pair breadth. SEEDS=random draws NPERPAIR seeds per
# round from 1-8 and 17-60, never 9-16, which are the evaluation seeds whose starting noise is
# shared across pairs.
SEEDS_SPEC="${SEEDS:-1 2}"
NPERPAIR="${NPERPAIR:-2}"
draw_seeds() {
  if [ "$SEEDS_SPEC" = "random" ]; then
    python3 -c "
import random
pool = list(range(1,9)) + list(range(17,61))
print(' '.join(str(x) for x in sorted(random.sample(pool, $NPERPAIR))))"
  else
    echo "$SEEDS_SPEC"
  fi
}

cd "$REPO" || exit 7
echo "=== conveyor on $(hostname) gpu $GPU, shard ${SHARD:-all}, $ROUNDS rounds of $PER pairs, $(date -Is)"
echo "PID $$"

for r in $(seq 1 "$ROUNDS"); do
  # Wait for MY build to finish, then for the card to actually be free. The first restart of this
  # belt skipped both checks: another user held the card, the build's device guard aborted, the
  # abort was hidden by the output filter, and ten sheets were written from cells that never
  # existed. So: never filter the build's output, and never build a sheet on a failed render.
  while pgrep -f build_training_cache > /dev/null; do sleep 20; done
  WAITED=0
  while :; do
    USED=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ,')
    case "$USED" in ''|*[!0-9]*) echo "=== gpu $GPU unreadable, belt stops"; exit 4;; esac
    [ "$USED" -le 1024 ] && break
    [ "$WAITED" -ge "$MAXWAIT" ] && { echo "=== gpu $GPU held for ${MAXWAIT}s, belt stops"; exit 0; }
    echo "=== round $r waiting: gpu $GPU has ${USED} MiB in use, $(date +%H:%M)"
    sleep 60; WAITED=$((WAITED+60))
  done

  # Tell a crashed generator apart from an exhausted one. Round 2 of the first run on this card hit
  # a NameError introduced while the belt was live, produced no output, and the belt read that as
  # "no candidates left" and stopped clean. Those two look identical unless the exit code is checked.
  PROP=$(XFORMERS_DISABLED=1 python3 "scripts/showcase/$PROPOSER" "$PER" 2>&1)
  if [ $? -ne 0 ]; then
    echo "=== round $r: the generator failed, belt stops. Its error:"
    echo "$PROP" | tail -5
    exit 5
  fi
  LINE=$(echo "$PROP" | grep '^PAIRS=' | sed 's/^PAIRS=//')
  if [ -z "$LINE" ]; then echo "=== round $r: generator ran clean and proposed nothing, belt stops"; break; fi

  echo "=== round $r  $(date -Is)"
  ROUND_SEEDS=$(draw_seeds)
  echo "=== round $r seeds: $ROUND_SEEDS"
  GPU="$GPU" PY="$PY" SEEDS="$ROUND_SEEDS" PAIRS="$LINE" bash "$OUT/$BUILDER" 2>&1 | tee /tmp/conv_round_$$.log
  if ! grep -q '^=== done' /tmp/conv_round_$$.log; then
    echo "=== round $r: the render did not complete, no sheet built"
    grep -E '^ABORT|^!!!' /tmp/conv_round_$$.log
    continue
  fi
  BUILT=$(grep -c '^\[built\]\|^\[cached\]' /tmp/conv_round_$$.log)
  if [ "$BUILT" -eq 0 ]; then echo "=== round $r: zero cells produced, no sheet built"; continue; fi

  SLUGS=$(echo "$LINE" | tr ';' '\n' | cut -d'|' -f1 | tr -d ' ' | tr '\n' ' ')
  # Sheets are mono-beside-poe, which is the layout a training pair is actually chosen from: the
  # target on the left, what PoE does without the adapter on the right. SHEETDIR keeps each belt's
  # output in its own folder so a judging pass is one directory rather than a mixed pile.
  SHEETNAME="${SHEETPREFIX:-belt}-$(date +%H%M%S)-r$r"
  HALF="${SHEETHALF:-210}"   python3 scripts/showcase/mono_poe_grids.py --pairs $SLUGS ${MONOONLY:+--mono-only} \
      --label "${SHEETLABEL:-belt} round $r, seeds $ROUND_SEEDS" \
      --pairs-per-figure "${SHEETROWS:-5}" --max-seeds "${SHEETCOLS:-8}" \
      --prefix "$SHEETNAME" --out "$G/$SHEETDIR" > /tmp/conv_sheet_$$.log 2>&1
  SHEET=$(grep -oE '[A-Za-z0-9._-]+\.png' /tmp/conv_sheet_$$.log | head -1)
  if [ -n "$SHEET" ]; then
    FULL="$G/$SHEETDIR/$SHEET"
    echo "$(date -Is)  round $r  $BUILT cells  $FULL" >> "$MANIFEST"
    echo "=== SHEET READY  $FULL"
  else
    echo "=== round $r: sheet build produced nothing"
  fi
done
echo "=== conveyor done $(date -Is)"
