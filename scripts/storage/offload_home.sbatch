#!/bin/bash
#SBATCH --job-name=offload
#SBATCH --partition=batch
#SBATCH --time=12:00:00
#SBATCH --output=/datasets/mmolefe/offload/logs/offload-%j.out
#SBATCH --error=/datasets/mmolefe/offload/logs/offload-%j.err
#
# Move bulk data off the home quota and onto /datasets, leaving a symlink behind so nothing
# that referenced the old path breaks.
#
# It runs on a compute node because `du` over this home directory times out from the login
# node: the survey is as much the reason for the job as the move is.
#
#   sbatch scripts/storage/offload_home.sbatch                 # survey only, moves nothing
#   sbatch --export=ALL,MODE=move scripts/storage/offload_home.sbatch
#
# MODE=survey (default) writes a report and stops. Read it, delete any line you want kept, then
# run MODE=move, which acts only on what the report still lists.
set -uo pipefail

HOME_DIR=/home-mscluster/mmolefe
DEST=/datasets/mmolefe/offload
REPORT=$DEST/survey.txt
PLAN=$DEST/to_move.txt
MODE="${MODE:-survey}"

mkdir -p "$DEST/logs"
echo "=== $MODE  $(date -Is)  node $(hostname -s)"

# --- the guard that has to pass before anything is written -------------------------------
USE=$(df --output=pcent "$DEST" | tail -1 | tr -dc '0-9')
echo "=== /datasets is ${USE}% full"
[ "${USE:-100}" -ge 90 ] && { echo "ABORT: destination is ${USE}% full" >&2; exit 3; }

# ==========================================================================================
if [ "$MODE" = "survey" ]; then
  : > "$REPORT"
  {
    echo "# Home survey, $(date -Is), node $(hostname -s)"
    echo "# Sizes are real du output, not estimates."
    echo
    echo "## Where the space is, top level"
    du -sh "$HOME_DIR"/* "$HOME_DIR"/.[!.]* 2>/dev/null | sort -rh | head -25
    echo
    echo "## Directories over 1 GB, anywhere under home"
    du -h --max-depth=4 "$HOME_DIR" 2>/dev/null | awk '$1 ~ /[0-9]G/' | sort -rh | head -40
    echo
    echo "## By file kind"
    for pat in "*.pt" "*.ckpt" "*.safetensors" "*.png" "*.jpg" "*.mp4" "*.tar" "*.zip"; do
      read -r n b < <(find "$HOME_DIR" -type f -name "$pat" -printf '%s\n' 2>/dev/null \
                      | awk '{n++; s+=$1} END {printf "%d %.1f\n", n+0, s/1073741824}')
      printf "  %-16s %6s files  %8s GB\n" "$pat" "$n" "$b"
    done
  } >> "$REPORT" 2>&1

  # The move plan: whole directories that are bulk output and safe to relocate behind a symlink.
  # Caches are included because they regenerate; source and notes are never listed.
  : > "$PLAN"
  for d in "$HOME_DIR/.cache/huggingface" "$HOME_DIR/.cache/torch" "$HOME_DIR/.cache/pip" \
           "$HOME_DIR/wandb"; do
    [ -d "$d" ] && [ ! -L "$d" ] && echo "$d" >> "$PLAN"
  done
  # Any checkpoint or render directory still living under home rather than on /datasets.
  find "$HOME_DIR" -type d \( -name checkpoints -o -name per_epoch -o -name wandb \) \
       -not -path "*/.git/*" 2>/dev/null | while read -r d; do
    sz=$(du -sm "$d" 2>/dev/null | cut -f1)
    [ "${sz:-0}" -ge 512 ] && echo "$d" >> "$PLAN"
  done
  sort -u "$PLAN" -o "$PLAN"
  echo
  echo "=== survey written to $REPORT"
  echo "=== move plan ($(wc -l < "$PLAN") entries) written to $PLAN"
  echo "=== nothing was moved. Edit $PLAN, then rerun with MODE=move"
  exit 0
fi

# ==========================================================================================
[ -s "$PLAN" ] || { echo "ABORT: $PLAN is empty or absent. Run the survey first." >&2; exit 4; }
moved=0; freed=0
while read -r src; do
  [ -d "$src" ] || { echo "skip (gone):     $src"; continue; }
  [ -L "$src" ]  && { echo "skip (already):  $src"; continue; }
  rel="${src#$HOME_DIR/}"
  dst="$DEST/$rel"
  sz=$(du -sm "$src" 2>/dev/null | cut -f1)
  echo "--- $src  (${sz} MB)  ->  $dst"
  mkdir -p "$(dirname "$dst")"
  # Copy first, verify, and only then remove. A move that half-succeeds over NFS is the one
  # failure mode that loses data, so the delete never runs on an unverified copy.
  if ! rsync -a --info=progress2 "$src/" "$dst/"; then
    echo "    rsync failed, source left untouched" >&2; continue
  fi
  a=$(du -sb "$src" | cut -f1); b=$(du -sb "$dst" | cut -f1)
  if [ "$a" != "$b" ]; then
    echo "    size mismatch after copy ($a vs $b), source left untouched" >&2; continue
  fi
  rm -rf "$src" && ln -s "$dst" "$src"
  echo "    moved and symlinked"
  moved=$((moved+1)); freed=$((freed+sz))
done < "$PLAN"
echo
echo "=== moved $moved directories, freed about $((freed/1024)) GB from home"
echo "=== every one left a symlink, so old paths still resolve"
