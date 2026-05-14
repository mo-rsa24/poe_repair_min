#!/usr/bin/env bash
# Rehydrate outputs/ symlinks from /datasets/mmolefe/poe_repair_min/outputs/.
# Safe to re-run. See docs/STORAGE.md.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORE="/datasets/mmolefe/poe_repair_min/outputs"

if [[ ! -d "$STORE" ]]; then
  echo "error: $STORE not found. Are you on a host with /datasets mounted?" >&2
  exit 1
fi

mkdir -p "$REPO_ROOT/outputs"
cd "$REPO_ROOT/outputs"

linked=0
skipped_real=0
already=0

for src in "$STORE"/*/; do
  name="$(basename "$src")"
  target="$src"
  target="${target%/}"

  if [[ -L "$name" ]]; then
    current="$(readlink "$name")"
    if [[ "$current" == "$target" ]]; then
      already=$((already + 1))
      continue
    fi
    echo "fixing stale symlink: $name -> $current  =>  $target"
    rm "$name"
    ln -s "$target" "$name"
    linked=$((linked + 1))
  elif [[ -e "$name" ]]; then
    # Real directory exists locally — leave it alone (likely a live run).
    echo "skip (real dir present, not a symlink): $name"
    skipped_real=$((skipped_real + 1))
  else
    ln -s "$target" "$name"
    echo "linked: $name -> $target"
    linked=$((linked + 1))
  fi
done

echo
echo "rehydrate summary: linked=$linked  already_ok=$already  skipped_real=$skipped_real"
