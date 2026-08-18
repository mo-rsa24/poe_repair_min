#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
paper_dir="$root/paper/iclr"
out="${1:-$root/paper/iclr-overleaf.zip}"

if [[ ! -d "$paper_dir" ]]; then
  echo "Cannot find paper directory: $paper_dir" >&2
  exit 1
fi

if [[ "$out" != *.zip ]]; then
  echo "Output path must end in .zip: $out" >&2
  exit 1
fi

if [[ -e "$out" ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  out="${out%.zip}-$stamp.zip"
fi

stage="$(mktemp -d)"
cleanup() {
  rm -rf "$stage"
}
trap cleanup EXIT

rsync -a \
  --include='/figures/' \
  --include='/figures/***' \
  --include='/*.tex' \
  --include='/*.sty' \
  --include='/*.bst' \
  --include='/*.bib' \
  --include='/README.md' \
  --include='/OVERLEAF.md' \
  --include='/figures.md' \
  --include='/SPINE.md' \
  --include='/DRAFT_MAP.md' \
  --exclude='*' \
  "$paper_dir/" "$stage/"

mkdir -p "$(dirname "$out")"
(
  cd "$stage"
  zip -qr "$out" .
)

echo "$out"
