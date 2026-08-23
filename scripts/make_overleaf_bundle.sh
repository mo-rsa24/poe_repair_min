#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
paper_dir="$root/paper/iclr"
slim=0
if [[ "${1:-}" == "--slim" ]]; then
  slim=1
  shift
fi
out="${1:-$root/paper/iclr-overleaf.zip}"
if [[ "$slim" == "1" && "$out" == "$root/paper/iclr-overleaf.zip" ]]; then
  out="$root/paper/iclr-overleaf-slim.zip"
fi

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

# --slim keeps only the figure files the manuscript actually includes. Everything
# else under figures/ is there for sections that have not been written yet, and it
# is what pushes the full bundle past what Overleaf will accept on upload.
if [[ "$slim" == "1" ]]; then
  used="$(grep -oE '\\includegraphics(\[[^]]*\])?\{[^}]+\}' "$paper_dir"/*.tex \
          | sed -E 's/.*\{([^}]+)\}/\1/' | sort -u)"
  find "$stage/figures" -type f -print0 | while IFS= read -r -d '' f; do
    base="${f#$stage/figures/}"
    if ! grep -qxF "$base" <<< "$used"; then
      rm -f "$f"
    fi
  done
  find "$stage/figures" -type d -empty -delete
  echo "slim bundle keeps:" >&2
  echo "$used" | sed 's/^/  /' >&2
fi

mkdir -p "$(dirname "$out")"
(
  cd "$stage"
  zip -qr "$out" .
)

echo "$out"
