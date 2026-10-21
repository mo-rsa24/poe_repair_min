#!/usr/bin/env bash
# Fetch the primary liver cancer CECT dataset (Luo et al. 2025, Scientific Data;
# Science Data Bank 10.57760/sciencedb.12207, CC BY 4.0) onto the cluster's /datasets mount.
# No login is needed: both files are public links with published MD5s. Resumable, so a
# dropped connection or a dead node is fixed by running this script again.
#
#   bash scripts/download_liver_cect.sh              download, verify, extract
#   bash scripts/download_liver_cect.sh --csv-only   fetch and verify only patient_data.csv
#
# Layout under DEST:
#   raw/CECT.tar  raw/patient_data.csv  raw/MD5SUMS   verified originals, kept
#   ct_files/ mask_files/ liver_mask_files/           extracted NIfTI volumes
#   logs/                                             one log per run
# Env override: DEST (default the path below).
set -euo pipefail

DEST="${DEST:-/datasets/mmolefe/poe_repair_min/data/liver-cect}"
BASE="https://china.scidb.cn/download?fileId"
TAR_ID=cfdd5e37e35a013dc0015aa361531b01;  TAR_MD5=71dc809a0221fe64616a2f0e915eca84;  TAR_BYTES=42051328000
CSV_ID=e0af91f0f91a3c20b3a8779f7dc03352;  CSV_MD5=f12a18577c714627afcafe1eb84b1f1a
NEED_GB=100   # tar (42 GB) + extracted volumes (about 42 GB) + headroom

die() { echo "download_liver_cect: $*" >&2; exit 1; }

# Disk guard on the filesystem DEST really lives on (the tar and the extraction both land there).
mkdir -p "$DEST/raw" "$DEST/logs"
read -r avail_kb used_pct < <(df -Pk "$DEST" | awk 'NR==2 {gsub("%","",$5); print $4, $5}')
[ "$used_pct" -lt 90 ] || die "$DEST filesystem is ${used_pct}% full"
[ "$avail_kb" -gt $((NEED_GB * 1024 * 1024)) ] || die "under ${NEED_GB} GB free at $DEST"

LOG="$DEST/logs/download-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "host $(hostname)  dest $DEST  start $(date -Is)"

fetch() {  # fetch <fileId> <outfile> <md5>
  local id="$1" out="$2" md5="$3"
  if [ -f "$out" ] && [ "$(md5sum "$out" | cut -d' ' -f1)" = "$md5" ]; then
    echo "ok already: $out"; return
  fi
  curl -fL --retry 20 --retry-delay 10 --retry-all-errors -C - -o "$out" "$BASE=$id"
  [ "$(md5sum "$out" | cut -d' ' -f1)" = "$md5" ] || die "md5 mismatch for $out (delete it and rerun)"
  echo "verified: $out"
}

fetch "$CSV_ID" "$DEST/raw/patient_data.csv" "$CSV_MD5"
[ "${1:-}" = "--csv-only" ] && { echo "csv only: done"; exit 0; }

fetch "$TAR_ID" "$DEST/raw/CECT.tar" "$TAR_MD5"
printf '%s  raw/CECT.tar\n%s  raw/patient_data.csv\n' "$TAR_MD5" "$CSV_MD5" > "$DEST/raw/MD5SUMS"

if [ ! -f "$DEST/.extracted" ]; then
  echo "extracting at $(date -Is)"
  tar --no-same-owner --no-same-permissions -xf "$DEST/raw/CECT.tar" -C "$DEST"
  touch "$DEST/.extracted"
fi

echo "files: ct $(ls "$DEST/ct_files" | wc -l)  mask $(ls "$DEST/mask_files" | wc -l)  liver_mask $(ls "$DEST/liver_mask_files" | wc -l)"
echo "done $(date -Is)"
