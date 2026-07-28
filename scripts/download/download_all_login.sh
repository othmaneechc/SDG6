#!/usr/bin/env bash
# Download all SDG6-Tracker data and weights (~48 GB).
#
# MUST run on a Rorqual LOGIN node: compute nodes have no outbound internet and
# batch download jobs are SIGTERM-killed almost immediately. Extraction needs no
# network and is done separately by scripts/slurm/extract_data.sbatch.
#
#   nohup bash scripts/download/download_all_login.sh > logs/download.log 2>&1 &
#
# Zenodo throttles each connection to roughly 1 MB/s, so transfers run in
# parallel (PAR workers) to raise aggregate throughput. Safe to re-run: every
# transfer resumes (curl -C -) and files already matching their published MD5
# are skipped.

set -uo pipefail

# Repo root: override with SDG6_ROOT, else derive from this script's location.
ROOT="${SDG6_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DL="$ROOT/downloads"
GALILEO_DIR="$ROOT/runs/pretrained/galileo/base"
MANIFEST="$DL/manifest.tsv"
PAR="${PAR:-6}"
SELF="$(readlink -f "$0")"

log() { echo "[$(date +'%F %T')] $*"; }

# --- worker mode: one file per invocation ----------------------------------
if [ "${1:-}" = "--worker" ]; then
  # NOTE: tab is an IFS *whitespace* character, so consecutive tabs collapse and
  # an empty field silently disappears, shifting every later field left. The
  # manifest therefore uses "-" (never empty) for "no published checksum".
  IFS=$'\t' read -r url out want label <<< "$2"
  [ "$want" = "-" ] && want=""
  mkdir -p "$(dirname "$out")"

  if [ -s "$out" ] && [ -n "$want" ]; then
    have=$(md5sum "$out" | awk '{print $1}')
    if [ "$have" = "$want" ]; then
      log "SKIP  $label (md5 ok)"
      exit 0
    fi
  fi

  log "GET   $label"
  if ! curl -fL --no-progress-meter --retry 8 --retry-delay 10 \
       --retry-all-errors -C - --connect-timeout 30 -o "$out" "$url"; then
    log "ERROR $label — download failed"
    exit 1
  fi

  if [ -n "$want" ]; then
    have=$(md5sum "$out" | awk '{print $1}')
    if [ "$have" != "$want" ]; then
      log "ERROR $label — md5 MISMATCH (want $want got $have)"
      exit 1
    fi
    log "OK    $label ($(du -h "$out" | cut -f1), md5 verified)"
  else
    log "OK    $label ($(du -h "$out" | cut -f1))"
  fi
  exit 0
fi

# --- build the manifest -----------------------------------------------------
mkdir -p "$DL/zenodo_19156085" "$DL/zenodo_14740420" "$GALILEO_DIR" "$ROOT/logs"
: > "$MANIFEST"

add_zenodo() {
  local rec="$1" dest="$2"
  log "indexing Zenodo record $rec"
  curl -sfL "https://zenodo.org/api/records/$rec" | python3 -c "
import json,sys,os
rec, dest = sys.argv[1], sys.argv[2]
d = json.load(sys.stdin)
for f in d.get('files', []):
    key = f['key']
    md5 = f.get('checksum','').replace('md5:','')
    url = f'https://zenodo.org/records/{rec}/files/{key}?download=1'
    print('\t'.join([url, os.path.join(dest, key), md5, f'{rec}/{key}']))
" "$rec" "$dest" >> "$MANIFEST" || { log "ERROR: Zenodo API failed for $rec"; exit 1; }
}

add_zenodo 19156085 "$DL/zenodo_19156085"
add_zenodo 14740420 "$DL/zenodo_14740420"

# Hugging Face Galileo base encoder (configs/galileo.yaml wants config.json + encoder.pt)
for f in config.json encoder.pt decoder.pt target_encoder.pt second_decoder.pt; do
  printf '%s\t%s\t%s\t%s\n' \
    "https://huggingface.co/nasaharvest/galileo/resolve/main/models/base/$f" \
    "$GALILEO_DIR/$f" "-" "galileo/base/$f" >> "$MANIFEST"
done

total=$(wc -l < "$MANIFEST")
log "manifest: $total files; starting $PAR parallel workers"
log "sizes: $(awk -F'\t' '{print $4}' "$MANIFEST" | tr '\n' ' ')"

# --- run ---------------------------------------------------------------------
start=$(date +%s)
xargs -a "$MANIFEST" -d '\n' -P "$PAR" -n 1 -I{} bash "$SELF" --worker "{}"
rc=$?
elapsed=$(( $(date +%s) - start ))

log "=== totals after $((elapsed/60))m$((elapsed%60))s"
du -sh "$DL"/zenodo_* "$GALILEO_DIR" 2>/dev/null | while read -r l; do log "  $l"; done

if [ "$rc" -ne 0 ]; then
  log "DONE WITH ERRORS (rc=$rc) — re-run; verified files are skipped."
  exit 1
fi
log "ALL DOWNLOADS COMPLETE AND VERIFIED"
log "Next: sbatch scripts/slurm/extract_data.sbatch"
