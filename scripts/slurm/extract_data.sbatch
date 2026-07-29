#!/bin/bash
#SBATCH --job-name=sdg6-extract
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Extract the downloaded SDG6-Tracker archives into the layout the configs
# expect. This runs as a batch job on purpose: unzipping ~48 GB is IO/CPU heavy
# and needs NO network, so it belongs on a compute node. (The *download* itself
# may need to run on a login node with outbound internet.)
#
#   sbatch scripts/slurm/extract_data.sbatch
#
# Archives self-nest (inference.zip contains inference/...), so each is
# extracted into its PARENT directory. Re-running is safe.

set -uo pipefail

ROOT="${SDG6_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
DL="$ROOT/downloads"
Z1="$DL/zenodo_19156085"
Z2="$DL/zenodo_14740420"
RAW="$ROOT/DATABASE/RAW"

log() { echo "[$(date +'%F %T')] $*"; }
fail=0

mkdir -p "$RAW" "$ROOT/runs/pretrained" "$ROOT/data"

# Which imagery modality to extract. Sentinel alone is 188,876 files against
# 137,778 for landsat; on a filesystem with a per-user inode quota the landsat
# half is dead weight unless the landsat datasets are actually being built.
MODALITY="${SDG6_MODALITY:-sentinel}"

# unzip <archive> <parent-dir> [extra unzip flags] [include-glob]
extract() {
  local zip="$1" parent="$2" mode="${3:--o}" include="${4:-}"
  if [ ! -s "$zip" ]; then
    log "MISSING $zip — skipping"
    fail=1
    return
  fi
  local top
  top=$(unzip -Z1 "$zip" | head -1 | cut -d/ -f1)
  log "UNZIP $(basename "$zip") -> $parent/  (top-level: $top, filter: ${include:-ALL})"
  mkdir -p "$parent"
  if [ -n "$include" ]; then
    unzip -q "$mode" "$zip" "$include" -x "__MACOSX/*" -d "$parent"
  else
    unzip -q "$mode" "$zip" -x "__MACOSX/*" -d "$parent"
  fi
  if [ $? -eq 0 ]; then
    log "  done: $(basename "$zip")"
  else
    log "  ERROR unzipping $(basename "$zip")"
    fail=1
  fi
}

log "=== survey imagery -> $RAW (modality: $MODALITY)"
# The three big rounds extract in parallel; they are independent.
extract "$Z2/R7.zip" "$RAW" "-o" "R7/$MODALITY/*" &
p7=$!
extract "$Z2/R8.zip" "$RAW" "-o" "R8/$MODALITY/*" &
p8=$!
extract "$Z2/R9.zip" "$RAW" "-o" "R9/$MODALITY/*" &
p9=$!

log "=== weights + auxiliary data (7 GB)"
extract "$Z1/dino.zip"    "$ROOT/runs/pretrained"
extract "$Z1/dinov2.zip"  "$ROOT/runs/pretrained"
extract "$Z1/inference.zip" "$ROOT/data"
# -n protects the repo's own data/meta_pop_data/build_infer_summary.py
extract "$Z1/meta_pop_data.zip" "$ROOT/data" "-n"

wait $p7 $p8 $p9 || fail=1

log "=== survey CSVs -> $RAW"
for c in "$Z2"/*.csv; do
  [ -s "$c" ] && cp -f "$c" "$RAW/" && log "  copied $(basename "$c")"
done

log "=== cleaning macOS cruft"
find "$ROOT/data" "$ROOT/runs" "$RAW" -name "__MACOSX" -type d -prune -exec rm -rf {} + 2>/dev/null
find "$ROOT/data" "$ROOT/runs" "$RAW" -name "._*" -type f -delete 2>/dev/null

log "=== resulting layout"
du -sh "$RAW"/* "$ROOT/runs/pretrained"/* "$ROOT/data/inference" \
       "$ROOT/data/meta_pop_data" 2>/dev/null | while read -r l; do log "  $l"; done

log "=== sanity checks (paths the configs expect)"
for p in "$ROOT/runs/pretrained/galileo/base/encoder.pt" \
         "$ROOT/runs/pretrained/galileo/base/config.json"; do
  [ -s "$p" ] && log "  OK   ${p#$ROOT/}" || { log "  MISS ${p#$ROOT/}"; fail=1; }
done
# DINOv2 checkpoint name/location varies inside the archive — locate it.
find "$ROOT/runs/pretrained" -name "*checkpoint*.pth" -o -name "*.pth" 2>/dev/null |
  head -10 | while read -r p; do log "  found weight: ${p#$ROOT/}"; done

if [ "$fail" -ne 0 ]; then
  log "FINISHED WITH ERRORS"
  exit 1
fi
log "EXTRACTION COMPLETE"
log "Next: build train/val/test splits ->"
log "  python data/prepare_datasets.py --base-dir DATABASE/RAW --output-dir data"
