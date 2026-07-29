#!/bin/bash
#SBATCH --job-name=sdg6-embed
#SBATCH --account=rrg-bengioy-ad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=124G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Extract encoder embeddings for every image in the manifest.
#
# DINOv2 here is built against xformers' memory_efficient_attention, which is
# CUDA-only, so this cannot be smoke-tested on a login node. The job therefore
# runs a tiny --limit pass first and aborts before the full run if it fails,
# which keeps a broken run from burning the whole allocation.
#
#   sbatch scripts/slurm/extract_embeddings.sbatch
#   MODEL=dino WEIGHTS=runs/pretrained/dino/checkpoint.pth \
#     sbatch scripts/slurm/extract_embeddings.sbatch

set -uo pipefail

ROOT="${SDG6_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
cd "$ROOT" || exit 1

MODEL="${MODEL:-dinov2}"
WEIGHTS="${WEIGHTS:-$ROOT/runs/pretrained/dinov2/teacher_checkpoint.pth}"
DINOV2_CONFIG="${DINOV2_CONFIG:-$ROOT/scripts/configs/sat_vit.yaml}"
MANIFEST="${MANIFEST:-$ROOT/data/manifest_sentinel.csv}"
OUT="${OUT:-$ROOT/runs/embeddings}"
BATCH="${BATCH:-128}"
WORKERS="${WORKERS:-8}"
PY="${PY:-$ROOT/.venv/bin/python}"

log() { echo "[$(date +'%F %T')] $*"; }

common=(--model "$MODEL" --weights "$WEIGHTS" --manifest "$MANIFEST"
        --num-workers "$WORKERS")
if [ "$MODEL" = "dinov2" ]; then
  common+=(--dinov2-config "$DINOV2_CONFIG")
fi

log "=== smoke pass (16 rows/split) on GPU"
if ! "$PY" scripts/extract_embeddings.py "${common[@]}" \
      --limit 16 --batch-size 8 --out "$OUT/_smoke"; then
  log "SMOKE FAILED — aborting before the full run"
  exit 1
fi
rm -rf "$OUT/_smoke"
log "smoke OK"

log "=== full extraction (model=$MODEL batch=$BATCH workers=$WORKERS)"
if ! "$PY" scripts/extract_embeddings.py "${common[@]}" \
      --batch-size "$BATCH" --out "$OUT"; then
  log "FULL EXTRACTION FAILED"
  exit 1
fi

log "=== outputs"
for f in "$OUT/$MODEL"/*.npz; do
  [ -e "$f" ] && log "  $(du -h "$f" | cut -f1)  $f"
done
log "DONE"
