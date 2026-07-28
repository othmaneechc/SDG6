#!/bin/bash
#SBATCH --job-name=sdg6-fullmetrics
#SBATCH --account=rrg-bengioy-ad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=124G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail
ROOT="${SDG6_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
cd "$ROOT" || exit 1
PY="${PY:-$ROOT/.venv/bin/python}"
for m in dinov2 dino galileo; do
  d="$ROOT/runs/embeddings/$m"
  [ -d "$d" ] || { echo "skip $m"; continue; }
  echo "======== $m ========"
  "$PY" scripts/eval_full_metrics.py --emb-dir "$d" --k 100 200 || exit 1
done
echo "DONE"
