#!/bin/bash
#SBATCH --job-name=sdg6-galfix
#SBATCH --account=rrg-bengioy-ad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=124G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail
ROOT="${SDG6_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
cd "$ROOT" || exit 1
PY="${PY:-$ROOT/.venv/bin/python}"
for tag in baseline scale39_norm; do
  echo "======== galileo_$tag ========"
  "$PY" scripts/eval_full_metrics.py --emb-dir "runs/embeddings/galileo_$tag" --k 5 10 20 50 100 200 || exit 1
done
echo DONE
