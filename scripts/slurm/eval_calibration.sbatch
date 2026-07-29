#!/bin/bash
#SBATCH --job-name=sdg6-calib
#SBATCH --account=rrg-bengioy-ad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=124G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Calibration of the access probabilities (Reviewer 1, point 3).
#   sbatch scripts/slurm/eval_calibration.sbatch

set -uo pipefail
ROOT="${SDG6_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
cd "$ROOT" || exit 1
PY="${PY:-$ROOT/.venv/bin/python}"

rm -f outputs/tables/calibration_summary.csv outputs/tables/calibration_reliability.csv
for task in pw sw; do
  echo "[$(date +'%F %T')] === task=$task"
  "$PY" scripts/eval_calibration.py --task "$task" || exit 1
done
echo "[$(date +%F\ %T)] DONE"
