#!/bin/bash
#SBATCH --job-name=sdg6-dinov3-knn
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/mila/e/echchabo/projects/SDG6-Tracker/runs/dinov3-knn-%j.out
#SBATCH --error=/home/mila/e/echchabo/projects/SDG6-Tracker/runs/dinov3-knn-%j.err

set -euo pipefail

module load cuda/12.1.1

REPO="/home/mila/e/echchabo/projects/SDG6-Tracker"

VENV_ACTIVATE="${VENV_ACTIVATE:-${REPO}/.venv/bin/activate}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-/home/mila/e/echchabo/scratch/PW-m}"
WEIGHTS_SOURCE="${WEIGHTS_SOURCE:-hf}"
WEIGHTS_TYPE="${WEIGHTS_TYPE:-auto}"
K_VALUES="${K_VALUES:-5,10,20}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"
CKPT_DIR="${CKPT_DIR:-${REPO}/runs/dinov3-knn-cache/$(basename "${DATA_DIR}")/${WEIGHTS_SOURCE}}"
WEIGHTS="${WEIGHTS:-facebook/dinov3-vitb16-pretrain-lvd1689m}"

CMD=(
  python "${REPO}/scripts/run_dinov3_knn.py"
  --data-dir "${DATA_DIR}"
  --source "${WEIGHTS_SOURCE}"
  --weights "${WEIGHTS}"
  --weights-type "${WEIGHTS_TYPE}"
  --k-values "${K_VALUES}"
  --ckpt-dir "${CKPT_DIR}"
  --embed-batch-size "${EMBED_BATCH_SIZE}"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"
