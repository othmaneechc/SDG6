#!/bin/bash
#SBATCH --job-name=dino-predict
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/mila/e/echchabo/projects/SDG6-Tracker/runs/dino-predict-%j.out
#SBATCH --error=/home/mila/e/echchabo/projects/SDG6-Tracker/runs/dino-predict-%j.err

set -euo pipefail

module load cuda/12.1.1

REPO="/home/mila/e/echchabo/projects/SDG6-Tracker"
PREDICT_PY="${REPO}/src/dino/predict.py"

# Ensure imports resolve
export PYTHONPATH="${REPO}/src/dino:${PYTHONPATH:-}"

# Optional: activate a venv if present
VENV_ACTIVATE="${VENV_ACTIVATE:-${REPO}/.venv/bin/activate}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

RUN_DATETIME="${RUN_DATETIME:-$(date +%Y%m%dT%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/mila/e/echchabo/scratch/SDG6-Tracker/runs/dino_PW-s/dino_PW-s}"
DATA_DIR="${DATA_DIR:-/home/mila/e/echchabo/scratch/PW-s/test}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/mila/e/echchabo/scratch/SDG6-Tracker/runs/dino_PW-s/checkpoint.pth}"
KNN_CLASSIFIER_PATH="${KNN_CLASSIFIER_PATH:-/home/mila/e/echchabo/scratch/SDG6-Tracker/runs/dino_PW-s/knn_classifier.pth}"
OUTPUT_DIR="${OUTPUT_ROOT}"

mkdir -p "${OUTPUT_ROOT}"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "DATA_DIR does not exist: ${DATA_DIR}"
  exit 1
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}"
  exit 1
fi

if [[ ! -f "${KNN_CLASSIFIER_PATH}" ]]; then
  echo "k-NN classifier not found: ${KNN_CLASSIFIER_PATH}"
  exit 1
fi

CMD=(
  python "${PREDICT_PY}"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --knn_classifier_path "${KNN_CLASSIFIER_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --directory "${DATA_DIR}"
)

echo "Running: ${CMD[*]}"
SRUN_CPUS="${SLURM_CPUS_PER_TASK:-8}"
SRUN_CMD=(srun --ntasks=1 --cpus-per-task="${SRUN_CPUS}" --cpu-bind=none)
"${SRUN_CMD[@]}" "${CMD[@]}"
