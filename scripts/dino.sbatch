#!/bin/bash
#SBATCH --job-name=sdg6-dino-knn
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

# Optional: activate venv
VENV_ACTIVATE="${VENV_ACTIVATE:-${REPO}/.venv/bin/activate}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

# ---- User-tunable knobs -----------------------------------------------------
DATA_ROOT="${DATA_ROOT:-/path/to/dataset}"      # expects train/val/test
WEIGHTS="${WEIGHTS:-/path/to/dino/checkpoint.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/runs/dino-knn}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
K_VALUES="${K_VALUES:-5,10,20,50,100}"
KNN_METRIC="${KNN_METRIC:-cosine}"              # cosine or l2
DINO_ARCH="${DINO_ARCH:-vit_base}"
DINO_PATCH_SIZE="${DINO_PATCH_SIZE:-8}"
DINO_CHECKPOINT_KEY="${DINO_CHECKPOINT_KEY:-teacher}"

DATASETS=(${DATASETS:-PW-s})  # e.g., "PW-s" or "PW-m"; adjust loop below

for DATASET in "${DATASETS[@]}"; do
  DATA_DIR="${DATA_ROOT}/${DATASET}"
  WEIGHT_TAG="$(basename "${WEIGHTS}")"
  OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}/${WEIGHT_TAG}"
  mkdir -p "${OUTPUT_DIR}"

  echo "============================================================"
  echo "Dataset : ${DATA_DIR}"
  echo "Weights : ${WEIGHTS}"
  echo "Metric  : ${KNN_METRIC}"
  echo "Output  : ${OUTPUT_DIR}"
  echo "============================================================"

  python3 -m sdg6.cli \
    --model dino \
    --weights "${WEIGHTS}" \
    --data-dir "${DATA_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --k-values "${K_VALUES}" \
    --knn-metric "${KNN_METRIC}" \
    --dino-arch "${DINO_ARCH}" \
    --dino-patch-size "${DINO_PATCH_SIZE}" \
    --dino-checkpoint-key "${DINO_CHECKPOINT_KEY}" \
    --output-dir "${OUTPUT_DIR}"
done

echo "All DINO k-NN runs completed."
