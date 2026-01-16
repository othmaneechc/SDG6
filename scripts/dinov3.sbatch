#!/bin/bash
#SBATCH --job-name=sdg6-dinov3-knn
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
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
DATA_ROOT="${DATA_ROOT:-/path/to/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/runs/dinov3-knn}"
MODELS=(${MODELS:-facebook/dinov3-vitb16-pretrain-lvd1689m})
DATASETS=(${DATASETS:-PW-s})
K_VALUES="${K_VALUES:-5,10,20,50,100}"
KNN_METRIC="${KNN_METRIC:-cosine}"   # cosine or l2
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
WEIGHTS_TYPE="${WEIGHTS_TYPE:-auto}" # auto|lvd|sat

for DATASET in "${DATASETS[@]}"; do
  DATA_DIR="${DATA_ROOT}/${DATASET}"
  for WEIGHTS in "${MODELS[@]}"; do
    MODEL_TAG="$(basename "${WEIGHTS}")"
    OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}/${MODEL_TAG}"
    mkdir -p "${OUTPUT_DIR}"

    echo "============================================================"
    echo "Dataset : ${DATA_DIR}"
    echo "Weights : ${WEIGHTS}"
    echo "Metric  : ${KNN_METRIC}"
    echo "Output  : ${OUTPUT_DIR}"
    echo "============================================================"

    python3 -m sdg6.cli \
      --model dinov3 \
      --weights "${WEIGHTS}" \
      --data-dir "${DATA_DIR}" \
      --weights-type "${WEIGHTS_TYPE}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --k-values "${K_VALUES}" \
      --knn-metric "${KNN_METRIC}" \
      --output-dir "${OUTPUT_DIR}"
  done
done

echo "All DINOv3 k-NN runs completed."
