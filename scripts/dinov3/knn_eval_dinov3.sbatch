#!/bin/bash
#SBATCH --job-name=sdg6-dinov3-knn
#SBATCH --partition=main
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/dkucc/home/oe23/SDG6-Tracker/runs/dinov3-knn-%j.out
#SBATCH --error=/dkucc/home/oe23/SDG6-Tracker/runs/dinov3-knn-%j.err

set -euo pipefail

REPO="/dkucc/home/oe23/SDG6-Tracker"

# ---- Activate venv -----------------------------------------------------------
VENV_ACTIVATE="${VENV_ACTIVATE:-${REPO}/.venv/bin/activate}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

# ---- Shared config -----------------------------------------------------------
DATA_ROOT="/work/lamlab/data"
WEIGHTS_SOURCE="hf"
WEIGHTS_TYPE="auto"
K_VALUES="5,10,20,50,100"
EMBED_BATCH_SIZE=256
CACHE_ROOT="${REPO}/runs/dinov3-knn-cache"
GPUS_PER_NODE=4
MASTER_PORT="${MASTER_PORT:-29500}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

# ---- Experiments -------------------------------------------------------------
DATASETS=(
  "PW-s"
  "PW-m"
  "PW-l"
  "SW-s"
  "SW-m"
  "SW-l"
)

MODELS=(
  "facebook/dinov3-vitl16-pretrain-lvd1689m"
  "facebook/dinov3-vitb16-pretrain-lvd1689m"
  "facebook/dinov3-vitl16-pretrain-sat493m"
)

# ---- Run grid ---------------------------------------------------------------
for DATASET in "${DATASETS[@]}"; do
  DATA_DIR="${DATA_ROOT}/${DATASET}"

  for WEIGHTS in "${MODELS[@]}"; do
    MODEL_NAME="$(basename "${WEIGHTS}")"
    CKPT_DIR="${CACHE_ROOT}/${DATASET}/${MODEL_NAME}"

    echo "============================================================"
    echo "Dataset : ${DATASET}"
    echo "Model   : ${WEIGHTS}"
    echo "Cache   : ${CKPT_DIR}"
    echo "============================================================"

    torchrun --nproc_per_node="${GPUS_PER_NODE}" --master_port="${MASTER_PORT}" "${REPO}/src/models/dinov3/knn_eval.py" \
      --data-dir "${DATA_DIR}" \
      --source "${WEIGHTS_SOURCE}" \
      --weights "${WEIGHTS}" \
      --weights-type "${WEIGHTS_TYPE}" \
      --dtype "bf16" \
      --k-values "${K_VALUES}" \
      --ckpt-dir "${CKPT_DIR}" \
      --embed-batch-size "${EMBED_BATCH_SIZE}"
  done
done

echo "All experiments completed."
