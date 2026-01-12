#!/bin/bash
#SBATCH --job-name=sdg6-galileo-knn
#SBATCH --partition=main
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=120:00:00
#SBATCH --output=/dkucc/home/oe23/SDG6-Tracker/runs/galileo-knn-%j.out
#SBATCH --error=/dkucc/home/oe23/SDG6-Tracker/runs/galileo-knn-%j.err

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# ---- Activate venv -----------------------------------------------------------
VENV_ACTIVATE="${VENV_ACTIVATE:-${REPO}/.venv/bin/activate}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

# ---- Shared config -----------------------------------------------------------
DATA_ROOT="/home/mila/e/echchabo/scratch/SDG6-Tracker/"
WEIGHTS_DIR="${WEIGHTS_DIR:-/home/mila/e/echchabo/scratch/SDG6-Tracker/models/base}"
K_VALUES="5,10,20,50,100"
EMBED_BATCH_SIZE=16
CACHE_ROOT="/home/mila/e/echchabo/scratch/SDG6-Tracker/runs/galileo-knn"
GPUS_PER_NODE=2
MASTER_PORT="${MASTER_PORT:-29500}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

INPUT_RESOLUTION_M=10
PATCH_SIZE=0
VALUE_SCALE=1.0
NORMALIZE=true
COMPUTE_NDVI=true
PAD_SQUARE=true
PAD_TO_PATCH=true
DEFAULT_MONTH_INDEX=5

# Optional overrides for band mapping
BAND_INDICES="${BAND_INDICES:-}"
BAND_NAMES="${BAND_NAMES:-}"

# ---- Experiments -------------------------------------------------------------
DATASETS=(
  "PW-s"
  "SW-s"
)

# ---- Run grid ---------------------------------------------------------------
for DATASET in "${DATASETS[@]}"; do
  DATA_DIR="${DATA_ROOT}/${DATASET}"
  MODEL_TAG="$(basename "${WEIGHTS_DIR}")"
  CKPT_DIR="${CACHE_ROOT}/${DATASET}/${MODEL_TAG}"

  EXTRA_ARGS=()
  if [[ -n "${BAND_INDICES}" ]]; then
    EXTRA_ARGS+=(--band-indices "${BAND_INDICES}")
  fi
  if [[ -n "${BAND_NAMES}" ]]; then
    EXTRA_ARGS+=(--band-names "${BAND_NAMES}")
  fi
  if [[ "${NORMALIZE}" == "true" ]]; then
    EXTRA_ARGS+=(--normalize)
  fi
  if [[ "${COMPUTE_NDVI}" == "true" ]]; then
    EXTRA_ARGS+=(--compute-ndvi)
  fi
  if [[ "${PAD_SQUARE}" != "true" ]]; then
    EXTRA_ARGS+=(--no-pad-square)
  fi
  if [[ "${PAD_TO_PATCH}" != "true" ]]; then
    EXTRA_ARGS+=(--no-pad-to-patch)
  fi

  echo "============================================================"
  echo "Dataset : ${DATASET}"
  echo "Weights : ${WEIGHTS_DIR}"
  echo "Cache   : ${CKPT_DIR}"
  echo "============================================================"

  torchrun --nproc_per_node="${GPUS_PER_NODE}" --master_port="${MASTER_PORT}" "${REPO}/src/models/galileo/knn_eval.py" \
    --data-dir "${DATA_DIR}" \
    --weights-dir "${WEIGHTS_DIR}" \
    --k-values "${K_VALUES}" \
    --ckpt-dir "${CKPT_DIR}" \
    --embed-batch-size "${EMBED_BATCH_SIZE}" \
    --input-resolution-m "${INPUT_RESOLUTION_M}" \
    --patch-size "${PATCH_SIZE}" \
    --value-scale "${VALUE_SCALE}" \
    --default-month-index "${DEFAULT_MONTH_INDEX}" \
    "${EXTRA_ARGS[@]}"

done

echo "All experiments completed."
