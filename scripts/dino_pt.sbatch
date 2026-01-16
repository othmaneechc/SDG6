#!/bin/bash
#SBATCH --job-name=sdg6-dino-train
#SBATCH --partition=main
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

# Optional: activate a venv
VENV_ACTIVATE="${VENV_ACTIVATE:-${REPO}/.venv/bin/activate}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

# Threading / NCCL defaults
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DEBUG=WARN
export TORCH_NCCL_TIMEOUT=1800

# ---- User-tunable knobs -----------------------------------------------------
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
DATA_PATH="${DATA_PATH:-/path/to/train_data}"   # ImageFolder-style
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/runs/dino-train}"
ARCH="${ARCH:-vit_base}"
PATCH_SIZE="${PATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-64}"
LR="${LR:-0.001}"
USE_FP16="${USE_FP16:-true}"

DATA_NAME="$(basename "${DATA_PATH}")"
RUN_TS="$(date +"%Y%m%d_%H%M%S")"
OUTPUT_DIR="${OUTPUT_ROOT}/${ARCH}_p${PATCH_SIZE}_${DATA_NAME}_${RUN_TS}"
mkdir -p "${OUTPUT_DIR}"

echo "Training DINO: arch=${ARCH}, patch=${PATCH_SIZE}, data=${DATA_PATH}, out=${OUTPUT_DIR}"

torchrun --nproc_per_node="${GPUS_PER_NODE}" --master_port="${MASTER_PORT}" -m dino.main_dino \
  --arch "${ARCH}" \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch_size_per_gpu "${BATCH_SIZE_PER_GPU}" \
  --patch_size "${PATCH_SIZE}" \
  --lr "${LR}" \
  --use_fp16 "${USE_FP16}"
