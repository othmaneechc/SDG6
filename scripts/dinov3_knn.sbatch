#!/bin/bash
#SBATCH --job-name=dinov3-knn
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/mila/e/echchabo/projects/SDG6-Tracker/runs/dinov3-knn-%A_%a.out
#SBATCH --error=/home/mila/e/echchabo/projects/SDG6-Tracker/runs/dinov3-knn-%A_%a.err

set -euo pipefail

module load cuda/12.1.1

REPO="/home/mila/e/echchabo/projects/SDG6-Tracker"
DINOV3_DIR="${REPO}/src/dinov3"
export PYTHONPATH="${DINOV3_DIR}:${PYTHONPATH:-}"

# Optional: activate a venv if present
VENV_ACTIVATE="${VENV_ACTIVATE:-${REPO}/.venv/bin/activate}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

CHECKPOINTS=(
  "${DINOV3_DIR}/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
  # "${DINOV3_DIR}/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
  # "${DINOV3_DIR}/checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
)

MODEL_NAMES=(
  "dinov3_vitb16"
  # "dinov3_vitl16"
  # "dinov3_vitl16"
)

# User-configurable inputs
DATA_DIR="${DATA_DIR:-/home/mila/e/echchabo/scratch/PW-m}"
RUN_DATETIME="${RUN_DATETIME:-$(date +%Y%m%dT%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/mila/e/echchabo/projects/SDG6-Tracker/runs/dinov3-knn-${RUN_DATETIME}}"
K_VALUES="${K_VALUES:-5,10,20}"
WORLD_SIZE=1
CKPT_ROOT="${CKPT_ROOT:-${REPO}/runs/dinov3-knn-cache/$(basename "${DATA_DIR}")}"
RUN_ALL="${RUN_ALL:-1}"

for split in train val test; do
  if [[ ! -d "${DATA_DIR}/${split}" ]]; then
    echo "Missing split directory: ${DATA_DIR}/${split}"
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}"

if [[ -z "${MASTER_ADDR:-}" ]]; then
  export MASTER_ADDR="127.0.0.1"
fi
if [[ -z "${MASTER_PORT:-}" ]]; then
  # Spread ports across array tasks to avoid collisions on a shared node.
  export MASTER_PORT=$((29500 + (${SLURM_ARRAY_TASK_ID:-0} % 1000)))
fi

run_one() {
  local idx="$1"
  local checkpoint="${CHECKPOINTS[idx]}"
  local model_name="${MODEL_NAMES[idx]}"
  local run_name
  run_name=$(basename "${checkpoint%.pth}")
local log_dir="${OUTPUT_ROOT}/logs"
  mkdir -p "${log_dir}"
  local log_stem
  log_stem=$(basename "${DATA_DIR}" | tr ' /' '__')
  local k_tag
  k_tag=$(echo "${K_VALUES}" | tr ',' '-')
  local log_file="${log_dir}/${run_name}-${log_stem}-k${k_tag}.log"
  local ckpt_dir="${CKPT_ROOT}/${run_name}"

  CMD=(
    python "${REPO}/scripts/run_dinov3_knn.py"
    --data-dir "${DATA_DIR}"
    --weights "${checkpoint}"
    --model-name "${model_name}"
    --k-values "${K_VALUES}"
    --ckpt-dir "${ckpt_dir}"
    --embed-batch-size 256
  )

  echo "Running: ${CMD[*]}"
  SRUN_CPUS="${SLURM_CPUS_PER_TASK:-8}"
  SRUN_CMD=(srun --ntasks=1 --cpus-per-task="${SRUN_CPUS}" --cpu-bind=none)
  "${SRUN_CMD[@]}" "${CMD[@]}" |& tee "${log_file}"
}

if [[ "${RUN_ALL}" -eq 1 ]]; then
  for i in "${!CHECKPOINTS[@]}"; do
    run_one "$i"
  done
else
  IDX=${SLURM_ARRAY_TASK_ID:-0}
  run_one "${IDX}"
fi
