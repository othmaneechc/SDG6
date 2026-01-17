#!/bin/bash
#SBATCH --job-name=sdg6-dinov3-knn
#SBATCH --partition=main
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
CONFIG="${CONFIG:-${REPO}/scripts/configs/dinov3.yaml}"

export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Optional: activate venv
VENV_ACTIVATE="${VENV_ACTIVATE:-${REPO}/.venv/bin/activate}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

cd "${REPO}"
mkdir -p "${REPO}/log"
echo "Starting dinov3 job at $(date) with CONFIG=${CONFIG}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"
if [[ "${GPUS_PER_NODE}" -gt 1 ]]; then
  torchrun --nproc_per_node="${GPUS_PER_NODE}" --master_port="${MASTER_PORT}" -m sdg6.cli --config "${CONFIG}" "$@"
else
  python3 -m sdg6.cli --config "${CONFIG}" "$@"
fi
