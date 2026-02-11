#!/bin/bash
#SBATCH --job-name=sdg6-dinov2-train
#SBATCH --partition=main
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
CONFIG="${CONFIG:-${REPO}/scripts/configs/dinov2_pt.yaml}"

export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

# Optional: activate a venv
VENV_ACTIVATE="${VENV_ACTIVATE:-${REPO}/.venv/bin/activate}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

cd "${REPO}"
mkdir -p "${REPO}/log"
python3 scripts/train_dinov2.py --config "${CONFIG}" "$@"
