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

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
CONFIG="${CONFIG:-${REPO}/scripts/configs/dino_pt.yaml}"

export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

# Optional: activate a venv
VENV_ACTIVATE="${VENV_ACTIVATE:-${REPO}/.venv/bin/activate}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

cd "${REPO}"
mkdir -p "${REPO}/log"
python3 scripts/train_dino.py --config "${CONFIG}" "$@"
