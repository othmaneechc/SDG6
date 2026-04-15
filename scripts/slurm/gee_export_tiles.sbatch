#!/bin/bash
#SBATCH --job-name=sdg6-gee-tiles
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
CONFIG="${CONFIG:-${REPO}/scripts/configs/gee_export_tiles.yaml}"

export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

VENV_ACTIVATE="${VENV_ACTIVATE:-${REPO}/.venv/bin/activate}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

cd "${REPO}"
mkdir -p "${REPO}/log"
echo "Starting GEE tile export at $(date) with CONFIG=${CONFIG}"

python3 -m gee_export.batch_tiles --config "${CONFIG}" "$@"
