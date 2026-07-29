#!/bin/bash
#SBATCH --job-name=sdg6-galbestk
#SBATCH --account=rrg-bengioy-ad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=124G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail
ROOT="${SDG6_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"; cd "$ROOT"||exit 1
"${PY:-$ROOT/.venv/bin/python}" scripts/galileo_bestk.py; echo DONE
