#!/bin/bash
#SBATCH --job-name=sdg6-dinov3-pretrain
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/mila/e/echchabo/projects/SDG6-Tracker/runs/dinov3-pretrain-%j.out
#SBATCH --error=/home/mila/e/echchabo/projects/SDG6-Tracker/runs/dinov3-pretrain-%j.err

set -euo pipefail

module load cuda/12.1.1

REPO="/home/mila/e/echchabo/projects/SDG6-Tracker"

VENV_ACTIVATE="${VENV_ACTIVATE:-${REPO}/.venv/bin/activate}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

# Expected data layout: DATA_ROOT/{train,val,test}/{class}/*.png (+ optional .mask.npy)
DATA_ROOT="${DATA_ROOT:-/network/scratch/e/echchabo/SDG6-Tracker/paired}"
DATA_NAME="$(basename "${DATA_ROOT}")"
OUTPUT_ROOT="${OUTPUT_ROOT:-/network/scratch/e/echchabo/SDG6-Tracker/runs/dinov3}"
RUN_TS="${RUN_TS:-$(date +%Y%m%dT%H%M%S)}"
OUTPUT_DIR="${OUTPUT_ROOT}/dinov3_${DATA_NAME}_${RUN_TS}"
mkdir -p "${OUTPUT_DIR}"

BACKBONE="${BACKBONE:-facebook/dinov3-convnext-tiny-pretrain-lvd1689m}"
CHECKPOINT="${CHECKPOINT:-}"

BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-20}"
LR_HEAD="${LR_HEAD:-1e-4}"
LR_BACKBONE="${LR_BACKBONE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-2}"
UNFREEZE_LAST_BLOCKS="${UNFREEZE_LAST_BLOCKS:-0}"

PAIR_COMPOSITION="${PAIR_COMPOSITION:-b_a_delta}"
CLASS_WEIGHTING="${CLASS_WEIGHTING:-none}"
LOSS="${LOSS:-ce}"

APPLY_COLOR_JITTER="${APPLY_COLOR_JITTER:-1}"
EXTRA_AUGMENT="${EXTRA_AUGMENT:-0}"
TRAIN_REPEAT="${TRAIN_REPEAT:-1}"
MAX_ROTATION="${MAX_ROTATION:-20}"
MAX_TRANSLATE="${MAX_TRANSLATE:-0.05}"
MAX_SCALE="${MAX_SCALE:-0.1}"
VFLIP_PROB="${VFLIP_PROB:-0.2}"

CMD=(
  python "${REPO}/scripts/train_dinov3.py"
  --data-root "${DATA_ROOT}"
  --backbone "${BACKBONE}"
  --output "${OUTPUT_DIR}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr-head "${LR_HEAD}"
  --lr-backbone "${LR_BACKBONE}"
  --weight-decay "${WEIGHT_DECAY}"
  --num-workers "${NUM_WORKERS}"
  --freeze-backbone-epochs "${FREEZE_BACKBONE_EPOCHS}"
  --pair-composition "${PAIR_COMPOSITION}"
  --class-weighting "${CLASS_WEIGHTING}"
  --loss "${LOSS}"
  --train-repeat "${TRAIN_REPEAT}"
  --max-rotation "${MAX_ROTATION}"
  --max-translate "${MAX_TRANSLATE}"
  --max-scale "${MAX_SCALE}"
  --vflip-prob "${VFLIP_PROB}"
  --unfreeze-last-blocks "${UNFREEZE_LAST_BLOCKS}"
  --device cuda
)

if [[ -n "${CHECKPOINT}" ]]; then
  CMD+=(--checkpoint "${CHECKPOINT}")
fi
if [[ "${APPLY_COLOR_JITTER}" == "1" ]]; then
  CMD+=(--apply-color-jitter)
fi
if [[ "${EXTRA_AUGMENT}" == "1" ]]; then
  CMD+=(--extra-augment)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
