#!/bin/bash
#SBATCH --job-name=galileo
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

# ---- Inputs -----------------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-/home/mila/e/echchabo/scratch/SDG6-Tracker}"                # expects train/val/test
WEIGHTS_DIR="${WEIGHTS_DIR:-/home/mila/e/echchabo/scratch/SDG6-Tracker/models/base}"     # folder with config.json + encoder.pt
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/runs/galileo}"
K_VALUES="${K_VALUES:-5,10,20,50,100}"
KNN_METRIC="${KNN_METRIC:-cosine}"   # cosine or l2
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"

INPUT_RESOLUTION_M="${INPUT_RESOLUTION_M:-10}"   # meters-per-pixel for positional encoding
PATCH_SIZE="${PATCH_SIZE:-0}"                     # override patch size (0 = use checkpoint config)
VALUE_SCALE="${VALUE_SCALE:-1.0}"                 # multiply raw band values (e.g., 0.0001 for reflectance)
NORMALIZE="${NORMALIZE:-true}"                    # apply Galileo pretraining normalization
COMPUTE_NDVI="${COMPUTE_NDVI:-true}"              # derive NDVI if B4/B8 present
PAD_SQUARE="${PAD_SQUARE:-true}"                  # pad H/W to square
PAD_TO_PATCH="${PAD_TO_PATCH:-true}"              # pad so H/W divisible by patch size
DEFAULT_MONTH_INDEX="${DEFAULT_MONTH_INDEX:-5}"   # fallback month (0-11) if parsing fails
BAND_INDICES="${BAND_INDICES:-}"                  # optional channel indices to select from TIFF
BAND_NAMES="${BAND_NAMES:-}"                      # optional names for selected channels (e.g., B2,B3,B4,B8)

DATASETS=(${DATASETS:-PW-s SW-s})
# ---- Run --------------------------------------------------------------------
for DATASET in "${DATASETS[@]}"; do
  DATA_DIR="${DATA_ROOT}/${DATASET}"
  MODEL_TAG="$(basename "${WEIGHTS_DIR}")"
  OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}/${MODEL_TAG}"

  EXTRA_ARGS=()
  if [[ -n "${BAND_INDICES}" ]]; then
    EXTRA_ARGS+=(--galileo-band-indices "${BAND_INDICES}")
  fi
  if [[ -n "${BAND_NAMES}" ]]; then
    EXTRA_ARGS+=(--galileo-band-names "${BAND_NAMES}")
  fi
  if [[ "${NORMALIZE}" == "true" ]]; then
    EXTRA_ARGS+=(--galileo-normalize)
  fi
  if [[ "${COMPUTE_NDVI}" == "true" ]]; then
    EXTRA_ARGS+=(--galileo-compute-ndvi)
  fi
  if [[ "${PAD_SQUARE}" != "true" ]]; then
    EXTRA_ARGS+=(--galileo-no-pad-square)
  fi
  if [[ "${PAD_TO_PATCH}" != "true" ]]; then
    EXTRA_ARGS+=(--galileo-no-pad-to-patch)
  fi

  echo "============================================================"
  echo "Dataset : ${DATASET}"
  echo "Weights : ${WEIGHTS_DIR}"
  echo "Metric  : ${KNN_METRIC}"
  echo "Output  : ${OUTPUT_DIR}"
  echo "============================================================"

  python3 -m sdg6.cli \
    --model galileo \
    --weights "${WEIGHTS_DIR}" \
    --data-dir "${DATA_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --k-values "${K_VALUES}" \
    --knn-metric "${KNN_METRIC}" \
    --galileo-input-resolution-m "${INPUT_RESOLUTION_M}" \
    --galileo-patch-size "${PATCH_SIZE}" \
    --galileo-value-scale "${VALUE_SCALE}" \
    --galileo-default-month-index "${DEFAULT_MONTH_INDEX}" \
    --output-dir "${OUTPUT_DIR}" \
    "${EXTRA_ARGS[@]}"
done

echo "All experiments completed."
