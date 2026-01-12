#!/bin/bash
#SBATCH --job-name=sdg6-dinov3-knn-hp
#SBATCH --partition=main
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/dkucc/home/oe23/SDG6-Tracker/runs/dinov3-knn-hp-%j.out
#SBATCH --error=/dkucc/home/oe23/SDG6-Tracker/runs/dinov3-knn-hp-%j.err

set -euo pipefail

REPO="/dkucc/home/oe23/SDG6-Tracker"

# ---- Activate venv -----------------------------------------------------------
VENV_ACTIVATE="${VENV_ACTIVATE:-${REPO}/.venv/bin/activate}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

# ---- Shared config -----------------------------------------------------------
DATA_ROOT="/work/lamlab/data"
WEIGHTS_SOURCE="hf"
WEIGHTS_TYPE="auto"
K_VALUES="5,10,20,50,100"
EMBED_BATCH_SIZE=256
CACHE_ROOT="${REPO}/runs/dinov3-knn-tuned"
GPUS_PER_NODE=8
MASTER_PORT="${MASTER_PORT:-29500}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

DATASETS=(
  "PW-s"
  "PW-l"
)

MODELS=(
  "facebook/dinov3-vitl16-pretrain-lvd1689m"
  "facebook/dinov3-vitb16-pretrain-lvd1689m"
  "facebook/dinov3-vitl16-pretrain-sat493m"
)

METRICS=("cosine" "l2")
WEIGHTS_SCHEMES=("uniform" "distance" "softmax")
SOFTMAX_TEMPS=("0.03" "0.07")
PCA_DIMS=("0" "128" "256")

for DATASET in "${DATASETS[@]}"; do
  DATA_DIR="${DATA_ROOT}/${DATASET}"

  for WEIGHTS in "${MODELS[@]}"; do
    MODEL_NAME="$(basename "${WEIGHTS}")"
    for METRIC in "${METRICS[@]}"; do
      for WEIGHTS_SCHEME in "${WEIGHTS_SCHEMES[@]}"; do
        temps=("n/a")
        if [[ "${WEIGHTS_SCHEME}" == "softmax" ]]; then
          temps=("${SOFTMAX_TEMPS[@]}")
        fi

        for TEMP in "${temps[@]}"; do
          for PCA_DIM in "${PCA_DIMS[@]}"; do
            RUN_TAG="metric-${METRIC}_w-${WEIGHTS_SCHEME}_temp-${TEMP}_pca-${PCA_DIM}"
            CKPT_DIR="${CACHE_ROOT}/${DATASET}/${MODEL_NAME}/${RUN_TAG}"
              
              # Skip if results already exist (use highest k marker)
              LAST_K="$(echo "${K_VALUES}" | awk -F',' '{print $NF}')"
              TEST_CM="${CKPT_DIR}/confusion/confusion_test_k${LAST_K}.txt"
              if [[ -f "${TEST_CM}" ]]; then
                echo "[SKIP] ${DATASET} / ${MODEL_NAME} / ${RUN_TAG} already has ${TEST_CM}"
                continue
              fi

            echo "============================================================"
            echo "Dataset : ${DATASET}"
            echo "Model   : ${WEIGHTS}"
            echo "Metric  : ${METRIC}"
            echo "Weights : ${WEIGHTS_SCHEME} (temp=${TEMP})"
            echo "PCA dim : ${PCA_DIM}"
            echo "Cache   : ${CKPT_DIR}"
            echo "============================================================"

            TEMP_ARG="${TEMP}"; [[ "${TEMP}" == "n/a" ]] && TEMP_ARG="0.07"

            torchrun --nproc_per_node="${GPUS_PER_NODE}" --master_port="${MASTER_PORT}" "${REPO}/src/models/dinov3/knn_eval.py" \
              --data-dir "${DATA_DIR}" \
              --source "${WEIGHTS_SOURCE}" \
              --weights "${WEIGHTS}" \
              --weights-type "${WEIGHTS_TYPE}" \
              --dtype "bf16" \
              --k-values "${K_VALUES}" \
              --ckpt-dir "${CKPT_DIR}" \
              --embed-batch-size "${EMBED_BATCH_SIZE}" \
              --knn-metric "${METRIC}" \
              --knn-weights "${WEIGHTS_SCHEME}" \
              --knn-softmax-temp "${TEMP_ARG}" \
              --pca-dim "${PCA_DIM}"
          done
        done
      done
    done
  done
done

echo "All experiments completed."
