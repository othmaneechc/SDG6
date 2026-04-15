#!/bin/bash
#SBATCH --job-name=sdg6-dinov2-infer
#SBATCH --partition=main
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
CONFIG="${CONFIG:-${REPO}/scripts/configs/dinov2_infer.yaml}"

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

COUNTRIES_ROOT="${COUNTRIES_ROOT:-${REPO}/data/countries}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/outputs/inference}"
DINOV2_WEIGHTS="${DINOV2_WEIGHTS:-${REPO}/runs/dinov2/teacher_checkpoint.pth}"
DINOV2_CONFIG="${DINOV2_CONFIG:-${REPO}/scripts/configs/sat_vit.yaml}"
DINOV2_CKPT_KEY="${DINOV2_CKPT_KEY:-teacher}"
KNN_SW="${KNN_SW:-${REPO}/runs/dinov2/knn_classifier_sw-s.npz}"
KNN_PW="${KNN_PW:-${REPO}/runs/dinov2/knn_classifier_pw-s.npz}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RESIZE="${RESIZE:-256}"
CROP="${CROP:-224}"

for country_dir in "${COUNTRIES_ROOT}"/*; do
  [[ -d "${country_dir}" ]] || continue
  country="$(basename "${country_dir}")"
  out_dir="${OUTPUT_ROOT}/${country}"
  mkdir -p "${out_dir}"

  cfg_sw="$(mktemp)"
  cat "${CONFIG}" > "${cfg_sw}"
  cat >> "${cfg_sw}" <<EOF
weights: ${DINOV2_WEIGHTS}
dinov2_config: ${DINOV2_CONFIG}
dinov2_checkpoint_key: ${DINOV2_CKPT_KEY}
knn_classifier_path: ${KNN_SW}
input_dir: ${country_dir}
output_csv: ${out_dir}/sw_predictions.csv
batch_size: ${BATCH_SIZE}
num_workers: ${NUM_WORKERS}
dinov2_resize: ${RESIZE}
dinov2_crop: ${CROP}
EOF

  cfg_pw="$(mktemp)"
  cat "${CONFIG}" > "${cfg_pw}"
  cat >> "${cfg_pw}" <<EOF
weights: ${DINOV2_WEIGHTS}
dinov2_config: ${DINOV2_CONFIG}
dinov2_checkpoint_key: ${DINOV2_CKPT_KEY}
knn_classifier_path: ${KNN_PW}
input_dir: ${country_dir}
output_csv: ${out_dir}/pw_predictions.csv
batch_size: ${BATCH_SIZE}
num_workers: ${NUM_WORKERS}
dinov2_resize: ${RESIZE}
dinov2_crop: ${CROP}
EOF

  python3 -m sdg6.infer_knn --config "${cfg_sw}" "$@"
  python3 -m sdg6.infer_knn --config "${cfg_pw}" "$@"

  rm -f "${cfg_sw}" "${cfg_pw}"
done
