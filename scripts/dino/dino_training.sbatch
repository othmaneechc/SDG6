#!/bin/bash
#SBATCH --job-name=sdg6-dino-train
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=120:00:00

set -euo pipefail

# mkdir -p logs 

# Optional: pin threads to avoid oversubscription
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

# NCCL settings for multi-GPU stability
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DEBUG=WARN
export TORCH_NCCL_TIMEOUT=1800

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# Activate env

# source /home/mila/e/echchabo/projects/SDG6-Tracker/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=1,2,4,5,6,7

# Timestamped output directory grouped by model (override MODEL_NAME to change)
MODEL_NAME=${MODEL_NAME:-dino}
DATA_PATH=${DATA_PATH:-/work/lamlab/data/PW-s}
DATA_NAME=$(basename "$DATA_PATH")
RUN_TS=$(date +"%Y%m%d_%H%M%S")
OUTPUT_ROOT=/work/lamlab/data/runs
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}_${DATA_NAME}"
mkdir -p "$OUTPUT_DIR"

torchrun --nproc_per_node=6 --module dino.main_dino \
	--arch vit_base \
	--data_path "$DATA_PATH" \
	--output_dir "$OUTPUT_DIR" \
	--epochs 200 \
	--batch_size_per_gpu 64 \
	--patch_size 8 \
	--lr 0.001 \
	--use_fp16 True
