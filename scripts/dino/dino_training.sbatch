#!/bin/bash
#SBATCH --job-name=sdg6-dino-train
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=110:00:00

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
source /home/mila/e/echchabo/projects/SDG6-Tracker/.venv/bin/activate

# Timestamped output directory grouped by model (override MODEL_NAME to change)
MODEL_NAME=${MODEL_NAME:-dino}
DATA_PATH=${DATA_PATH:-/network/scratch/e/echchabo/SDG6-Tracker/PW-s}
DATA_NAME=$(basename "$DATA_PATH")
RUN_TS=$(date +"%Y%m%d_%H%M%S")
OUTPUT_ROOT=/network/scratch/e/echchabo/SDG6-Tracker/runs
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}_${DATA_NAME}"
mkdir -p "$OUTPUT_DIR"

torchrun --nproc_per_node=2 /home/mila/e/echchabo/projects/SDG6-Tracker/src/dino/main_dino.py \
	--arch vit_base \
	--data_path "$DATA_PATH" \
	--output_dir "$OUTPUT_DIR" \
	--epochs 50 \
	--batch_size_per_gpu 32 \
	--patch_size 8 \
	--lr 0.001 \
	--use_fp16 True

DATA_PATH=${DATA_PATH:-/network/scratch/e/echchabo/SDG6-Tracker/PW-m}
DATA_NAME=$(basename "$DATA_PATH")
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}_${DATA_NAME}_${RUN_TS}"
mkdir -p "$OUTPUT_DIR"

# Main training (use the 8 GPUs requested above)
torchrun --standalone --nproc_per_node=4 /dkucc/home/oe23/SDG6-Tracker/src/dino/main_dino.py \
	--arch vit_base \
	--data_path "$DATA_PATH" \
	--output_dir "$OUTPUT_DIR" \
	--epochs 50 \
	--batch_size_per_gpu 16 \
	--patch_size 8 \
	--lr 0.001 \
	--use_fp16 True


# ---------------------------------------------------------------------------
# Reference commands (kept commented; single example per script)
# ---------------------------------------------------------------------------

# k-NN eval example (2x GPUs, cosine metric)
# torchrun --nproc_per_node=8 --master_port=29501 /dkucc/home/oe23/MODELS/dino/eval_knn.py \
#   --pretrained_weights /dkucc/home/oe23/DATABASE/RESULTS/base/checkpoint.pth \
#   --checkpoint_key teacher \
#   --data_path /work/lamlab/TEST_C/Sentinel \
#   --patch_size 8 \
#   --arch vit_base \
#   --output /dkucc/home/oe23/DATABASE/RESULTS/val/rural_sew

# Attention viz example
# python /home/mila/e/echchabo/projects/SDG6-Tracker/scripts/visualize_attention.py \
#   --pretrained_weights /home/mila/e/echchabo/projects/SDG6-Tracker/runs/checkpoint.pth \
#   --image_path /network/scratch/e/echchabo/PW-s/train/sample.tif \
#   --output_dir /home/mila/e/echchabo/projects/SDG6-Tracker/runs/viz \
#   --patch_size 8 \
#   --image_size 256 256 \
#   --arch vit_base
