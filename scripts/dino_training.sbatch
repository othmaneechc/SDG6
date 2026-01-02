#!/bin/bash
#SBATCH --job-name=sdg6-dino-train
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00

set -euo pipefail

mkdir -p logs

# Optional: pin threads to avoid oversubscription
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# NCCL settings for multi-GPU stability
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DEBUG=WARN
export TORCH_NCCL_TIMEOUT=1800

# Activate env
source /home/mila/e/echchabo/projects/SDG6-Tracker/.venv/bin/activate

# Main training (2x A100)
torchrun --nproc_per_node=2 /home/mila/e/echchabo/projects/SDG6-Tracker/src/dino/main_dino.py \
	--arch vit_base \
	--data_path /home/mila/e/echchabo/scratch/PW-s \
	--output_dir /home/mila/e/echchabo/projects/SDG6-Tracker/runs \
	--epochs 50 \
	--batch_size_per_gpu 32 \
	--patch_size 8 \
	--lr 0.001 \
	--use_fp16 True

# ---------------------------------------------------------------------------
# Reference commands (kept commented; single example per script)
# ---------------------------------------------------------------------------

# k-NN eval example (2x GPUs, cosine metric)
# torchrun --nproc_per_node=2 /home/mila/e/echchabo/projects/SDG6-Tracker/scripts/knn_eval.py \
#   --data_dir /home/mila/e/echchabo/scratch/PW-m \
#   --model_name dinov3_vitl16 \
#   --weights /home/mila/e/echchabo/projects/SDG6-Tracker/src/dinov3/checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth \
#   --k_values 1,5,10,20,50,100

# Attention viz example
# python /home/mila/e/echchabo/projects/SDG6-Tracker/scripts/visualize_attention.py \
#   --pretrained_weights /home/mila/e/echchabo/projects/SDG6-Tracker/runs/checkpoint.pth \
#   --image_path /home/mila/e/echchabo/scratch/PW-s/train/sample.tif \
#   --output_dir /home/mila/e/echchabo/projects/SDG6-Tracker/runs/viz \
#   --patch_size 8 \
#   --image_size 256 256 \
#   --arch vit_base

