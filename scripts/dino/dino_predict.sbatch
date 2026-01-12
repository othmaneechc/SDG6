#!/bin/bash
#SBATCH --job-name=sdg6-dino-train
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=120:00:00

set -euo pipefail

# mkdir -p logs 

# Timestamped output directory grouped by model (override MODEL_NAME to change)
MODEL_NAME=${MODEL_NAME:-dino}
DATA_PATH=${DATA_PATH:-/network/scratch/e/echchabo/SDG6-Tracker/PW-s}
DATA_NAME=$(basename "$DATA_PATH")
RUN_TS=$(date +"%Y%m%d_%H%M%S")
OUTPUT_ROOT=/network/scratch/e/echchabo/SDG6-Tracker/val_pws_old
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}_${DATA_NAME}"
mkdir -p "$OUTPUT_DIR"

# k-NN eval example (2x GPUs, cosine metric)
torchrun --nproc_per_node=2 --master_port=29501 --module dino.eval_knn \
  --pretrained_weights /home/mila/e/echchabo/scratch/checkpoint.pth \
  --checkpoint_key teacher \
  --data_path "$DATA_PATH" \
  --patch_size 8 \
  --arch vit_base \
  --dump_features "$OUTPUT_DIR" \

# Attention viz example
# python /home/mila/e/echchabo/projects/SDG6-Tracker/scripts/visualize_attention.py \
#   --pretrained_weights /home/mila/e/echchabo/projects/SDG6-Tracker/runs/checkpoint.pth \
#   --image_path /network/scratch/e/echchabo/PW-s/train/sample.tif \
#   --output_dir /home/mila/e/echchabo/projects/SDG6-Tracker/runs/viz \
#   --patch_size 8 \
#   --image_size 256 256 \
#   --arch vit_base
