# DINOv3 k-NN batch job

- Script: `scripts/dinov3_knn.sbatch` launches an array job (one task per checkpoint in `src/dinov3/checkpoints`).
- Driver: `scripts/run_dinov3_knn.py` loads a DINOv3 backbone via torch.hub, applies an existing k-NN classifier (or builds one from a labeled `TRAIN_DIR`), and writes predictions.

## Required inputs
- `IMAGES_DIR` (env var): folder with images to classify (default `/home/mila/e/echchabo/scratch/PW-s`).
- Either `KNN_CLASSIFIER` (path to a `.pth` with `train_features`/`train_labels`) **or** `TRAIN_DIR` (ImageFolder-style train set to build the classifier; default `/home/mila/e/echchabo/scratch/PW-s`).
- Optional env vars: `OUTPUT_ROOT` (default `runs/dinov3-knn`), `BATCH_SIZE`, `NUM_WORKERS`, `N_NEIGHBORS`, `VENV_ACTIVATE` to source a venv.

## Submit
```bash
cd /home/mila/e/echchabo/projects/SDG6-Tracker
sbatch scripts/dinov3_knn.sbatch
```

Outputs land in `runs/dinov3-knn/<checkpoint-name>.csv`; per-run logs (named with checkpoint, image folder, k) go to `runs/dinov3-knn/logs/`, and Slurm logs stay in `scripts/logs/`. The script uses `srun --cpu-bind=none` (with `SLURM_CPUS_PER_TASK` if set) to avoid CPU binding issues on interactive `salloc` runs.
