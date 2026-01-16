# Batch jobs (path agnostic)

All jobs call the unified CLI (`python -m sdg6.cli`). `PYTHONPATH` is set relative to the script location; override `DATA_ROOT`, `WEIGHTS`, etc. via env vars before `sbatch`.

- `scripts/dino_pt.sbatch` — DINO pretraining/fine-tuning via `dino.main_dino` (data + hyperparams configurable).
- `scripts/dino.sbatch` — k-NN eval for DINO checkpoints.
- `scripts/dinov3.sbatch` — k-NN eval for Hugging Face DINOv3 weights.
- `scripts/galileo.sbatch` — k-NN eval for local Galileo encoder weights.

K-NN metric choices are limited to `cosine` or `l2` (set `KNN_METRIC`), with shared knobs like `K_VALUES`, `BATCH_SIZE`, and `OUTPUT_ROOT`.
