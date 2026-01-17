# Batch jobs (YAML-driven)

All k-NN jobs now read hyperparameters from `scripts/configs/*.yaml` and simply run the unified CLI (`python -m sdg6.cli --config <file>`). `PYTHONPATH` is set relative to the script; logs go to `log/%x-%j.*`.

- `scripts/dino.sbatch` — k-NN for DINO checkpoints (config: `scripts/configs/dino.yaml`).
- `scripts/dinov3.sbatch` — k-NN for Hugging Face DINOv3 (config: `scripts/configs/dinov3.yaml`).
- `scripts/galileo.sbatch` — k-NN for local Galileo encoder weights (config: `scripts/configs/galileo.yaml`).
- `scripts/dino_pt.sbatch` — DINO pretraining/fine-tuning via `dino.main_dino` (config: `scripts/configs/dino_pt.yaml`).

Override the config with `CONFIG=/path/to/your.yaml sbatch scripts/galileo.sbatch` or append extra CLI flags after `--`, e.g. `sbatch scripts/dinov3.sbatch -- --device cuda:1`.
