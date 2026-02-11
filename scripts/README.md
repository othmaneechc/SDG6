# Batch jobs (YAML-driven)

All k-NN jobs now read hyperparameters from `scripts/configs/*.yaml` and simply run the unified CLI (`python -m sdg6.cli --config <file>`). `PYTHONPATH` is set relative to the script; logs go to `log/%x-%j.*`.

- `scripts/dino.sbatch` — k-NN for DINO checkpoints (config: `scripts/configs/dino.yaml`).
- `scripts/dinov2.sbatch` — k-NN for DINOv2 checkpoints (config: `scripts/configs/dinov2.yaml`).
- `scripts/dinov3.sbatch` — k-NN for Hugging Face DINOv3 (config: `scripts/configs/dinov3.yaml`).
- `scripts/galileo.sbatch` — k-NN for local Galileo encoder weights (config: `scripts/configs/galileo.yaml`).
- `scripts/dino_pt.sbatch` — DINO pretraining/fine-tuning via `dino.main_dino` (config: `scripts/configs/dino_pt.yaml`).
- `scripts/dinov2_pt.sbatch` — DINOv2 pretraining via local repo + uv (config: `scripts/configs/dinov2_pt.yaml`).
 - `scripts/gee_export_tiles.sbatch` — batch Sentinel downloads per-country from tiles CSVs (config: `scripts/configs/gee_export_tiles.yaml`).


GEE image export:
- `python -m gee_export.cli --config scripts/configs/gee_export.yaml` — downloads tiles with the UM6P service account key. Override fields on the CLI to switch datasets or coordinate files.
- `python -m gee_export.batch_tiles --config scripts/configs/gee_export_tiles.yaml` — builds centroids from per-country `*_tiles.csv` and downloads 256×256 Sentinel tiles into country subfolders.

Override the config with `CONFIG=/path/to/your.yaml sbatch scripts/galileo.sbatch` or append extra CLI flags after `--`, e.g. `sbatch scripts/dinov3.sbatch -- --device cuda:1`.
