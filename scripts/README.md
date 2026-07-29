# Scripts

This directory contains the reproducible command-line workflows for the paper
and appendix.

## Core Scripts

- `build_manifest.py`: build `data/manifest_sentinel.csv`.
- `extract_embeddings.py`: encode manifest images once for both tasks.
- `eval_splits.py`: original, random, spatial-block, and country-held-out AUROC.
- `eval_baselines.py`: urban/rural baseline and settlement-stratified AUROC.
- `eval_calibration.py`: calibration methods, Brier score, ECE, and reliability.
- `burden_uncertainty.py`: bootstrap intervals for LGA burden estimates.
- `eval_full_metrics.py`: accuracy, recall, F1, and AUROC at a chosen `k`.
- `galileo_bestk.py`: validation-based `k` selection for Galileo diagnostics.

## Subdirectories

- `configs/`: YAML configs consumed by the CLI and training wrappers.
- `slurm/`: cluster launchers for long GPU or IO jobs.
- `training/`: DINO and DINOv2 pretraining wrappers.
- `analysis/`: figure, table, and appendix plotting scripts.
- `download/`: data and weight download helpers.

## Common Commands

```bash
bash scripts/download/download_all_login.sh
sbatch scripts/slurm/extract_data.sbatch
sbatch scripts/slurm/extract_embeddings.sbatch
sbatch scripts/slurm/eval_splits.sbatch
sbatch scripts/slurm/eval_baselines.sbatch
sbatch scripts/slurm/eval_calibration.sbatch
python scripts/analysis/plot_spatial_blocks.py
```

Outputs are written under `outputs/figures/`, `outputs/tables/`, and
`outputs/reports/`.
