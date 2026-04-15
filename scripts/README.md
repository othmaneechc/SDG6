# Scripts Layout

This directory is organized for camera-ready use with portable paths.

- `scripts/slurm/`:
  SLURM launchers for training, evaluation, inference, and GEE export.
- `scripts/configs/`:
  YAML configs consumed by `sdg6.cli`, `sdg6.infer_knn`, and training wrappers.
- `scripts/training/`:
  Python wrappers for DINO and DINOv2 pretraining.
- `scripts/analysis/`:
  Plotting/statistics scripts (including converted notebook workflows).

## SLURM launchers

- `scripts/slurm/dino.sbatch` — k-NN for DINO checkpoints (`scripts/configs/dino.yaml`)
- `scripts/slurm/dinov2.sbatch` — k-NN for DINOv2 checkpoints (`scripts/configs/dinov2.yaml`)
- `scripts/slurm/dinov3.sbatch` — k-NN for DINOv3 checkpoints (`scripts/configs/dinov3.yaml`)
- `scripts/slurm/prithvi.sbatch` — k-NN for Prithvi checkpoints (`scripts/configs/prithvi.yaml`)
- `scripts/slurm/galileo.sbatch` — k-NN for Galileo checkpoints (`scripts/configs/galileo.yaml`)
- `scripts/slurm/dino_pt.sbatch` — DINO pretraining (`scripts/configs/dino_pt.yaml`)
- `scripts/slurm/dinov2_pt.sbatch` — DINOv2 pretraining (`scripts/configs/dinov2_pt.yaml`)
- `scripts/slurm/dinov2_infer.sbatch` — batched country inference (`scripts/configs/dinov2_infer.yaml`)
- `scripts/slurm/gee_export_tiles.sbatch` — Sentinel tile export (`scripts/configs/gee_export_tiles.yaml`)

Examples:

```bash
sbatch scripts/slurm/dinov2.sbatch
CONFIG=scripts/configs/galileo.yaml sbatch scripts/slurm/galileo.sbatch
sbatch scripts/slurm/dinov3.sbatch -- --device cuda:1
```

## Analysis scripts

- `python scripts/analysis/plot_nigeria_access_hotspots.py`
- `python scripts/analysis/plot_no_survey_population_cdf.py`
- `python scripts/analysis/compute_dino_family_auroc.py`
- `python scripts/analysis/figures.py`
- `python scripts/analysis/stats.py`
- `python scripts/analysis/urban_rural_split_analysis.py`
- `python scripts/analysis/count_unique_countries.py`

## Portable path defaults

Analysis scripts default to repo-relative paths and can be overridden with:

- `SDG6_DATA_ROOT` (defaults to `data/`)
- `SDG6_RUNS_ROOT` (defaults to `runs/`)

Generated artifacts are written under `outputs/`:

- `outputs/figures/`
- `outputs/tables/`
- `outputs/reports/`
