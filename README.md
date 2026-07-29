# SDG6 Tracker

Code and reproducible experiment artifacts for:

**Seeing SDG 6 from space: local-scale monitoring of piped water and sewage
system access across Africa using satellite imagery and self-supervised
learning**

Paper: https://arxiv.org/abs/2411.19093

The repository supports three tasks:

- train or load satellite image encoders;
- extract reusable image embeddings;
- evaluate, calibrate, and run k-NN access prediction experiments.

## Layout

- `src/models/`: model adapters for DINO, DINOv2, DINOv3, Prithvi, and Galileo.
- `src/sdg6/`: datasets, embedding extraction, k-NN, inference, and split helpers.
- `scripts/configs/`: YAML configs for training, evaluation, export, and inference.
- `scripts/slurm/`: cluster launchers.
- `scripts/analysis/`: figure, table, and appendix analysis scripts.
- `outputs/`: committed figures, tables, reports, and calibrators.

## Setup

```bash
uv sync
```

The repo uses Python 3.10/3.11 and PyTorch. Most full experiments are intended
for a GPU node.

## Data And Weights

Published data and weights are external:

- DINO/DINOv2 weights, inference results, and population patches:
  https://zenodo.org/records/19156085
- Afrobarometer imagery tiles:
  https://zenodo.org/records/14740420
- Galileo weights:
  https://huggingface.co/nasaharvest/galileo

Download and extract the released artifacts:

```bash
bash scripts/download/download_all_login.sh
sbatch scripts/slurm/extract_data.sbatch
```

Build the manifest used by the current experiments:

```bash
python scripts/build_manifest.py --base-dir DATABASE/RAW --out data/manifest_sentinel.csv
```

The manifest stores one row per image with both labels, split, coordinates,
country, and settlement type. It replaces duplicated ImageFolder symlink trees.

## Main Experiments

Extract DINOv2 embeddings once:

```bash
sbatch scripts/slurm/extract_embeddings.sbatch
```

Run the released split, random folds, spatial-block folds, and
leave-one-country-out:

```bash
sbatch scripts/slurm/eval_splits.sbatch
```

Run the urban/rural baseline and settlement-stratified AUROC:

```bash
sbatch scripts/slurm/eval_baselines.sbatch
```

Run probability calibration:

```bash
sbatch scripts/slurm/eval_calibration.sbatch
```

Run LGA burden uncertainty:

```bash
python scripts/burden_uncertainty.py --task pw
python scripts/burden_uncertainty.py --task sw
```

Run country inference from a saved k-NN classifier:

```bash
sbatch scripts/slurm/dinov2_infer.sbatch
```

## Appendix Artifacts

The appendix experiments are part of the reproducible artifact set:

- `outputs/tables/split_scheme_auroc.csv`: original, random, spatial-block, and
  leave-one-country-out AUROC.
- `outputs/tables/baseline_stratified_auroc.csv`: urban/rural baseline and
  within-settlement AUROC.
- `outputs/tables/calibration_summary.csv` and
  `outputs/tables/calibration_reliability.csv`: Brier, ECE, AUROC, and
  reliability data.
- `outputs/tables/burden_uncertainty.csv`: LGA burden point estimates and
  bootstrap intervals.
- `outputs/tables/equal_k_auroc.csv`: matched-k DINO/DINOv2 comparison.
- `outputs/tables/galileo_norm_test_auroc.csv`: Galileo input-scale diagnostic.
- `outputs/figures/spatial_blocks_cv.png`: spatial-block 5-fold assignment.

Regenerate the spatial-block figure and its block assignment table:

```bash
python scripts/analysis/plot_spatial_blocks.py
```

## k-NN Prediction

The classifier stores training embeddings and labels. For a new image it:

1. extracts an encoder embedding;
2. finds the nearest training embeddings by cosine similarity;
3. softmax-weights the top `k` neighbors;
4. predicts the class with the larger weighted probability.

For binary access tasks, `prob_positive` / `prob_pos_k*` is `P(access)`.

## Adding A Model

Add `src/models/<name>.py` with a `ModelAdapter`, register it in
`src/models/__init__.py`, then run:

```bash
python -m sdg6.cli --model <name> ...
```
