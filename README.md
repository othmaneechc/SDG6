# SDG6 Tracker

Code and reproducible experiment artifacts for:

**Seeing SDG 6 from space: local-scale monitoring of piped water and sewage
system access across Africa using satellite imagery and self-supervised
learning**

Paper: https://doi.org/10.1016/j.wroa.2026.100600

Arxiv: https://arxiv.org/abs/2411.19093

The framework learns self-supervised representations of Sentinel-2 imagery,
classifies area-level piped-water and sewage-system infrastructure presence with
a k-nearest-neighbor classifier, and combines the predictions with gridded
population data to produce population-weighted national estimates and
subnational burden and severity maps.

## Layout

- `src/models/`: encoder adapters (DINO, DINOv2, DINOv3, Prithvi, Galileo).
- `src/sdg6/`: datasets, embedding extraction, k-NN, inference, and split helpers.
- `src/gee_export/`: Google Earth Engine export of imagery and population tiles.
- `scripts/`: command-line experiments (see the pipeline below).
- `scripts/configs/`: YAML configs for training, evaluation, export, and inference.
- `scripts/slurm/`: cluster launchers for the GPU/IO jobs.
- `scripts/analysis/`: figure and table scripts.
- `outputs/`: committed figures, tables, reports, and fitted calibrators.

## Setup

```bash
uv sync
```

Python 3.10/3.11 with PyTorch. The embedding-extraction and inference jobs are
intended for a GPU node; the evaluation and analysis scripts run on CPU.

## Data and weights

Published data and weights are external:

- DINO/DINOv2 weights, inference results, and population patches:
  https://zenodo.org/records/19156085
- Afrobarometer imagery tiles: https://zenodo.org/records/14740420
- Galileo weights: https://huggingface.co/nasaharvest/galileo

```bash
bash scripts/download/download_all_login.sh   # download released archives
sbatch scripts/slurm/extract_data.sbatch       # extract them
```

## Pipeline

Run these once, in order. Each step's outputs feed the next.

```bash
# 1. Build the manifest (one row per image: labels, split, coords, settlement type)
python scripts/build_manifest.py --base-dir DATABASE/RAW --out data/manifest_sentinel.csv

# 2. Reverse-geocode each location to its country (adds `country_name`, used by the
#    region-out split and the population-density baseline)
python scripts/geocode_countries.py

# 3. Extract DINOv2 embeddings once (reused by every downstream experiment)
sbatch scripts/slurm/extract_embeddings.sbatch
```

## Reproducing the paper

Every result below is regenerated from the DINOv2 embeddings at `k = 200`.
Committed outputs live in `outputs/`.

| Paper element | Command | Output |
| --- | --- | --- |
| Table 1 — held-out metrics by encoder | `sbatch scripts/slurm/eval_full_metrics.sbatch`; `python scripts/analysis/compute_dino_family_auroc.py` | `outputs/tables/dino_family_auroc_pws_sws.csv` |
| Matched-k DINO/DINOv2 comparison | `sbatch scripts/slurm/eval_equalk.sbatch` | `outputs/tables/equal_k_auroc.csv` |
| Galileo input-scale diagnostic | `sbatch scripts/slurm/galileo_norm_test.sbatch`; `sbatch scripts/slurm/galileo_bestk.sbatch` | `outputs/tables/galileo_norm_test_auroc.csv` |
| Split schemes (original / random / leave-one-region-out) | `sbatch scripts/slurm/eval_splits.sbatch` | `outputs/tables/split_scheme_auroc.csv` |
| Region-fold map | `python scripts/analysis/plot_region_folds.py` | `outputs/figures/region_folds_cv.png`, `outputs/tables/region_folds_cv.csv` |
| Confound baselines (urban/rural, density, combined) + within-strata | `sbatch scripts/slurm/eval_baselines.sbatch` | `outputs/tables/baseline_stratified_auroc.csv` |
| Probability calibration (Brier, ECE, reliability) | `sbatch scripts/slurm/eval_calibration.sbatch` | `outputs/tables/calibration_summary.csv`, `outputs/tables/calibration_reliability.csv`, `outputs/tables/calibrator_{pw,sw}.npz` |
| National JMP validation + no-survey coverage | `python scripts/analysis/figures.py`; `python scripts/analysis/plot_no_survey_population_cdf.py` | `outputs/figures/*africa*`, `outputs/figures/no_survey_*` |
| Nigeria burden/severity maps + summary table | `python scripts/analysis/plot_nigeria_access_hotspots.py` | `outputs/figures/nigeria_hotspots_composite_abcd.png`, `outputs/tables/nigeria_lga_metric_summary*.{csv,tex}` |
| Nigeria LGA burden bootstrap intervals | `python scripts/burden_uncertainty.py --task pw`; `--task sw` | `outputs/tables/burden_uncertainty.csv` |

Notes:

- The split, calibration, and Nigeria burden experiments all use **isotonic**
  recalibration (`calibrator_{pw,sw}.npz`, fit on validation) so probabilities are
  consistent across the paper. `plot_nigeria_access_hotspots.py` applies it by
  default (`--calibrator isotonic`).
- The generalization metric is **leave-one-region-out** over the five UN African
  subregions (Northern, Western, Middle, Eastern, Southern); the fixed
  country-to-subregion mapping is archived in `outputs/tables/region_folds_cv.csv`.

## Country inference

```bash
sbatch scripts/slurm/dinov2_infer.sbatch   # gridded inference from a saved k-NN classifier
```

## k-NN classifier

For a new image the classifier: (1) extracts an encoder embedding; (2) finds the
nearest training embeddings by cosine similarity; (3) softmax-weights the top `k`
neighbors (temperature 0.07); (4) predicts the class with the larger weighted
probability. For the binary access tasks, `prob_positive` is `P(access)`.

## Adding a model

Add `src/models/<name>.py` with a `ModelAdapter`, register it in
`src/models/__init__.py`, then run:

```bash
python -m sdg6.cli --model <name> ...
```
