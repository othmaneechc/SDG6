# Analysis Scripts

These scripts regenerate paper and appendix figures, tables, and reports from
repo-relative inputs.

## Figures

- `figures.py`: main paper figure workflows.
- `plot_nigeria_access_hotspots.py`: Nigeria burden hotspot maps.
- `plot_no_survey_population_cdf.py`: no-survey population coverage CDF.
- `plot_spatial_blocks.py`: spatial-block 5-fold assignment figure and table.

## Tables And Reports

- `compute_dino_family_auroc.py`: DINO-family AUROC table and figure.
- `stats.py`: summary statistics used in the manuscript.
- `urban_rural_split_analysis.py`: settlement split summaries.
- `count_unique_countries.py`: survey country counts.

## Paths

Defaults are repo-relative. Optional overrides:

- `SDG6_DATA_ROOT` defaults to `data/`.
- `SDG6_RUNS_ROOT` defaults to `runs/`.

Generated files go to:

- `outputs/figures/`
- `outputs/tables/`
- `outputs/reports/`
