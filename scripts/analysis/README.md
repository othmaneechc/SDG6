# Analysis Scripts

These scripts were converted from the former notebook workflows:
- `figures.py` (from `figures.ipynb`)
- `stats.py` (from `stats.ipynb`)
- `urban_rural_split_analysis.py` (from `urban_rural_split_analysis.ipynb`)

Additional analysis utilities in this folder:
- `plot_nigeria_access_hotspots.py`
- `plot_no_survey_population_cdf.py`
- `compute_dino_family_auroc.py`
- `count_unique_countries.py`

Conventions:
- Use `sewage system access` wording in figure titles/labels.
- Write figures to `outputs/figures/`.
- Write tabular artifacts to `outputs/tables/` (and reports to `outputs/reports/` when applicable).
- Use portable defaults with optional env overrides:
  - `SDG6_DATA_ROOT` (defaults to `data/`)
  - `SDG6_RUNS_ROOT` (defaults to `runs/`)

Some scripts still include notebook-style `display(...)` calls and cell separators (`# %%`), but they are now plain Python modules that can be run from the command line.
