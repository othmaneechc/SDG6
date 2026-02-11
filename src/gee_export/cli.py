"""CLI entrypoint for Google Earth Engine exports."""

from __future__ import annotations

import argparse
from pathlib import Path

from gee_export.config import ExporterConfig, merge_config
from gee_export.datasets import DATASETS
from gee_export.exporter import run_export


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export imagery from Google Earth Engine.")
    parser.add_argument("--config", type=Path, default=None, help="YAML/JSON config file with defaults.")
    parser.add_argument("--service-account", type=str, dest="service_account", help="Service account email.")
    parser.add_argument("--key-path", type=Path, dest="key_path", help="Path to service account key JSON.")
    parser.add_argument("--dataset", type=str, choices=sorted(DATASETS.keys()), help="Dataset name to export.")
    parser.add_argument("--coords-csv", type=Path, dest="coords_csv", default=None, help="CSV containing lon,lat rows.")
    parser.add_argument("--output-dir", type=Path, dest="output_dir", default=None, help="Directory for downloads.")
    parser.add_argument("--start-date", type=str, dest="start_date", default=None, help="YYYY-MM-DD start date.")
    parser.add_argument("--end-date", type=str, dest="end_date", default=None, help="YYYY-MM-DD end date.")
    parser.add_argument("--height", type=int, default=None, help="Tile height in pixels.")
    parser.add_argument("--width", type=int, default=None, help="Tile width in pixels.")
    parser.add_argument("--band", type=str, default=None, help="Band alias to visualize (e.g., RGB, NIR).")
    parser.add_argument("--sharpened", action="store_true", default=None, help="Pan-sharpen Landsat composites when available.")
    parser.add_argument("--parallel-workers", type=int, dest="parallel_workers", default=None, help="Thread count for point exports.")
    parser.add_argument("--redownload", action="store_true", default=None, help="Force re-download even if files exist.")
    parser.add_argument("--country", type=str, default=None, help="Country name for region modes (default: Morocco).")
    parser.add_argument("--mode", type=str, choices=["point", "region-tiles", "region-summary", "soilgrids"], default=None, help="Override auto-selected mode.")
    parser.add_argument("--tile-nx", type=int, dest="tile_nx", default=None, help="Number of columns for region tiling.")
    parser.add_argument("--tile-ny", type=int, dest="tile_ny", default=None, help="Number of rows for region tiling.")
    parser.add_argument("--crs", type=str, default=None, help="CRS for exports (default: EPSG:3857).")
    parser.add_argument("--opt-url", type=str, dest="opt_url", default=None, help="Custom Earth Engine endpoint.")
    return parser


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, remaining = pre.parse_known_args()

    parser = build_parser()
    if pre_args.config and pre_args.config.exists():
        try:
            from gee_export.config import load_config_dict

            cfg_defaults = load_config_dict(pre_args.config)
            if isinstance(cfg_defaults, dict):
                parser.set_defaults(**cfg_defaults)
        except Exception:
            pass

    args = parser.parse_args(remaining)
    cli_kwargs = vars(args)
    cfg = merge_config(pre_args.config, cli_kwargs)
    run_export(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
