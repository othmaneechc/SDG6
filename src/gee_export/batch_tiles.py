#!/usr/bin/env python3
"""Batch-download Sentinel tiles per country using centroid CSVs.

Reads a config (YAML/JSON) describing service account creds, the countries
root (folders containing `*_tiles.csv` with centroid columns), and an output
root. For each country folder it builds a lon/lat CSV and invokes the
standard `run_export` in point mode.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd
from tqdm import tqdm

from gee_export.config import ExporterConfig, load_config_dict
from gee_export.exporter import run_export


@dataclass(frozen=True)
class BatchConfig:
    service_account: str
    key_path: Path
    countries_dir: Path
    output_root: Path
    dataset: str = "sentinel"
    band: str = "RGB"
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    tile_size: int = 256
    tiles_glob: str = "*_tiles.csv"
    parallel_workers: int = 8
    redownload: bool = False
    sharpened: bool = False
    countries: list[str] | None = None
    lon_fields: Sequence[str] = field(default_factory=lambda: ("centroid_lon", "lon", "longitude", "x"))
    lat_fields: Sequence[str] = field(default_factory=lambda: ("centroid_lat", "lat", "latitude", "y"))

    def resolved(self, base: Path) -> "BatchConfig":
        def _resolve(p: Path) -> Path:
            return p if p.is_absolute() else (base / p).resolve()

        return BatchConfig(
            service_account=self.service_account,
            key_path=_resolve(self.key_path),
            countries_dir=_resolve(self.countries_dir),
            output_root=_resolve(self.output_root),
            dataset=self.dataset,
            band=self.band,
            start_date=self.start_date,
            end_date=self.end_date,
            tile_size=self.tile_size,
            tiles_glob=self.tiles_glob,
            parallel_workers=self.parallel_workers,
            redownload=self.redownload,
            sharpened=self.sharpened,
            countries=self.countries,
            lon_fields=self.lon_fields,
            lat_fields=self.lat_fields,
        )


def _merge_batch_config(config_path: Path | None, overrides: Mapping[str, object]) -> BatchConfig:
    base_dir = config_path.parent if config_path else Path.cwd()
    cfg_dict = load_config_dict(config_path) if config_path else {}
    merged: dict[str, object] = {**cfg_dict, **{k: v for k, v in overrides.items() if v is not None}}

    required = {"service_account", "key_path", "countries_dir", "output_root"}
    missing = [k for k in required if k not in merged]
    if missing:
        raise ValueError(f"Missing required config fields: {', '.join(missing)}")

    countries_arg = merged.get("countries")
    if isinstance(countries_arg, str):
        merged["countries"] = [c.strip() for c in countries_arg.split(",") if c.strip()]

    return BatchConfig(
        service_account=str(merged["service_account"]),
        key_path=Path(merged["key_path"]),
        countries_dir=Path(merged["countries_dir"]),
        output_root=Path(merged["output_root"]),
        dataset=str(merged.get("dataset", "sentinel")),
        band=str(merged.get("band", "RGB")),
        start_date=str(merged.get("start_date", "2024-01-01")),
        end_date=str(merged.get("end_date", "2024-12-31")),
        tile_size=int(merged.get("tile_size", merged.get("height", 256))),
        tiles_glob=str(merged.get("tiles_glob", "*_tiles.csv")),
        parallel_workers=int(merged.get("parallel_workers", 8)),
        redownload=bool(merged.get("redownload", False)),
        sharpened=bool(merged.get("sharpened", False)),
        countries=merged.get("countries"),
    ).resolved(base_dir)


def _choose_field(columns: Iterable[str], candidates: Sequence[str]) -> str:
    cols = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    raise ValueError(f"Required columns not found; tried: {', '.join(candidates)}")


def _pick_tiles_file(country_dir: Path, glob_pattern: str) -> Path | None:
    candidates = sorted(country_dir.glob(glob_pattern))
    if not candidates:
        return None
    preferred = [p for p in candidates if "general" in p.stem]
    return preferred[0] if preferred else candidates[0]


def _build_coords_csv(tiles_csv: Path, lon_fields: Sequence[str], lat_fields: Sequence[str], out_path: Path) -> Path:
    df = pd.read_csv(tiles_csv)
    lon_col = _choose_field(df.columns, lon_fields)
    lat_col = _choose_field(df.columns, lat_fields)
    coords = df[[lon_col, lat_col]].rename(columns={lon_col: "lon", lat_col: "lat"})
    coords = coords.dropna(subset=["lon", "lat"]).drop_duplicates()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coords.to_csv(out_path, index=False)
    return out_path


def run_batch(cfg: BatchConfig) -> None:
    if not cfg.countries_dir.exists():
        raise FileNotFoundError(f"Countries directory not found: {cfg.countries_dir}")

    country_dirs = [p for p in sorted(cfg.countries_dir.iterdir()) if p.is_dir()]
    if cfg.countries:
        allowed = {c.lower() for c in cfg.countries}
        country_dirs = [p for p in country_dirs if p.name.lower() in allowed]

    if not country_dirs:
        print("No country folders found to process.")
        return

    for country_dir in tqdm(country_dirs, desc="Countries", unit="country"):
        tiles_csv = _pick_tiles_file(country_dir, cfg.tiles_glob)
        if not tiles_csv:
            print(f"[skip] No tiles CSV in {country_dir}")
            continue

        output_dir = cfg.output_root / country_dir.name
        coords_csv = _build_coords_csv(
            tiles_csv,
            lon_fields=cfg.lon_fields,
            lat_fields=cfg.lat_fields,
            out_path=output_dir / "centroids.csv",
        )

        export_cfg = ExporterConfig(
            service_account=cfg.service_account,
            key_path=cfg.key_path,
            dataset=cfg.dataset,
            coords_csv=coords_csv,
            output_dir=output_dir,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            height=cfg.tile_size,
            width=cfg.tile_size,
            band=cfg.band,
            sharpened=cfg.sharpened,
            parallel_workers=cfg.parallel_workers,
            redownload=cfg.redownload,
            mode="point",
        )

        print(f"[run] {country_dir.name}: {tiles_csv.name} -> {output_dir}")
        run_export(export_cfg)


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, remaining = pre.parse_known_args()

    parser = argparse.ArgumentParser(description="Batch Sentinel downloads from tiles CSVs.")
    parser.add_argument("--config", type=Path, default=None, help="YAML/JSON config path.")
    parser.add_argument("--countries", type=str, default=None, help="Optional comma-separated country names to filter.")
    parser.add_argument("--countries-dir", type=Path, default=None, help="Override countries root.")
    parser.add_argument("--output-root", type=Path, default=None, help="Override output root.")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--parallel-workers", type=int, default=None)
    parser.add_argument("--redownload", action="store_true", default=None)
    parser.add_argument("--sharpened", action="store_true", default=None)
    parser.add_argument("--tiles-glob", type=str, default=None)
    parser.add_argument("--band", type=str, default=None)
    parser.add_argument("--service-account", type=str, default=None)
    parser.add_argument("--key-path", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default=None)

    if pre_args.config and pre_args.config.exists():
        cfg_defaults = load_config_dict(pre_args.config)
        if isinstance(cfg_defaults, dict):
            parser.set_defaults(**cfg_defaults)

    args = parser.parse_args(remaining)
    overrides = vars(args)
    cfg = _merge_batch_config(pre_args.config, overrides)
    run_batch(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
