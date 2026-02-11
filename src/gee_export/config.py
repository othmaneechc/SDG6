"""Config helpers for the GEE exporter."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from gee_export.datasets import DEFAULT_OPT_URL, REGION_SUMMARY_DATASETS, SPLIT_TILE_DATASETS


@dataclass(frozen=True)
class ExporterConfig:
    service_account: str
    key_path: Path
    dataset: str
    coords_csv: Path | None = None
    output_dir: Path = Path("data/gee_exports")
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    height: int = 512
    width: int = 512
    band: str = "RGB"
    sharpened: bool = False
    parallel_workers: int = 8
    redownload: bool = True
    country: str = "Morocco"
    mode: str | None = None
    tile_nx: int = 5
    tile_ny: int = 4
    crs: str = "EPSG:3857"
    opt_url: str = DEFAULT_OPT_URL

    def infer_mode(self) -> str:
        if self.mode:
            return self.mode
        if self.dataset == "soilgrids":
            return "soilgrids"
        if self.dataset in SPLIT_TILE_DATASETS:
            return "region-tiles"
        if self.dataset in REGION_SUMMARY_DATASETS:
            return "region-summary"
        return "point"

    def resolved(self, base_dir: Path | None = None) -> "ExporterConfig":
        base = base_dir or Path.cwd()
        return replace(
            self,
            key_path=_resolve_path(self.key_path, base),
            coords_csv=_resolve_path(self.coords_csv, base) if self.coords_csv else None,
            output_dir=_resolve_path(self.output_dir, base),
        )

    def with_overrides(self, overrides: Mapping[str, Any]) -> "ExporterConfig":
        data = {k: getattr(self, k) for k in self.__dataclass_fields__}
        for key, val in overrides.items():
            if val is None:
                continue
            if key in {"key_path", "coords_csv", "output_dir"} and val:
                data[key] = Path(val)
            else:
                data[key] = val
        return ExporterConfig(**data)


def _resolve_path(path_like: Path | str, base: Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (base / p).resolve()


def load_config_dict(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    if path.suffix.lower() == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported config format: {path}")


def merge_config(config_path: Path | None, cli_kwargs: dict[str, Any]) -> ExporterConfig:
    cfg_dict = load_config_dict(config_path)
    base_dir = config_path.parent if config_path else Path.cwd()
    if not cfg_dict:
        raise ValueError("Config file is empty or missing required keys.") if config_path else None
    merged = {**cfg_dict, **{k: v for k, v in cli_kwargs.items() if v is not None}}
    required = {"service_account", "key_path", "dataset"}
    missing = [k for k in required if k not in merged]
    if missing:
        raise ValueError(f"Missing required config fields: {', '.join(missing)}")
    return ExporterConfig(**merged).resolved(base_dir)
