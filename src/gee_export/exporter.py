"""High-level Earth Engine export routines inspired by the UM6P gee-exporter."""

from __future__ import annotations

import csv
import io
import logging
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

import zipfile

import ee
import geopandas as gpd
import pandas as pd
import requests
from tqdm import tqdm

from gee_export.config import ExporterConfig
from gee_export.datasets import (
    DATASETS,
    LANDSAT_CLOUD_THRESHOLD,
    NE_COUNTRIES_URL,
    RAW_EXPORT_DATASETS,
    REGION_SUMMARY_DATASETS,
    SENTINEL_CLOUD_THRESHOLD,
    SPLIT_TILE_DATASETS,
)

logger = logging.getLogger(__name__)


def initialize_ee(cfg: ExporterConfig) -> None:
    # ee client expects string path for the key file
    try:
        creds = ee.ServiceAccountCredentials(cfg.service_account, str(cfg.key_path))
        ee.Initialize(creds, opt_url=cfg.opt_url)
    except Exception as exc:  # pragma: no cover - network/billing dependent
        raise RuntimeError(
            f"Failed to initialize Earth Engine. Check service_account={cfg.service_account} "
            f"and key_path={cfg.key_path}. Original error: {exc}"
        ) from exc
    logger.info("Initialized Earth Engine with %s", cfg.service_account)


def bounding_box(lat: float, lon: float, size: int, res: float) -> tuple[float, float, float, float]:
    earth_radius = 6_371_000
    ang = math.degrees(0.5 * ((size * res) / earth_radius))
    return (lon - ang, lon + ang, lat - ang, lat + ang)


def _download_bytes(url: str, retries: int = 8, backoff: float = 2.0) -> bytes:
    delay = 1.0
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=180)
            if resp.status_code == 429:
                if attempt == retries:
                    resp.raise_for_status()
                wait = delay + random.uniform(0, delay * 0.1)
                logger.warning("429 Too Many Requests (attempt %s/%s); sleeping %.1fs", attempt, retries, wait)
                time.sleep(wait)
                delay *= backoff
                continue
            resp.raise_for_status()
            content = resp.content
            _validate_geotiff_bytes(content, resp.headers.get("Content-Type"))
            return content
        except Exception as exc:  # pragma: no cover - network dependent
            last_err = exc
            wait = delay + random.uniform(0, delay * 0.1)
            logger.warning("Download failed (attempt %s/%s): %s; retrying in %.1fs", attempt, retries, exc, wait)
            if attempt == retries:
                break
            time.sleep(wait)
            delay *= backoff

    if last_err:
        raise last_err
    raise RuntimeError("Download failed with no additional context")


def _validate_geotiff_bytes(content: bytes, content_type: str | None) -> None:
    if not content:
        raise ValueError("Empty response body")

    sig = content[:4]
    if sig in (b"II*\x00", b"MM\x00*", b"PK\x03\x04"):
        return

    head = content[:256].lower()
    if b"<html" in head or b"<!doctype" in head or b"quota" in head or b"error" in head or b"forbidden" in head:
        raise ValueError("Received HTML/JSON error payload instead of GeoTIFF")

    raise ValueError(f"Unexpected content signature {sig!r} (Content-Type={content_type})")


def _write_geotiff_bytes(content: bytes, output: Path) -> None:
    """Persist GeoTIFF bytes; if zipped, extract the first .tif inside.

    EE sometimes returns a zip (PK header) even when requesting GEO_TIFF; if so,
    we extract the first .tif entry to the target path to keep downstream reads simple.
    """
    if content[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            tif_names = [n for n in zf.namelist() if n.lower().endswith(('.tif', '.tiff'))]
            if not tif_names:
                raise ValueError("Zip payload contained no TIFF files")
            name = tif_names[0]
            with zf.open(name) as src, output.open('wb') as dst:
                dst.write(src.read())
        return

    output.write_bytes(content)


def _get_info_with_retry(obj, retries: int = 8, backoff: float = 2.0):
    delay = 1.0
    for attempt in range(1, retries + 1):
        try:
            return obj.getInfo()
        except Exception as exc:  # pragma: no cover - network dependent
            if attempt == retries:
                raise
            wait = delay + random.uniform(0, delay * 0.1)
            logger.warning("getInfo retry %s/%s after error: %s (sleep %.1fs)", attempt, retries, exc, wait)
            time.sleep(wait)
            delay *= backoff
    return None


def _get_download_url_with_retry(img, params: dict, retries: int = 6, backoff: float = 2.0):
    delay = 1.0
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return img.getDownloadUrl(params)
        except Exception as exc:  # pragma: no cover - network dependent
            last_err = exc
            if attempt == retries:
                break
            wait = delay + random.uniform(0, delay * 0.1)
            logger.warning("getDownloadUrl retry %s/%s after error: %s (sleep %.1fs)", attempt, retries, exc, wait)
            time.sleep(wait)
            delay *= backoff
    if last_err:
        raise last_err
    raise RuntimeError("getDownloadUrl failed with no additional context")


def _prepare_collection(dataset: str, geometry: ee.Geometry, start_date: str, end_date: str) -> ee.ImageCollection:
    meta = DATASETS[dataset]
    coll = ee.ImageCollection(meta["dataset"])
    select = meta.get("select")
    if select:
        coll = coll.select(select)
    coll = coll.filterDate(start_date, end_date).filterBounds(geometry)
    if dataset == "sentinel":
        coll = coll.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", SENTINEL_CLOUD_THRESHOLD))
    if dataset == "landsat":
        coll = coll.filter(ee.Filter.lte("CLOUD_COVER", LANDSAT_CLOUD_THRESHOLD))
    if dataset == "sentinel1":
        filt = meta.get("filter", {})
        if "instrumentMode" in filt:
            coll = coll.filter(ee.Filter.eq("instrumentMode", filt["instrumentMode"]))
    return coll


def _export_visual(
    dataset: str,
    geom: ee.Geometry,
    img: ee.Image,
    band: str,
    desc: str,
    height: int,
    width: int,
    output: Path,
    sharpened: bool,
) -> None:
    meta = DATASETS[dataset]
    bands_list = _get_info_with_retry(img.bandNames())
    if not bands_list:
        logger.warning("Failed to fetch band list for %s (%s); skipping", dataset, desc)
        return
    channels = meta.get(band, [])
    if not channels or not all(b in bands_list for b in channels):
        logger.info("Missing bands for %s on %s. Available: %s", dataset, desc, bands_list)
        return

    vis = img.visualize(bands=channels, min=meta.get("min"), max=meta.get("max"))
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        url = _get_download_url_with_retry(
            vis,
            {
                "description": desc,
                "region": geom,
                "crs": "EPSG:3857",
                "fileFormat": "GEO_TIFF",
                "dimensions": [height, width],
            },
        )
        try:
            data = _download_bytes(url)
        except Exception as exc:
            logger.error("Download failed for %s: %s", output.name, exc)
            raise
        _write_geotiff_bytes(data, output)
        logger.info("Downloaded %s", output.name)
    else:
        logger.info("Skipping %s (exists)", output.name)

    panc = meta.get("panchromatic")
    if sharpened and panc and panc[0] in bands_list:
        hsv = img.select(channels).rgbToHsv()
        pan_img = ee.Image.cat([hsv.select("hue"), hsv.select("saturation"), img.select(panc)]).hsvToRgb()
        pan_path = output.parent / f"sharpened_{output.name}"
        if not pan_path.exists():
            url2 = _get_download_url_with_retry(
                pan_img,
                {
                    "description": "sharpened_" + desc,
                    "region": geom,
                    "crs": "EPSG:3857",
                    "fileFormat": "GEO_TIFF",
                    "dimensions": [height, width],
                },
            )
            try:
                data = _download_bytes(url2)
            except Exception as exc:
                logger.error("Download failed for %s: %s", pan_path.name, exc)
                raise
            _write_geotiff_bytes(data, pan_path)
            logger.info("Downloaded %s", pan_path.name)
        else:
            logger.info("Skipping %s (exists)", pan_path.name)


def _export_raw(dataset: str, geom: ee.Geometry, img: ee.Image, desc: str, output: Path) -> None:
    if output.exists():
        logger.info("Skipping %s (exists)", output.name)
        return
    url = _get_download_url_with_retry(
        img,
        {
            "description": desc,
            "region": geom,
            "crs": "EPSG:3857",
            "fileFormat": "GEO_TIFF",
        },
    )
    try:
        data = _download_bytes(url)
    except Exception as exc:
        logger.error("Download failed for %s: %s", output.name, exc)
        raise
    _write_geotiff_bytes(data, output)
    logger.info("Downloaded %s", output.name)


def _load_coords(csv_path: Path) -> list[tuple[float, float]]:
    df = pd.read_csv(csv_path)
    for lon_key, lat_key in (("lon", "lat"), ("longitude", "latitude"), ("x", "y")):
        if lon_key in df.columns and lat_key in df.columns:
            return list(zip(df[lon_key].astype(float), df[lat_key].astype(float)))
    if df.shape[1] >= 2:
        return list(zip(df.iloc[:, 0].astype(float), df.iloc[:, 1].astype(float)))
    with csv_path.open() as cf:
        reader = csv.reader(cf, quoting=csv.QUOTE_NONNUMERIC)
        next(reader, None)
        return [(float(r[0]), float(r[1])) for r in reader if len(r) >= 2]


def export_points(cfg: ExporterConfig) -> None:
    if not cfg.coords_csv:
        raise ValueError("coords_csv must be provided for point exports.")
    coords = _load_coords(cfg.coords_csv)
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    existing: set[tuple[float, float]] = set()
    if not cfg.redownload:
        existing = {
            tuple(map(float, f.stem.split("_")[-2:]))
            for f in out_dir.iterdir()
            if f.suffix.lower() in {".tif", ".zip"}
        }
        coords = [c for c in coords if (c[0], c[1]) not in existing]

    logger.info("Running dataset=%s on %s (%d coords remaining)", cfg.dataset, cfg.coords_csv, len(coords))
    meta = DATASETS[cfg.dataset]

    def _worker(coord: tuple[float, float]) -> None:
        lon, lat = coord
        res = 20 if cfg.dataset == "sentinel" and cfg.band not in {"RGB", "NIR", "IR"} else meta.get("resolution", 30)
        minx, maxx, miny, maxy = bounding_box(lat, lon, cfg.height, res)
        geom = ee.Geometry.Rectangle([[minx, miny], [maxx, maxy]])
        coll = _prepare_collection(cfg.dataset, geom, cfg.start_date, cfg.end_date)
        if cfg.dataset in RAW_EXPORT_DATASETS:
            img = coll.mean().clip(geom)
            target = out_dir / f"{cfg.dataset}_{lat:.5f}_{lon:.5f}.zip"
            if not cfg.redownload and target.exists():
                logger.info("Skipping existing file %s", target)
                return
            _export_raw(cfg.dataset, geom, img, target.stem, target)
            return
        img = coll.median().clip(geom)
        target = out_dir / f"{cfg.dataset}_{lat:.5f}_{lon:.5f}.tif"
        if not cfg.redownload and target.exists():
            logger.info("Skipping existing file %s", target)
            return
        _export_visual(cfg.dataset, geom, img, cfg.band, target.stem, cfg.height, cfg.width, target, cfg.sharpened)

    if cfg.parallel_workers > 1:
        with ThreadPoolExecutor(max_workers=cfg.parallel_workers) as ex:
            futures = [ex.submit(_worker, c) for c in coords]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Point exports"):
                fut.result()
    else:
        for c in tqdm(coords, desc="Point exports"):
            _worker(c)
            time.sleep(1)

    logger.info("Completed point exports to %s", out_dir)


def _country_bounds(country: str) -> tuple[float, float, float, float]:
    fc = gpd.read_file(NE_COUNTRIES_URL)
    region = fc[fc.ADMIN == country].to_crs("EPSG:4326")
    if region.empty:
        raise ValueError(f"Country '{country}' not found in Natural Earth dataset")
    minx, miny, maxx, maxy = region.total_bounds
    return minx, miny, maxx, maxy


def export_soilgrids(cfg: ExporterConfig) -> None:
    minx, miny, maxx, maxy = _country_bounds(cfg.country)
    bbox = [minx, miny, maxx, maxy]
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for prop in tqdm(DATASETS["soilgrids"]["layers"], desc="SoilGrids layers"):
        target = out_dir / f"{prop}.zip"
        if target.exists() and not cfg.redownload:
            logger.info("Skipping %s (exists)", target.name)
            continue
        img = ee.Image(f"projects/soilgrids-isric/{prop}")
        url = img.getDownloadUrl(
            {
                "region": bbox,
                "crs": DATASETS["soilgrids"].get("crs", "EPSG:3857"),
                "scale": DATASETS["soilgrids"].get("scale", 9300),
                "fileFormat": "GEO_TIFF",
            }
        )
        target.write_bytes(_download_bytes(url))
        logger.info("Downloaded %s", target.name)


def export_region_tiles(cfg: ExporterConfig) -> None:
    minx, miny, maxx, maxy = _country_bounds(cfg.country)
    nx, ny = cfg.tile_nx, cfg.tile_ny
    dx = (maxx - minx) / nx
    dy = (maxy - miny) / ny
    xs = [minx + i * dx for i in range(nx + 1)]
    ys = [miny + j * dy for j in range(ny + 1)]

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    coll_full = _prepare_collection(cfg.dataset, ee.Geometry.Rectangle([minx, miny, maxx, maxy]), cfg.start_date, cfg.end_date)

    metrics = [
        ("sum", lambda sub: sub.sum()),
        ("mean", lambda sub: sub.mean()),
        ("min", lambda sub: sub.min()),
        ("max", lambda sub: sub.max()),
        ("stdDev", lambda sub: sub.reduce(ee.Reducer.stdDev())),
        ("p10", lambda sub: sub.reduce(ee.Reducer.percentile([10]))),
        ("p50", lambda sub: sub.reduce(ee.Reducer.percentile([50]))),
        ("p90", lambda sub: sub.reduce(ee.Reducer.percentile([90]))),
    ]

    year = cfg.start_date[:4]
    tiles = [(r, c) for r in range(ny) for c in range(nx)]
    for metric_name, fn in tqdm(metrics, desc=f"{cfg.dataset} metrics"):
        for idx, (row, col) in enumerate(tiles, start=1):
            tile = [xs[col], ys[row], xs[col + 1], ys[row + 1]]
            geom = ee.Geometry.Rectangle(tile)
            sub = coll_full.filterBounds(geom)
            img = fn(sub).clip(geom)
            desc = f"{cfg.dataset}_{metric_name}_{year}_T{idx}"
            target = out_dir / f"{desc}.zip"
            if target.exists() and not cfg.redownload:
                logger.info("Skipping %s (exists)", target.name)
                continue
            url = _get_download_url_with_retry(
                img,
                {
                    "description": desc,
                    "region": tile,
                    "crs": "EPSG:3857",
                    "scale": DATASETS[cfg.dataset].get("resolution", 30),
                    "fileFormat": "GEO_TIFF",
                },
            )
            target.write_bytes(_download_bytes(url))
            logger.info("Downloaded %s", target.name)



def export_region_summary(cfg: ExporterConfig) -> None:
    minx, miny, maxx, maxy = _country_bounds(cfg.country)
    bbox = [minx, miny, maxx, maxy]
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    year = cfg.start_date[:4]
    dataset = cfg.dataset

    if dataset == "chirps":
        coll = _prepare_collection(dataset, ee.Geometry.Rectangle(bbox), cfg.start_date, cfg.end_date)
        total = coll.sum().rename("precip_total")
        mean = coll.mean().rename("precip_mean")
        mx = coll.max().rename("precip_max")
        std = coll.reduce(ee.Reducer.stdDev()).rename("precip_stdDev")
        pct = coll.reduce(ee.Reducer.percentile([10, 50, 90])).rename(["precip_p10", "precip_p50", "precip_p90"])
        rainy = coll.map(lambda i: i.gt(1).rename("rainy_day")).sum().rename("rainy_days")
        summary = ee.Image.cat([total, mean, mx, std, pct, rainy])
    elif dataset == "era5":
        coll = _prepare_collection(dataset, ee.Geometry.Rectangle(bbox), cfg.start_date, cfg.end_date)
        mean_t = coll.mean().rename("temp_mean")
        max_t = coll.max().rename("temp_max")
        std_t = coll.reduce(ee.Reducer.stdDev()).rename("temp_stdDev")
        pct_t = coll.reduce(ee.Reducer.percentile([10, 50, 90])).rename(["temp_p10", "temp_p50", "temp_p90"])
        frost = coll.map(lambda i: i.lt(273.15).rename("frost_day")).sum().rename("frost_days")
        gdd = coll.map(lambda i: i.subtract(283.15).max(0).rename("gdd_day")).sum().rename("gdd")
        summary = ee.Image.cat([mean_t, max_t, std_t, pct_t, frost, gdd])
    elif dataset == "terraclimate":
        vars = [
            "aet",
            "def",
            "pdsi",
            "pet",
            "pr",
            "ro",
            "soil",
            "srad",
            "swe",
            "tmmn",
            "tmmx",
            "vap",
            "vpd",
            "vs",
        ]
        coll = _prepare_collection(dataset, ee.Geometry.Rectangle(bbox), cfg.start_date, cfg.end_date)
        stats = []
        for v in vars:
            img = coll.select(v)
            stats += [
                img.mean().rename(f"{v}_mean"),
                img.min().rename(f"{v}_min"),
                img.max().rename(f"{v}_max"),
                img.reduce(ee.Reducer.stdDev()).rename(f"{v}_stdDev"),
            ]
        summary = ee.Image.cat(stats)
    elif dataset == "gpw_population":
        summary = _prepare_collection(dataset, ee.Geometry.Rectangle(bbox), cfg.start_date, cfg.end_date).mean().rename("pop_density")
    else:
        raise ValueError(f"Dataset {dataset} is not supported for region-summary mode")

    for band in tqdm(summary.bandNames().getInfo(), desc=f"{dataset} summary bands"):
        img = summary.select(band).clip(ee.Geometry.Rectangle(bbox))
        desc = f"{dataset}_{band}_{year}"
        target = out_dir / f"{desc}.zip"
        if target.exists() and not cfg.redownload:
            logger.info("Skipping %s (exists)", target.name)
            continue
        url = _get_download_url_with_retry(
            img,
            {
                "description": desc,
                "region": bbox,
                "crs": "EPSG:3857",
                "scale": DATASETS[dataset].get("resolution", 30),
                "fileFormat": "GEO_TIFF",
            },
        )
        target.write_bytes(_download_bytes(url))
        logger.info("Downloaded %s", target.name)


def run_export(cfg: ExporterConfig) -> None:
    cfg = cfg.resolved()
    cfg_mode = cfg.infer_mode()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(cfg.output_dir / f"{cfg.dataset}.log"),
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s-%(levelname)s-%(message)s",
    )
    logger.info("Starting export mode=%s dataset=%s", cfg_mode, cfg.dataset)

    initialize_ee(cfg)

    if cfg_mode == "soilgrids":
        export_soilgrids(cfg)
    elif cfg_mode == "region-tiles":
        export_region_tiles(cfg)
    elif cfg_mode == "region-summary":
        export_region_summary(cfg)
    elif cfg_mode == "point":
        export_points(cfg)
    else:
        raise ValueError(f"Unknown export mode: {cfg_mode}")

    logger.info("Finished export to %s", cfg.output_dir)
