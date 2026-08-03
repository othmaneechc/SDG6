#!/usr/bin/env python3
"""Plot Nigeria access hotspots for piped water and sewage predictions.

This script merges Nigeria tile-level inference outputs with tile population,
aggregates to regions, and draws a map highlighting dense regions with low
predicted access.

Region modes:
- If `--admin-boundaries` is provided, aggregate by those polygons.
- Otherwise, aggregate by a fixed lat/lon grid (county-like proxy).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shapely
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import matplotlib.patheffects as pe
from shapely.geometry import box

os.environ["SHAPE_RESTORE_SHX"] = "YES"
BURDEN_CMAP = "YlGnBu"
SEVERITY_CMAP = "magma_r"
HOTSPOT_COLOR = "#FF0000"  # pure red
PW_HOTSPOT_COLOR = HOTSPOT_COLOR
SW_HOTSPOT_COLOR = HOTSPOT_COLOR
PW_LABEL_LINE_COLOR = HOTSPOT_COLOR
HOTSPOT_OUTLINE_WIDTH = 1.6
LABELED_CLUSTER_OUTLINE_COLOR = "#111111"
LABELED_CLUSTER_OUTLINE_WIDTH = 2.8

SEVERITY_VMIN_QUANTILE = 0.05
SEVERITY_VMAX_QUANTILE = 0.98


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    data_root = Path(os.environ.get("SDG6_DATA_ROOT", repo_root / "data")).expanduser()
    admin_default = repo_root / "data" / "admin_boundaries" / "gadm41_NGA_2.json"
    fig_dir = repo_root / "outputs" / "figures"
    tab_dir = repo_root / "outputs" / "tables"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=data_root,
        help="Root directory that contains inference/ and meta_pop_data/.",
    )
    parser.add_argument(
        "--country",
        type=str,
        default="Nigeria",
        help="Country folder name under inference/ and countries_2x2/.",
    )
    parser.add_argument(
        "--pw-csv",
        type=Path,
        default=None,
        help="Piped water inference CSV. Defaults to inference/<country>/pw_predictions.csv.",
    )
    parser.add_argument(
        "--sw-csv",
        type=Path,
        default=None,
        help="Sewage inference CSV. Defaults to inference/<country>/sw_predictions.csv.",
    )
    parser.add_argument(
        "--tiles-csv",
        type=Path,
        default=None,
        help="Tile population CSV for the target country.",
    )
    parser.add_argument(
        "--country-shapefile",
        type=Path,
        default=None,
        help="Country boundaries shapefile containing Nigeria.",
    )
    parser.add_argument(
        "--admin-boundaries",
        type=Path,
        default=admin_default,
        help="Optional admin boundaries (state/county/LGA polygons) to aggregate by.",
    )
    parser.add_argument(
        "--force-grid",
        action="store_true",
        help="Force grid aggregation even when --admin-boundaries exists.",
    )
    parser.add_argument(
        "--admin-name-column",
        type=str,
        default=None,
        help="Column in --admin-boundaries used as region name.",
    )
    parser.add_argument(
        "--grid-deg",
        type=float,
        default=0.5,
        help="Grid size in degrees for fallback county-like aggregation.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="k for prediction columns (e.g. 200). Defaults to largest available.",
    )
    parser.add_argument(
        "--density-quantile",
        type=float,
        default=0.70,
        help="Quantile threshold for dense regions.",
    )
    parser.add_argument(
        "--hotspot-top-fraction",
        type=float,
        default=0.10,
        help="Fraction of LGAs selected as hotspots based on expected people without access.",
    )
    parser.add_argument(
        "--min-population",
        type=float,
        default=9000.0,
        help="Minimum population to be eligible as a hotspot.",
    )
    parser.add_argument(
        "--top-labels",
        type=int,
        default=3,
        help="Maximum number of hotspot-area labels shown on each map.",
    )
    parser.add_argument(
        "--output-figure-burden",
        type=Path,
        default=fig_dir / "nigeria_pw_sw_hotspots_burden.png",
        help="Output figure path for hotspots based on expected people without access.",
    )
    parser.add_argument(
        "--output-figure-severity",
        type=Path,
        default=fig_dir / "nigeria_pw_sw_hotspots_no_access_probability.png",
        help="Output figure path for hotspots based on no-access probability.",
    )
    parser.add_argument(
        "--output-figure-composite",
        type=Path,
        default=fig_dir / "nigeria_hotspots_composite_abcd.png",
        help="Output figure path for the 2x2 composite hotspot map with panel labels.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=tab_dir / "nigeria_access_hotspots.csv",
        help="Output CSV with region-level metrics.",
    )
    parser.add_argument(
        "--output-clusters-csv",
        type=Path,
        default=tab_dir / "nigeria_access_hotspot_clusters.csv",
        help="Output CSV with hotspot-cluster summaries.",
    )
    parser.add_argument(
        "--output-summary-csv",
        type=Path,
        default=tab_dir / "nigeria_lga_metric_summary.csv",
        help="Output CSV with LGA-level distribution summaries for key burden and severity metrics.",
    )
    parser.add_argument(
        "--output-summary-tex",
        type=Path,
        default=tab_dir / "nigeria_lga_metric_summary_table.tex",
        help="Output LaTeX table with LGA-level inequality and dispersion metrics.",
    )
    parser.add_argument(
        "--cluster-min-shared-boundary-km",
        type=float,
        default=0.0,
        help="Minimum shared boundary length in km for merging two hotspot LGAs into one contiguous cluster; use 0 to merge any touching/intersecting hotspot LGAs.",
    )
    parser.add_argument(
        "--cluster-min-shared-boundary-fraction",
        type=float,
        default=0.05,
        help="Minimum shared-boundary fraction for merging two hotspot LGAs into one cluster, computed as shared boundary length divided by the smaller LGA perimeter; use 0 to disable this criterion.",
    )
    parser.add_argument(
        "--label-cluster-distance-km",
        type=float,
        default=70.0,
        help="Distance threshold in km for grouping nearby top-decile hotspot LGAs into labeled hotspot concentrations for visualization.",
    )
    parser.add_argument(
        "--label-cluster-min-size",
        type=int,
        default=3,
        help="Minimum number of hotspot LGAs required for a labeled hotspot concentration.",
    )
    parser.add_argument(
        "--label-component-min-size",
        type=int,
        default=3,
        help="Minimum number of boundary-contiguous LGAs required for a component to be kept inside a labeled hotspot concentration.",
    )
    parser.add_argument(
        "--output-population-map",
        type=Path,
        default=fig_dir / "nigeria_lga_population_map.png",
        help="Output figure path for the LGA-level population map.",
    )
    parser.add_argument(
        "--texture-alpha",
        type=float,
        default=0.34,
        help="Alpha for the green satellite-like texture background.",
    )

    args = parser.parse_args()
    args.data_root = Path(args.data_root).expanduser()
    country = str(args.country)
    if args.pw_csv is None:
        args.pw_csv = args.data_root / "inference" / country / "pw_predictions.csv"
    if args.sw_csv is None:
        args.sw_csv = args.data_root / "inference" / country / "sw_predictions.csv"
    if args.tiles_csv is None:
        if country.lower() == "nigeria":
            tile_name = "nga_general_2020_tiles.csv"
        else:
            tile_name = f"{country.lower()}_general_2020_tiles.csv"
        args.tiles_csv = args.data_root / "meta_pop_data" / "countries_2x2" / country / tile_name
    if args.country_shapefile is None:
        args.country_shapefile = args.data_root / "meta_pop_data" / "natural_earth_data" / "ne_110m_admin_0_countries.shp"
    return args
# --- POPULATION MAP ---
def plot_population_map(
    country_geom: gpd.GeoDataFrame,
    regions: gpd.GeoDataFrame,
    output_path: Path,
    texture_alpha: float,
) -> None:
    population = pd.to_numeric(regions["population"], errors="coerce")
    vmax = float(population.quantile(0.98)) if population.notna().any() else 1.0
    vmax = max(vmax, 1.0)

    fig, ax = plt.subplots(figsize=(13.0, 12.2), constrained_layout=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.10)

    _add_country_texture(ax, country_geom, texture_alpha)
    regions.plot(
        column="population",
        ax=ax,
        cmap="YlGnBu",
        norm=Normalize(vmin=0.0, vmax=vmax),
        edgecolor="#d7dde2",
        linewidth=0.8,
        alpha=0.88,
        legend=False,
        zorder=2,
    )
    country_geom.boundary.plot(ax=ax, color="#1f1f1f", linewidth=1, zorder=3)

    ax.set_title("Nigeria LGA-level population", fontsize=22, pad=12)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_frame_on(False)

    shape = country_geom.geometry.union_all()
    minx, miny, maxx, maxy = shape.bounds
    width = maxx - minx
    height = maxy - miny
    ax.set_xlim(minx - 0.05 * width, maxx + 0.05 * width)
    ax.set_ylim(miny - 0.03 * height, maxy + 0.03 * height)

    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=vmax), cmap="YlGnBu")
    sm.set_array([])
    cax = fig.add_axes([0.20, 0.055, 0.60, 0.018])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("LGA population from aggregated 2020 tile population", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    stats_text = (
        f"LGAs: {population.notna().sum():,}\n"
        f"Median: {population.median():,.0f}\n"
        f"Mean: {population.mean():,.0f}\n"
        f"90th pct: {population.quantile(0.90):,.0f}\n"
        f"Max: {population.max():,.0f}"
    )
    ax.text(
        0.985,
        0.02,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=13,
        color="#2f2f2f",
        bbox={"facecolor": "white", "edgecolor": "#d7d7d7", "alpha": 0.92, "pad": 6},
        zorder=10,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def _extract_lon_lat(path_text: str) -> tuple[float | None, float | None]:
    patterns = (
        r"sentinel_image_([-+]?\d+(?:\.\d+)?)_([-+]?\d+(?:\.\d+)?)",
        r"sentinel_([-+]?\d+(?:\.\d+)?)_([-+]?\d+(?:\.\d+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, path_text)
        if match:
            lat = float(match.group(1))
            lon = float(match.group(2))
            return lon, lat
    return None, None


def _select_k(df: pd.DataFrame, col_prefix: str, k: int | None) -> str:
    candidates = [c for c in df.columns if c.startswith(col_prefix)]
    if not candidates:
        raise ValueError(f"No columns found with prefix '{col_prefix}'")

    def parse_k(col: str) -> int:
        return int(col.replace(col_prefix, ""))

    if k is None:
        return max(candidates, key=parse_k)
    target = f"{col_prefix}{k}"
    if target not in candidates:
        raise ValueError(f"Requested k={k} not found in columns {sorted(candidates)}")
    return target


def _as_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _prediction_to_access_prob(pred_class: pd.Series, pred_prob: pd.Series) -> pd.Series:
    labels = pred_class.astype(str).str.lower()
    probs = pd.to_numeric(pred_prob, errors="coerce").clip(lower=0.0, upper=1.0)
    no_access_mask = labels.str.contains("no_") | labels.str.startswith("no")
    return pd.Series(np.where(no_access_mask, 1.0 - probs, probs), index=pred_class.index)


def load_country_geometry(country_shapefile: Path, country: str) -> gpd.GeoDataFrame:
    countries = gpd.read_file(country_shapefile)
    if countries.crs is None:
        countries = countries.set_crs("EPSG:4326")
    countries = countries.to_crs("EPSG:4326")

    candidates = []
    if "NAME" in countries.columns:
        candidates.append(countries["NAME"].astype(str).str.lower() == country.lower())
    if "ADMIN" in countries.columns:
        candidates.append(countries["ADMIN"].astype(str).str.lower() == country.lower())
    if "SOVEREIGNT" in countries.columns:
        candidates.append(countries["SOVEREIGNT"].astype(str).str.lower() == country.lower())
    if "ISO_A3" in countries.columns and country.lower() == "nigeria":
        candidates.append(countries["ISO_A3"].astype(str).str.upper() == "NGA")
    if "ADM0_A3" in countries.columns and country.lower() == "nigeria":
        candidates.append(countries["ADM0_A3"].astype(str).str.upper() == "NGA")

    if not candidates:
        raise ValueError(f"Could not find country-identifying columns in {country_shapefile}")

    mask = candidates[0]
    for extra in candidates[1:]:
        mask = mask | extra

    subset = countries.loc[mask, ["geometry"]].copy()
    if subset.empty:
        raise ValueError(f"Country '{country}' not found in {country_shapefile}")

    subset["country"] = country
    subset = subset.dissolve(by="country").reset_index()
    return subset[["country", "geometry"]]


def load_predictions(pred_csv: Path, service_name: str, k: int | None) -> pd.DataFrame:
    df = pd.read_csv(pred_csv)
    pred_col = _select_k(df, "pred_class_k", k)
    prob_col = _select_k(df, "prob_k", k)
    coords = df["path"].astype(str).map(_extract_lon_lat)
    df["lon"] = [c[0] for c in coords]
    df["lat"] = [c[1] for c in coords]
    df = _as_numeric(df, ["lon", "lat", prob_col]).dropna(subset=["lon", "lat", prob_col]).copy()
    df["lon_round"] = df["lon"].round(5)
    df["lat_round"] = df["lat"].round(5)
    df[f"{service_name}_access_prob"] = _prediction_to_access_prob(df[pred_col], df[prob_col])
    return df[["lon_round", "lat_round", f"{service_name}_access_prob"]]


def build_tile_frame(args: argparse.Namespace) -> gpd.GeoDataFrame:
    tiles = pd.read_csv(args.tiles_csv)
    required_cols = {"centroid_lon", "centroid_lat", "total_population"}
    missing = required_cols - set(tiles.columns)
    if missing:
        raise ValueError(f"Missing required columns in tiles CSV: {sorted(missing)}")

    tiles = _as_numeric(tiles, ["centroid_lon", "centroid_lat", "total_population"]).dropna(
        subset=["centroid_lon", "centroid_lat", "total_population"]
    )
    tiles["lon_round"] = tiles["centroid_lon"].round(5)
    tiles["lat_round"] = tiles["centroid_lat"].round(5)
    tiles = tiles.rename(
        columns={
            "centroid_lon": "lon",
            "centroid_lat": "lat",
            "total_population": "population",
        }
    )

    pw = load_predictions(args.pw_csv, "pw", args.k)
    sw = load_predictions(args.sw_csv, "sw", args.k)

    merged = tiles.merge(pw, on=["lon_round", "lat_round"], how="left").merge(
        sw, on=["lon_round", "lat_round"], how="left"
    )
    merged = merged.dropna(subset=["pw_access_prob", "sw_access_prob"]).copy()
    merged["population"] = merged["population"].clip(lower=0.0)
    merged["either_access_prob"] = np.maximum(merged["pw_access_prob"], merged["sw_access_prob"])

    geom = gpd.points_from_xy(merged["lon"], merged["lat"])
    return gpd.GeoDataFrame(merged, geometry=geom, crs="EPSG:4326")


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce")
    wts = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    valid = vals.notna() & wts.notna()
    vals = vals[valid].to_numpy(dtype=float)
    wts = wts[valid].to_numpy(dtype=float)
    if len(vals) == 0:
        return float("nan")
    wsum = wts.sum()
    if wsum <= 0:
        return float(np.mean(vals))
    return float(np.average(vals, weights=wts))


def _summarize_group_frame(group_df: pd.DataFrame) -> dict[str, float | int]:
    return {
        "tiles": int(len(group_df)),
        "population": float(group_df["population"].sum()),
        "pw_access_prob": _weighted_mean(group_df["pw_access_prob"], group_df["population"]),
        "sw_access_prob": _weighted_mean(group_df["sw_access_prob"], group_df["population"]),
        "either_access_prob": _weighted_mean(group_df["either_access_prob"], group_df["population"]),
    }


def _finalize_region_metrics(regions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    regions = regions.copy()
    if regions.crs is None:
        regions = regions.set_crs("EPSG:4326")
    regions = regions.to_crs("EPSG:4326")

    area_km2 = regions.to_crs("EPSG:6933").geometry.area / 1_000_000.0
    regions["area_km2"] = area_km2
    regions["population_density_km2"] = regions["population"] / regions["area_km2"].replace(0.0, np.nan)
    return regions


def _choose_name_column(boundaries: gpd.GeoDataFrame, explicit_col: str | None) -> str:
    if explicit_col:
        if explicit_col not in boundaries.columns:
            raise ValueError(f"--admin-name-column '{explicit_col}' not found in boundaries file.")
        return explicit_col

    preferred = (
        "NAME_2",
        "ADM2_EN",
        "shapeName",
        "NAME_1",
        "ADM1_EN",
        "NAME",
        "name",
    )
    for col in preferred:
        if col in boundaries.columns:
            return col
    return boundaries.columns[0]


def aggregate_with_admin_boundaries(
    points: gpd.GeoDataFrame,
    admin_boundaries_path: Path,
    admin_name_column: str | None,
    country_geom: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, str]:
    boundaries = gpd.read_file(admin_boundaries_path)
    if boundaries.crs is None:
        boundaries = boundaries.set_crs("EPSG:4326")
    boundaries = boundaries.to_crs("EPSG:4326")
    boundaries = boundaries[boundaries.geometry.notna()].copy()

    # Ensure regions are inside Nigeria even when a global boundary file is used.
    nigeria_only = gpd.overlay(boundaries, country_geom[["geometry"]], how="intersection")
    if nigeria_only.empty:
        raise ValueError("No admin polygons intersect Nigeria in --admin-boundaries.")

    name_col = _choose_name_column(nigeria_only, admin_name_column)
    nigeria_only["region_name"] = nigeria_only[name_col].astype(str).str.strip()

    joined = gpd.sjoin(
        points,
        nigeria_only[["region_name", "geometry"]],
        how="left",
        predicate="intersects",
    )
    joined = joined.dropna(subset=["region_name"]).copy()
    if joined.empty:
        raise ValueError("No points could be assigned to admin boundaries.")

    grouped_rows: list[dict[str, float | str]] = []
    for region_name, group_df in joined.groupby("region_name"):
        row = {"region_name": str(region_name)}
        row.update(_summarize_group_frame(group_df))
        grouped_rows.append(row)
    grouped = pd.DataFrame(grouped_rows)

    region_geom = nigeria_only[["region_name", "geometry"]].dissolve(by="region_name").reset_index()
    regions = region_geom.merge(grouped, on="region_name", how="left").dropna(
        subset=["population", "either_access_prob"]
    )
    regions = _finalize_region_metrics(regions)
    return regions, "admin"


def aggregate_with_grid(
    points: gpd.GeoDataFrame,
    grid_deg: float,
    country_geom: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, str]:
    if grid_deg <= 0:
        raise ValueError("--grid-deg must be > 0.")

    pts = points.copy()
    pts["grid_lon"] = np.floor(pts.geometry.x / grid_deg) * grid_deg
    pts["grid_lat"] = np.floor(pts.geometry.y / grid_deg) * grid_deg
    pts["region_name"] = (
        "cell_"
        + pts["grid_lat"].map(lambda v: f"{v:.2f}")
        + "_"
        + pts["grid_lon"].map(lambda v: f"{v:.2f}")
    )

    grouped_rows: list[dict[str, float | str]] = []
    for (region_name, grid_lon, grid_lat), group_df in pts.groupby(["region_name", "grid_lon", "grid_lat"]):
        row: dict[str, float | str] = {
            "region_name": str(region_name),
            "grid_lon": float(grid_lon),
            "grid_lat": float(grid_lat),
        }
        row.update(_summarize_group_frame(group_df))
        grouped_rows.append(row)
    grouped = pd.DataFrame(grouped_rows)

    grouped["geometry"] = grouped.apply(
        lambda row: box(row["grid_lon"], row["grid_lat"], row["grid_lon"] + grid_deg, row["grid_lat"] + grid_deg),
        axis=1,
    )
    regions = gpd.GeoDataFrame(grouped, geometry="geometry", crs="EPSG:4326")
    nigeria_shape = country_geom.iloc[0].geometry
    regions["geometry"] = regions.geometry.intersection(nigeria_shape)
    regions = regions[~regions.geometry.is_empty & regions.geometry.notna()].copy()
    regions = _finalize_region_metrics(regions)
    return regions, "grid"


def add_hotspot_flags(regions: gpd.GeoDataFrame, args: argparse.Namespace) -> gpd.GeoDataFrame:
    out = regions.copy()
    out["density_rank"] = out["population_density_km2"].rank(pct=True).fillna(0.0)

    top_fraction = float(np.clip(args.hotspot_top_fraction, 0.0, 1.0))
    burden_quantile = 1.0 - top_fraction

    for service in ("pw", "sw"):
        access_col = f"{service}_access_prob"
        no_access_col = f"{service}_no_access_prob"
        burden_col = f"{service}_people_without_access"
        hotspot_col = f"is_hotspot_{service}"
        score_col = f"hotspot_score_{service}"
        burden_threshold_col = f"hotspot_score_threshold_{service}"

        severity_hotspot_col = f"is_severity_hotspot_{service}"
        severity_threshold_col = f"no_access_prob_threshold_{service}"

        out[no_access_col] = (1.0 - out[access_col]).clip(lower=0.0, upper=1.0)
        out[burden_col] = out["population"] * out[no_access_col]

        # Use expected people without access directly as the hotspot score.
        out[score_col] = out[burden_col].fillna(0.0)

        burden_thr = float(out[burden_col].quantile(burden_quantile))
        out[burden_threshold_col] = burden_thr
        out[hotspot_col] = out[burden_col] >= burden_thr

        severity_thr = float(out[no_access_col].quantile(burden_quantile))
        out[severity_threshold_col] = severity_thr
        out[severity_hotspot_col] = out[no_access_col] >= severity_thr

    return out


def _add_country_texture(ax: plt.Axes, country_geom: gpd.GeoDataFrame, alpha: float) -> None:
    shape = country_geom.geometry.union_all()
    minx, miny, maxx, maxy = shape.bounds
    width = max(maxx - minx, 1e-6)
    height = max(maxy - miny, 1e-6)

    nx = 900
    ny = int(np.clip(nx * (height / width), 400, 1200))
    xs = np.linspace(minx, maxx, nx)
    ys = np.linspace(miny, maxy, ny)
    xx, yy = np.meshgrid(xs, ys)

    x_norm = (xx - minx) / width
    y_norm = (yy - miny) / height

    rng = np.random.default_rng(2026)
    noise = rng.normal(0.0, 1.0, size=(ny, nx))
    for _ in range(3):
        noise = (
            noise
            + np.roll(noise, 1, axis=0)
            + np.roll(noise, -1, axis=0)
            + np.roll(noise, 1, axis=1)
            + np.roll(noise, -1, axis=1)
        ) / 5.0

    texture = (
        0.52
        + 0.20 * np.sin(10.0 * np.pi * x_norm) * np.cos(8.0 * np.pi * y_norm)
        + 0.12 * np.sin(18.0 * np.pi * x_norm + 2.0 * np.pi * y_norm)
        + 0.08 * np.cos(15.0 * np.pi * y_norm)
        + 0.09 * noise
    )
    texture = (texture - np.nanmin(texture)) / (np.nanmax(texture) - np.nanmin(texture) + 1e-12)

    mask = shapely.contains_xy(shape, xx, yy)
    texture = np.where(mask, texture, np.nan)

    cmap = plt.cm.Greens.copy()
    cmap.set_bad(alpha=0.0)
    ax.imshow(
        texture,
        extent=(minx, maxx, miny, maxy),
        origin="lower",
        cmap=cmap,
        alpha=float(np.clip(alpha, 0.0, 1.0)),
        zorder=0,
    )


def _place_outside_labels(
    ax: plt.Axes,
    labels_df: gpd.GeoDataFrame,
    *,
    minx: float,
    maxx: float,
    miny: float,
    maxy: float,
    color: str,
    side: str = "both",
    shorten_bottom: bool = False,
    y_shift_frac: float = 0.0,
) -> None:
    if labels_df.empty:
        return

    width = maxx - minx
    height = maxy - miny
    mid_x = 0.5 * (minx + maxx)
    y0 = miny + 0.05 * height
    y1 = maxy - 0.05 * height
    x_left = minx - 0.095 * width
    x_right = maxx + 0.050 * width
    text_gap = 0.007 * width
    # Keep label elbows outside the map so text does not sit on top of polygons.
    elbow_left = minx - 0.020 * width
    elbow_right = maxx + 0.020 * width

    labels_df = labels_df.copy()
    labels_df["pt"] = labels_df.geometry.representative_point()
    labels_df["pt_x"] = labels_df["pt"].x
    labels_df["pt_y"] = labels_df["pt"].y

    if side == "left":
        left = labels_df.sort_values("pt_y", ascending=False).copy()
        right = labels_df.iloc[0:0].copy()
    elif side == "right":
        left = labels_df.iloc[0:0].copy()
        right = labels_df.sort_values("pt_y", ascending=False).copy()
    else:
        left = labels_df[labels_df["pt_x"] < mid_x].sort_values("pt_y", ascending=False).copy()
        right = labels_df[labels_df["pt_x"] >= mid_x].sort_values("pt_y", ascending=False).copy()

    def assign_y(side_df: gpd.GeoDataFrame) -> np.ndarray:
        n = len(side_df)
        if n <= 0:
            return np.array([], dtype=float)
        if n == 1:
            return np.array([0.5 * (y0 + y1)], dtype=float)
        # Start from natural y, then greedily enforce a minimum gap for cleaner labels.
        ys = np.clip(side_df["pt_y"].to_numpy(dtype=float), y0, y1)
        min_gap = 0.045 * height
        ys[0] = min(ys[0], y1)
        for i in range(1, n):
            ys[i] = min(ys[i], ys[i - 1] - min_gap)
        if ys[-1] < y0:
            ys = ys + (y0 - ys[-1])
        ys = np.clip(ys, y0, y1)
        for i in range(1, n):
            if ys[i] > ys[i - 1] - min_gap:
                ys[i] = ys[i - 1] - min_gap
        return ys

    left_y = assign_y(left)
    right_y = assign_y(right)
    if y_shift_frac != 0.0:
        shift = float(y_shift_frac) * height
        left_y = np.clip(left_y + shift, y0, y1)
        right_y = np.clip(right_y + shift, y0, y1)

    for (_, row), label_y in zip(left.iterrows(), left_y):
        y_lab = float(label_y)
        x0 = float(row["pt_x"])
        y0_row = float(row["pt_y"])
        x1 = elbow_left
        x2 = x_left
        ax.plot([x0, x1, x2], [y0_row, y_lab, y_lab], color=color, lw=1, alpha=0.95, zorder=6, clip_on=False)
        text = ax.text(
            x_left - text_gap,
            y_lab,
            row["region_name"],
            ha="right",
            va="center",
            fontsize=26.0,
            color="#303030",
            zorder=7,
            clip_on=False,
        )
        text.set_path_effects([pe.Stroke(linewidth=10, foreground="white", alpha=0.94), pe.Normal()])

    for (_, row), label_y in zip(right.iterrows(), right_y):
        y_lab = float(label_y)
        x0 = float(row["pt_x"])
        y0_row = float(row["pt_y"])
        x1 = elbow_right
        x2 = x_right
        if shorten_bottom:
            y_frac = (y_lab - y0) / max(y1 - y0, 1e-6)
            if y_frac < 0.45:
                delta = (0.45 - y_frac) / 0.45 * (0.032 * width)
                x1 -= delta * 0.85
                x2 -= delta
        ax.plot([x0, x1, x2], [y0_row, y_lab, y_lab], color=color, lw=1, alpha=0.95, zorder=6, clip_on=False)
        text = ax.text(
            x2 + text_gap,
            y_lab,
            row["region_name"],
            ha="left",
            va="center",
            fontsize=26.0,
            color="#303030",
            zorder=7,
            clip_on=False,
        )
        text.set_path_effects([pe.Stroke(linewidth=10, foreground="white", alpha=0.94), pe.Normal()])


# --- HOTSPOT CONCENTRATION LABELS ---
def _make_cluster_labels(
    hotspots: gpd.GeoDataFrame,
    *,
    score_col: str,
    service: str,
    min_shared_boundary_m: float = 0.0,
    min_shared_boundary_fraction: float = 0.05,
    max_centroid_distance_m: float = 55_000.0,
    min_cluster_size: int = 2,
) -> gpd.GeoDataFrame:
    """Group nearby hotspot LGAs into labeled hotspot concentrations.

    This function is used for map labeling and visual emphasis only. The formal
    hotspot definition remains the red top-decile LGA outline. Labeled hotspot
    concentrations are formed by connecting hotspot LGAs whose representative
    points are within `max_centroid_distance_m` of one another.
    """
    if hotspots.empty:
        return hotspots.iloc[0:0].copy()

    projected = hotspots[["region_name", score_col, "population", "geometry"]].copy().to_crs("EPSG:6933")
    projected = projected.reset_index(drop=True)
    projected[score_col] = pd.to_numeric(projected[score_col], errors="coerce").fillna(0.0)
    projected["population"] = pd.to_numeric(projected["population"], errors="coerce").fillna(0.0)
    projected["label_point"] = projected.geometry.representative_point()

    parent = list(range(len(projected)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    max_centroid_distance_m = float(max(max_centroid_distance_m, 0.0))
    points = list(projected["label_point"])
    for i, point_i in enumerate(points):
        for j in range(i + 1, len(points)):
            if point_i.distance(points[j]) <= max_centroid_distance_m:
                union(i, j)

    projected["cluster_id"] = [find(i) for i in range(len(projected))]

    rows: list[dict[str, object]] = []
    min_cluster_size = max(int(min_cluster_size), 1)
    for cluster_id, group in projected.groupby("cluster_id"):
        if len(group) < min_cluster_size:
            continue
        group = group.sort_values(score_col, ascending=False)
        top_lga = str(group.iloc[0]["region_name"])
        lga_count = int(len(group))
        total_score = float(group[score_col].sum())
        total_population = float(group["population"].sum())
        cluster_geom = group.geometry.union_all()
        member_lgas = group["region_name"].astype(str).tolist()
        label = f"{top_lga} hotspot area"
        if lga_count > 1:
            label = f"{top_lga} hotspot area ({lga_count} LGAs)"
        rows.append(
            {
                "region_name": label,
                "cluster_id": int(cluster_id),
                "top_lga": top_lga,
                "lga_count": lga_count,
                "service": "PW" if service == "pw" else "SW",
                "cluster_score": total_score,
                "cluster_population": total_population,
                "member_lgas": "; ".join(member_lgas),
                "geometry": cluster_geom.representative_point(),
            }
        )

    if not rows:
        empty = hotspots.iloc[0:0].copy()
        for col in [
            "cluster_id",
            "top_lga",
            "lga_count",
            "service",
            "cluster_score",
            "cluster_population",
            "member_lgas",
        ]:
            empty[col] = pd.Series(dtype="object")
        return empty

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=projected.crs).to_crs(hotspots.crs)

def _substantial_contiguous_components(
    hotspot_members: gpd.GeoDataFrame,
    *,
    score_col: str,
    min_component_size: int = 3,
) -> gpd.GeoDataFrame:
    """Keep substantial boundary-contiguous components within a labeled hotspot area.

    Distance-based grouping is useful for identifying nearby hotspot concentrations,
    but it can chain together tiny detached islands. This helper keeps all
    boundary-contiguous components with at least `min_component_size` LGAs, rather
    than only the largest component.
    """
    if hotspot_members.empty or len(hotspot_members) <= 1:
        return hotspot_members.copy()

    projected = hotspot_members[["region_name", score_col, "geometry"]].copy().to_crs("EPSG:6933").reset_index(drop=True)
    projected[score_col] = pd.to_numeric(projected[score_col], errors="coerce").fillna(0.0)
    adjacency: list[set[int]] = [set() for _ in range(len(projected))]
    sindex = projected.sindex

    for i, geom_i in enumerate(projected.geometry):
        if geom_i is None or geom_i.is_empty:
            continue
        candidate_idx = list(sindex.query(geom_i, predicate="intersects"))
        for j in candidate_idx:
            if j <= i:
                continue
            geom_j = projected.geometry.iloc[j]
            if geom_j is None or geom_j.is_empty:
                continue
            shared = geom_i.boundary.intersection(geom_j.boundary)
            shared_length = float(shared.length) if not shared.is_empty else 0.0
            if shared_length > 0:
                adjacency[i].add(j)
                adjacency[j].add(i)

    remaining = set(range(len(projected)))
    components: list[set[int]] = []
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor not in component:
                    component.add(neighbor)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(component)

    min_component_size = max(int(min_component_size), 1)
    kept_components = [component for component in components if len(component) >= min_component_size]
    if not kept_components:
        kept_components = [
            max(
                components,
                key=lambda comp: (
                    len(comp),
                    float(projected.loc[sorted(comp), score_col].sum()),
                ),
            )
        ]

    keep_idx = set().union(*kept_components)
    keep_names = set(projected.loc[sorted(keep_idx), "region_name"].astype(str))
    return hotspot_members[hotspot_members["region_name"].astype(str).isin(keep_names)].copy()

# --- CONNECTED CORE LGAS THICK OUTLINE ---
def _connected_core_lgas(
    hotspot_members: gpd.GeoDataFrame,
    *,
    min_shared_boundary_fraction: float,
    min_degree: int = 2,
    anchor_region_name: str | None = None,
) -> gpd.GeoDataFrame:
    """Return the k-core of a labeled hotspot area.

    This is used only for the thick visual outline and map labels. It removes
    dangling LGAs that are technically part of a transitive cluster but are not
    locally embedded in the hotspot area.
    """
    if hotspot_members.empty or len(hotspot_members) <= min_degree:
        return hotspot_members.iloc[0:0].copy()

    projected = hotspot_members[["region_name", "geometry"]].copy().to_crs("EPSG:6933").reset_index(drop=True)
    projected["perimeter_m"] = projected.geometry.length
    adjacency: list[set[int]] = [set() for _ in range(len(projected))]
    sindex = projected.sindex

    threshold = float(np.clip(min_shared_boundary_fraction, 0.0, 1.0))
    for i, geom_i in enumerate(projected.geometry):
        if geom_i is None or geom_i.is_empty:
            continue
        candidate_idx = list(sindex.query(geom_i, predicate="intersects"))
        for j in candidate_idx:
            if j <= i:
                continue
            geom_j = projected.geometry.iloc[j]
            if geom_j is None or geom_j.is_empty:
                continue
            shared = geom_i.boundary.intersection(geom_j.boundary)
            shared_length = float(shared.length) if not shared.is_empty else 0.0
            perimeter_i = float(projected.at[i, "perimeter_m"])
            perimeter_j = float(projected.at[j, "perimeter_m"])
            smaller_perimeter = max(min(perimeter_i, perimeter_j), 1e-9)
            shared_fraction = shared_length / smaller_perimeter
            if shared_fraction >= threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)

    keep = set(range(len(projected)))
    changed = True
    while changed:
        changed = False
        remove = {idx for idx in keep if len(adjacency[idx] & keep) < min_degree}
        if remove:
            keep -= remove
            changed = True

    if not keep:
        return hotspot_members.iloc[0:0].copy()

    # The k-core can still contain multiple disconnected components. For visual
    # consistency, keep only the largest connected component, or the one containing
    # the anchor region if provided.
    remaining = set(keep)
    components: list[set[int]] = []
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node] & keep:
                if neighbor not in component:
                    component.add(neighbor)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(component)

    selected_component: set[int] | None = None
    if anchor_region_name is not None:
        anchor_matches = projected.index[projected["region_name"].astype(str) == str(anchor_region_name)].tolist()
        if anchor_matches:
            anchor_idx = int(anchor_matches[0])
            for component in components:
                if anchor_idx in component:
                    selected_component = component
                    break

    if selected_component is None:
        selected_component = max(
            components,
            key=lambda comp: (
                len(comp),
                float(projected.loc[sorted(comp), "perimeter_m"].sum()),
            ),
        )

    core_names = set(projected.loc[sorted(selected_component), "region_name"].astype(str))
    return hotspot_members[hotspot_members["region_name"].astype(str).isin(core_names)].copy()


def _plot_service_panel(
    ax: plt.Axes,
    country_geom: gpd.GeoDataFrame,
    regions: gpd.GeoDataFrame,
    mode: str,
    args: argparse.Namespace,
    *,
    service: str,
    hotspot_color: str,
    label_side: str,
    y_shift_frac: float = 0.0,
    metric: str = "burden",
    panel_label: str | None = None,
) -> None:
    service_name = "Piped Water" if service == "pw" else "Sewage"
    burden_col = f"{service}_people_without_access"
    no_access_col = f"{service}_no_access_prob"
    if metric == "severity":
        plot_col = no_access_col
        hotspot_col = f"is_severity_hotspot_{service}"
        score_col = no_access_col
        threshold_col = f"no_access_prob_threshold_{service}"
    else:
        plot_col = burden_col
        hotspot_col = f"is_hotspot_{service}"
        score_col = f"hotspot_score_{service}"
        threshold_col = f"hotspot_score_threshold_{service}"

    _add_country_texture(ax, country_geom, args.texture_alpha)
    if metric == "severity":
        severity_values = pd.to_numeric(regions[no_access_col], errors="coerce").dropna()
        if severity_values.empty:
            severity_vmin, severity_vmax = 0.0, 1.0
        else:
            severity_vmin = float(severity_values.quantile(SEVERITY_VMIN_QUANTILE))
            severity_vmax = float(severity_values.quantile(SEVERITY_VMAX_QUANTILE))
            if severity_vmax <= severity_vmin:
                severity_vmin, severity_vmax = 0.0, 1.0
        panel_norm = Normalize(vmin=severity_vmin, vmax=severity_vmax)
        panel_cmap = SEVERITY_CMAP
    else:
        panel_norm = Normalize(vmin=0.0, vmax=max(float(regions[burden_col].quantile(0.98)), 1.0))
        panel_cmap = BURDEN_CMAP
    regions.plot(
        column=plot_col,
        ax=ax,
        cmap=panel_cmap,
        norm=panel_norm,
        edgecolor="#d7dde2",
        linewidth=1,
        alpha=0.86,
        legend=False,
        zorder=2,
    )
    country_geom.boundary.plot(ax=ax, color="#1f1f1f", linewidth=1, zorder=3)

    hotspots = regions[regions[hotspot_col]].copy()
    if not hotspots.empty:
        hotspots.boundary.plot(ax=ax, color=hotspot_color, linewidth=HOTSPOT_OUTLINE_WIDTH, zorder=4)
        hotspot_count = int(regions[hotspot_col].sum())
        score_thr = float(regions[threshold_col].iloc[0])
        if metric == "severity":
            summary_lines = [
                f"Top {args.hotspot_top_fraction:.0%} severity hotspots",
                f"p(no access) ≥ {score_thr:.2f}",
            ]
            summary_x, summary_y = 0.965, 0.005
            summary_fontsize = 30
            summary_alpha = 0.86
            summary_pad = 3.2
        else:
            summary_lines = [
                f"Top {args.hotspot_top_fraction:.0%} burden hotspots",
                f"≥ {score_thr:,.0f} people w/o access",
            ]
            summary_x, summary_y = 0.985, -0.020
            summary_fontsize = 29
            summary_alpha = 0.84
            summary_pad = 3.1
        ax.text(
            summary_x,
            summary_y,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            fontsize=summary_fontsize,
            color="#2f2f2f",
            ha="right",
            va="bottom",
            linespacing=1.02,
            bbox={
                "facecolor": "white",
                "edgecolor": "#d0d0d0",
                "alpha": summary_alpha,
                "pad": summary_pad,
                "boxstyle": "round,pad=0.30",
            },
            zorder=10,
        )

        if panel_label is not None:
            ax.text(
                0.035,
                0.965,
                f"({panel_label})",
                transform=ax.transAxes,
                fontsize=34,
                fontweight="bold",
                color="#111111",
                ha="left",
                va="top",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "#333333",
                    "alpha": 0.92,
                    "pad": 2.8,
                    "boxstyle": "round,pad=0.20",
                },
                zorder=20,
            )

        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_frame_on(False)
        shape = country_geom.geometry.union_all()
        minx, miny, maxx, maxy = shape.bounds
        width = maxx - minx
        height = maxy - miny
        ax.set_xlim(minx - 0.18 * width, maxx + 0.14 * width)
        ax.set_ylim(miny - 0.03 * height, maxy + 0.03 * height)
# --- HOTSPOT CLUSTER SUMMARY ---

def build_hotspot_cluster_summary(regions: gpd.GeoDataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Create a table identifying contiguous clusters of hotspot LGAs."""
    rows: list[dict[str, object]] = []
    specs = [
        ("pw", "burden", "is_hotspot_pw", "hotspot_score_pw"),
        ("sw", "burden", "is_hotspot_sw", "hotspot_score_sw"),
        ("pw", "severity", "is_severity_hotspot_pw", "pw_no_access_prob"),
        ("sw", "severity", "is_severity_hotspot_sw", "sw_no_access_prob"),
    ]
    for service, metric, hotspot_col, score_col in specs:
        hotspots = regions[regions[hotspot_col]].copy()
        if hotspots.empty:
            continue
        clusters = _make_cluster_labels(
            hotspots,
            score_col=score_col,
            service=service,
            max_centroid_distance_m=float(args.label_cluster_distance_km) * 1000.0,
            min_cluster_size=int(args.label_cluster_min_size),
        )
        if clusters.empty:
            continue
        for _, cluster in clusters.sort_values("cluster_score", ascending=False).iterrows():
            rows.append(
                {
                    "service": "piped_water" if service == "pw" else "sewage",
                    "metric": metric,
                    "cluster_id": int(cluster["cluster_id"]),
                    "cluster_label": str(cluster["region_name"]),
                    "top_lga": str(cluster["top_lga"]),
                    "lga_count": int(cluster["lga_count"]),
                    "cluster_score": float(cluster["cluster_score"]),
                    "cluster_population": float(cluster["cluster_population"]),
                    "member_lgas": str(cluster["member_lgas"]),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "service",
                "metric",
                "cluster_id",
                "cluster_label",
                "top_lga",
                "lga_count",
                "cluster_score",
                "cluster_population",
                "member_lgas",
            ]
        )
    return pd.DataFrame(rows).sort_values(["service", "metric", "cluster_score"], ascending=[True, True, False])


# --- LGA METRIC SUMMARY ---
def build_lga_metric_summary(regions: gpd.GeoDataFrame) -> pd.DataFrame:
    """Summarize LGA-level distributions for population, severity, and burden metrics."""
    metric_specs = [
        ("pw_people_without_access", "Piped water burden", "million people", 1_000_000.0),
        ("sw_people_without_access", "Sewage burden", "million people", 1_000_000.0),
        ("pw_no_access_prob", "Piped water severity", "probability", 1.0),
        ("sw_no_access_prob", "Sewage severity", "probability", 1.0),
        ("population", "LGA population", "million people", 1_000_000.0),
    ]
    rows: list[dict[str, object]] = []
    for col, label, unit, scale in metric_specs:
        values_raw = pd.to_numeric(regions[col], errors="coerce").dropna()
        if values_raw.empty:
            continue
        values = values_raw / float(scale)
        mean_value = float(values.mean())
        std_value = float(values.std(ddof=1))
        median_value = float(values.median())
        p10_value = float(values.quantile(0.10))
        p90_value = float(values.quantile(0.90))
        max_value = float(values.max())
        min_value = float(values.min())
        top10_threshold = p90_value
        top10_mask = values >= top10_threshold
        top10_share = float(values.loc[top10_mask].sum() / values.sum()) if values.sum() > 0 else float("nan")
        rows.append(
            {
                "metric": col,
                "label": label,
                "unit": unit,
                "n_lgas": int(values.shape[0]),
                "min": min_value,
                "mean": mean_value,
                "std": std_value,
                "mean_plus_minus_std": f"{mean_value:.3f} ± {std_value:.3f}",
                "median": median_value,
                "p10": p10_value,
                "p90": p90_value,
                "max": max_value,
                "range": max_value - min_value,
                "coefficient_of_variation": float(std_value / mean_value) if mean_value != 0 else float("nan"),
                "p90_to_median_ratio": float(p90_value / median_value) if median_value != 0 else float("nan"),
                "max_to_mean_ratio": float(max_value / mean_value) if mean_value != 0 else float("nan"),
                "max_to_median_ratio": float(max_value / median_value) if median_value != 0 else float("nan"),
                "top10_share_of_total": top10_share,
            }
        )
    return pd.DataFrame(rows)


# --- LGA METRIC SUMMARY LATEX OUTPUT ---
def write_lga_metric_summary_latex(summary: pd.DataFrame, output_path: Path) -> None:
    """Write a compact LaTeX table emphasizing LGA-level inequality and concentration."""
    if summary.empty:
        return

    display = summary.copy()
    keep_metrics = [
        "pw_people_without_access",
        "sw_people_without_access",
        "pw_no_access_prob",
        "sw_no_access_prob",
        "population",
    ]
    display = display[display["metric"].isin(keep_metrics)].copy()
    display["Mean $\\pm$ SD"] = display.apply(lambda r: f"{r['mean']:.3f} $\\pm$ {r['std']:.3f}", axis=1)
    display["Median"] = display["median"].map(lambda v: f"{v:.3f}")
    display["90th pct."] = display["p90"].map(lambda v: f"{v:.3f}")
    display["Max"] = display["max"].map(lambda v: f"{v:.3f}")
    display["Max/median"] = display["max_to_median_ratio"].map(lambda v: f"{v:.1f}")
    display["Top 10\\% share"] = display["top10_share_of_total"].map(lambda v: f"{100 * v:.1f}\\%")

    table = display[
        [
            "label",
            "unit",
            "Mean $\\pm$ SD",
            "Median",
            "90th pct.",
            "Max",
            "Max/median",
            "Top 10\\% share",
        ]
    ].rename(columns={"label": "Metric", "unit": "Unit"})

    latex = table.to_latex(
        index=False,
        escape=False,
        column_format="llrrrrrr",
        caption=(
            "LGA-level dispersion and concentration metrics for the Nigeria application. "
            "Burden metrics are reported in millions of people; severity metrics are predicted probabilities of no access."
        ),
        label="tab:nigeria_lga_metric_summary",
    )
    latex = latex.replace("\\toprule", "\\toprule")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex, encoding="utf-8")


def plot_combined_map(
    country_geom: gpd.GeoDataFrame,
    regions: gpd.GeoDataFrame,
    mode: str,
    args: argparse.Namespace,
    *,
    metric: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(27.0, 12.2), constrained_layout=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.993, bottom=0.125, wspace=0.06)
    pw_color = PW_HOTSPOT_COLOR
    sw_color = SW_HOTSPOT_COLOR

    _plot_service_panel(
        axes[0],
        country_geom,
        regions,
        mode,
        args,
        service="pw",
        hotspot_color=pw_color,
        label_side="left",
        y_shift_frac=0.0,
        metric=metric,
        panel_label="a" if metric == "burden" else "c",
    )
    _plot_service_panel(
        axes[1],
        country_geom,
        regions,
        mode,
        args,
        service="sw",
        hotspot_color=sw_color,
        label_side="right",
        y_shift_frac=-0.045,
        metric=metric,
        panel_label="b" if metric == "burden" else "d",
    )

    if metric == "severity":
        severity_values = pd.concat(
            [
                regions["pw_no_access_prob"],
                regions["sw_no_access_prob"],
            ],
            ignore_index=True,
        ).dropna()
        if severity_values.empty:
            severity_vmin, severity_vmax = 0.0, 1.0
        else:
            severity_vmin = float(severity_values.quantile(SEVERITY_VMIN_QUANTILE))
            severity_vmax = float(severity_values.quantile(SEVERITY_VMAX_QUANTILE))
            if severity_vmax <= severity_vmin:
                severity_vmin, severity_vmax = 0.0, 1.0
        norm = Normalize(vmin=severity_vmin, vmax=severity_vmax)
        cbar_label = "Predicted probability of no access"
        cbar_cmap = SEVERITY_CMAP
    else:
        all_burden = pd.concat(
            [
                regions["pw_people_without_access"],
                regions["sw_people_without_access"],
            ],
            ignore_index=True,
        ).dropna()
        vmax = float(all_burden.quantile(0.98)) if not all_burden.empty else 1.0
        vmax = max(vmax, 1.0)
        norm = Normalize(vmin=0.0, vmax=vmax)
        cbar_label = "Expected people without access"
        cbar_cmap = BURDEN_CMAP
    sm = ScalarMappable(norm=norm, cmap=cbar_cmap)
    sm.set_array([])
    cax = fig.add_axes([0.20, 0.045, 0.60, 0.012])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(cbar_label, fontsize=28)
    cbar.ax.tick_params(labelsize=26)

    legend_handles = [
        Line2D([0], [0], color=HOTSPOT_COLOR, lw=HOTSPOT_OUTLINE_WIDTH, label="Top 10% hotspot LGAs"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.089),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#d7d7d7",
        framealpha=0.95,
        fontsize=28.0,
        borderpad=0.75,
        labelspacing=0.55,
        handlelength=2.4,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
def plot_composite_hotspot_map(
    country_geom: gpd.GeoDataFrame,
    regions: gpd.GeoDataFrame,
    mode: str,
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(27.0, 22.5), constrained_layout=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.985, bottom=0.255, wspace=0.06, hspace=0.035)

    panel_specs = [
        (axes[0, 0], "pw", "burden", "a", "left", 0.0, PW_HOTSPOT_COLOR),
        (axes[0, 1], "sw", "burden", "b", "right", -0.045, SW_HOTSPOT_COLOR),
        (axes[1, 0], "pw", "severity", "c", "left", 0.0, PW_HOTSPOT_COLOR),
        (axes[1, 1], "sw", "severity", "d", "right", -0.045, SW_HOTSPOT_COLOR),
    ]
    for ax, service, metric, label, label_side, y_shift, hotspot_color in panel_specs:
        _plot_service_panel(
            ax,
            country_geom,
            regions,
            mode,
            args,
            service=service,
            hotspot_color=hotspot_color,
            label_side=label_side,
            y_shift_frac=y_shift,
            metric=metric,
            panel_label=label,
        )

    burden_values = pd.concat(
        [regions["pw_people_without_access"], regions["sw_people_without_access"]],
        ignore_index=True,
    ).dropna()
    burden_vmax = float(burden_values.quantile(0.98)) if not burden_values.empty else 1.0
    burden_vmax = max(burden_vmax, 1.0)

    burden_sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=burden_vmax), cmap=BURDEN_CMAP)
    burden_sm.set_array([])
    burden_cax = fig.add_axes([0.10, 0.170, 0.37, 0.020])
    burden_cbar = fig.colorbar(burden_sm, cax=burden_cax, orientation="horizontal")
    burden_cbar.set_label("Expected people without access (millions)", fontsize=30, labelpad=14)
    burden_cbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1_000_000:.1f}"))
    burden_cbar.ax.tick_params(labelsize=24, pad=4)

    severity_values = pd.concat(
        [regions["pw_no_access_prob"], regions["sw_no_access_prob"]],
        ignore_index=True,
    ).dropna()
    if severity_values.empty:
        severity_vmin, severity_vmax = 0.0, 1.0
    else:
        severity_vmin = float(severity_values.quantile(SEVERITY_VMIN_QUANTILE))
        severity_vmax = float(severity_values.quantile(SEVERITY_VMAX_QUANTILE))
        if severity_vmax <= severity_vmin:
            severity_vmin, severity_vmax = 0.0, 1.0
    severity_sm = ScalarMappable(norm=Normalize(vmin=severity_vmin, vmax=severity_vmax), cmap=SEVERITY_CMAP)
    severity_sm.set_array([])
    severity_cax = fig.add_axes([0.53, 0.170, 0.37, 0.020])
    severity_cbar = fig.colorbar(severity_sm, cax=severity_cax, orientation="horizontal")
    severity_cbar.set_label("Predicted probability of no access (5th–98th pct. scale)", fontsize=30, labelpad=14)
    severity_cbar.ax.tick_params(labelsize=24, pad=4)

    legend_handles = [
        Line2D([0], [0], color=HOTSPOT_COLOR, lw=HOTSPOT_OUTLINE_WIDTH, label="Top 10% hotspot LGAs"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#d7d7d7",
        framealpha=0.95,
        fontsize=32.0,
        borderpad=0.65,
        handlelength=2.4,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    country_geom = load_country_geometry(args.country_shapefile, args.country)
    tile_points = build_tile_frame(args)

    # Keep only points inside country boundary.
    tile_points = gpd.sjoin(tile_points, country_geom[["geometry"]], how="inner", predicate="within").drop(
        columns=["index_right"]
    )
    if tile_points.empty:
        raise ValueError("No tile points fall inside the country geometry.")

    use_admin = (not args.force_grid) and (args.admin_boundaries is not None)
    if use_admin:
        if not args.admin_boundaries.exists():
            raise FileNotFoundError(
                f"Admin boundary file not found: {args.admin_boundaries}. "
                "Provide --admin-boundaries or use --force-grid."
            )
        regions, mode = aggregate_with_admin_boundaries(
            tile_points,
            args.admin_boundaries,
            args.admin_name_column,
            country_geom,
        )
    else:
        regions, mode = aggregate_with_grid(tile_points, args.grid_deg, country_geom)

    regions = add_hotspot_flags(regions, args)
    regions["tiles"] = pd.to_numeric(regions["tiles"], errors="coerce").fillna(0).astype(int)
    regions = regions.sort_values("hotspot_score_pw", ascending=False).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_cols = [
        "region_name",
        "tiles",
        "population",
        "area_km2",
        "population_density_km2",
        "density_rank",
        "pw_access_prob",
        "pw_no_access_prob",
        "pw_people_without_access",
        "sw_access_prob",
        "sw_no_access_prob",
        "sw_people_without_access",
        "either_access_prob",
        "hotspot_score_threshold_pw",
        "hotspot_score_threshold_sw",
        "no_access_prob_threshold_pw",
        "no_access_prob_threshold_sw",
        "is_severity_hotspot_pw",
        "is_severity_hotspot_sw",
        "is_hotspot_pw",
        "is_hotspot_sw",
        "hotspot_score_pw",
        "hotspot_score_sw",
    ]
    regions[csv_cols].to_csv(args.output_csv, index=False)
    clusters = build_hotspot_cluster_summary(regions, args)
    clusters.to_csv(args.output_clusters_csv, index=False)
    metric_summary = build_lga_metric_summary(regions)
    metric_summary.to_csv(args.output_summary_csv, index=False)
    write_lga_metric_summary_latex(metric_summary, args.output_summary_tex)
    plot_population_map(country_geom, regions, args.output_population_map, args.texture_alpha)
    plot_combined_map(country_geom, regions, mode, args, metric="burden", output_path=args.output_figure_burden)
    plot_combined_map(country_geom, regions, mode, args, metric="severity", output_path=args.output_figure_severity)
    plot_composite_hotspot_map(country_geom, regions, mode, args, args.output_figure_composite)

    print(f"Mode: {mode}")
    print(f"Regions: {len(regions)}")
    print(f"Piped hotspots: {int(regions['is_hotspot_pw'].sum())}")
    print(f"Sewage hotspots: {int(regions['is_hotspot_sw'].sum())}")
    print(f"Saved burden figure: {args.output_figure_burden}")
    print(f"Saved no-access probability figure: {args.output_figure_severity}")
    print(f"Saved composite hotspot figure: {args.output_figure_composite}")
    print(f"Saved population map: {args.output_population_map}")
    print(f"Saved table: {args.output_csv}")
    print(f"Saved hotspot cluster table: {args.output_clusters_csv}")
    print(f"Saved LGA metric summary table: {args.output_summary_csv}")
    print(f"Saved LGA metric summary LaTeX table: {args.output_summary_tex}")


if __name__ == "__main__":
    main()
