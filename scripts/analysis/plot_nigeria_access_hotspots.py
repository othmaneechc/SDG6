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
import os
import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from shapely.geometry import box


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    data_root = Path(os.environ.get("SDG6_DATA_ROOT", repo_root / "data")).expanduser()
    admin_default = repo_root / "data" / "admin_boundaries" / "gadm41_NGA_2.json"

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
        help="Piped water inference CSV.",
    )
    parser.add_argument(
        "--sw-csv",
        type=Path,
        default=None,
        help="Sewage inference CSV.",
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
        "--low-access-quantile",
        type=float,
        default=0.30,
        help="Quantile threshold for low combined access probability.",
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
        default=12,
        help="Maximum number of hotspot labels shown on the map.",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=repo_root / "outputs" / "figures" / "nigeria_pw_sw_hotspots.png",
        help="Output figure path for the combined side-by-side PW/SW map (PNG).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo_root / "outputs" / "tables" / "nigeria_access_hotspots.csv",
        help="Output CSV with region-level metrics.",
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
    density_thr = float(out["population_density_km2"].quantile(args.density_quantile))
    out["dense_threshold"] = density_thr

    density_norm = out["population_density_km2"] / out["population_density_km2"].max()
    density_norm = density_norm.fillna(0.0)

    for service in ("pw", "sw"):
        access_col = f"{service}_access_prob"
        threshold_col = f"low_access_threshold_{service}"
        hotspot_col = f"is_hotspot_{service}"
        score_col = f"hotspot_score_{service}"
        low_thr = float(out[access_col].quantile(args.low_access_quantile))
        out[threshold_col] = low_thr
        out[hotspot_col] = (
            (out["population"] >= args.min_population)
            & (out["population_density_km2"] >= density_thr)
            & (out[access_col] <= low_thr)
        )
        out[score_col] = density_norm * (1.0 - out[access_col].fillna(0.0))

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
    x_left = minx - 0.045 * width
    x_right = maxx + 0.030 * width
    text_gap = 0.007 * width
    # Start the horizontal segment early from inside the map.
    elbow_left = minx + 0.025 * width
    elbow_right = maxx - 0.025 * width

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
            fontsize=20.0,
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
            fontsize=20.0,
            color="#303030",
            zorder=7,
            clip_on=False,
        )
        text.set_path_effects([pe.Stroke(linewidth=10, foreground="white", alpha=0.94), pe.Normal()])


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
) -> None:
    service_name = "Piped Water" if service == "pw" else "Sewage System Access"
    access_col = f"{service}_access_prob"
    hotspot_col = f"is_hotspot_{service}"
    score_col = f"hotspot_score_{service}"
    threshold_col = f"low_access_threshold_{service}"

    _add_country_texture(ax, country_geom, args.texture_alpha)
    regions.plot(
        column=access_col,
        ax=ax,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        edgecolor="#d7dde2",
        linewidth=1,
        alpha=0.86,
        legend=False,
        zorder=2,
    )
    country_geom.boundary.plot(ax=ax, color="#1f1f1f", linewidth=1, zorder=3)

    hotspots = regions[regions[hotspot_col]].copy()
    if not hotspots.empty:
        hotspots.boundary.plot(ax=ax, color=hotspot_color, linewidth=1, zorder=4)
        top = hotspots.sort_values(score_col, ascending=False).head(max(0, args.top_labels))
        shape = country_geom.geometry.union_all()
        minx, miny, maxx, maxy = shape.bounds
        _place_outside_labels(
            ax,
            top,
            minx=minx,
            maxx=maxx,
            miny=miny,
            maxy=maxy,
            color=hotspot_color,
            side=label_side,
            shorten_bottom=(service == "pw"),
            y_shift_frac=y_shift_frac,
        )

    low_thr = float(regions[threshold_col].iloc[0])
    dense_thr = float(regions["dense_threshold"].iloc[0])
    hotspot_count = int(regions[hotspot_col].sum())
    labeled_count = int(min(args.top_labels, hotspot_count))
    summary_lines = [
        # f"Color = predicted {service_name.lower()} access probability (0-1)",
        f"Hotspot rule ({hotspot_count}):",
        f"population >= {args.min_population:,.0f}\n"
        f"density >= {dense_thr:.1f}/km² (q={args.density_quantile:.2f})",
        f"{service_name.lower()} access <= {low_thr:.3f} (q={args.low_access_quantile:.2f})",
    ]
    ax.text(
        0.985,
        0.015,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        fontsize=20,
        color="#2f2f2f",
        ha="right",
        multialignment="left",
        bbox={"facecolor": "white", "edgecolor": "#d7d7d7", "alpha": 0.9, "pad": 3.5},
        zorder=10,
    )

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_frame_on(False)
    shape = country_geom.geometry.union_all()
    minx, miny, maxx, maxy = shape.bounds
    width = maxx - minx
    height = maxy - miny
    ax.set_xlim(minx - 0.11 * width, maxx + 0.11 * width)
    ax.set_ylim(miny - 0.03 * height, maxy + 0.03 * height)


def plot_combined_map(
    country_geom: gpd.GeoDataFrame,
    regions: gpd.GeoDataFrame,
    mode: str,
    args: argparse.Namespace,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(27.0, 12.2), constrained_layout=False)
    fig.subplots_adjust(left=0.004, right=0.996, top=0.993, bottom=0.125, wspace=0.028)
    pw_color = "#d7263d"
    sw_color = "#5f0f99"

    _plot_service_panel(
        axes[0],
        country_geom,
        regions,
        mode,
        args,
        service="pw",
        hotspot_color=pw_color,
        label_side="right",
        y_shift_frac=0.045,
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
    )

    norm = Normalize(vmin=0.0, vmax=1.0)
    sm = ScalarMappable(norm=norm, cmap="YlGnBu")
    sm.set_array([])
    cax = fig.add_axes([0.20, 0.045, 0.60, 0.012])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Predicted access probability", fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    legend_handles = [
        Line2D([0], [0], color=pw_color, lw=3.2, label="PW hotspots"),
        Line2D([0], [0], color=sw_color, lw=3.2, label="SW hotspots"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.089),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#d7d7d7",
        framealpha=0.95,
        fontsize=20.0,
        borderpad=0.75,
        labelspacing=0.55,
        handlelength=2.4,
    )

    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_figure, dpi=260, bbox_inches="tight", pad_inches=0.18)
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
        "pw_access_prob",
        "sw_access_prob",
        "either_access_prob",
        "dense_threshold",
        "low_access_threshold_pw",
        "low_access_threshold_sw",
        "is_hotspot_pw",
        "is_hotspot_sw",
        "hotspot_score_pw",
        "hotspot_score_sw",
    ]
    regions[csv_cols].to_csv(args.output_csv, index=False)
    plot_combined_map(country_geom, regions, mode, args)

    print(f"Mode: {mode}")
    print(f"Regions: {len(regions)}")
    print(f"Piped hotspots: {int(regions['is_hotspot_pw'].sum())}")
    print(f"Sewage system access hotspots: {int(regions['is_hotspot_sw'].sum())}")
    print(f"Saved combined figure: {args.output_figure}")
    print(f"Saved table: {args.output_csv}")


if __name__ == "__main__":
    main()
