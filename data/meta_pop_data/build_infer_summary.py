#!/usr/bin/env python3
"""Build country-level water/sewage access summaries from k-NN inference outputs.

This script merges per-tile population from *_general_2020_tiles.csv with
inference predictions (sw_predictions.csv / pw_predictions.csv), then reports
absolute counts and percentages per country.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_root = repo_root / "data" / "meta_pop_data"
    p = argparse.ArgumentParser(description="Summarize k-NN inference per country using tile population.")
    p.add_argument(
        "--data-root",
        type=Path,
        default=default_root,
        help="Root folder containing countries_2x2.",
    )
    p.add_argument(
        "--infer-root",
        type=Path,
        default=default_root / "inference",
        help="Root folder containing inference results per country.",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=default_root / "inference_summary.csv",
        help="Output summary CSV path.",
    )
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="Which k to use for predictions. Defaults to max k available in CSV.",
    )
    return p.parse_args()


def _extract_lon_lat(path: str) -> tuple[str | None, str | None]:
    patterns = [
        r"sentinel_image_([-+]?\d*\.\d+)_([-+]?\d*\.\d+)",
        r"sentinel_([-+]?\d*\.\d+)_([-+]?\d*\.\d+)",
    ]
    for pat in patterns:
        match = re.search(pat, path)
        if match:
            return match.group(2), match.group(1)
    return None, None


def _select_k_column(df: pd.DataFrame, kind: str, k: int | None) -> str:
    prefix = f"{kind}_k"
    candidates = [c for c in df.columns if c.startswith(prefix)]
    if not candidates:
        raise ValueError(f"No columns found with prefix '{prefix}'")

    def _parse_k(col: str) -> int:
        return int(col.replace(prefix, ""))

    if k is None:
        best = max(candidates, key=_parse_k)
        return best
    col = f"{prefix}{k}"
    if col not in df.columns:
        raise ValueError(f"Requested k={k} not found in columns: {sorted(candidates)}")
    return col


def _normalize_coords(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["centroid_lon"] = df["centroid_lon"].astype(float).round(5).astype(str).str.strip()
    df["centroid_lat"] = df["centroid_lat"].astype(float).round(5).astype(str).str.strip()
    return df


def _prediction_to_labels(series: pd.Series, task: str) -> pd.Series:
    values = series.astype(str).str.lower()
    is_missing = series.isna()
    if task == "pw":
        no_mask = values.str.contains("no") & (values.str.contains("piped") | values.str.contains("water"))
    else:
        no_mask = values.str.contains("no") & values.str.contains("sew")

    labels = pd.Series(["access"] * len(values), index=values.index)
    labels[no_mask] = "no_access"
    labels[is_missing] = pd.NA
    return labels


def _summarize_task(
    tiles: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    task: str,
    k: int | None,
) -> dict[str, float | int]:
    pred_class_col = _select_k_column(predictions, "pred_class", k)
    predictions = predictions.copy()
    predictions[["lon", "lat"]] = predictions["path"].apply(
        lambda x: pd.Series(_extract_lon_lat(str(x)))
    )
    predictions = predictions.dropna(subset=["lon", "lat"])
    predictions["lon"] = predictions["lon"].astype(float).round(5).astype(str).str.strip()
    predictions["lat"] = predictions["lat"].astype(float).round(5).astype(str).str.strip()

    merged = pd.merge(
        tiles,
        predictions,
        left_on=["centroid_lon", "centroid_lat"],
        right_on=["lon", "lat"],
        how="left",
    )

    merged["_label"] = _prediction_to_labels(merged[pred_class_col], task)
    matched = merged.dropna(subset=["_label"]).copy()

    access_pop = matched.loc[matched["_label"] == "access", "total_population"].sum()
    no_access_pop = matched.loc[matched["_label"] == "no_access", "total_population"].sum()
    matched_pop = access_pop + no_access_pop
    total_pop = merged["total_population"].sum()
    access_pct = (access_pop / matched_pop * 100) if matched_pop else 0.0
    no_access_pct = (no_access_pop / matched_pop * 100) if matched_pop else 0.0
    coverage_pct = (matched_pop / total_pop * 100) if total_pop else 0.0

    return {
        "access_pop": int(round(access_pop)),
        "no_access_pop": int(round(no_access_pop)),
        "access_pct": round(access_pct, 2),
        "no_access_pct": round(no_access_pct, 2),
        "total_pop": int(round(total_pop)),
        "matched_pop": int(round(matched_pop)),
        "coverage_pct": round(coverage_pct, 2),
    }


def main() -> None:
    args = parse_args()
    countries_dir = args.data_root / "countries_2x2"
    infer_root = args.infer_root

    rows: list[dict] = []
    for country_dir in tqdm(sorted(countries_dir.iterdir()), desc="Countries"):
        if not country_dir.is_dir():
            continue
        country = country_dir.name

        tiles = list(country_dir.glob("*_general_2020_tiles.csv"))
        if not tiles:
            continue
        tiles_df = pd.read_csv(tiles[0])
        if not {"centroid_lon", "centroid_lat", "total_population"}.issubset(tiles_df.columns):
            continue
        tiles_df = _normalize_coords(tiles_df)

        sw_path = infer_root / country / "sw_predictions.csv"
        pw_path = infer_root / country / "pw_predictions.csv"
        if not sw_path.exists() or not pw_path.exists():
            continue

        sw_pred = pd.read_csv(sw_path)
        pw_pred = pd.read_csv(pw_path)

        sw_stats = _summarize_task(tiles_df, sw_pred, task="sw", k=args.k)
        pw_stats = _summarize_task(tiles_df, pw_pred, task="pw", k=args.k)

        rows.append(
            {
                "Country": country,
                "Piped Water Access": pw_stats["access_pop"],
                "Piped Water Access (%)": pw_stats["access_pct"],
                "No Piped Water Access": pw_stats["no_access_pop"],
                "No Piped Water Access (%)": pw_stats["no_access_pct"],
                "Piped Water Matched Population": pw_stats["matched_pop"],
                "Piped Water Coverage (%)": pw_stats["coverage_pct"],
                "Sewage Access": sw_stats["access_pop"],
                "Sewage Access (%)": sw_stats["access_pct"],
                "No Sewage Access": sw_stats["no_access_pop"],
                "No Sewage Access (%)": sw_stats["no_access_pct"],
                "Sewage Matched Population": sw_stats["matched_pop"],
                "Sewage Coverage (%)": sw_stats["coverage_pct"],
                "Total Population": pw_stats["total_pop"],
            }
        )

    out_df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Wrote summary to {args.output_csv}")


if __name__ == "__main__":
    main()
