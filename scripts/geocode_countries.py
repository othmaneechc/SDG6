#!/usr/bin/env python3
"""Add a real country name to the manifest for the region-based (leave-one-region-out) split and the population-density baseline.

The Afrobarometer ``COUNTRY`` field is coded *per round* -- R7, R8 and R9 each
renumber their countries 1..N -- so the same numeric code denotes different
countries in different rounds (code 2 is Angola in one round and Botswana in
another). Grouping locations by that raw code therefore pools
unrelated countries and is wrong.

This script assigns each survey location its true country by reverse-geocoding
its coordinates against the Natural Earth admin-0 boundaries, then taking, for
each ``(round, code)`` group, the modal geocoded country. The mode makes the
assignment robust to the ~2% of points that fall just across a border or
offshore. The resulting ``country_name`` column is written back to the manifest.

Usage:
    python scripts/geocode_countries.py --manifest data/manifest_sentinel.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

REPO = Path(__file__).resolve().parents[1]
NE = REPO / "data" / "meta_pop_data" / "natural_earth_data" / "ne_110m_admin_0_countries.shp"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=REPO / "data" / "manifest_sentinel.csv")
    p.add_argument("--shapefile", type=Path, default=NE)
    return p.parse_args()


def geocode(locs: pd.DataFrame, shp: Path) -> pd.Series:
    """Return a geocoded country name per row of ``locs`` (has lat, lon)."""
    ne = gpd.read_file(shp)
    namecol = "NAME" if "NAME" in ne.columns else "ADMIN"
    g = gpd.GeoDataFrame(
        locs.reset_index(drop=True),
        geometry=[Point(xy) for xy in zip(locs.lon, locs.lat)], crs="EPSG:4326")
    j = gpd.sjoin(g, ne[[namecol, "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    # nearest polygon for points that fell outside every boundary (offshore)
    ne_p = ne.to_crs(3857)
    for i, r in j[j[namecol].isna()].iterrows():
        pt = gpd.GeoSeries([Point(r.lon, r.lat)], crs=4326).to_crs(3857).iloc[0]
        j.loc[i, namecol] = ne.iloc[ne_p.distance(pt).values.argmin()][namecol]
    return j[namecol].rename("geo_country")


def main() -> int:
    args = parse_args()
    man = pd.read_csv(args.manifest)
    for col in ("lat", "lon", "country", "round"):
        if col not in man.columns:
            raise SystemExit(f"manifest missing required column: {col}")

    uniq = man.drop_duplicates(["lat", "lon"])[["lat", "lon"]].reset_index(drop=True)
    uniq["geo_country"] = geocode(uniq, args.shapefile).values
    man = man.merge(uniq, on=["lat", "lon"], how="left")

    # canonical country per (round, code): mode of the geocoded country
    canon = (man.groupby(["round", "country"])["geo_country"]
                .agg(lambda s: s.value_counts().index[0])
                .rename("country_name").reset_index())
    man = man.drop(columns=["geo_country"]).merge(canon, on=["round", "country"], how="left")

    n = man["country_name"].nunique()
    man.to_csv(args.manifest, index=False)
    print(f"wrote country_name to {args.manifest} ({n} distinct countries, "
          f"{man['country_name'].isna().sum()} unmatched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
