#!/usr/bin/env python3
"""Plot the five African subregions used by the region-based cross-validation.

Each survey location is colored by the UN African subregion (Northern, Western,
Middle, Eastern, Southern) that its reverse-geocoded country belongs to. Under
the ``region`` scheme in scripts/eval_splits.py, every subregion is held out of
training in turn (leave-one-region-out), so this map shows exactly which
locations move together between the training and test folds.

The country-to-region assignment is fixed in sdg6.splits.AFRICA_UN_SUBREGION and
archived here as a CSV so the split can be explained rather than being an
arbitrary coordinate partition.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from sdg6.splits import REGION_ORDER, region_labels  # noqa: E402

# One stable color per subregion, ordered north-to-south.
REGION_COLORS = {
    "Northern": "#3b82f6",
    "Western": "#ef4444",
    "Middle": "#22c55e",
    "Eastern": "#f59e0b",
    "Southern": "#8b5cf6",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=REPO / "data" / "manifest_sentinel.csv")
    p.add_argument(
        "--shapefile",
        type=Path,
        default=REPO / "data" / "meta_pop_data" / "natural_earth_data" / "ne_110m_admin_0_countries.shp",
        help="Natural Earth countries shapefile, used only for a faint continent outline.",
    )
    p.add_argument("--out", type=Path, default=REPO / "outputs" / "figures" / "region_folds_cv.png")
    p.add_argument(
        "--table",
        type=Path,
        default=REPO / "outputs" / "tables" / "region_folds_cv.csv",
    )
    return p.parse_args()


def load_manifest(path: Path) -> tuple[pd.DataFrame, str]:
    man = pd.read_csv(path)
    name_col = "country_name" if "country_name" in man.columns else "country"
    df = man.dropna(subset=["lat", "lon", name_col]).copy()
    df["region"] = region_labels(df[name_col])
    return df, name_col


def build_assignment_table(df: pd.DataFrame, name_col: str) -> pd.DataFrame:
    """One row per country: its subregion (= held-out fold) and coverage."""
    img_counts = df.groupby([name_col, "region"]).size().rename("n_images")
    loc_counts = (
        df.drop_duplicates(["lat", "lon"]).groupby([name_col, "region"]).size().rename("n_locations")
    )
    tbl = pd.concat([img_counts, loc_counts], axis=1).reset_index()
    tbl = tbl.rename(columns={name_col: "country"})
    tbl["fold"] = tbl["region"]
    region_rank = {r: i for i, r in enumerate(REGION_ORDER)}
    tbl = tbl.sort_values(
        by=["region", "country"], key=lambda s: s.map(region_rank) if s.name == "region" else s
    ).reset_index(drop=True)
    return tbl[["fold", "region", "country", "n_locations", "n_images"]]


def draw_continent_outline(ax, shapefile: Path) -> None:
    """Faint African landmass outline for context; skipped if it cannot be read."""
    if not shapefile.exists():
        print(f"[warn] shapefile not found, drawing scatter only: {shapefile}")
        return
    try:
        import geopandas as gpd

        world = gpd.read_file(shapefile)
        continent_col = next((c for c in ("CONTINENT", "REGION_UN") if c in world.columns), None)
        africa = world[world[continent_col] == "Africa"] if continent_col else world
        africa.boundary.plot(ax=ax, color="#9aa5b1", linewidth=0.6, zorder=1)
    except Exception as exc:  # outline is decorative; never fail the figure for it
        print(f"[warn] could not draw continent outline ({exc}); scatter only")


def main() -> int:
    args = parse_args()
    df, name_col = load_manifest(args.manifest)
    locs = df.drop_duplicates(["lat", "lon"]).reset_index(drop=True)

    tbl = build_assignment_table(df, name_col)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(args.table, index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "figure.dpi": 150,
        "savefig.dpi": 200,
    })
    fig, ax = plt.subplots(figsize=(7.2, 7.4))
    ax.set_facecolor("#eef2f7")

    draw_continent_outline(ax, args.shapefile)

    counts = {r: 0 for r in REGION_ORDER}
    for region in REGION_ORDER:
        sub = locs[locs["region"] == region]
        counts[region] = len(sub)
        ax.scatter(
            sub["lon"], sub["lat"],
            s=3.0, c=REGION_COLORS[region], alpha=0.55, linewidth=0,
            rasterized=True, zorder=2, label=region,
        )

    pad = 2.0
    ax.set_xlim(locs["lon"].min() - pad, locs["lon"].max() + pad)
    ax.set_ylim(locs["lat"].min() - pad, locs["lat"].max() + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Region-based 5-fold cross-validation\n(UN African subregions, leave-one-region-out)")
    # No lat/lon grid or axis frame, to match the other distribution maps.
    ax.set_axis_off()

    n_countries = tbl.groupby("region")["country"].nunique().to_dict()
    handles = [
        Patch(
            facecolor=REGION_COLORS[r], edgecolor="white",
            label=f"{r} ({n_countries.get(r, 0)} countries, {counts[r]:,} loc.)",
        )
        for r in REGION_ORDER
    ]
    ax.legend(handles=handles, title="Held-out subregion", loc="lower left", frameon=True, fontsize=8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")
    print(f"wrote {args.table}")
    print(f"{len(locs):,} unique locations across {len(REGION_ORDER)} subregions "
          f"and {tbl['country'].nunique()} countries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
