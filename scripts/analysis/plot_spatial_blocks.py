#!/usr/bin/env python3
"""Plot the spatial blocks used by the spatial cross-validation experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from sdg6.splits import assign_balanced_group_folds, spatial_block_keys  # noqa: E402


PALETTE = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=REPO / "data" / "manifest_sentinel.csv")
    p.add_argument("--block-deg", type=float, default=0.5)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=REPO / "outputs" / "figures" / "spatial_blocks_cv.png")
    p.add_argument("--show-locations", action="store_true")
    p.add_argument(
        "--block-table",
        type=Path,
        default=REPO / "outputs" / "tables" / "spatial_blocks_cv.csv",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.manifest, usecols=["lat", "lon"]).dropna()
    locs = df.drop_duplicates(["lat", "lon"]).reset_index(drop=True)
    keys = spatial_block_keys(locs["lat"], locs["lon"], args.block_deg)

    blocks = pd.DataFrame(keys, columns=["block_lat", "block_lon"]).drop_duplicates()
    block_keys = list(zip(blocks["block_lat"], blocks["block_lon"]))
    blocks["fold"] = assign_balanced_group_folds(block_keys, args.folds, args.seed)
    blocks["lat_min"] = blocks["block_lat"] * args.block_deg
    blocks["lat_max"] = blocks["lat_min"] + args.block_deg
    blocks["lon_min"] = blocks["block_lon"] * args.block_deg
    blocks["lon_max"] = blocks["lon_min"] + args.block_deg
    blocks = blocks.sort_values(["fold", "block_lat", "block_lon"]).reset_index(drop=True)

    args.block_table.parent.mkdir(parents=True, exist_ok=True)
    blocks.to_csv(args.block_table, index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "figure.dpi": 150,
        "savefig.dpi": 200,
    })
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    ax.set_facecolor("#eef2f7")

    lat0 = int(blocks["block_lat"].min())
    lat1 = int(blocks["block_lat"].max())
    lon0 = int(blocks["block_lon"].min())
    lon1 = int(blocks["block_lon"].max())
    grid = np.full((lat1 - lat0 + 1, lon1 - lon0 + 1), np.nan)
    for row in blocks.itertuples(index=False):
        grid[int(row.block_lat) - lat0, int(row.block_lon) - lon0] = int(row.fold)

    fold_colors = [PALETTE[i % len(PALETTE)] for i in range(args.folds)]
    cmap = ListedColormap(fold_colors)
    cmap.set_bad((1, 1, 1, 0))
    norm = BoundaryNorm(np.arange(-0.5, args.folds + 0.5, 1), cmap.N)
    ax.imshow(
        np.ma.masked_invalid(grid),
        origin="lower",
        extent=[
            lon0 * args.block_deg,
            (lon1 + 1) * args.block_deg,
            lat0 * args.block_deg,
            (lat1 + 1) * args.block_deg,
        ],
        cmap=cmap,
        norm=norm,
        alpha=0.72,
        interpolation="nearest",
        zorder=1,
    )

    if args.show_locations:
        ax.scatter(
            locs["lon"],
            locs["lat"],
            s=1.6,
            c="#111827",
            alpha=0.18,
            linewidth=0,
            rasterized=True,
            zorder=2,
        )

    lon_pad = 2.0
    lat_pad = 2.0
    ax.set_xlim(locs["lon"].min() - lon_pad, locs["lon"].max() + lon_pad)
    ax.set_ylim(locs["lat"].min() - lat_pad, locs["lat"].max() + lat_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Spatial-block 5-fold cross-validation ({args.block_deg:g} deg grid)")
    ax.grid(color="white", linewidth=0.5, alpha=0.7)

    handles = [
        Patch(facecolor=fold_colors[i], edgecolor="white", alpha=0.55, label=f"Fold {i}")
        for i in range(args.folds)
    ]
    ax.legend(handles=handles, title="Held-out fold", loc="lower left", frameon=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")
    print(f"wrote {args.block_table}")
    print(f"{len(locs):,} unique locations across {len(blocks):,} spatial blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
