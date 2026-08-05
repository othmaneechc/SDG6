#!/usr/bin/env python3
"""Uncertainty on the LGA-level burden estimates.

The paper reports burden figures as bare point estimates -- 1.155 and 1.452
million people in the highest-burden LGAs, top-decile shares of 21.2% and 20.9%
-- with no interval. Two things drive the uncertainty and both are propagated
here:

  sampling     which tiles fall in an LGA, and what the model says about them.
               Resampled with replacement within each LGA.

  calibration  the mapping from raw k-NN score to probability is itself an
               estimate fit on a finite validation set. Refit on a bootstrap
               resample of validation in every replicate, so its wobble is
               carried through rather than treated as exact.

Burden is population x P(no access), summed within an LGA, so it inherits
uncertainty from the probability scale directly -- which is exactly why the
calibration fix has to come first.

Usage:
    python scripts/burden_uncertainty.py --task pw --n-boot 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

EPS = 1e-6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["pw", "sw"], default="pw")
    p.add_argument("--country", default="Nigeria")
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--calibrator", choices=["isotonic", "platt", "temperature", "none"],
                   default="isotonic")
    p.add_argument("--emb-dir", type=Path, default=REPO / "runs" / "embeddings" / "dinov2")
    p.add_argument("--data-root", type=Path, default=REPO / "data")
    p.add_argument("--boundaries", type=Path,
                   default=REPO / "data" / "admin_boundaries" / "gadm41_NGA_2.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=REPO / "outputs" / "tables" / "burden_uncertainty.csv")
    return p.parse_args()


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def parse_tile_lonlat(path_series: pd.Series) -> pd.DataFrame:
    """Tile filenames encode sentinel_<lat>_<lon>.tif."""
    stem = path_series.str.rsplit("/", n=1).str[-1].str.replace(".tif", "", regex=False)
    parts = stem.str.split("_")
    return pd.DataFrame({
        "lat": pd.to_numeric(parts.str[1], errors="coerce"),
        "lon": pd.to_numeric(parts.str[2], errors="coerce"),
    })


def access_prob(df: pd.DataFrame, k: int) -> np.ndarray:
    """Reconstruct P(access) from the max-class probability and predicted class.

    Valid only because the analysis uses k == max_k, where the two class scores
    sum to 1. At smaller k they do not, and 1 - p would be wrong.
    """
    p = pd.to_numeric(df[f"prob_k{k}"], errors="coerce").clip(0, 1).to_numpy()
    cls = df[f"pred_class_k{k}"].astype(str).str.lower()
    no_access = cls.str.startswith("no").to_numpy()
    return np.where(no_access, 1.0 - p, p)


def load_calibrator(kind: str, task: str, out_dir: Path):
    if kind == "none":
        return lambda p: p
    f = out_dir / f"calibrator_{task}.npz"
    if not f.exists():
        raise SystemExit(f"missing {f}; run scripts/eval_calibration.py --task {task} first")
    d = np.load(f)
    if kind == "temperature":
        T = float(d["temperature"][0])
        return lambda p: sigmoid(_logit(p) / T)
    if kind == "platt":
        a, b = float(d["platt_a"][0]), float(d["platt_b"][0])
        return lambda p: sigmoid(a * _logit(p) + b)
    xs, ys = d["iso_x"], d["iso_y"]
    return lambda p: np.interp(np.clip(p, 0, 1), xs, ys)


def fit_isotonic_boot(p_val, y_val, rng):
    """Refit isotonic on a bootstrap resample of validation."""
    from sklearn.isotonic import IsotonicRegression
    idx = rng.integers(0, len(p_val), len(p_val))
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_val[idx], y_val[idx])
    return lambda p: iso.predict(np.clip(p, 0, 1))


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    import geopandas as gpd
    from shapely.geometry import Point

    svc = "pw" if args.task == "pw" else "sw"
    pred_csv = args.data_root / "inference" / args.country / f"{svc}_predictions.csv"
    preds = pd.read_csv(pred_csv)
    preds = pd.concat([preds, parse_tile_lonlat(preds["path"])], axis=1).dropna(subset=["lat", "lon"])
    preds["p_access_raw"] = access_prob(preds, args.k)
    print(f"{len(preds)} {args.country} tiles with predictions (k={args.k})")

    tiles = pd.read_csv(args.data_root / "meta_pop_data" / "countries_2x2" /
                        args.country / "nga_general_2020_tiles.csv",
                        usecols=["centroid_lon", "centroid_lat", "total_population"])
    tiles = tiles.dropna()

    # Join each tile's population to its prediction on the shared 2 km grid,
    # matching plot_nigeria_access_hotspots.py (round to 5 decimals, tile-based),
    # so the bootstrap point burden equals the value reported in the figure/table.
    preds["kx"] = preds["lon"].round(5)
    preds["ky"] = preds["lat"].round(5)
    tiles["kx"] = tiles["centroid_lon"].round(5)
    tiles["ky"] = tiles["centroid_lat"].round(5)
    merged = tiles.merge(preds[["kx", "ky", "p_access_raw"]], on=["kx", "ky"], how="inner")
    merged = merged.rename(columns={"centroid_lon": "lon", "centroid_lat": "lat"})
    print(f"  joined predictions for {len(merged)}/{len(tiles)} tiles")
    if merged.empty:
        raise SystemExit("population join produced no rows; check the centroid grids")
    merged["population"] = merged["total_population"].clip(lower=0)

    gdf = gpd.GeoDataFrame(
        merged, geometry=[Point(xy) for xy in zip(merged["lon"], merged["lat"])], crs="EPSG:4326")
    admin = gpd.read_file(args.boundaries)[["NAME_2", "geometry"]].rename(columns={"NAME_2": "lga"})
    gdf = gpd.sjoin(gdf, admin, how="inner", predicate="within").drop(columns=["index_right"])
    # sjoin preserves the pre-join labels; reset so that label and positional
    # indices agree, since the bootstrap indexes numpy arrays positionally.
    gdf = gdf.reset_index(drop=True)
    print(f"  {gdf['lga'].nunique()} LGAs, {len(gdf)} tiles inside boundaries")

    # Validation scores, for refitting the calibrator inside the bootstrap.
    vp = out_dir / f"valprobs_{args.task}.npy"
    vy = out_dir / f"valy_{args.task}.npy"
    p_val = np.load(vp) if vp.exists() else None
    y_val = np.load(vy) if vy.exists() else None
    if p_val is None or y_val is None:
        print("  [warn] no cached val scores; calibration uncertainty NOT propagated "
              "(interval reflects sampling only)")
        p_val = y_val = None

    calib = load_calibrator(args.calibrator, args.task, out_dir)
    gdf["p_access"] = np.clip(calib(gdf["p_access_raw"].to_numpy()), 0, 1)

    def lga_burden(frame, p_col):
        g = frame.groupby("lga")
        pop = g["population"].sum()
        acc = g.apply(lambda x: np.average(x[p_col], weights=x["population"])
                      if x["population"].sum() > 0 else np.nan, include_groups=False)
        return pop * (1.0 - acc)

    point = lga_burden(gdf, "p_access").rename("burden")
    raw_point = lga_burden(gdf.assign(p_raw=gdf["p_access_raw"]), "p_raw").rename("burden_raw")

    # Bootstrap: resample tiles within LGA, and refit the calibrator each replicate.
    lgas = point.index.to_numpy()
    boot = np.full((args.n_boot, len(lgas)), np.nan)
    # .indices gives POSITIONAL indices, which is what the numpy arrays below need.
    groups = gdf.groupby("lga").indices
    pop_all = gdf["population"].to_numpy()
    praw_all = gdf["p_access_raw"].to_numpy()

    for b in range(args.n_boot):
        cal_b = (fit_isotonic_boot(p_val, y_val, rng)
                 if (p_val is not None and args.calibrator == "isotonic") else calib)
        for i, lga in enumerate(lgas):
            idx = groups[lga]
            take = rng.choice(idx, size=len(idx), replace=True)
            w = pop_all[take]
            if w.sum() <= 0:
                continue
            p = np.clip(cal_b(praw_all[take]), 0, 1)
            boot[b, i] = w.sum() * (1.0 - np.average(p, weights=w))
        if (b + 1) % max(1, args.n_boot // 5) == 0:
            print(f"  bootstrap {b+1}/{args.n_boot}")

    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)
    res = pd.DataFrame({
        "task": args.task, "country": args.country, "lga": lgas,
        "burden": point.to_numpy(), "burden_raw_uncalibrated": raw_point.reindex(lgas).to_numpy(),
        "ci_lo": lo, "ci_hi": hi,
        "population": gdf.groupby("lga")["population"].sum().reindex(lgas).to_numpy(),
    }).sort_values("burden", ascending=False)
    res.to_csv(args.out, mode="a", header=not args.out.exists(), index=False)

    top = res.head(5)
    print(f"\n  highest-burden LGAs ({args.task}, calibrator={args.calibrator}):")
    for r in top.itertuples():
        print(f"    {r.lga:<24} {r.burden/1e6:6.3f} M  [{r.ci_lo/1e6:5.3f}-{r.ci_hi/1e6:5.3f}]  "
              f"(uncalibrated {r.burden_raw_uncalibrated/1e6:6.3f} M)")
    tot = res["burden"].sum()
    dec = res.nlargest(max(1, len(res) // 10), "burden")["burden"].sum()
    print(f"\n  total burden {tot/1e6:.3f} M; top-decile share {dec/tot*100:.1f}%")

    # Bootstrap CIs for the across-LGA summary statistics used in the burden text.
    med_b = np.nanmedian(boot, axis=1)
    ratio_b = np.nanmax(boot, axis=1) / np.nanmean(boot, axis=1)
    med_lo, med_hi = np.nanpercentile(med_b, [2.5, 97.5])
    ratio_lo, ratio_hi = np.nanpercentile(ratio_b, [2.5, 97.5])
    print(f"  median LGA burden {point.median() / 1e6:.3f} M  "
          f"95% CI [{med_lo / 1e6:.3f}, {med_hi / 1e6:.3f}] M")
    print(f"  max/mean ratio {point.max() / point.mean():.2f}  "
          f"95% CI [{ratio_lo:.2f}, {ratio_hi:.2f}]")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
