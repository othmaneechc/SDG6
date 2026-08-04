#!/usr/bin/env python3
"""Urban/rural baseline and stratified AUROC.

Access is strongly correlated with settlement type, so this script separates
imagery signal from the urban/rural shortcut. It runs two checks on the same
folds and corrected k-NN scoring:

  marginal    AUROC of a trivial classifier using ONLY the survey's urban/rural
              indicator against AUROC of the image embeddings. Answers
              "how much does imagery beat the
              confounder?"

  stratified  AUROC of the embeddings computed WITHIN urban areas and WITHIN
              rural areas separately. If discrimination collapses inside a
              stratum, the model was largely separating urban from rural; if it
              survives, the representation carries information beyond settlement
              type. This is the stronger of the two tests.

The urban/rural baseline is fit exactly like any other model: class frequencies
are estimated on the training fold only and applied to the held-out fold, so it
never sees test labels.

Usage:
    python scripts/eval_baselines.py --scheme spatial --task pw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sdg6.knn import _knn_softmax_vote_with_probs  # noqa: E402
from sdg6.splits import (  # noqa: E402
    assign_balanced_group_folds,
    location_keys,
    spatial_block_keys,
)

K_EVAL = 200
URBAN, RURAL, SEMI = 1.0, 2.0, 3.0
VALID_URBRUR = {URBAN, RURAL, SEMI}   # 460.0 is an out-of-range junk code


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb-dir", type=Path, default=REPO / "runs" / "embeddings" / "dinov2")
    p.add_argument("--model", default="dinov2")
    p.add_argument("--task", choices=["pw", "sw"], default="pw")
    p.add_argument("--scheme", choices=["random", "spatial"], default="random")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--block-deg", type=float, default=0.5)
    p.add_argument("--grid-root", type=Path,
                   default=REPO / "data" / "meta_pop_data" / "countries_2x2",
                   help="Meta population grid root, for the population-density baseline.")
    p.add_argument("--temp", type=float, default=0.07)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-n", type=int, default=200)
    p.add_argument("--out", type=Path,
                   default=REPO / "outputs" / "tables" / "baseline_stratified_auroc.csv")
    return p.parse_args()


def load_all(emb_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frames, feats = [], []
    for split in ("train", "val", "test"):
        f = emb_dir / f"{split}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        feats.append(d["features"])
        frames.append(pd.DataFrame({
            "path": d["paths"], "pw": d["pw_label"], "sw": d["sw_label"],
            "lat": d["lat"], "lon": d["lon"],
        }))
    df = pd.concat(frames, ignore_index=True)
    man = pd.read_csv(REPO / "data" / "manifest_sentinel.csv",
                      usecols=["path", "country", "country_name", "urbrur"])
    df = df.merge(man, on="path", how="left")
    return df, np.concatenate(feats, axis=0)


def assign_folds(df, scheme, n_folds, block_deg, seed):
    if scheme == "random":
        keys = location_keys(df["lat"], df["lon"])
    else:
        keys = spatial_block_keys(df["lat"], df["lon"], block_deg)
    return assign_balanced_group_folds(keys, n_folds, seed)


def join_population(df: pd.DataFrame, grid_root: Path) -> np.ndarray:
    """Nearest-Meta-tile total population per survey location (NaN if no grid file).

    Provides the continuous population-density covariate for the confound
    baselines, taken from the same Meta grid the framework uses for inference.
    """
    from scipy.spatial import cKDTree

    pop = np.full(len(df), np.nan)
    lon = df["lon"].to_numpy()
    lat = df["lat"].to_numpy()
    for country, sub_idx in df.groupby("country_name").indices.items():
        cdir = grid_root / str(country)
        files = sorted(cdir.glob("*_general_2020_tiles.csv")) if cdir.is_dir() else []
        if not files:
            continue
        tiles = pd.read_csv(files[0], usecols=["centroid_lon", "centroid_lat", "total_population"]).dropna()
        if tiles.empty:
            continue
        tree = cKDTree(tiles[["centroid_lon", "centroid_lat"]].to_numpy())
        _, nn = tree.query(np.column_stack([lon[sub_idx], lat[sub_idx]]), k=1)
        pop[sub_idx] = tiles["total_population"].to_numpy()[nn]
    return pop


def knn_probs(train_X, train_y, test_X, temp) -> np.ndarray:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr = torch.nn.functional.normalize(torch.as_tensor(train_X, device=dev, dtype=torch.float32), dim=1)
    Xte = torch.nn.functional.normalize(torch.as_tensor(test_X, device=dev, dtype=torch.float32), dim=1)
    ytr = torch.as_tensor(train_y, device=dev, dtype=torch.long)
    out = _knn_softmax_vote_with_probs(Xtr, ytr, Xte, num_classes=2,
                                       k_values=[K_EVAL], temperature=temp)
    return out[K_EVAL][2][:, 1]


def safe_auroc(y, s) -> float:
    y = np.asarray(y)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def main() -> int:
    args = parse_args()
    df, X = load_all(args.emb_dir)
    keep = df["urbrur"].isin(VALID_URBRUR).to_numpy()
    dropped = int((~keep).sum())
    df, X = df[keep].reset_index(drop=True), X[keep]
    print(f"{len(df)} rows (dropped {dropped} with invalid urbrur codes)")

    pop = join_population(df, args.grid_root)
    has_pop = ~np.isnan(pop)
    log_pop = np.log1p(np.clip(np.nan_to_num(pop, nan=0.0), 0, None))
    print(f"population joined for {int(has_pop.sum())}/{len(df)} locations "
          f"({100 * has_pop.mean():.1f}% coverage); "
          f"missing: {sorted(df.loc[~has_pop, 'country_name'].unique())}")

    y = df[args.task].to_numpy()
    ur = df["urbrur"].to_numpy()
    folds = assign_folds(df, args.scheme, args.folds, args.block_deg, args.seed)
    rows = []

    for f in sorted(set(folds)):
        te, tr = folds == f, folds != f
        if te.sum() < args.min_n or len(np.unique(y[te])) < 2:
            continue

        p_img = knn_probs(X[tr], y[tr], X[te], args.temp)

        # P(access | urban/rural), with frequencies estimated from train only.
        prior = float(y[tr].mean())
        rates = {u: float(y[tr & (ur == u)].mean()) if (tr & (ur == u)).sum() else prior
                 for u in VALID_URBRUR}
        p_ur = np.array([rates[u] for u in ur[te]])

        def rec(features, stratum, mask):
            s_img = p_img[mask] if features == "dinov2" else p_ur[mask]
            rows.append({
                "scheme": args.scheme, "task": args.task, "fold": int(f),
                "features": features, "stratum": stratum,
                "n": int(mask.sum()), "pos_rate": float(y[te][mask].mean()),
                "auroc": safe_auroc(y[te][mask], s_img),
            })

        allm = np.ones(int(te.sum()), dtype=bool)
        urb = ur[te] == URBAN
        rur = ur[te] == RURAL
        for feats in ("dinov2", "urbrur"):
            rec(feats, "all", allm)
            if urb.sum() >= args.min_n:
                rec(feats, "urban", urb)
            if rur.sum() >= args.min_n:
                rec(feats, "rural", rur)

        # Continuous-covariate confound baselines (overall stratum only), on the
        # density-available subset of the held-out fold; fit on train, scored on test.
        pop_te = has_pop[te]
        if pop_te.sum() >= args.min_n and len(np.unique(y[te][pop_te])) > 1:
            rows.append({
                "scheme": args.scheme, "task": args.task, "fold": int(f),
                "features": "density", "stratum": "all",
                "n": int(pop_te.sum()), "pos_rate": float(y[te][pop_te].mean()),
                "auroc": safe_auroc(y[te][pop_te], log_pop[te][pop_te]),
            })
            tr_p, te_p = tr & has_pop, te & has_pop
            if len(np.unique(y[tr_p])) > 1:
                def dummies(u):
                    return pd.get_dummies(u).reindex(
                        columns=sorted(VALID_URBRUR), fill_value=0).to_numpy(dtype=float)
                scaler = StandardScaler().fit(log_pop[tr_p].reshape(-1, 1))
                Xtr = np.column_stack([dummies(ur[tr_p]), scaler.transform(log_pop[tr_p].reshape(-1, 1))])
                Xte = np.column_stack([dummies(ur[te_p]), scaler.transform(log_pop[te_p].reshape(-1, 1))])
                lr = LogisticRegression(max_iter=1000).fit(Xtr, y[tr_p])
                rows.append({
                    "scheme": args.scheme, "task": args.task, "fold": int(f),
                    "features": "urbrur_density", "stratum": "all",
                    "n": int(te_p.sum()), "pos_rate": float(y[te_p].mean()),
                    "auroc": safe_auroc(y[te_p], lr.predict_proba(Xte)[:, 1]),
                })

        got = {(r["features"], r["stratum"]): r["auroc"] for r in rows if r["fold"] == f}
        print(f"  fold {f}: img_all={got.get(('dinov2','all'),float('nan'))*100:.2f}% "
              f"ur_all={got.get(('urbrur','all'),float('nan'))*100:.2f}% | "
              f"img_urban={got.get(('dinov2','urban'),float('nan'))*100:.2f}% "
              f"img_rural={got.get(('dinov2','rural'),float('nan'))*100:.2f}%")

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, mode="a", header=not args.out.exists(), index=False)
    print(f"\nwrote {len(out)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
