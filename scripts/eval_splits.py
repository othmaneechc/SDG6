#!/usr/bin/env python3
"""Evaluate k-NN under different split schemes on pre-extracted embeddings.

Reviewer 1 point 4: the released split shuffles survey locations at random, so a
test location can sit beside a training location. Given how strongly settlement
structure is spatially autocorrelated, that can inflate held-out AUROC, and the
paper's claim is transfer to *unsampled* areas and countries. This script scores
the same encoder under:

    original  the released 80/10/10 split (train -> test), for continuity
    random    K-fold over locations, shuffled at random
    spatial   K-fold over spatial blocks, so whole neighbourhoods are held out
    country   leave-one-country-out, matching the transfer claim

All schemes share one code path and the corrected k-NN scoring, so differences
are attributable to the split and not to the metric. Folds are over *locations*,
never images, so repeat imagery of one place cannot straddle a fold boundary.

Usage:
    python scripts/eval_splits.py --scheme spatial --task pw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sdg6.knn import _knn_softmax_vote_with_probs  # noqa: E402

K_VALUES = [5, 10, 20, 50, 100, 200]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb-dir", type=Path, default=REPO / "runs" / "embeddings" / "dinov2")
    p.add_argument("--model", default="dinov2")
    p.add_argument("--task", choices=["pw", "sw"], default="pw")
    p.add_argument("--scheme", choices=["original", "random", "spatial", "country"],
                   default="spatial")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--block-deg", type=float, default=0.5,
                   help="Spatial block size in degrees (0.5 deg ~ 55 km).")
    p.add_argument("--min-test", type=int, default=200,
                   help="Skip folds with fewer than this many test images.")
    p.add_argument("--temp", type=float, default=0.07)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=REPO / "outputs" / "tables" / "split_scheme_auroc.csv")
    return p.parse_args()


def load_embeddings(emb_dir: Path) -> pd.DataFrame:
    frames, feats = [], []
    for split in ("train", "val", "test"):
        f = emb_dir / f"{split}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        feats.append(d["features"])
        frames.append(pd.DataFrame({
            "path": d["paths"], "orig_split": split,
            "pw": d["pw_label"], "sw": d["sw_label"],
            "lat": d["lat"], "lon": d["lon"],
        }))
    if not frames:
        raise SystemExit(f"No embeddings under {emb_dir}")
    return pd.concat(frames, ignore_index=True), np.concatenate(feats, axis=0)


def assign_folds(df: pd.DataFrame, scheme: str, n_folds: int,
                 block_deg: float, seed: int) -> np.ndarray:
    """Fold id per row. Grouping is always at location (or coarser) level."""
    rng = np.random.default_rng(seed)
    if scheme == "random":
        keys = list(zip(df["lat"], df["lon"]))
    elif scheme == "spatial":
        keys = list(zip(np.floor(df["lat"] / block_deg).astype(int),
                        np.floor(df["lon"] / block_deg).astype(int)))
    elif scheme == "country":
        keys = list(df["country"])
    else:
        raise ValueError(scheme)

    uniq = sorted(set(keys), key=lambda x: str(x))
    if scheme == "country":
        mapping = {g: i for i, g in enumerate(uniq)}   # one fold per country
    else:
        order = rng.permutation(len(uniq))
        mapping = {g: int(order[i] % n_folds) for i, g in enumerate(uniq)}
    return np.array([mapping[k] for k in keys], dtype=int)


def knn_auroc(train_X, train_y, test_X, test_y, temp: float) -> dict[int, float]:
    """Corrected k-NN scoring -> AUROC on P(y=1) for each k."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr = torch.nn.functional.normalize(torch.as_tensor(train_X, device=dev, dtype=torch.float32), dim=1)
    Xte = torch.nn.functional.normalize(torch.as_tensor(test_X, device=dev, dtype=torch.float32), dim=1)
    ytr = torch.as_tensor(train_y, device=dev, dtype=torch.long)

    ks = [k for k in K_VALUES if k <= len(train_y)]
    out = _knn_softmax_vote_with_probs(
        Xtr, ytr, Xte, num_classes=2, k_values=ks, temperature=temp)
    res = {}
    for k, (_, _, probs) in out.items():
        res[k] = float(roc_auc_score(test_y, probs[:, 1])) if len(np.unique(test_y)) > 1 else float("nan")
    return res


def main() -> int:
    args = parse_args()
    df, X = load_embeddings(args.emb_dir)

    # country/urbrur live in the manifest, not the embedding files
    man = pd.read_csv(REPO / "data" / "manifest_sentinel.csv")
    # country identity for LOCO: the reverse-geocoded real country, since the
    # raw Afrobarometer code is per-round and not comparable across rounds.
    cc = "country_name" if "country_name" in man.columns else "country"
    man = man[["path", cc, "urbrur"]].rename(columns={cc: "country"})
    df = df.merge(man, on="path", how="left")
    if df["country"].isna().any():
        raise SystemExit("Some rows have no country; rebuild the manifest.")
    print(f"loaded {len(df)} embeddings, dim={X.shape[1]}")

    y = df[args.task].to_numpy()
    rows = []

    if args.scheme == "original":
        tr = (df["orig_split"] == "train").to_numpy()
        te = (df["orig_split"] == "test").to_numpy()
        for k, auc in knn_auroc(X[tr], y[tr], X[te], y[te], args.temp).items():
            rows.append({"model": args.model, "scheme": "original", "task": args.task, "fold": 0,
                         "k": k, "n_train": int(tr.sum()), "n_test": int(te.sum()),
                         "auroc": auc})
            print(f"  original k={k:<4} auroc={auc*100:.2f}%")
    else:
        folds = assign_folds(df, args.scheme, args.folds, args.block_deg, args.seed)
        for f in sorted(set(folds)):
            te = folds == f
            tr = ~te
            if te.sum() < args.min_test or len(np.unique(y[te])) < 2 or tr.sum() < max(K_VALUES):
                continue
            aucs = knn_auroc(X[tr], y[tr], X[te], y[te], args.temp)
            label = df.loc[te, "country"].iloc[0] if args.scheme == "country" else f
            for k, auc in aucs.items():
                rows.append({"model": args.model, "scheme": args.scheme, "task": args.task, "fold": label,
                             "k": k, "n_train": int(tr.sum()), "n_test": int(te.sum()),
                             "auroc": auc})
            print(f"  {args.scheme} fold={label} n_test={int(te.sum()):>6} "
                  f"auroc@200={aucs.get(200, float('nan'))*100:.2f}%")

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = not args.out.exists()
    out.to_csv(args.out, mode="a", header=header, index=False)
    print(f"\nwrote {len(out)} rows -> {args.out}")

    if not out.empty:
        best = out[out["k"] == out["k"].max()]
        print(f"summary ({args.scheme}, {args.task}, k={int(best['k'].iloc[0])}): "
              f"mean AUROC={best['auroc'].mean()*100:.2f}%  "
              f"median={best['auroc'].median()*100:.2f}%  n_folds={len(best)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
