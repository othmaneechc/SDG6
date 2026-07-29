#!/usr/bin/env python3
"""Urban/rural baseline and stratified AUROC (Reviewer 1, point 1).

The reviewer's decisive objection is that the embeddings may be detecting
urbanisation rather than infrastructure, since access is strongly confounded
with settlement type. Two complementary tests, both on the same folds and the
same corrected k-NN scoring:

  marginal    AUROC of a trivial classifier using ONLY the survey's urban/rural
              indicator -- baseline (iv) in the review -- against AUROC of the
              image embeddings. Answers "how much does imagery beat the
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
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sdg6.knn import _knn_softmax_vote_with_probs  # noqa: E402

K_EVAL = 200
URBAN, RURAL, SEMI = 1.0, 2.0, 3.0
VALID_URBRUR = {URBAN, RURAL, SEMI}   # 460.0 is an out-of-range junk code


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb-dir", type=Path, default=REPO / "runs" / "embeddings" / "dinov2")
    p.add_argument("--model", default="dinov2")
    p.add_argument("--task", choices=["pw", "sw"], default="pw")
    p.add_argument("--scheme", choices=["random", "spatial"], default="spatial")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--block-deg", type=float, default=0.5)
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
                      usecols=["path", "country", "urbrur"])
    df = df.merge(man, on="path", how="left")
    return df, np.concatenate(feats, axis=0)


def assign_folds(df, scheme, n_folds, block_deg, seed):
    rng = np.random.default_rng(seed)
    if scheme == "random":
        keys = list(zip(df["lat"], df["lon"]))
    else:
        keys = list(zip(np.floor(df["lat"] / block_deg).astype(int),
                        np.floor(df["lon"] / block_deg).astype(int)))
    uniq = sorted(set(keys))
    order = rng.permutation(len(uniq))
    mapping = {g: int(order[i] % n_folds) for i, g in enumerate(uniq)}
    return np.array([mapping[k] for k in keys], dtype=int)


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

    y = df[args.task].to_numpy()
    ur = df["urbrur"].to_numpy()
    folds = assign_folds(df, args.scheme, args.folds, args.block_deg, args.seed)
    rows = []

    for f in sorted(set(folds)):
        te, tr = folds == f, folds != f
        if te.sum() < args.min_n or len(np.unique(y[te])) < 2:
            continue

        p_img = knn_probs(X[tr], y[tr], X[te], args.temp)

        # Baseline (iv): P(access | urban/rural), frequencies from TRAIN only.
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
