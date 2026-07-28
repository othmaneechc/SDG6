#!/usr/bin/env python3
"""Full held-out metrics (accuracy, macro recall, macro F1, AUROC) at a given k.

Used to complete Table 1 at the matched neighborhood size k=200. Hard predictions
are argmax over the k-NN class probabilities and are invariant to the per-k
normalization fix, so accuracy/recall/F1 at k=200 are well defined; AUROC uses the
corrected P(y=1).

Reports every requested k so the run can be validated against the published table
(e.g. DINO at k=100 must reproduce the existing row) before the k=200 numbers are
trusted.

    python scripts/eval_full_metrics.py --emb-dir runs/embeddings/dino --k 100 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             roc_auc_score)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sdg6.knn import _knn_softmax_vote_with_probs  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb-dir", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", default=["pw", "sw"])
    p.add_argument("--k", nargs="+", type=int, default=[100, 200])
    p.add_argument("--temp", type=float, default=0.07)
    return p.parse_args()


def load(emb_dir: Path, split: str):
    d = np.load(emb_dir / f"{split}.npz", allow_pickle=True)
    return d["features"], d


def main() -> int:
    args = parse_args()
    Xtr, dtr = load(args.emb_dir, "train")
    Xte, dte = load(args.emb_dir, "test")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.nn.functional.normalize(torch.as_tensor(Xtr, device=dev, dtype=torch.float32), dim=1)
    B = torch.nn.functional.normalize(torch.as_tensor(Xte, device=dev, dtype=torch.float32), dim=1)

    print(f"model dir: {args.emb_dir.name}  train={len(Xtr)} test={len(Xte)}")
    print(f"{'task':<5} {'k':>4} {'Accuracy':>9} {'Recall':>8} {'F1':>8} {'AUROC':>8}")
    for task in args.tasks:
        ytr = dtr[f"{task}_label"].astype(np.int64)
        yte = dte[f"{task}_label"].astype(np.int64)
        y = torch.as_tensor(ytr, device=dev, dtype=torch.long)
        out = _knn_softmax_vote_with_probs(
            A, y, B, num_classes=2, k_values=list(args.k), temperature=args.temp)
        for k in args.k:
            preds, _, probs = out[k]
            acc = accuracy_score(yte, preds) * 100
            rec = recall_score(yte, preds, average="macro", zero_division=0) * 100
            f1 = f1_score(yte, preds, average="macro", zero_division=0) * 100
            auc = roc_auc_score(yte, probs[:, 1]) * 100
            print(f"{task:<5} {k:>4} {acc:>8.2f} {rec:>7.2f} {f1:>7.2f} {auc:>7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
