#!/usr/bin/env python3
"""Calibration of the access probabilities.

The paper reports both ranking quality and probability-scale quantities. These
are different things:

  ranking      does a higher score mean a higher chance of access?  (AUROC)
  calibration  when the model says 0.8, is it right 80% of the time? (Brier/ECE)

AUROC is invariant to any monotone transform of the score, so a model can rank
almost perfectly and still be badly calibrated. The severity thresholds in the
paper (p >= 0.805, p >= 0.952) are cuts on the probability scale, so if that
scale is distorted the thresholds are not cutting where they claim to.

This script measures calibration first, without fitting anything, then compares
four corrections. Every calibrator is fit on the validation split and reported on
the test split; fitting and reporting on the same data manufactures a good
reliability diagram that means nothing.

  none         raw k-NN vote fractions
  temperature  p' = sigmoid(logit(p) / T), one parameter, preserves ranking
  platt        p' = sigmoid(a*logit(p) + b), two parameters, preserves ranking
  isotonic     monotone step function, non-parametric, may reorder ties
  logistic     a logistic head trained directly on the embeddings, which is
               natively calibrated because it optimises a proper scoring rule

Usage:
    python scripts/eval_calibration.py --task pw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sdg6.knn import _knn_softmax_vote_with_probs  # noqa: E402

EPS = 1e-6
K_EVAL = 200


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb-dir", type=Path, default=REPO / "runs" / "embeddings" / "dinov2")
    p.add_argument("--task", choices=["pw", "sw"], default="pw")
    p.add_argument("--temp", type=float, default=0.07)
    p.add_argument("--bins", type=int, default=15)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "tables")
    return p.parse_args()


# --- metrics ---------------------------------------------------------------

def brier(y, p):
    return float(np.mean((p - y) ** 2))


def ece(y, p, n_bins: int) -> float:
    """Expected calibration error: bin-size-weighted |confidence - accuracy|."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def reliability(y, p, n_bins: int) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({"bin": b, "lo": edges[b], "hi": edges[b + 1],
                     "n": int(m.sum()), "mean_pred": float(p[m].mean()),
                     "obs_freq": float(y[m].mean())})
    return pd.DataFrame(rows)


# --- calibrators -----------------------------------------------------------

def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_temperature(p_val, y_val) -> float:
    """One-parameter scaling of the logit, fit by minimising NLL on val."""
    z = torch.tensor(_logit(p_val), dtype=torch.float64)
    y = torch.tensor(y_val, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            z / torch.exp(log_t), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).item())


def fit_platt(p_val, y_val) -> tuple[float, float]:
    z = torch.tensor(_logit(p_val), dtype=torch.float64)
    y = torch.tensor(y_val, dtype=torch.float64)
    a = torch.ones(1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=0.1, max_iter=300)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(a * z + b, y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(a.item()), float(b.item())


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# --- models ----------------------------------------------------------------

def knn_probs(Xtr, ytr, Xev, temp):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.nn.functional.normalize(torch.as_tensor(Xtr, device=dev, dtype=torch.float32), dim=1)
    B = torch.nn.functional.normalize(torch.as_tensor(Xev, device=dev, dtype=torch.float32), dim=1)
    y = torch.as_tensor(ytr, device=dev, dtype=torch.long)
    out = _knn_softmax_vote_with_probs(A, y, B, num_classes=2,
                                       k_values=[K_EVAL], temperature=temp)
    return out[K_EVAL][2][:, 1].astype(np.float64)


def logistic_head(Xtr, ytr, evals, *, epochs, lr):
    """Linear head trained with BCE -- a proper scoring rule, so natively calibrated."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.nn.functional.normalize(torch.as_tensor(Xtr, device=dev, dtype=torch.float32), dim=1)
    y = torch.as_tensor(ytr, device=dev, dtype=torch.float32)
    model = torch.nn.Linear(A.shape[1], 1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = A.shape[0]
    bs = 4096
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(A[j]).squeeze(-1), y[j])
            loss.backward()
            opt.step()
            tot += float(loss) * len(j)
        if (ep + 1) % 10 == 0:
            print(f"    logistic epoch {ep+1}/{epochs} loss={tot/n:.4f}")
    outs = []
    with torch.no_grad():
        for Xe in evals:
            E = torch.nn.functional.normalize(
                torch.as_tensor(Xe, device=dev, dtype=torch.float32), dim=1)
            outs.append(torch.sigmoid(model(E).squeeze(-1)).cpu().numpy().astype(np.float64))
    return outs


def main() -> int:
    args = parse_args()
    d = {s: np.load(args.emb_dir / f"{s}.npz", allow_pickle=True)
         for s in ("train", "val", "test")}
    X = {s: d[s]["features"] for s in d}
    y = {s: d[s][f"{args.task}_label"].astype(np.float64) for s in d}
    print(f"task={args.task}  train={len(y['train'])} val={len(y['val'])} test={len(y['test'])}")

    print("  k-NN on val and test ...")
    p_val = knn_probs(X["train"], y["train"].astype(int), X["val"], args.temp)
    p_test = knn_probs(X["train"], y["train"].astype(int), X["test"], args.temp)

    print("  logistic head on embeddings ...")
    lg_val, lg_test = logistic_head(X["train"], y["train"], [X["val"], X["test"]],
                                    epochs=args.epochs, lr=args.lr)

    T = fit_temperature(p_val, y["val"])
    a, b = fit_platt(p_val, y["val"])
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_val, y["val"])
    print(f"  fitted on val: temperature T={T:.3f}  platt a={a:.3f} b={b:.3f}")

    methods = {
        "none": p_test,
        "temperature": sigmoid(_logit(p_test) / T),
        "platt": sigmoid(a * _logit(p_test) + b),
        "isotonic": iso.predict(p_test),
        "logistic": lg_test,
    }

    rows, rel_rows = [], []
    yt = y["test"]
    for name, p in methods.items():
        rows.append({"task": args.task, "method": name,
                     "auroc": roc_auc_score(yt, p),
                     "brier": brier(yt, p), "ece": ece(yt, p, args.bins),
                     "mean_pred": float(p.mean()), "base_rate": float(yt.mean()),
                     "frac_above_0.9": float((p > 0.9).mean()),
                     "frac_below_0.1": float((p < 0.1).mean())})
        r = reliability(yt, p, args.bins)
        r.insert(0, "method", name)
        r.insert(0, "task", args.task)
        rel_rows.append(r)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summ = pd.DataFrame(rows)
    rel = pd.concat(rel_rows, ignore_index=True)
    f1 = args.out_dir / "calibration_summary.csv"
    f2 = args.out_dir / "calibration_reliability.csv"
    summ.to_csv(f1, mode="a", header=not f1.exists(), index=False)
    rel.to_csv(f2, mode="a", header=not f2.exists(), index=False)

    print(f"\n  {'method':<12} {'AUROC':>8} {'Brier':>8} {'ECE':>8} {'mean p':>8} "
          f"{'base':>7} {'p>0.9':>7} {'p<0.1':>7}")
    for r in rows:
        print(f"  {r['method']:<12} {r['auroc']*100:>7.2f}% {r['brier']:>8.4f} "
              f"{r['ece']:>8.4f} {r['mean_pred']:>8.3f} {r['base_rate']:>7.3f} "
              f"{r['frac_above_0.9']:>7.3f} {r['frac_below_0.1']:>7.3f}")

    # Persist the calibrator so the burden analysis applies the identical mapping.
    np.savez(args.out_dir / f"calibrator_{args.task}.npz",
             temperature=np.array([T]), platt_a=np.array([a]), platt_b=np.array([b]),
             iso_x=iso.X_thresholds_, iso_y=iso.y_thresholds_)
    # Cache the validation scores so the burden bootstrap can refit the
    # calibrator per replicate and propagate its uncertainty rather than
    # treating the fitted mapping as exact.
    np.save(args.out_dir / f"valprobs_{args.task}.npy", p_val)
    np.save(args.out_dir / f"valy_{args.task}.npy", y["val"])
    print(f"  wrote calibrator -> {args.out_dir / f'calibrator_{args.task}.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
