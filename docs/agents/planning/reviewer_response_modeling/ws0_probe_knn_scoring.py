#!/usr/bin/env python3
"""WS-0 investigation probe — READ-ONLY, touches no pipeline code.

Reproduces the exact scoring path in src/sdg6/knn.py::_knn_softmax_vote_with_probs
and compares it against the DINOv2-correct scoring (softmax renormalized over the
k actual neighbours), to quantify how much AUROC is distorted for k < max_k.

Hypothesis under test
---------------------
The reported AUROC rises monotonically with k, and in 5 of 6 model/dataset combos
peaks exactly at k == max_k — the single k where the current code's probabilities
happen to sum to 1. If the rise is partly an artifact of shrinking normalization
error rather than a genuine benefit of more neighbours, then in simulation the
"current" curve should sit below the "correct" curve and converge to it at max_k.

Run:  python3 ws0_probe_knn_scoring.py
"""

from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(0)
K_VALUES = [5, 10, 20, 50, 100, 200]
MAX_K = max(K_VALUES)
TEMP = 0.07  # scripts/configs/*.yaml knn_softmax_temp


def auc_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney AUC with average tie handling (same as the repo's analysis)."""
    y_true = np.asarray(y_true, np.int64)
    y_score = np.asarray(y_score, np.float64)
    n_pos = int(y_true.sum())
    n_neg = y_true.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(y_true.size, np.float64)
    s = y_score[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def make_data(n_train=4000, n_test=2000, d=64, sep=0.55, pos_rate=0.45):
    """Two class-conditional Gaussians on the unit sphere."""
    def sample(n):
        y = (RNG.random(n) < pos_rate).astype(np.int64)
        mu = np.zeros((2, d))
        mu[1, 0] = sep
        mu[0, 0] = -sep
        X = mu[y] + RNG.normal(0, 1.0, (n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X, y
    return (*sample(n_train), *sample(n_test))


def main() -> None:
    Xtr, ytr, Xte, yte = make_data()
    sims = Xte @ Xtr.T
    idx = np.argpartition(-sims, MAX_K, axis=1)[:, :MAX_K]
    part = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-part, axis=1)
    top_sims = np.take_along_axis(part, order, axis=1)          # (n_test, MAX_K)
    top_labels = ytr[np.take_along_axis(idx, order, axis=1)]    # (n_test, MAX_K)

    # softmax over ALL max_k neighbours — exactly what the current code does once
    e_all = np.exp((top_sims - top_sims.max(1, keepdims=True)) / TEMP)
    w_all = e_all / e_all.sum(1, keepdims=True)

    print(f"{'k':>5} {'sum(p) mean':>12} {'sum(p) min':>11} "
          f"{'AUROC current':>14} {'AUROC correct':>14} {'delta':>8}")
    print("-" * 70)
    rows = []
    for k in K_VALUES:
        # ---- current code path: sum the first k of the max_k-normalized weights
        wk = w_all[:, :k]
        lk = top_labels[:, :k]
        p1_cur = (wk * (lk == 1)).sum(1)
        p0_cur = (wk * (lk == 0)).sum(1)
        pred = (p1_cur >= p0_cur).astype(int)
        conf = np.maximum(p1_cur, p0_cur)
        signed = np.where(pred == 1, conf, -conf)   # analysis script line 145
        auc_cur = auc_rank(yte, signed)
        mass = p1_cur + p0_cur

        # ---- correct: renormalize the softmax over the k actual neighbours
        s_k = top_sims[:, :k]
        e_k = np.exp((s_k - s_k.max(1, keepdims=True)) / TEMP)
        w_k = e_k / e_k.sum(1, keepdims=True)
        p1_ok = (w_k * (lk == 1)).sum(1)
        auc_ok = auc_rank(yte, p1_ok)

        rows.append((k, auc_cur, auc_ok))
        print(f"{k:>5} {mass.mean():>12.4f} {mass.min():>11.4f} "
              f"{auc_cur*100:>13.2f}% {auc_ok*100:>13.2f}% "
              f"{(auc_ok-auc_cur)*100:>+7.2f}")

    print("-" * 70)
    cur = [r[1] for r in rows]
    ok = [r[2] for r in rows]
    print(f"current  path: AUROC rises {cur[0]*100:.2f}% -> {cur[-1]*100:.2f}% "
          f"({(cur[-1]-cur[0])*100:+.2f} pts across k)")
    print(f"correct  path: AUROC rises {ok[0]*100:.2f}% -> {ok[-1]*100:.2f}% "
          f"({(ok[-1]-ok[0])*100:+.2f} pts across k)")
    print(f"\nAt k == max_k ({MAX_K}) the two agree to "
          f"{abs(cur[-1]-ok[-1])*100:.4f} pts (probabilities sum to 1 there).")
    artifact = (cur[-1] - cur[0]) - (ok[-1] - ok[0])
    print(f"Apparent gain from k that is NORMALIZATION ARTIFACT, not signal: "
          f"{artifact*100:+.2f} pts")


if __name__ == "__main__":
    main()
