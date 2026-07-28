"""Regression tests for k-NN probability normalization.

Guards the defect fixed in WS-0: the softmax was normalized once over max_k and
then summed over only the first k, so class scores summed to S_k < 1 for every
k < max_k and were not comparable across samples.

Run:  uv run pytest tests/test_knn_probs.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sdg6.knn import _knn_softmax_vote, _knn_softmax_vote_with_probs

K_VALUES = [5, 10, 20, 50, 100, 200]
TEMP = 0.07


def _fixture(n_train=800, n_eval=200, d=32, seed=0):
    rng = np.random.default_rng(seed)
    Xtr = rng.normal(size=(n_train, d)).astype(np.float32)
    Xev = rng.normal(size=(n_eval, d)).astype(np.float32)
    Xtr /= np.linalg.norm(Xtr, axis=1, keepdims=True)
    Xev /= np.linalg.norm(Xev, axis=1, keepdims=True)
    ytr = rng.integers(0, 2, size=n_train)
    return (
        torch.from_numpy(Xtr),
        torch.from_numpy(ytr).long(),
        torch.from_numpy(Xev),
    )


def test_probabilities_sum_to_one_for_every_k():
    """The whole point of the fix: valid distribution at EVERY k, not just max_k."""
    Xtr, ytr, Xev = _fixture()
    out = _knn_softmax_vote_with_probs(
        Xtr, ytr, Xev, num_classes=2, k_values=K_VALUES, temperature=TEMP
    )
    for k in K_VALUES:
        _, _, class_probs = out[k]
        totals = class_probs.sum(axis=1)
        assert np.allclose(totals, 1.0, atol=1e-5), (
            f"k={k}: probabilities sum to {totals.min():.4f}..{totals.max():.4f}, "
            "expected 1.0"
        )


def test_probabilities_are_in_unit_interval():
    Xtr, ytr, Xev = _fixture()
    out = _knn_softmax_vote_with_probs(
        Xtr, ytr, Xev, num_classes=2, k_values=K_VALUES, temperature=TEMP
    )
    for k in K_VALUES:
        _, conf, class_probs = out[k]
        assert class_probs.min() >= -1e-6 and class_probs.max() <= 1 + 1e-6
        # reported confidence must equal the max class probability
        assert np.allclose(conf, class_probs.max(axis=1), atol=1e-6)


def test_hard_predictions_unchanged_by_normalization():
    """Normalization divides by a per-sample constant, so argmax must not move.

    This is what lets us say accuracy/confusion numbers in the manuscript are
    unaffected by the fix.
    """
    Xtr, ytr, Xev = _fixture()
    with_probs = _knn_softmax_vote_with_probs(
        Xtr, ytr, Xev, num_classes=2, k_values=K_VALUES, temperature=TEMP
    )
    plain = _knn_softmax_vote(
        Xtr, ytr, Xev, num_classes=2, k_values=K_VALUES, temperature=TEMP
    )
    for k in K_VALUES:
        np.testing.assert_array_equal(with_probs[k][0], plain[k])


def test_single_k_equals_max_k_case():
    """When k == max_k the old and new paths agree - the reason the headline
    AUROC values (dinov2, k=200) needed no restatement."""
    Xtr, ytr, Xev = _fixture()
    out = _knn_softmax_vote_with_probs(
        Xtr, ytr, Xev, num_classes=2, k_values=[200], temperature=TEMP
    )
    _, _, class_probs = out[200]
    sims = Xev @ Xtr.T
    top_sims, idx = sims.topk(200, dim=1, largest=True, sorted=True)
    legacy_w = torch.softmax(top_sims / TEMP, dim=1)
    legacy = (
        torch.nn.functional.one_hot(ytr[idx], num_classes=2) * legacy_w.unsqueeze(-1)
    ).sum(dim=1)
    np.testing.assert_allclose(class_probs, legacy.numpy(), atol=1e-5)


@pytest.mark.parametrize("k", [1, 3, 7])
def test_small_k_still_normalized(k):
    Xtr, ytr, Xev = _fixture()
    out = _knn_softmax_vote_with_probs(
        Xtr, ytr, Xev, num_classes=2, k_values=[k, 50], temperature=TEMP
    )
    totals = out[k][2].sum(axis=1)
    assert np.allclose(totals, 1.0, atol=1e-5)
