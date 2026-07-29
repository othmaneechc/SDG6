"""Shared split helpers for location-based evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Hashable

import numpy as np


def location_keys(lat: Iterable[float], lon: Iterable[float]) -> list[tuple[float, float]]:
    """One key per survey location."""
    return list(zip(lat, lon))


def spatial_block_keys(
    lat: Iterable[float], lon: Iterable[float], block_deg: float
) -> list[tuple[int, int]]:
    """Map coordinates to regular latitude/longitude grid blocks."""
    lat_arr = np.asarray(list(lat), dtype=float)
    lon_arr = np.asarray(list(lon), dtype=float)
    return list(
        zip(
            np.floor(lat_arr / block_deg).astype(int),
            np.floor(lon_arr / block_deg).astype(int),
        )
    )


def assign_balanced_group_folds(
    keys: Iterable[Hashable], n_folds: int, seed: int
) -> np.ndarray:
    """Assign repeated group keys to deterministic, approximately balanced folds."""
    key_list = list(keys)
    uniq = sorted(set(key_list), key=lambda x: str(x))
    order = np.random.default_rng(seed).permutation(len(uniq))
    mapping = {group: int(order[i] % n_folds) for i, group in enumerate(uniq)}
    return np.array([mapping[key] for key in key_list], dtype=int)


def assign_one_group_per_fold(keys: Iterable[Hashable]) -> np.ndarray:
    """Assign each unique group to its own fold."""
    key_list = list(keys)
    uniq = sorted(set(key_list), key=lambda x: str(x))
    mapping = {group: i for i, group in enumerate(uniq)}
    return np.array([mapping[key] for key in key_list], dtype=int)
