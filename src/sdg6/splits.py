"""Shared split helpers for location-based evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Hashable

import numpy as np


def location_keys(lat: Iterable[float], lon: Iterable[float]) -> list[tuple[float, float]]:
    """One key per survey location."""
    return list(zip(lat, lon))


def assign_balanced_group_folds(
    keys: Iterable[Hashable], n_folds: int, seed: int
) -> np.ndarray:
    """Assign repeated group keys to deterministic, approximately balanced folds."""
    key_list = list(keys)
    uniq = sorted(set(key_list), key=lambda x: str(x))
    order = np.random.default_rng(seed).permutation(len(uniq))
    mapping = {group: int(order[i] % n_folds) for i, group in enumerate(uniq)}
    return np.array([mapping[key] for key in key_list], dtype=int)


# Five African subregions (UN M49 geoscheme), keyed by the reverse-geocoded
# country_name in the manifest. Used to form the region-based (leave-one-region-out)
# cross-validation folds: each subregion is held out in turn. The grouping is fixed
# and documented so the split can be explained rather than being an arbitrary
# coordinate partition.
AFRICA_UN_SUBREGION: dict[str, str] = {
    # Northern Africa
    "Morocco": "Northern", "Tunisia": "Northern", "Sudan": "Northern",
    # Western Africa
    "Nigeria": "Western", "Ghana": "Western", "Senegal": "Western", "Mali": "Western",
    "Niger": "Western", "Benin": "Western", "Togo": "Western", "Burkina Faso": "Western",
    "Guinea": "Western", "Sierra Leone": "Western", "Liberia": "Western",
    "Côte d'Ivoire": "Western", "Gambia": "Western", "Mauritania": "Western",
    # Middle (Central) Africa
    "Angola": "Middle", "Cameroon": "Middle", "Congo": "Middle", "Gabon": "Middle",
    # Eastern Africa
    "Kenya": "Eastern", "Tanzania": "Eastern", "Uganda": "Eastern", "Ethiopia": "Eastern",
    "Madagascar": "Eastern", "Malawi": "Eastern", "Mozambique": "Eastern",
    "Zambia": "Eastern", "Zimbabwe": "Eastern",
    # Southern Africa
    "South Africa": "Southern", "Namibia": "Southern", "Botswana": "Southern",
    "Lesotho": "Southern", "eSwatini": "Southern",
}

# North-to-south ordering, used for stable fold labels and figure legends.
REGION_ORDER = ["Northern", "Western", "Middle", "Eastern", "Southern"]


def region_labels(country_names: Iterable[str]) -> np.ndarray:
    """Map each location's country to its UN African subregion.

    Raises if any country lacks a mapping, so a new survey country cannot be
    silently dropped from the region folds.
    """
    names = list(country_names)
    missing = sorted({n for n in names if n not in AFRICA_UN_SUBREGION})
    if missing:
        raise ValueError(f"No African subregion mapping for: {missing}")
    return np.array([AFRICA_UN_SUBREGION[n] for n in names], dtype=object)
