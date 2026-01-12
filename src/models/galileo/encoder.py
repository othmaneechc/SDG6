from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from . import single_file as galileo

DEFAULT_MONTH = 5

# Normalization stats adapted from nasaharvest/galileo (MIT):
# /home/mila/e/echchabo/projects/galileo/src/data/utils.py
SPACE_TIME_STATS = {
    "mean": [
        -11.728724389184965,
        -18.85558188024017,
        1395.3408730676722,
        1338.4026921784578,
        1343.09883810357,
        1543.8607982512297,
        2186.2022069512263,
        2525.0932853316694,
        2410.3377187373408,
        2750.2854646886753,
        2234.911100061487,
        1474.5311266077113,
        0.2892116502999044,
    ],
    "std": [
        4.887145774840316,
        5.730270320384293,
        917.7041440370853,
        913.2988423581528,
        1092.678723527555,
        1047.2206083460424,
        1048.0101611156767,
        1143.6903026819996,
        1098.979177731649,
        1204.472755085893,
        1145.9774063078878,
        980.2429840007796,
        0.2720939024500081,
    ],
}


@dataclass(frozen=True)
class GalileoInput:
    s_t_x: torch.Tensor
    sp_x: torch.Tensor
    t_x: torch.Tensor
    st_x: torch.Tensor
    s_t_m: torch.Tensor
    sp_m: torch.Tensor
    t_m: torch.Tensor
    st_m: torch.Tensor
    months: torch.Tensor

    def to(self, device: torch.device) -> "GalileoInput":
        return GalileoInput(
            s_t_x=self.s_t_x.to(device, non_blocking=True),
            sp_x=self.sp_x.to(device, non_blocking=True),
            t_x=self.t_x.to(device, non_blocking=True),
            st_x=self.st_x.to(device, non_blocking=True),
            s_t_m=self.s_t_m.to(device, non_blocking=True),
            sp_m=self.sp_m.to(device, non_blocking=True),
            t_m=self.t_m.to(device, non_blocking=True),
            st_m=self.st_m.to(device, non_blocking=True),
            months=self.months.to(device, non_blocking=True),
        )

    @classmethod
    def stack(cls, items: Sequence["GalileoInput"]) -> "GalileoInput":
        return GalileoInput(
            s_t_x=torch.stack([i.s_t_x for i in items], dim=0),
            sp_x=torch.stack([i.sp_x for i in items], dim=0),
            t_x=torch.stack([i.t_x for i in items], dim=0),
            st_x=torch.stack([i.st_x for i in items], dim=0),
            s_t_m=torch.stack([i.s_t_m for i in items], dim=0),
            sp_m=torch.stack([i.sp_m for i in items], dim=0),
            t_m=torch.stack([i.t_m for i in items], dim=0),
            st_m=torch.stack([i.st_m for i in items], dim=0),
            months=torch.stack([i.months for i in items], dim=0),
        )


def load_encoder(weights_dir: Path, device: torch.device) -> tuple[galileo.Encoder, dict]:
    config_path = weights_dir / galileo.CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    with config_path.open("r") as f:
        config = json.load(f)
    encoder = galileo.Encoder.load_from_folder(weights_dir, device)
    encoder = encoder.to(device).eval()
    return encoder, config


def normalize_space_time(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(SPACE_TIME_STATS["mean"], device=x.device, dtype=x.dtype)
    std = torch.tensor(SPACE_TIME_STATS["std"], device=x.device, dtype=x.dtype)
    shift = mean - 2 * std
    div = 4 * std
    return (x - shift) / div


def _ensure_tdim(s2: torch.Tensor) -> torch.Tensor:
    if s2.ndim == 3:
        return s2.unsqueeze(2)
    if s2.ndim == 4:
        return s2
    raise ValueError(f"Expected s2 shape [H,W,C] or [H,W,T,C], got {s2.shape}")


def build_s2_input(
    s2: torch.Tensor,
    *,
    band_names: Sequence[str],
    months: torch.Tensor | None = None,
    normalize: bool = False,
    compute_ndvi: bool = False,
) -> GalileoInput:
    s2 = _ensure_tdim(s2)
    device = s2.device
    h, w, t, c = s2.shape

    if len(band_names) != c:
        raise ValueError(f"band_names has length {len(band_names)} but input has {c} bands")

    s2_full = torch.zeros(
        (h, w, t, len(galileo.S2_BANDS)), dtype=s2.dtype, device=device
    )
    for idx, name in enumerate(band_names):
        if name in galileo.S2_BANDS:
            s2_full[..., galileo.S2_BANDS.index(name)] = s2[..., idx]

    s_t_x = torch.zeros(
        (h, w, t, len(galileo.SPACE_TIME_BANDS)), dtype=s2.dtype, device=device
    )
    offset = len(galileo.S1_BANDS)
    s_t_x[..., offset : offset + len(galileo.S2_BANDS)] = s2_full

    has_b4 = "B4" in band_names
    has_b8 = "B8" in band_names
    if compute_ndvi and has_b4 and has_b8:
        red = s2_full[..., galileo.S2_BANDS.index("B4")]
        nir = s2_full[..., galileo.S2_BANDS.index("B8")]
        ndvi = (nir - red) / (nir + red + 1e-6)
        s_t_x[..., galileo.SPACE_TIME_BANDS.index("NDVI")] = ndvi

    s_t_m = torch.ones(
        (h, w, t, len(galileo.SPACE_TIME_BANDS_GROUPS_IDX)), dtype=s2.dtype, device=device
    )
    for g_idx, (g_name, band_idxs) in enumerate(galileo.SPACE_TIME_BANDS_GROUPS_IDX.items()):
        if g_name == "NDVI":
            if compute_ndvi and has_b4 and has_b8:
                s_t_m[..., g_idx] = 0
            continue
        if not g_name.startswith("S2"):
            continue
        group_band_names = [galileo.SPACE_TIME_BANDS[i] for i in band_idxs]
        if any(name in band_names for name in group_band_names):
            s_t_m[..., g_idx] = 0

    sp_x = torch.zeros(
        (h, w, len(galileo.SPACE_BANDS)), dtype=s2.dtype, device=device
    )
    sp_m = torch.ones(
        (h, w, len(galileo.SPACE_BAND_GROUPS_IDX)), dtype=s2.dtype, device=device
    )
    t_x = torch.zeros((t, len(galileo.TIME_BANDS)), dtype=s2.dtype, device=device)
    t_m = torch.ones((t, len(galileo.TIME_BAND_GROUPS_IDX)), dtype=s2.dtype, device=device)
    st_x = torch.zeros((len(galileo.STATIC_BANDS)), dtype=s2.dtype, device=device)
    st_m = torch.ones(
        (len(galileo.STATIC_BAND_GROUPS_IDX)), dtype=s2.dtype, device=device
    )

    if months is None:
        months = torch.ones((t,), dtype=torch.long, device=device) * DEFAULT_MONTH
    else:
        months = months.to(device)
        if months.shape[0] != t:
            raise ValueError("Incorrect number of input months")

    if normalize:
        s_t_x = normalize_space_time(s_t_x)

    return GalileoInput(
        s_t_x=s_t_x,
        sp_x=sp_x,
        t_x=t_x,
        st_x=st_x,
        s_t_m=s_t_m,
        sp_m=sp_m,
        t_m=t_m,
        st_m=st_m,
        months=months,
    )


def encode_batch(
    encoder: galileo.Encoder,
    batch: GalileoInput,
    *,
    patch_size: int,
    input_resolution_m: int,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    device = next(encoder.parameters()).device
    with torch.no_grad(), torch.amp.autocast(
        device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"
    ):
        out = encoder(
            batch.s_t_x,
            batch.sp_x,
            batch.t_x,
            batch.st_x,
            batch.s_t_m,
            batch.sp_m,
            batch.t_m,
            batch.st_m,
            batch.months,
            patch_size=patch_size,
            input_resolution_m=input_resolution_m,
        )
        s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, _ = out
        emb = galileo.Encoder.average_tokens(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)
        emb = F.normalize(emb, dim=-1)
        return emb


def infer_band_names(num_bands: int) -> list[str]:
    if num_bands >= len(galileo.S2_BANDS):
        return list(galileo.S2_BANDS)
    if num_bands == 3:
        return ["B2", "B3", "B4"]
    if num_bands == 4:
        return ["B2", "B3", "B4", "B8"]
    raise ValueError(f"Cannot infer band names for {num_bands} bands")


def map_s2_bands(arr: np.ndarray, band_names: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    if arr.ndim != 3:
        raise ValueError(f"Expected array shape [H,W,C], got {arr.shape}")
    if len(band_names) != arr.shape[2]:
        raise ValueError("Band name count does not match input channels")
    mapped_names = [name for name in band_names if name in galileo.S2_BANDS]
    if not mapped_names:
        raise ValueError("No S2 bands found in provided band names")
    return arr, list(band_names)


def to_month_index(month: int) -> int:
    if month < 1 or month > 12:
        return DEFAULT_MONTH
    return month - 1
