"""Prithvi wrapper using Terratorch backbones.

Sources:
- TerraTorch backbone registry: https://github.com/IBM/terratorch
- Prithvi checkpoints: https://huggingface.co/ibm-nasa-geospatial
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import torch
from torchvision.transforms import v2 as T

from models.base import ModelAdapter, resolve_device, resolve_dtype
from sdg6.data import collate_samples, read_rgb_image

_WEIGHT_ALIASES = {
    "ibm-nasa-geospatial/prithvi-eo-2.0-600m": "prithvi_eo_v2_600",
    "ibm-nasa-geospatial/prithvi-eo-2.0-300m": "prithvi_eo_v2_300",
    "ibm-nasa-geospatial/prithvi-eo-2.0-300m-tl": "prithvi_eo_v2_300_tl",
    "ibm-nasa-geospatial/prithvi-eo-2.0-600m-tl": "prithvi_eo_v2_600_tl",
    "ibm-nasa-geospatial/prithvi-eo-2.0-100m-tl": "prithvi_eo_v2_100_tl",
    "ibm-nasa-geospatial/prithvi-eo-2.0-tiny-tl": "prithvi_eo_v2_tiny_tl",
    "prithvi-eo-2.0-600m": "prithvi_eo_v2_600",
    "prithvi-eo-2.0-300m": "prithvi_eo_v2_300",
    "prithvi-eo-2.0-300m-tl": "prithvi_eo_v2_300_tl",
    "prithvi-eo-2.0-600m-tl": "prithvi_eo_v2_600_tl",
    "prithvi-eo-2.0-100m-tl": "prithvi_eo_v2_100_tl",
    "prithvi-eo-2.0-tiny-tl": "prithvi_eo_v2_tiny_tl",
}


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _patch_torchgeo_compat() -> None:
    """Handle torchgeo enum differences required by some terratorch versions."""
    try:
        from torchgeo.models.resnet import ResNet50_Weights  # type: ignore
    except Exception:
        return

    if (
        not hasattr(ResNet50_Weights, "SENTINEL2_ALL_SOFTCON")
        and hasattr(ResNet50_Weights, "SENTINEL2_ALL_MOCO")
    ):
        setattr(ResNet50_Weights, "SENTINEL2_ALL_SOFTCON", ResNet50_Weights.SENTINEL2_ALL_MOCO)


def _resolve_backbone_name(weights: str) -> str:
    if not weights:
        raise ValueError("Prithvi weights cannot be empty.")

    alias = _WEIGHT_ALIASES.get(weights.lower())
    if alias:
        return alias

    stem = weights.split("/")[-1]
    alias = _WEIGHT_ALIASES.get(stem.lower())
    if alias:
        return alias

    normalized = _normalize_name(weights).replace("__", "_")
    if normalized.startswith("terratorch_"):
        normalized = normalized[len("terratorch_") :]

    if normalized.startswith("prithvi"):
        if normalized.startswith("prithvi_eo_v"):
            return normalized

        if normalized.startswith("prithvi_eo_2_0_"):
            suffix = normalized.replace("prithvi_eo_2_0_", "", 1)
            suffix = suffix.replace("_m", "")
            suffix = suffix.replace("m", "")
            if suffix in {"600", "300"}:
                return f"prithvi_eo_v2_{suffix}"
            if suffix in {"600_tl", "300_tl", "100_tl", "tiny_tl"}:
                return f"prithvi_eo_v2_{suffix}"

        # Already some variant of prithvi key (e.g., prithvi_eo_tiny).
        return normalized

    raise ValueError(
        "Unsupported Prithvi weights string. "
        "Use a Terratorch backbone name like 'prithvi_eo_v2_600' "
        "or an HF id like 'ibm-nasa-geospatial/Prithvi-EO-2.0-600M'."
    )


def _infer_in_channels(model: torch.nn.Module) -> int | None:
    for attr in ("in_chans", "num_channels", "num_input_channels"):
        value = getattr(model, attr, None)
        if isinstance(value, int) and value > 0:
            return value

    cfg = getattr(model, "config", None)
    for attr in ("in_chans", "num_channels", "num_input_channels"):
        value = getattr(cfg, attr, None) if cfg is not None else None
        if isinstance(value, int) and value > 0:
            return value

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
            return int(module.in_channels)
    return None


def _infer_output_dim(model: torch.nn.Module) -> int | None:
    for attr in ("embed_dim", "hidden_size", "num_features", "feature_dim"):
        value = getattr(model, attr, None)
        if isinstance(value, int) and value > 0:
            return value

    cfg = getattr(model, "config", None)
    for attr in ("hidden_size", "embed_dim", "projection_dim"):
        value = getattr(cfg, attr, None) if cfg is not None else None
        if isinstance(value, int) and value > 0:
            return value
    return None


def _resolve_img_size(model: torch.nn.Module, default: int) -> int:
    size = getattr(model, "img_size", None)
    if isinstance(size, int) and size > 0:
        return int(size)
    if isinstance(size, (tuple, list)) and size:
        if isinstance(size[0], int) and size[0] > 0:
            return int(size[0])
    return int(default)


def _extract_last_tensor(outputs) -> torch.Tensor | None:
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, dict):
        for key in ("last_hidden_state", "pooler_output", "features", "x"):
            if key in outputs:
                tensor = _extract_last_tensor(outputs[key])
                if tensor is not None:
                    return tensor
        for value in reversed(list(outputs.values())):
            tensor = _extract_last_tensor(value)
            if tensor is not None:
                return tensor
        return None
    if isinstance(outputs, (list, tuple)):
        for item in reversed(outputs):
            tensor = _extract_last_tensor(item)
            if tensor is not None:
                return tensor
        return None
    return None


def _resolve_model_bands(band_indices: list[int] | None):
    """Map input channel indices to Prithvi HLS bands when possible."""
    if not band_indices:
        return None
    try:
        from terratorch.datasets import HLSBands  # type: ignore
    except Exception:
        return None

    hls_order = [
        HLSBands.BLUE,
        HLSBands.GREEN,
        HLSBands.RED,
        HLSBands.NIR_NARROW,
        HLSBands.SWIR_1,
        HLSBands.SWIR_2,
    ]
    bands = []
    for idx in [int(i) for i in band_indices]:
        if 0 <= idx < len(hls_order):
            bands.append(hls_order[idx])
    return bands or None


def _expand_stats(values: Iterable[float], channels: int) -> torch.Tensor:
    vals = [float(v) for v in values]
    if not vals:
        vals = [0.5]
    if len(vals) < channels:
        vals = vals + [vals[-1]] * (channels - len(vals))
    elif len(vals) > channels:
        vals = vals[:channels]
    return torch.tensor(vals, dtype=torch.float32).view(channels, 1, 1)


class PrithviAdapter(ModelAdapter):
    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(self.device, non_blocking=True)
        outputs = self.encoder(batch)
        tokens = _extract_last_tensor(outputs)
        if tokens is None:
            raise KeyError("Prithvi model outputs missing tensor features.")

        if tokens.ndim == 2:
            cls = tokens
        elif tokens.ndim == 3:
            cls = tokens[:, 0]
        elif tokens.ndim == 4:
            cls = tokens.mean(dim=(-2, -1))
        else:
            raise ValueError(f"Unsupported Prithvi output shape: {tuple(tokens.shape)}")

        return torch.nn.functional.normalize(cls, dim=-1)


def load_model(
    *,
    weights: str,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype = "auto",
    img_size: int = 224,
    band_indices: list[int] | None = None,
    nodata_value: float = -9999.0,
    fill_value: float = 1e-4,
    value_scale: float = 1.0,
) -> ModelAdapter:
    device = resolve_device(str(device)) if isinstance(device, str) or device is None else device
    dtype = dtype if isinstance(dtype, torch.dtype) else resolve_dtype(str(dtype), device)
    requested_band_indices = [int(i) for i in band_indices] if band_indices else None

    _patch_torchgeo_compat()
    try:
        from terratorch.registry import BACKBONE_REGISTRY  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "terratorch is required for Prithvi loading. "
            "Install it in this environment (for example: `uv pip install terratorch`)."
        ) from exc

    base_key = _resolve_backbone_name(weights)
    model_bands = _resolve_model_bands(requested_band_indices)
    candidate_keys = [base_key]
    if base_key.startswith("terratorch_"):
        candidate_keys.append(base_key[len("terratorch_") :])
    else:
        candidate_keys.append(f"terratorch_{base_key}")

    model = None
    build_error: Exception | None = None
    for key in dict.fromkeys(candidate_keys):
        try:
            if model_bands is not None:
                model = BACKBONE_REGISTRY.build(key, pretrained=True, bands=model_bands)
            else:
                model = BACKBONE_REGISTRY.build(key, pretrained=True)
            break
        except Exception as exc:
            build_error = exc

    if model is None:
        tried = ", ".join(dict.fromkeys(candidate_keys))
        raise RuntimeError(
            f"Failed to build Prithvi backbone for weights='{weights}'. Tried: {tried}. "
            f"Last error: {build_error}"
        ) from build_error

    model = model.to(device).eval()
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        model = model.to(dtype=dtype)

    expected_channels = _infer_in_channels(model)
    if expected_channels is None:
        expected_channels = len(band_indices) if band_indices else 3

    if not band_indices:
        band_indices = list(range(min(expected_channels, 3)))

    img_size = _resolve_img_size(model, img_size)

    base_transform = T.Compose(
        [
            T.ToImage(),
            T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToDtype(torch.float32, scale=True),
        ]
    )

    mean = _expand_stats([0.5], expected_channels)
    std = torch.clamp(_expand_stats([0.5], expected_channels), min=1e-6)

    def transform(image, path: Path | str | None = None) -> torch.Tensor:
        del path
        x = base_transform(image)
        x = torch.nan_to_num(x, nan=fill_value, posinf=fill_value, neginf=fill_value)

        selected = [idx for idx in band_indices if 0 <= int(idx) < x.shape[0]]
        if selected:
            x = x[selected]

        if nodata_value is not None:
            x = torch.where(x == float(nodata_value), torch.tensor(fill_value, dtype=x.dtype), x)

        if value_scale != 1.0:
            x = x * float(value_scale)

        if x.shape[0] < expected_channels:
            pad = torch.full(
                (expected_channels - x.shape[0], x.shape[1], x.shape[2]),
                float(fill_value),
                dtype=x.dtype,
            )
            x = torch.cat([x, pad], dim=0)
        elif x.shape[0] > expected_channels:
            x = x[:expected_channels]

        x = (x - mean.to(dtype=x.dtype)) / std.to(dtype=x.dtype)
        return x

    output_dim = _infer_output_dim(model)
    if output_dim is None:
        raise ValueError("Could not determine Prithvi hidden size.")

    return PrithviAdapter(
        name="prithvi",
        encoder=model,
        device=device,
        transform=transform,
        reader=read_rgb_image,
        collate_fn=collate_samples,
        output_dim=output_dim,
        dtype=dtype,
    )
