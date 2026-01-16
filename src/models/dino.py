"""Wrapper around the DINO ViT backbone (Caron et al. 2021) via the `dino` dependency."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torchvision.transforms import v2 as T

from models.base import ModelAdapter, resolve_device, resolve_dtype
from sdg6.data import collate_samples, read_rgb_image

# Reuse the upstream DINO implementation from the pip dependency.
from dino import utils as dino_utils
from dino import vision_transformer as vits

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(resize_size: int = 256, crop_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.ToImage(),
            T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(crop_size),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class DinoAdapter(ModelAdapter):
    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(batch.to(self.device, non_blocking=True))
        return torch.nn.functional.normalize(outputs, dim=-1)


def load_model(
    *,
    weights: str | Path,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype = "auto",
    arch: str = "vit_base",
    patch_size: int = 8,
    checkpoint_key: str | None = "teacher",
    resize_size: int = 256,
    crop_size: int = 224,
) -> ModelAdapter:
    device = resolve_device(str(device)) if isinstance(device, str) or device is None else device
    dtype = dtype if isinstance(dtype, torch.dtype) else resolve_dtype(str(dtype), device)

    if arch not in vits.__dict__:
        raise ValueError(f"Unknown DINO architecture '{arch}'")
    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    dino_utils.load_pretrained_weights(
        model,
        str(weights),
        checkpoint_key=checkpoint_key,
        model_name=arch,
        patch_size=patch_size,
    )
    model.to(device).eval()

    output_dim = int(getattr(model, "embed_dim", getattr(model, "num_features", 0)))
    transform = build_transform(resize_size=resize_size, crop_size=crop_size)

    return DinoAdapter(
        name="dino",
        encoder=model,
        device=device,
        transform=lambda img, path=None: transform(img),
        reader=read_rgb_image,
        collate_fn=collate_samples,
        output_dim=output_dim,
        dtype=dtype,
    )
