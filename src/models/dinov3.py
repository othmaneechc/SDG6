"""DINOv3 wrapper using Hugging Face checkpoints (Oquab et al. 2023)."""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision.transforms import v2 as T

from models.base import ModelAdapter, resolve_device, resolve_dtype
from sdg6.data import collate_samples, read_rgb_image

STATS = {
    "lvd": (
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    ),
    "sat": (
        (0.430, 0.411, 0.296),
        (0.213, 0.156, 0.143),
    ),
}


def resolve_hf_sizes(processor) -> tuple[int, int]:
    resize_size = None
    crop_size = None

    size = getattr(processor, "size", None)
    if isinstance(size, dict):
        resize_size = size.get("shortest_edge") or size.get("height") or size.get("width")

    crop = getattr(processor, "crop_size", None)
    if isinstance(crop, dict):
        crop_size = crop.get("height") or crop.get("width")

    return int(resize_size or 256), int(crop_size or 224)


def build_transform(mean: tuple[float, float, float], std: tuple[float, float, float], resize: int, crop: int) -> T.Compose:
    return T.Compose(
        [
            T.ToImage(),
            T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(crop),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=mean, std=std),
        ]
    )


class Dinov3Adapter(ModelAdapter):
    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(self.device, non_blocking=True)
        outputs = self.encoder(pixel_values=batch)

        if isinstance(outputs, (tuple, list)):
            last_hidden = outputs[0] if outputs else None
            pooler = None
        else:
            last_hidden = getattr(outputs, "last_hidden_state", None)
            pooler = getattr(outputs, "pooler_output", None)

        if pooler is not None:
            cls = pooler
        elif last_hidden is not None:
            cls = last_hidden[:, 0]
        else:
            raise KeyError("HF model outputs missing last_hidden_state")

        return torch.nn.functional.normalize(cls, dim=-1)


def load_model(
    *,
    weights: str,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype = "auto",
    weights_type: str = "auto",
) -> ModelAdapter:
    device = resolve_device(str(device)) if isinstance(device, str) or device is None else device
    dtype = dtype if isinstance(dtype, torch.dtype) else resolve_dtype(str(dtype), device)

    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise RuntimeError("transformers is required for DINOv3 loading") from exc

    processor = AutoImageProcessor.from_pretrained(weights)
    model = AutoModel.from_pretrained(weights)

    resize_size, crop_size = resolve_hf_sizes(processor)
    if weights_type != "auto":
        mean, std = STATS[weights_type]
    else:
        mean = tuple(getattr(processor, "image_mean", STATS["lvd"][0]))
        std = tuple(getattr(processor, "image_std", STATS["lvd"][1]))

    transform = build_transform(mean, std, resize_size, crop_size)

    model = model.to(device).eval()
    if device.type == "cuda" and dtype == torch.float16:
        model = model.half()

    output_dim = int(getattr(model.config, "hidden_size", 0)) or None
    if output_dim is None:
        raise ValueError("Could not determine DINOv3 hidden size.")

    return Dinov3Adapter(
        name="dinov3",
        encoder=model,
        device=device,
        transform=lambda img, path=None: transform(img),
        reader=read_rgb_image,
        collate_fn=collate_samples,
        output_dim=output_dim,
        dtype=dtype,
    )
