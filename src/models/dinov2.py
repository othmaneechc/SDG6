"""DINOv2 adapter for locally trained checkpoints.

Sources:
- Codebase: https://github.com/facebookresearch/dinov2
- Paper: Oquab et al., 2023, https://arxiv.org/abs/2304.07193
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import torch
from torchvision.transforms import v2 as T

from models.base import ModelAdapter, resolve_device, resolve_dtype
from sdg6.data import collate_samples, read_rgb_image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _ensure_dinov2_importable(repo_dir: str | Path | None) -> None:
    if repo_dir is None:
        return
    repo_path = Path(repo_dir).expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"DINOv2 repo not found: {repo_path}")
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))


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


class Dinov2Adapter(ModelAdapter):
    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(batch.to(self.device, non_blocking=True))
        return torch.nn.functional.normalize(outputs, dim=-1)


def _load_cfg(config_file: Path, opts: Sequence[str] | None):
    from omegaconf import OmegaConf
    from dinov2.configs import dinov2_default_config

    cfg = OmegaConf.create(dinov2_default_config)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(str(config_file)))
    if opts:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(list(opts)))
    return cfg


def load_model(
    *,
    weights: str | Path,
    config_file: str | Path,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype = "auto",
    repo_dir: str | Path | None = None,
    checkpoint_key: str | None = "teacher",
    resize_size: int = 256,
    crop_size: int = 224,
    opts: Sequence[str] | None = None,
) -> ModelAdapter:
    device = resolve_device(str(device)) if isinstance(device, str) or device is None else device
    dtype = dtype if isinstance(dtype, torch.dtype) else resolve_dtype(str(dtype), device)

    repo_dir = repo_dir or os.environ.get("DINOV2_REPO")
    if repo_dir:
        _ensure_dinov2_importable(repo_dir)

    try:
        from dinov2.models import build_model_from_cfg
        from dinov2.utils import utils as dinov2_utils
    except ImportError as exc:
        raise RuntimeError(
            "dinov2 is not importable. Set DINOV2_REPO to the local repo or install the package."
        ) from exc

    config_path = Path(config_file).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"DINOv2 config not found: {config_path}")

    cfg = _load_cfg(config_path, opts)
    model, _ = build_model_from_cfg(cfg, only_teacher=True)
    dinov2_utils.load_pretrained_weights(model, str(weights), checkpoint_key)
    model = model.to(device).eval()
    if dtype == torch.float16 and device.type == "cuda":
        model = model.half()
    elif dtype == torch.bfloat16 and device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)

    output_dim = int(getattr(model, "embed_dim", getattr(model, "num_features", 0)))
    transform = build_transform(resize_size=resize_size, crop_size=crop_size)

    return Dinov2Adapter(
        name="dinov2",
        encoder=model,
        device=device,
        transform=lambda img, path=None: transform(img),
        reader=read_rgb_image,
        collate_fn=collate_samples,
        output_dim=output_dim,
        dtype=dtype,
    )
