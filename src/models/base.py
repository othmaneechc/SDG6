"""Shared model adapter definitions for embedding and k-NN workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

# Images are loaded first with a reader, then passed through a model-specific transform.
Transform = Callable[[Any, Path | str | None], Any]
Reader = Callable[[Path], Any]
CollateFn = Callable[[list[dict[str, Any]]], dict[str, Any]]


def resolve_device(preferred: str | None = None) -> torch.device:
    """Choose an available device."""
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(choice: str | torch.dtype, device: torch.device) -> torch.dtype:
    """Map dtype hints to torch dtypes."""
    if isinstance(choice, torch.dtype):
        return choice
    choice = str(choice)
    if choice == "bf16":
        return torch.bfloat16
    if choice == "fp16":
        return torch.float16
    if choice == "fp32":
        return torch.float32
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


@dataclass
class ModelAdapter:
    """Lightweight wrapper exposing a uniform encode interface."""

    name: str
    encoder: torch.nn.Module
    device: torch.device
    transform: Transform
    reader: Reader
    collate_fn: CollateFn
    output_dim: int
    dtype: torch.dtype

    def encode(self, batch: Any) -> torch.Tensor:  # pragma: no cover - interface only
        raise NotImplementedError

    def to(self, device: torch.device) -> "ModelAdapter":
        self.device = device
        self.encoder = self.encoder.to(device)
        return self


# Common image extensions accepted by the unified dataloader.
DEFAULT_EXTS: Iterable[str] = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
