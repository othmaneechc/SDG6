"""Embedding extraction helpers shared across models."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm

from models.base import ModelAdapter


def extract_embeddings(
    adapter: ModelAdapter,
    dataloader,
    *,
    amp_dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    desc: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Run a dataloader through an encoder and return (features, labels, paths)."""
    device = device or adapter.device
    amp_dtype = amp_dtype or adapter.dtype
    encoder = adapter.encoder.to(device).eval()

    feats: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    paths: list[str] = []

    for batch in tqdm(dataloader, desc=desc or f"{adapter.name} embeddings"):
        images = batch["image"]
        batch_labels = batch["label"]
        paths.extend(batch["path"])

        if images is None:
            continue
        if hasattr(images, "to"):
            images = images.to(device, non_blocking=True)

        with torch.no_grad(), torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"
        ):
            outputs = adapter.encode(images)

        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(0)

        feats.append(outputs.detach().cpu())
        labels.append(batch_labels.detach().cpu())

    if not feats:
        return np.zeros((0, adapter.output_dim), dtype=np.float32), np.zeros((0,), dtype=np.int64), paths

    feat_tensor = torch.cat(feats, dim=0)
    label_tensor = torch.cat(labels, dim=0)
    return feat_tensor.numpy(), label_tensor.numpy(), paths


def extract_all_splits(
    adapter: ModelAdapter,
    dataloaders: Dict[str, any],
    *,
    amp_dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    """Extract embeddings for every split present in the dataloader dict."""
    outputs: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    for split, loader in dataloaders.items():
        outputs[split] = extract_embeddings(
            adapter, loader, amp_dtype=amp_dtype, device=device, desc=f"{split} ({adapter.name})"
        )
    return outputs
