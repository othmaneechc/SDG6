"""Embedding extraction helpers shared across models."""

from __future__ import annotations

import os
import time
from typing import Dict, Tuple

import numpy as np
import torch

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

    total = len(dataloader) if hasattr(dataloader, "__len__") else None
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    start = time.time()
    for idx, batch in enumerate(dataloader, start=1):
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

        done = sum(f.shape[0] for f in feats)
        if total:
            frac = idx / total
            elapsed = time.time() - start
            avg = elapsed / idx
            eta = avg * (total - idx)
            prefix = f"[rank {rank} gpu {local_rank}]" if rank >= 0 else "[local]"
            print(
                f"{prefix} [progress] {desc or adapter.name} batch {idx}/{total} "
                f"features={done} (~{frac*100:4.1f}%) "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
            )
        else:
            elapsed = time.time() - start
            prefix = f"[rank {rank} gpu {local_rank}]" if rank >= 0 else "[local]"
            print(f"{prefix} [progress] {desc or adapter.name} batch {idx} features={done} elapsed={elapsed:.1f}s")

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
