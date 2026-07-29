"""Manifest-driven dataset: read imagery in place instead of via ImageFolder trees.

``ImageDataset`` discovers samples by walking ``<split>/<class>/`` directories,
which requires materializing a symlink per image per dataset. Since PW-s and SW-s
share the same imagery and the same split, and only their labels differ, that
costs two symlinks per image for no benefit -- expensive on a filesystem with a
per-user inode quota.

``ManifestDataset`` takes the (path, label) pairs directly from the manifest
produced by ``scripts/build_manifest.py``. It reuses the same ``Sample`` schema,
reader, transform and collate function, so every model adapter works unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from models.base import CollateFn, Reader, Transform
from sdg6.data import Sample, collate_samples, read_rgb_image

# Folder-sorted class order, so label index 1 is the "has access" class exactly
# as ImageFolder would have assigned it.
CLASS_NAMES = {
    "pw": ["no_pipedwater", "pipedwater"],
    "sw": ["no_sewage", "sewage"],
}


class ManifestDataset(Dataset):
    """Dataset over explicit (path, label) rows."""

    def __init__(
        self,
        paths: Sequence[str],
        labels: Sequence[int],
        *,
        transform: Transform,
        reader: Reader = read_rgb_image,
    ) -> None:
        if len(paths) != len(labels):
            raise ValueError(f"paths/labels length mismatch: {len(paths)} vs {len(labels)}")
        if not len(paths):
            raise ValueError("Manifest selection is empty.")
        self.paths = [Path(p) for p in paths]
        self.labels = [int(x) for x in labels]
        self.transform = transform
        self.reader = reader

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Sample:
        path = self.paths[index]
        label = self.labels[index]
        image = self.reader(path)
        if self.transform is not None:
            try:
                image = self.transform(image, path=path)
            except TypeError:
                image = self.transform(image)
            except Exception as exc:  # matches ImageDataset: skip, do not abort
                print(f"[warn] Skipping sample {path}: {exc}")
                image = None
        return Sample(image=image, label=label, path=str(path))


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"path", "split", "pw_label", "sw_label", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest {path} missing columns: {sorted(missing)}")
    return df


def build_manifest_dataloaders(
    manifest: pd.DataFrame,
    *,
    task: str,
    transform: Transform,
    reader: Reader = read_rgb_image,
    batch_size: int = 64,
    num_workers: int = 4,
    splits: Sequence[str] = ("train", "val", "test"),
    collate_fn: CollateFn | None = None,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> tuple[dict[str, DataLoader], list[str]]:
    """Build one loader per split for ``task`` in {"pw", "sw"}."""
    if task not in CLASS_NAMES:
        raise ValueError(f"task must be one of {sorted(CLASS_NAMES)}, got {task!r}")
    label_col = f"{task}_label"

    loaders: dict[str, DataLoader] = {}
    for split in splits:
        sub = manifest[manifest["split"] == split]
        if sub.empty:
            continue
        dataset = ManifestDataset(
            sub["path"].tolist(), sub[label_col].tolist(),
            transform=transform, reader=reader,
        )
        sampler = None
        if distributed and world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,          # embedding extraction must preserve order
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            sampler=sampler,
            collate_fn=collate_fn or collate_samples,
        )
    return loaders, list(CLASS_NAMES[task])
