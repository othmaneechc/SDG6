"""Unified dataloader utilities shared across all encoders."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models.base import CollateFn, DEFAULT_EXTS, Reader, Transform


def read_rgb_image(path: Path) -> Image.Image:
    """Load a standard RGB image."""
    with Image.open(path) as img:
        return img.convert("RGB")


def discover_classes(data_root: Path, splits: Sequence[str]) -> list[str]:
    """Find class folder names across provided splits."""
    classes = set()
    for split in splits:
        split_dir = data_root / split
        if not split_dir.is_dir():
            continue
        for child in split_dir.iterdir():
            if child.is_dir():
                classes.add(child.name)
    return sorted(classes)


@dataclass(frozen=True)
class Sample:
    image: Any
    label: int
    path: str


class ImageDataset(Dataset):
    """ImageFolder-style dataset with model-agnostic sample schema."""

    def __init__(
        self,
        root: Path,
        *,
        transform: Transform,
        reader: Reader = read_rgb_image,
        class_to_idx: dict[str, int] | None = None,
        allow_unlabeled: bool = False,
        exts: Iterable[str] = DEFAULT_EXTS,
    ) -> None:
        if not root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {root}")
        self.root = root
        self.transform = transform
        self.reader = reader
        self.allow_unlabeled = allow_unlabeled
        self.exts = tuple(exts)
        self.class_to_idx = class_to_idx or {}
        self.samples = self._gather_samples()

    def _gather_samples(self) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []
        if self.class_to_idx:
            for cls, idx in self.class_to_idx.items():
                cls_dir = self.root / cls
                if not cls_dir.is_dir():
                    continue
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith(self.exts):
                        samples.append((cls_dir / fname, idx))
        else:
            for fname in os.listdir(self.root):
                if fname.lower().endswith(self.exts):
                    samples.append((self.root / fname, -1))
        if not samples and not self.allow_unlabeled:
            raise RuntimeError(f"No samples found under {self.root}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        path, label = self.samples[index]
        image = self.reader(path)
        if self.transform is not None:
            try:
                image = self.transform(image, path=path)
            except TypeError:
                image = self.transform(image)
            except Exception as exc:
                # Log and skip samples that fail transform (e.g., zero-band inputs).
                print(f"[warn] Skipping sample {path}: {exc}")
                image = None
        return Sample(image=image, label=label, path=str(path))


def collate_samples(samples: list[Sample]) -> dict[str, Any]:
    """Collate list of Sample objects into a training/eval batch."""
    # Drop samples that failed to load/transform.
    dropped = len([s for s in samples if s.image is None])
    samples = [s for s in samples if s.image is not None]
    if dropped:
        print(f"[warn] Dropped {dropped} samples in batch due to transform failures.")
    if not samples:
        return {"image": None, "label": None, "path": []}

    images = [s.image for s in samples]
    labels = torch.tensor([s.label for s in samples], dtype=torch.long)
    paths = [s.path for s in samples]

    first = images[0]
    if hasattr(first, "__class__") and hasattr(first.__class__, "stack"):
        # GalileoInput exposes a classmethod stack for batching.
        images = first.__class__.stack(images)  # type: ignore[arg-type]
    elif isinstance(first, torch.Tensor):
        images = torch.stack(images, dim=0)
    else:
        images = images

    return {"image": images, "label": labels, "path": paths}


def build_dataloader(
    split_dir: Path,
    *,
    transform: Transform,
    reader: Reader = read_rgb_image,
    class_to_idx: dict[str, int] | None,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    allow_unlabeled: bool = False,
    collate_fn: CollateFn | None = None,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> DataLoader:
    dataset = ImageDataset(
        split_dir,
        transform=transform,
        reader=reader,
        class_to_idx=class_to_idx,
        allow_unlabeled=allow_unlabeled,
    )
    sampler = None
    if distributed and world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if (sampler is None and class_to_idx) else False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=sampler,
        collate_fn=collate_fn or collate_samples,
    )


def build_dataloaders(
    data_root: Path,
    *,
    transform: Transform,
    reader: Reader = read_rgb_image,
    batch_size: int = 64,
    num_workers: int = 4,
    splits: Sequence[str] = ("train", "val", "test"),
    shuffle_train: bool = False,
    allow_unlabeled: bool = False,
    collate_fn: CollateFn | None = None,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> tuple[dict[str, DataLoader], list[str]]:
    """Create DataLoaders for available splits. Missing splits are skipped."""
    class_names = discover_classes(data_root, splits) if not allow_unlabeled else []
    class_to_idx = {name: idx for idx, name in enumerate(class_names)} if class_names else None

    loaders: dict[str, DataLoader] = {}
    for split in splits:
        split_dir = data_root / split
        if split_dir.is_dir():
            target_dir = split_dir
        elif len(splits) == 1 and data_root.is_dir():
            target_dir = data_root
        else:
            continue
        shuffle = shuffle_train and split == "train"
        loaders[split] = build_dataloader(
            target_dir,
            transform=transform,
            reader=reader,
            class_to_idx=class_to_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            allow_unlabeled=allow_unlabeled,
            collate_fn=collate_fn,
            distributed=distributed,
            world_size=world_size,
            rank=rank,
        )
    return loaders, class_names
