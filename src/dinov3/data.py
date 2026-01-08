from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

Image.init()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def pad_to_patch(x: torch.Tensor, mask: torch.Tensor, patch: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad image and mask so H,W are multiples of patch."""
    _, h, w = x.shape
    h_pad = (patch - h % patch) % patch
    w_pad = (patch - w % patch) % patch
    if h_pad == 0 and w_pad == 0:
        return x, mask
    pad = (0, w_pad, 0, h_pad)  # (left,right,top,bottom) right/bottom only
    x = F.pad(x, pad, value=0.0)
    mask = F.pad(mask, pad, value=0.0)
    return x, mask


@dataclass
class PairSample:
    img_before: Path
    mask_before: Path
    img_after: Path
    mask_after: Path
    label: int


def _parse_gap_and_date(path: Path) -> Tuple[str, datetime]:
    stem = path.stem
    if "__" not in stem:
        raise ValueError(f"Cannot parse gap/date from {path}")
    gap_part, rest = stem.split("__", 1)
    date_str = rest.split("_")[0]
    date = datetime.fromisoformat(date_str)
    return gap_part, date


class MaskedPairDataset(Dataset):
    """
    Loads paired images (before/after) with optional masks from a classification folder tree:
    data_root/{train,val,test}/{class}/*.png and matching *.mask.npy (if use_masks is True).
    Pairs are formed by grouping files with the same gap prefix before '__'
    and picking the earliest and latest timestamps. Missing masks fall back to all-ones.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        patch_size: int = 16,
        apply_color_jitter: bool = True,
        extra_augment: bool = False,
        max_rotation: float = 20.0,
        max_translate: float = 0.05,
        max_scale: float = 0.1,
        vflip_prob: float = 0.2,
        train_repeat: int = 1,
        use_masks: bool = True,
        image_mean: List[float] | None = None,
        image_std: List[float] | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        self.extra_augment = extra_augment
        self.max_rotation = max_rotation
        self.max_translate = max_translate
        self.max_scale = max_scale
        self.vflip_prob = vflip_prob
        self.train_repeat = max(1, train_repeat) if split == "train" else 1
        self.use_masks = use_masks

        split_dir = self.root / split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")

        classes = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
        if not classes:
            raise ValueError(f"No class folders under {split_dir}")
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        samples: List[PairSample] = []
        for cls in classes:
            by_gap: defaultdict[str, List[Tuple[Path, datetime]]] = defaultdict(list)
            for img_path in (split_dir / cls).glob("*"):
                if img_path.suffix.lower() not in {".png", ".tif", ".tiff", ".jpg", ".jpeg"}:
                    continue
                mask_path = img_path.with_suffix(".mask.npy")
                gap, dt = _parse_gap_and_date(img_path)
                by_gap[gap].append((img_path, dt))

            for gap, entries in by_gap.items():
                if len(entries) < 2:
                    continue
                entries_sorted = sorted(entries, key=lambda t: t[1])
                img_before = entries_sorted[0][0]
                img_after = entries_sorted[-1][0]
                samples.append(
                    PairSample(
                        img_before=img_before,
                        mask_before=img_before.with_suffix(".mask.npy"),
                        img_after=img_after,
                        mask_after=img_after.with_suffix(".mask.npy"),
                        label=self.class_to_idx[cls],
                    )
                )

        if not samples:
            raise ValueError(f"No paired samples found under {split_dir}")

        self.samples = samples * self.train_repeat if self.train_repeat > 1 else samples

        aug_list = [transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)] if apply_color_jitter else []
        self.color_aug: Callable = transforms.Compose(aug_list)
        self.to_tensor = transforms.ToTensor()
        mean = image_mean if image_mean is not None else IMAGENET_MEAN
        std = image_std if image_std is not None else IMAGENET_STD
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def _geom_params(self, w: int, h: int):
        angle = random.uniform(-self.max_rotation, self.max_rotation)
        translate = (
            int(random.uniform(-self.max_translate, self.max_translate) * w),
            int(random.uniform(-self.max_translate, self.max_translate) * h),
        )
        scale = 1.0 + random.uniform(-self.max_scale, self.max_scale)
        return angle, translate, scale

    def __len__(self) -> int:
        return len(self.samples)

    def _load_pair(self, img_path: Path, mask_path: Path):
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        if self.use_masks and mask_path.exists():
            mask_np = np.load(mask_path).astype(np.float32)
            if mask_np.ndim != 2:
                mask_np = mask_np.squeeze()
            if mask_np.shape != img_np.shape[:2]:
                raise ValueError(
                    f"Mask shape {mask_np.shape} does not match image {img_np.shape[:2]} for {img_path}"
                )
        else:
            mask_np = np.ones(img_np.shape[:2], dtype=np.float32)
        return img_np, mask_np

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img_b_np, mask_b_np = self._load_pair(s.img_before, s.mask_before)
        img_a_np, mask_a_np = self._load_pair(s.img_after, s.mask_after)

        if self.split == "train" and self.extra_augment and random.random() < self.vflip_prob:
            img_b_np = np.ascontiguousarray(np.flipud(img_b_np))
            mask_b_np = np.ascontiguousarray(np.flipud(mask_b_np))
            img_a_np = np.ascontiguousarray(np.flipud(img_a_np))
            mask_a_np = np.ascontiguousarray(np.flipud(mask_a_np))

        if self.split == "train" and random.random() < 0.5:
            img_b_np = np.ascontiguousarray(np.fliplr(img_b_np))
            mask_b_np = np.ascontiguousarray(np.fliplr(mask_b_np))
            img_a_np = np.ascontiguousarray(np.fliplr(img_a_np))
            mask_a_np = np.ascontiguousarray(np.fliplr(mask_a_np))

        angle = translate = scale = None
        if self.split == "train" and self.extra_augment:
            h, w = img_b_np.shape[:2]
            angle, translate, scale = self._geom_params(w, h)

        img_b = Image.fromarray(img_b_np)
        img_a = Image.fromarray(img_a_np)
        if self.split == "train":
            img_b = self.color_aug(img_b)
            img_a = self.color_aug(img_a)

        img_b_t = self.to_tensor(img_b)
        img_a_t = self.to_tensor(img_a)
        mask_b_t = torch.from_numpy(mask_b_np).float()
        mask_a_t = torch.from_numpy(mask_a_np).float()

        if self.split == "train" and self.extra_augment and angle is not None:
            img_b_t = TF.affine(
                img_b_t,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=0.0,
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            img_a_t = TF.affine(
                img_a_t,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=0.0,
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            mask_b_t = TF.affine(
                mask_b_t.unsqueeze(0),
                angle=angle,
                translate=translate,
                scale=scale,
                shear=0.0,
                interpolation=InterpolationMode.NEAREST,
                fill=0.0,
            ).squeeze(0)
            mask_a_t = TF.affine(
                mask_a_t.unsqueeze(0),
                angle=angle,
                translate=translate,
                scale=scale,
                shear=0.0,
                interpolation=InterpolationMode.NEAREST,
                fill=0.0,
            ).squeeze(0)

        img_b_t = img_b_t * mask_b_t.unsqueeze(0)
        img_a_t = img_a_t * mask_a_t.unsqueeze(0)

        img_b_t = self.normalize(img_b_t)
        img_a_t = self.normalize(img_a_t)

        img_b_t, mask_b_t = pad_to_patch(img_b_t, mask_b_t, self.patch_size)
        img_a_t, mask_a_t = pad_to_patch(img_a_t, mask_a_t, self.patch_size)

        return img_b_t, mask_b_t, img_a_t, mask_a_t, s.label


def collate_pad_pairs(batch):
    imgs_b, masks_b, imgs_a, masks_a, labels = zip(*batch)
    max_h = max(max(i.shape[-2], j.shape[-2]) for i, j in zip(imgs_b, imgs_a))
    max_w = max(max(i.shape[-1], j.shape[-1]) for i, j in zip(imgs_b, imgs_a))

    def _pad(x: torch.Tensor):
        pad = (0, max_w - x.shape[-1], 0, max_h - x.shape[-2])
        return F.pad(x, pad, value=0.0)

    imgs_b = torch.stack([_pad(x) for x in imgs_b])
    imgs_a = torch.stack([_pad(x) for x in imgs_a])
    masks_b = torch.stack([_pad(x) for x in masks_b])
    masks_a = torch.stack([_pad(x) for x in masks_a])
    labels = torch.tensor(labels, dtype=torch.long)
    return imgs_b, masks_b, imgs_a, masks_a, labels
