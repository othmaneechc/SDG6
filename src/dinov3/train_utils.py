from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler


@dataclass
class TrainState:
    epoch: int = 0
    best_val_acc: float = 0.0
    global_step: int = 0


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    freeze_backbone: bool,
) -> Tuple[float, float]:
    model.train()
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
    else:
        for p in model.backbone.parameters():
            p.requires_grad = True

    total_loss = 0.0
    total_acc = 0.0
    count = 0
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    for imgs, masks, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            logits = model(imgs, masks)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        total_acc += accuracy(logits.detach(), labels) * imgs.size(0)
        count += imgs.size(0)

    dt = time.time() - start
    return total_loss / count, total_acc / count


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for imgs, masks, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs, masks)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            total_acc += accuracy(logits, labels) * imgs.size(0)
            count += imgs.size(0)
    return total_loss / count, total_acc / count


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    state: TrainState,
    label_to_idx: Dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "state": state.__dict__,
            "label_to_idx": label_to_idx,
        },
        path,
    )
