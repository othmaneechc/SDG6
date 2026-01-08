#!/usr/bin/env python
"""
Train a DINO-based classifier on paired masked images (before/after).

Structure: data_root/{train,val,test}/{class}/*.png (or tif) with *.mask.npy alongside.
Pairs are built by grouping files sharing the prefix before '__' and picking earliest vs latest date.
Backbone is shared; the head uses pooled before/after features and their signed delta.

Example:
    uv run python scripts/train_dinov3.py \
        --data-root data/labels_lvl4 \
        --backbone facebook/dinov3-convnext-tiny-pretrain-lvd1689m \
        --apply-color-jitter
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from datetime import datetime
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from dinov3.data import MaskedPairDataset, collate_pad_pairs  # noqa: E402
from dinov3.model import Dinov3HFBackbone, DinoPairedClassifier, detect_patch_size  # noqa: E402
from dinov3.train_utils import TrainState, save_checkpoint  # noqa: E402


class FocalLoss(torch.nn.Module):
    """Multi-class focal loss with optional per-class alpha and weight."""

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_logp = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = target_logp.exp()
        focal = (1 - pt).pow(self.gamma)
        loss = -focal * target_logp
        if self.weight is not None:
            loss = loss * self.weight[target]
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def set_backbone_trainable(backbone: torch.nn.Module, freeze: bool, last_blocks: int) -> None:
    if freeze:
        for p in backbone.parameters():
            p.requires_grad = False
        return

    if last_blocks <= 0:
        for p in backbone.parameters():
            p.requires_grad = True
        return

    # freeze all, then unfreeze only the last N blocks (+final norm)
    for p in backbone.parameters():
        p.requires_grad = False

    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        for p in backbone.parameters():
            p.requires_grad = True
        return

    for block in list(blocks)[-last_blocks:]:
        for p in block.parameters():
            p.requires_grad = True

    if hasattr(backbone, "norm"):
        for p in backbone.norm.parameters():
            p.requires_grad = True


def format_confusion_table(conf: torch.Tensor, idx_to_class: dict[int, str]) -> str:
    """Format confusion matrix with per-class precision/recall as a table."""
    conf = conf.cpu()
    n = conf.shape[0]
    classes = [idx_to_class[i] for i in range(n)]
    row_totals = conf.sum(dim=1)
    col_totals = conf.sum(dim=0)
    diag = conf.diag()
    recall = [
        (diag[i] / row_totals[i]).item() if row_totals[i].item() > 0 else 0.0  # type: ignore[arg-type]
        for i in range(n)
    ]
    precision = [
        (diag[i] / col_totals[i]).item() if col_totals[i].item() > 0 else 0.0  # type: ignore[arg-type]
        for i in range(n)
    ]

    max_count = int(conf.max().item())
    digits = max(3, len(str(max_count)))
    first_w = max(len("true\\pred"), max(len(c) for c in classes))
    col_widths = [max(len(c), digits) for c in classes]
    total_w = max(len("total"), len(str(int(row_totals.max().item())) if row_totals.numel() > 0 else 1))
    recall_w = max(len("recall"), 6)

    lines = []
    header_cells = ["true\\pred".ljust(first_w)]
    header_cells += [c.rjust(col_widths[i]) for i, c in enumerate(classes)]
    header_cells += ["total".rjust(total_w), "recall".rjust(recall_w)]
    lines.append(" ".join(header_cells))

    for i, cls in enumerate(classes):
        row_cells = [cls.ljust(first_w)]
        row_cells += [f"{int(conf[i, j].item())}".rjust(col_widths[j]) for j in range(n)]
        row_cells += [f"{int(row_totals[i].item())}".rjust(total_w), f"{recall[i]:>{recall_w}.3f}"]
        lines.append(" ".join(row_cells))

    prec_cells = ["precision".ljust(first_w)]
    prec_cells += [f"{precision[j]:>{col_widths[j]}.3f}" for j in range(n)]
    prec_cells += ["".rjust(total_w), "".rjust(recall_w)]
    lines.append(" ".join(prec_cells))

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True, help="Root with train/val/test splits and masks.")
    p.add_argument(
        "--backbone",
        type=str,
        default="facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        help="Hugging Face model id for the DINOv3 backbone.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional local checkpoint directory or HF snapshot; defaults to --backbone id.",
    )
    p.add_argument("--output", type=Path, default=Path("outputs/dino"), help="Where to store checkpoints/logs.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr-head", type=float, default=1e-4)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--freeze-backbone-epochs", type=int, default=2, help="Freeze backbone for first N epochs.")
    p.add_argument("--apply-color-jitter", action="store_true", help="Enable color jitter during training.")
    p.add_argument(
        "--extra-augment",
        action="store_true",
        help="Enable extra geometric aug (rotation/translate/scale + occasional vertical flip).",
    )
    p.add_argument("--max-rotation", type=float, default=20.0, help="Max abs rotation (degrees) for extra aug.")
    p.add_argument(
        "--max-translate",
        type=float,
        default=0.05,
        help="Max fraction of width/height for translation in extra aug.",
    )
    p.add_argument("--max-scale", type=float, default=0.1, help="Max relative scaling for extra aug.")
    p.add_argument("--vflip-prob", type=float, default=0.2, help="Vertical flip probability in extra aug.")
    p.add_argument(
        "--train-repeat",
        type=int,
        default=1,
        help="Repeat/upsample the training set N times to expose more augment variations.",
    )
    p.add_argument(
        "--class-weighting",
        type=str,
        default="none",
        choices=["none", "inverse_freq", "balanced"],
        help="Optional class weighting scheme for CrossEntropyLoss.",
    )
    p.add_argument(
        "--class-weights",
        type=str,
        default=None,
        help="Comma-separated manual class weights (overrides --class-weighting).",
    )
    p.add_argument(
        "--loss",
        type=str,
        default="ce",
        choices=["ce", "focal"],
        help="Loss type: cross-entropy (ce) or focal loss (focal).",
    )
    p.add_argument("--focal-gamma", type=float, default=2.0, help="Gamma for focal loss.")
    p.add_argument(
        "--focal-alpha",
        type=str,
        default=None,
        help="Comma-separated alpha per class for focal loss (optional).",
    )
    p.add_argument(
        "--pair-composition",
        type=str,
        default="b_a_delta",
        choices=["b_a_delta", "b_a_delta_abs", "delta_only", "delta_abs", "delta_tokens", "delta_tokens_abs"],
        help="How to combine before/after pooled features.",
    )
    mask_group = p.add_mutually_exclusive_group()
    mask_group.add_argument(
        "--use-masks",
        action="store_true",
        dest="use_masks",
        default=True,
        help="Load and apply .mask.npy files (default).",
    )
    mask_group.add_argument(
        "--no-masks",
        action="store_false",
        dest="use_masks",
        help="Ignore .mask.npy files; fall back to all-ones masks.",
    )
    p.add_argument(
        "--unfreeze-last-blocks",
        type=int,
        default=0,
        help="When unfreezing, only unfreeze the last N transformer blocks (0 = unfreeze all).",
    )
    return p.parse_args()


def load_backbone(model_id: str, checkpoint: str | None, device: torch.device):
    target = str(checkpoint) if checkpoint else model_id
    processor = AutoImageProcessor.from_pretrained(target)
    hf_model = AutoModel.from_pretrained(target)
    backbone = Dinov3HFBackbone(hf_model).to(device)
    return backbone, processor.image_mean, processor.image_std

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    amp_enabled = amp_device_type == "cuda"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output / f"dino_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    logger = logging.getLogger("train_dinov3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False

    backbone, image_mean, image_std = load_backbone(args.backbone, args.checkpoint, device)
    patch_size = detect_patch_size(backbone)
    num_classes = len([p for p in (args.data_root / "train").iterdir() if p.is_dir()])

    train_ds = MaskedPairDataset(
        args.data_root,
        "train",
        patch_size=patch_size,
        image_mean=image_mean,
        image_std=image_std,
        apply_color_jitter=args.apply_color_jitter,
        extra_augment=args.extra_augment,
        max_rotation=args.max_rotation,
        max_translate=args.max_translate,
        max_scale=args.max_scale,
        vflip_prob=args.vflip_prob,
        train_repeat=args.train_repeat,
        use_masks=args.use_masks,
    )
    val_ds = MaskedPairDataset(
        args.data_root,
        "val",
        patch_size=patch_size,
        image_mean=image_mean,
        image_std=image_std,
        apply_color_jitter=False,
        use_masks=args.use_masks,
    )
    test_ds = MaskedPairDataset(
        args.data_root,
        "test",
        patch_size=patch_size,
        image_mean=image_mean,
        image_std=image_std,
        apply_color_jitter=False,
        use_masks=args.use_masks,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_pad_pairs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_pad_pairs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_pad_pairs,
    )

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Logging to: {log_path}")
    logger.info(f"Device: {device} (cuda available: {torch.cuda.is_available()})")
    logger.info(f"Backbone: {args.backbone} | checkpoint/model source: {args.checkpoint or 'hub'}")
    logger.info(f"Patch size: {patch_size} | Num classes: {num_classes}")
    logger.info(
        "Hyperparameters: batch_size=%d, epochs=%d, lr_head=%.2e, lr_backbone=%.2e, "
        "weight_decay=%.2e, freeze_backbone_epochs=%d, num_workers=%d, color_jitter=%s, "
        "extra_augment=%s, use_masks=%s, max_rotation=%.1f, max_translate=%.3f, max_scale=%.3f, vflip_prob=%.2f, "
        "train_repeat=%d, pair_composition=%s, class_weighting=%s, class_weights=%s, loss=%s, focal_gamma=%.2f, "
        "focal_alpha=%s, unfreeze_last_blocks=%d",
        args.batch_size,
        args.epochs,
        args.lr_head,
        args.lr_backbone,
        args.weight_decay,
        args.freeze_backbone_epochs,
        args.num_workers,
        args.apply_color_jitter,
        args.extra_augment,
        args.use_masks,
        args.max_rotation,
        args.max_translate,
        args.max_scale,
        args.vflip_prob,
        args.train_repeat,
        args.pair_composition,
        args.class_weighting,
        args.class_weights,
        args.loss,
        args.focal_gamma,
        args.focal_alpha,
        args.unfreeze_last_blocks,
    )
    logger.info(
        "Dataset sizes (pairs): train=%d, val=%d, test=%d | augmentation: flip p=0.5, color jitter=%s, "
        "extra_augment=%s, train_repeat=%d",
        len(train_ds),
        len(val_ds),
        len(test_ds),
        args.apply_color_jitter,
        args.extra_augment,
        args.train_repeat,
    )
    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    train_counts = Counter(s.label for s in train_ds.samples)
    val_counts = Counter(s.label for s in val_ds.samples)
    test_counts = Counter(s.label for s in test_ds.samples)
    logger.info(
        "Class distribution (train): %s",
        ", ".join(f"{idx_to_class[i]}:{c}" for i, c in sorted(train_counts.items())),
    )
    logger.info(
        "Class distribution (val): %s",
        ", ".join(f"{idx_to_class[i]}:{c}" for i, c in sorted(val_counts.items())),
    )
    logger.info(
        "Class distribution (test): %s",
        ", ".join(f"{idx_to_class[i]}:{c}" for i, c in sorted(test_counts.items())),
    )
    class_weights_t = None
    if args.class_weights:
        weights = [float(w.strip()) for w in args.class_weights.split(",") if w.strip()]
        if len(weights) != num_classes:
            raise ValueError(f"--class-weights expected {num_classes} values, got {len(weights)}")
        class_weights_t = torch.tensor(weights, dtype=torch.float, device=device)
        logger.info(
            "Class weights (manual): %s",
            ", ".join(f"{idx_to_class[i]}:{w:.4f}" for i, w in enumerate(weights)),
        )
    elif args.class_weighting != "none":
        weights = []
        for i in range(num_classes):
            count = train_counts.get(i, 1)
            if args.class_weighting == "inverse_freq":
                weights.append(1.0 / max(1, count))
            elif args.class_weighting == "balanced":
                weights.append(len(train_ds) / (num_classes * max(1, count)))
        class_weights_t = torch.tensor(weights, dtype=torch.float, device=device)
        logger.info(
            "Class weights (%s): %s",
            args.class_weighting,
            ", ".join(f"{idx_to_class[i]}:{w:.4f}" for i, w in enumerate(weights)),
        )

    focal_alpha_t = None
    if args.focal_alpha:
        focal_alpha = [float(w.strip()) for w in args.focal_alpha.split(",") if w.strip()]
        if len(focal_alpha) != num_classes:
            raise ValueError(f"--focal-alpha expected {num_classes} values, got {len(focal_alpha)}")
        focal_alpha_t = torch.tensor(focal_alpha, dtype=torch.float, device=device)
        logger.info(
            "Focal alpha: %s",
            ", ".join(f"{idx_to_class[i]}:{w:.4f}" for i, w in enumerate(focal_alpha)),
        )

    model = DinoPairedClassifier(
        backbone=backbone,
        num_classes=num_classes,
        composition=args.pair_composition,
    )
    model.to(device)

    head_params = list(model.head.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("head")]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": args.lr_head},
            {"params": backbone_params, "lr": args.lr_backbone},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(enabled=amp_enabled)
    state = TrainState()
    if args.loss == "ce":
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_t)
    else:
        criterion = FocalLoss(weight=class_weights_t, gamma=args.focal_gamma, alpha=focal_alpha_t)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head_params_count = sum(p.numel() for p in model.head.parameters())
    backbone_params_count = total_params - head_params_count
    logger.info(
        "Parameters: total=%d | trainable=%d | backbone=%d | head=%d",
        total_params,
        trainable_params,
        backbone_params_count,
        head_params_count,
    )

    # Persist run hyperparameters/metadata for later inspection.
    def _ser(o):
        if isinstance(o, Path):
            return str(o)
        return o

    hparams = {
        "args": {k: _ser(v) for k, v in vars(args).items()},
        "backbone": args.backbone,
        "checkpoint": _ser(args.checkpoint),
        "patch_size": patch_size,
        "num_classes": num_classes,
        "split_sizes": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "class_counts": {
            "train": {idx_to_class[i]: train_counts.get(i, 0) for i in range(num_classes)},
            "val": {idx_to_class[i]: val_counts.get(i, 0) for i in range(num_classes)},
            "test": {idx_to_class[i]: test_counts.get(i, 0) for i in range(num_classes)},
        },
        "class_to_idx": train_ds.class_to_idx,
        "image_mean": image_mean,
        "image_std": image_std,
        "parameters": {
            "total": total_params,
            "trainable": trainable_params,
            "backbone": backbone_params_count,
            "head": head_params_count,
        },
        "timestamp": timestamp,
    }
    with (run_dir / "hparams.json").open("w") as f:
        json.dump(hparams, f, indent=2, sort_keys=True)

    best_path = run_dir / "best.pt"

    for epoch in range(args.epochs):
        state.epoch = epoch
        freeze_bb = epoch < args.freeze_backbone_epochs

        model.train()
        set_backbone_trainable(model.backbone, freeze_bb, args.unfreeze_last_blocks)

        total_loss = total_acc = total_count = 0
        for img_b, mask_b, img_a, mask_a, labels in train_loader:
            img_b = img_b.to(device, non_blocking=True)
            mask_b = mask_b.to(device, non_blocking=True)
            img_a = img_a.to(device, non_blocking=True)
            mask_a = mask_a.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=amp_device_type, enabled=amp_enabled):
                logits = model(img_b, mask_b, img_a, mask_a)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            total_acc += (preds == labels).float().sum().item()
            total_count += labels.size(0)

        train_loss = total_loss / total_count
        train_acc = total_acc / total_count

        model.eval()
        val_loss = val_acc = val_count = 0
        with torch.no_grad():
            for img_b, mask_b, img_a, mask_a, labels in val_loader:
                img_b = img_b.to(device, non_blocking=True)
                mask_b = mask_b.to(device, non_blocking=True)
                img_a = img_a.to(device, non_blocking=True)
                mask_a = mask_a.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(img_b, mask_b, img_a, mask_a)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                val_acc += (preds == labels).float().sum().item()
                val_count += labels.size(0)

        val_loss /= max(1, val_count)
        val_acc /= max(1, val_count)

        logger.info(
            "Epoch %d/%d train_loss %.4f acc %.4f | val_loss %.4f acc %.4f | freeze_backbone=%s",
            epoch + 1,
            args.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            freeze_bb,
        )

        if val_acc > state.best_val_acc:
            state.best_val_acc = val_acc
            save_checkpoint(best_path, model, optimizer, scaler, state, train_ds.class_to_idx)

    # Save final state as last checkpoint
    save_checkpoint(run_dir / "last.pt", model, optimizer, scaler, state, train_ds.class_to_idx)

    # Load best checkpoint before evaluating on test
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(best_ckpt["model"])
        optimizer.load_state_dict(best_ckpt["optimizer"])
        scaler.load_state_dict(best_ckpt["scaler"])
        best_state = best_ckpt.get("state", {})
        best_epoch = best_state.get("epoch", None)
        best_val_acc = best_state.get("best_val_acc", None)
        if best_epoch is not None and best_val_acc is not None:
            logger.info("Loaded best checkpoint (epoch %d, val_acc %.4f)", best_epoch + 1, best_val_acc)
        else:
            logger.info("Loaded best checkpoint")
    else:
        logger.warning("Best checkpoint not found at %s; using last model state", best_path)

    # Test with best checkpoint
    model.eval()
    test_loss = test_acc = test_count = 0
    test_conf = torch.zeros((num_classes, num_classes), dtype=torch.long)
    with torch.no_grad():
        for img_b, mask_b, img_a, mask_a, labels in test_loader:
            img_b = img_b.to(device, non_blocking=True)
            mask_b = mask_b.to(device, non_blocking=True)
            img_a = img_a.to(device, non_blocking=True)
            mask_a = mask_a.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(img_b, mask_b, img_a, mask_a)
            loss = criterion(logits, labels)
            test_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            test_acc += (preds == labels).float().sum().item()
            test_count += labels.size(0)
            labels_cpu = labels.detach().cpu()
            preds_cpu = preds.detach().cpu()
            test_conf.index_put_((labels_cpu, preds_cpu), torch.ones_like(labels_cpu, dtype=torch.long), accumulate=True)
    test_loss /= max(1, test_count)
    test_acc /= max(1, test_count)
    logger.info("Test loss %.4f acc %.4f", test_loss, test_acc)
    logger.info("Test confusion matrix (rows=true, cols=pred):\n%s", format_confusion_table(test_conf, idx_to_class))


if __name__ == "__main__":
    main()
