#!/usr/bin/env python3
"""Distributed k-NN classification with DINOv3 backbones."""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from torchvision.transforms import v2 as T
from datetime import datetime, timedelta
from dinov3.data.transforms import make_classification_eval_transform

# -----------------------
# Limit CPU threading
# -----------------------
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")

# NCCL settings (for multi-GPU stability)
os.environ.pop("NCCL_BLOCKING_WAIT", None)
os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_NCCL_DEBUG", "WARN")
os.environ.setdefault("TORCH_NCCL_TIMEOUT", "1800")

REPO_DIR = "/home/mila/e/echchabo/projects/SDG6-Tracker/src/dinov3"
RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs"
DEFAULT_MASTER_ADDR = "127.0.0.1"
DEFAULT_MASTER_PORT = "29500"

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


def log_msg(message: str, *, rank: int | None = None) -> None:
    """Timestamped logging helper with optional rank prefix."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = f"[{ts}]"
    if rank is not None:
        prefix += f"[GPU {rank}]"
    print(f"{prefix} {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed DINOv3 k-NN evaluation.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory containing train/val/test splits.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to the DINOv3 checkpoint (.pth).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="DINOv3 hub backbone name (e.g., dinov3_vitl16).",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="1,5,10,20,50,100",
        help="Comma-separated list of k values.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=None,
        help="Directory to save/load embeddings (default: DATA_DIR/_embeddings).",
    )
    parser.add_argument(
        "--weights-type",
        type=str,
        choices=["auto", "lvd", "sat"],
        default="auto",
        help="Which normalization stats to use; 'auto' infers from weights filename.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=500,
        help="Emit per-GPU progress every N samples; 0 disables progress logs.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="Batch size for embedding extraction (higher uses more GPU memory).",
    )
    return parser.parse_args()


def resolve_weights_type(weights: Path, override: str) -> str:
    if override != "auto":
        return override
    name = weights.name.lower()
    if "lvd" in name:
        return "lvd"
    if "sat" in name:
        return "sat"
    return "sat"


def make_transform(weights_type: str, resize_size: int = 256) -> T.Compose:
    mean, std = STATS[weights_type]
    # Match upstream eval: resize shorter side then center crop
    return make_classification_eval_transform(
        resize_size=resize_size,
        crop_size=224,
        mean=mean,
        std=std,
    )


def discover_class_names(data_dir: Path) -> list[str]:
    classes = set()
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.is_dir():
            continue
        for child in split_dir.iterdir():
            if child.is_dir():
                classes.add(child.name)
    if not classes:
        raise FileNotFoundError(f"No class folders found under {data_dir}")
    return sorted(classes)


def extract_cls(model: torch.nn.Module, img_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad(), torch.amp.autocast(
        device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
    ):
        img_tensor = img_tensor.to(device, non_blocking=True)
        out = model.forward_features(img_tensor)

        if isinstance(out, list):
            out = out[0]

        if "x_norm_clstoken" in out:
            cls = out["x_norm_clstoken"]
        elif "x_cls" in out:
            cls = out["x_cls"]
        elif "x_prenorm" in out:
            cls = out["x_prenorm"][:, 0]
        else:
            raise KeyError(f"No CLS token in forward_features output: {out.keys()}")

        # L2 normalize like upstream k-NN eval
        cls = torch.nn.functional.normalize(cls, dim=-1)
        return cls.cpu()


def build_embeddings(
    split_dir: Path,
    model: torch.nn.Module,
    transform: T.Compose,
    device: torch.device,
    class_to_idx: dict[str, int],
    log_interval: int,
    embed_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    files: list[tuple[Path, int]] = []
    for class_name, label_int in class_to_idx.items():
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            continue
        for fname in os.listdir(class_dir):
            files.append((class_dir / fname, label_int))

    random.shuffle(files)
    total = len(files)
    log_msg(f"{split_dir.name}: assigned {total} files", rank=0)
    X_local: list[np.ndarray] = []
    y_local: list[int] = []
    start_time = time.time()
    batch_imgs: list[torch.Tensor] = []
    batch_labels: list[int] = []

    for idx, (fpath, label_int) in enumerate(files):
        try:
            img = Image.open(fpath).convert("RGB")
        except Exception:
            log_msg(f"Skipping corrupted file {fpath}", rank=0)
            continue

        img_t = transform(img)
        batch_imgs.append(img_t)
        batch_labels.append(label_int)

        flush = (len(batch_imgs) == embed_batch_size) or (idx == total - 1)
        if flush:
            batch_tensor = torch.stack(batch_imgs, dim=0)
            feats = extract_cls(model, batch_tensor, device)
            # feats shape: (B, dim)
            if feats.ndim == 1:
                feats = feats[None, :]
            X_local.extend([feat.numpy() for feat in feats])
            y_local.extend(batch_labels)
            batch_imgs.clear()
            batch_labels.clear()

        if log_interval and (idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0.0
            remaining = total - (idx + 1)
            eta = remaining / rate if rate > 0 else float("inf")
            log_msg(
                f"{split_dir.name}: processed {idx + 1}/{total} "
                f"({(idx + 1) / total * 100:.1f}%) | "
                f"elapsed {elapsed:.1f}s | "
                f"ETA {eta:.1f}s",
                rank=0,
            )

    elapsed = time.time() - start_time
    log_msg(
        f"{split_dir.name}: finished {len(X_local)}/{total} samples "
        f"in {elapsed:.1f}s",
        rank=0,
    )

    if len(X_local) == 0:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    return np.array(X_local), np.array(y_local, dtype=np.int64)


def run_single(args: argparse.Namespace) -> None:
    device = torch.device("cuda:0")
    weights_type = resolve_weights_type(args.weights, args.weights_type)
    transform = make_transform(weights_type)

    mean, std = STATS[weights_type]
    log_msg(f"Loading model {args.model_name}", rank=0)
    log_msg(f"Using weights: {args.weights}", rank=0)
    log_msg(f"Normalization ({weights_type}): mean={mean}, std={std}", rank=0)

    model = torch.hub.load(
        REPO_DIR,
        args.model_name,
        source="local",
        trust_repo=True,
        weights=str(args.weights),
    )
    model = model.to(device).eval().half()

    splits = ["train", "val", "test"]
    Xs: dict[str, np.ndarray] = {}
    ys: dict[str, np.ndarray] = {}

    ckpt_dir = args.ckpt_dir
    if not ckpt_dir.exists():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir = ckpt_dir / "confusion"
    confusion_dir.mkdir(parents=True, exist_ok=True)
    class_names_path = confusion_dir / "class_names.txt"
    if not class_names_path.exists():
        class_names_path.write_text("\n".join(args.class_names))

    class_to_idx = {name: idx for idx, name in enumerate(args.class_names)}

    for split in splits:
        split_dir = args.data_dir / split
        X_path = ckpt_dir / f"X_{split}.npy"
        y_path = ckpt_dir / f"y_{split}.npy"

        use_ckpt = False
        if X_path.exists() and y_path.exists():
            log_msg(f"Found cached embeddings for {split} in {ckpt_dir}. Loading...", rank=0)
            Xs[split] = np.load(X_path, mmap_mode="r")
            ys[split] = np.load(y_path, mmap_mode="r")
            use_ckpt = True

        if use_ckpt:
            continue

        log_msg(f"Computing embeddings for: {split}", rank=0)

        X_local, y_local = build_embeddings(
            split_dir,
            model,
            transform,
            device=device,
            class_to_idx=class_to_idx,
            log_interval=args.log_interval,
            embed_batch_size=args.embed_batch_size,
        )

        Xs[split] = X_local
        ys[split] = y_local

        log_msg(
            f"{split}: {Xs[split].shape[0]} samples, feat_dim={Xs[split].shape[1]}",
            rank=0,
        )

        np.save(X_path, Xs[split])
        np.save(y_path, ys[split])

    def save_confusion(split_name: str, preds: np.ndarray, targets: np.ndarray, k_val: int) -> Path:
        cm = confusion_matrix(targets, preds, labels=list(range(len(args.class_names))))
        report = classification_report(
            targets,
            preds,
            labels=list(range(len(args.class_names))),
            target_names=args.class_names,
            zero_division=0,
        )
        out_path = confusion_dir / f"confusion_{split_name}_k{k_val}.txt"
        with open(out_path, "w") as f:
            f.write(f"Split: {split_name}, k={k_val}\n")
            f.write("Confusion matrix (rows=true, cols=pred):\n")
            for row in cm:
                f.write(" ".join(str(int(x)) for x in row) + "\n")
            f.write("\nClassification report:\n")
            f.write(report)
            f.write("\n")
        return out_path

    log_msg("==============================", rank=0)
    log_msg("   DINOv3 k-NN (cosine) Eval  ", rank=0)
    log_msg("==============================", rank=0)

    Xtr = Xs["train"]
    ytr = ys["train"]
    Xval = Xs["val"]
    yval = ys["val"]
    Xte = Xs["test"]
    yte = ys["test"]

    log_msg(f"Train size: {len(ytr)}, Val size: {len(yval)}, Test size: {len(yte)}", rank=0)

    k_list = [int(k) for k in args.k_values.split(",") if k.strip()]
    results: list[tuple[int, float, float]] = []

    for k in k_list:
        log_msg(f"Training k-NN with k = {k} ...", rank=0)
        knn = KNeighborsClassifier(
            n_neighbors=k,
            metric="cosine",
            n_jobs=8,
        )
        knn.fit(Xtr, ytr)
        val_preds = knn.predict(Xval)
        test_preds = knn.predict(Xte)
        val_acc = float(np.mean(val_preds == yval))
        test_acc = float(np.mean(test_preds == yte))

        log_msg(f"  -> Val:  {val_acc*100:.2f}% | Test: {test_acc*100:.2f}%", rank=0)
        val_cm_path = save_confusion("val", val_preds, yval, k)
        test_cm_path = save_confusion("test", test_preds, yte, k)
        log_msg(f"  -> Confusion saved: val={val_cm_path.name}, test={test_cm_path.name}", rank=0)
        results.append((k, val_acc, test_acc))

    log_msg("==== Summary (cosine k-NN) ====", rank=0)
    log_msg("   k    |   Val Acc   |  Test Acc", rank=0)
    log_msg("---------------------------------", rank=0)
    for k, v, t in results:
        log_msg(f"{k:6d} | {v*100:9.2f}% | {t*100:9.2f}%", rank=0)
    log_msg("================================", rank=0)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for k-NN evaluation.")

    device = torch.device("cuda:0")

    args.data_dir = args.data_dir.resolve()
    args.weights = args.weights.resolve()
    dataset_name = args.data_dir.name
    weight_id = args.weights.stem
    default_ckpt = RUNS_ROOT / "dinov3-knn-cache" / dataset_name / f"{args.model_name}-{weight_id}"
    args.ckpt_dir = (args.ckpt_dir or default_ckpt).resolve()
    args.class_names = discover_class_names(args.data_dir)
    args.weights_type = resolve_weights_type(args.weights, args.weights_type)

    os.environ.setdefault("MASTER_ADDR", DEFAULT_MASTER_ADDR)
    os.environ.setdefault("MASTER_PORT", DEFAULT_MASTER_PORT)

    log_msg("Using single GPU: cuda:0")
    log_msg(f"Data directory: {args.data_dir}")
    run_single(args)


if __name__ == "__main__":
    main()
